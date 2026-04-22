#!/usr/bin/env python3
"""
V423: Differentiated Learning Rates for 2.7B Training

HYPOTHESIS: The universal 2.7B embedding collapse is caused by ALL model
parameters (encoder + heads) receiving the same learning rate. The encoder
at 2.7B scale has ~2.5B parameters vs heads with ~50M parameters. With
identical LR, encoder updates are too aggressive and destroy embedding
diversity (iso: 0.69 -> 0.57 while loss increases).

FIX: Give the encoder a 5x lower learning rate than the heads.
- Encoder params: lr = 1e-5
- Head params: lr = 5e-5
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch import nn
import bitsandbytes as bnb
from tqdm import tqdm

from src.models.jepa import JEPAModel, JEPAConfig
from src.training.phase4 import _proxy_config, _scaled_training_hparams
from src.utils.data import SimpleTokenizer
from src.losses.alignment import paired_alignment_loss, ignorance_penalty
from src.losses.sigreg import sigreg_loss, isotropic_score
from src.utils.data import make_text_code_pairs, sample_ood_queries


class LatentBuffer:
    """Inlined from train_production.py"""
    def __init__(self, size: int, dim: int, device: str):
        self.buffer = torch.zeros(size, dim, device=device)
        self.ptr = 0
        self.size = size
        self.is_full = False

    def push(self, x: torch.Tensor):
        x = x.detach()
        batch = x.shape[0]
        if self.ptr + batch <= self.size:
            self.buffer[self.ptr:self.ptr + batch] = x
            self.ptr += batch
        else:
            remaining = self.size - self.ptr
            self.buffer[self.ptr:] = x[:remaining]
            self.buffer[:batch - remaining] = x[remaining:]
            self.ptr = batch - remaining
            self.is_full = True
        if self.ptr >= self.size:
            self.ptr = 0
            self.is_full = True

    def get(self) -> torch.Tensor:
        return self.buffer if self.is_full else self.buffer[:self.ptr]


def build_param_groups(model: nn.Module, encoder_lr: float, head_lr: float):
    """
    Build AdamW8bit parameter groups with differentiated learning rates.
    Encoder gets encoder_lr, heads get head_lr.
    """
    encoder_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.split('.')
        if parts[0] == 'encoder':
            encoder_params.append(param)
        elif parts[0] in ('predictor', 'decoder', 'query_head', 'retrieval_project',
                           'retrieval_facets', 'confidence_head', 'ranking_head',
                           'action_embed', 'actionEmb'):
            head_params.append(param)
        else:
            # Fallback: check if param name contains encoder modules
            if any(enc in name for enc in ('token_embed', 'pos_embed', 'transformer',
                                           'final_ln', 'final_bn', 'proj')):
                encoder_params.append(param)
            else:
                head_params.append(param)
    
    total_enc = sum(p.numel() for p in encoder_params)
    total_head = sum(p.numel() for p in head_params)
    print(f"Param groups: encoder={len(encoder_params)} params ({total_enc/1e9:.2f}B) @ lr={encoder_lr:.2e}, "
          f"heads={len(head_params)} params ({total_head/1e6:.1f}M) @ lr={head_lr:.2e}")
    
    return [
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': head_params, 'lr': head_lr},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/overnight.yaml')
    parser.add_argument('--size', type=int, default=2700000000)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=711)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=5e-5)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    import yaml
    from src.utils.config import Phase4Config
    
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    phase4_dict = raw.get('phase4', raw)
    phase4_dict['sizes'] = [args.size]
    # Force v5_distinct proxy recipe (embed_dim=384) — critical for 2.7B compatibility
    phase4_dict['proxy_recipe'] = 'v5_distinct'
    phase4_dict['steps'] = 500
    phase4_dict['batch_size'] = 1
    phase4_dict['lr'] = 5e-5  # Required field; actual LR set via param groups
    config = Phase4Config(**phase4_dict)
    
    device = args.device
    proxy_recipe = config.proxy_recipe
    microbatch_size = 1  # Hardcoded for 2.7B single-GPU training
    ood_weight = float(getattr(config, 'ignorance_ood_weight', 0.2))
    clf_weight = float(getattr(config, 'classifier_weight', 0.25))
    
    model_config = _proxy_config(args.size, proxy_recipe)
    scaled_steps, scaled_lr, step_mult, lr_div = _scaled_training_hparams(config, args.size)
    
    encoder_lr = args.encoder_lr
    head_lr = args.head_lr
    
    print(f"=== V423: Differentiated LR Training ===")
    print(f"Size: {args.size:,} params")
    print(f"Recipe: {proxy_recipe}")
    print(f"Encoder LR: {encoder_lr:.2e}, Head LR: {head_lr:.2e}")
    print(f"Steps: {scaled_steps}, Base LR div: {lr_div:.2f}")
    print(f"Proxy config: embed_dim={model_config.embed_dim}, "
          f"enc_layers={model_config.encoder_layers}, pred_layers={model_config.predictor_layers}")
    print(f"Batch size: {config.batch_size}, microbatch: {microbatch_size}")
    print(f"OOD weight: {ood_weight}, clf_weight: {clf_weight}")
    
    tokenizer = SimpleTokenizer(vocab_size=4096)
    model = JEPAModel(model_config).to(device).to(torch.bfloat16)
    
    # Build differentiated param groups
    param_groups = build_param_groups(model, encoder_lr, head_lr)
    optimizer = bnb.optim.AdamW8bit(param_groups)
    
    buffer = LatentBuffer(size=1024, dim=model_config.embed_dim, device=device)
    code_buffer = LatentBuffer(size=2048, dim=model_config.embed_dim, device=device)
    
    pairs = make_text_code_pairs(repeats=max(config.batch_size * 32, 512))
    
    model.train()
    pbar = tqdm(total=scaled_steps, desc="Training")
    
    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    start_time = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
    end_time = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
    if start_time:
        start_time.record()
    
    for step in range(scaled_steps):
        batch_pairs = [pairs[(step * config.batch_size + offset) % len(pairs)]
                       for offset in range(config.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        num_microbatches = 0
        
        for start in range(0, len(batch_pairs), microbatch_size):
            micro_pairs = batch_pairs[start:start + microbatch_size]
            texts = tokenizer.batch_encode([p[0] for p in micro_pairs],
                                          model_config.max_seq_len, device)
            codes = tokenizer.batch_encode([p[1] for p in micro_pairs],
                                           model_config.max_seq_len, device)
            ood = tokenizer.batch_encode(sample_ood_queries(len(micro_pairs)),
                                        model_config.max_seq_len, device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_text = model.encode(texts)
                z_code = model.encode(codes)
                z_ood = model.encode(ood)
                z_pred = model.predict(z_text, action_id=1)
                z_ood_pred = model.predict(z_ood, action_id=1)
                coding_logits = model.query_logits(z_text)
                ood_logits = model.query_logits(z_ood)
                
                pred_loss, _ = paired_alignment_loss(
                    z_text, z_code, z_pred, negative_pool=code_buffer.get())
                code_candidates = (torch.cat([z_code.detach(), code_buffer.get()], dim=0)
                                   if code_buffer.get().numel() else z_code.detach())
                ignorance_loss = (ignorance_penalty(z_ood, code_candidates) +
                                ignorance_penalty(z_ood_pred, code_candidates))
                clf_loss = nn.functional.binary_cross_entropy_with_logits(
                    coding_logits, torch.ones_like(coding_logits))
                clf_loss = clf_loss + nn.functional.binary_cross_entropy_with_logits(
                    ood_logits, torch.zeros_like(ood_logits))
                
                buffer.push(z_text)
                buffer.push(z_code)
                code_buffer.push(z_code)
                
                z_pool = buffer.get()
                if z_pool.shape[0] >= 128:
                    lambda_reg = 0.5
                    reg_loss = sigreg_loss(z_pool.unsqueeze(1), m=1024, lambda_reg=lambda_reg)
                    micro_loss = (pred_loss + ood_weight * ignorance_loss +
                                 clf_weight * clf_loss + lambda_reg * reg_loss)
                else:
                    micro_loss = (pred_loss + ood_weight * ignorance_loss +
                                 clf_weight * clf_loss)
            
            num_microbatches += 1
            loss_value += float(micro_loss.detach().cpu().item())
            (micro_loss / max((config.batch_size + microbatch_size - 1) // microbatch_size, 1)).backward()
        
        optimizer.step()
        loss = torch.tensor(loss_value / max(num_microbatches, 1), device=device)
        
        pbar.update(1)
        if step % 10 == 0:
            z_stat = buffer.get()
            iso = isotropic_score(z_stat) if z_stat.shape[0] > 4 else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "iso": f"{iso:.2f}"})
        
        if (step + 1) % 500 == 0:
            torch.save(model.state_dict(), args.output + f".step{(step+1)}.tmp")
    
    pbar.close()
    
    if start_time and end_time:
        end_time.record()
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time) / 1000
    else:
        elapsed = 0
    
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0
    
    print(f"\nTraining complete in {elapsed/60:.2f} minutes.")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Saving final model to {args.output}...")
    torch.save(model.state_dict(), args.output)
    
    # Clean up .tmp files
    for f in Path(args.output).parent.glob(Path(args.output).name + ".step*.tmp"):
        f.unlink()
    
    print("Done.")


if __name__ == "__main__":
    main()
