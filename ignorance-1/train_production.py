from __future__ import annotations
import torch
import torch.nn.functional as F
import time
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.jepa import JEPAConfig, JEPAModel, approximate_model_params
from src.losses.alignment import ignorance_penalty, paired_alignment_loss
from src.utils.data import SimpleTokenizer, make_text_code_pairs, sample_ood_queries
from src.training.phase4 import _proxy_config, _scaled_training_hparams
from src.losses.sigreg import sigreg_loss, isotropic_score, collapse_detected

class LatentBuffer:
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

class AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

def train_production(config_path: str, size: int, output_path: str, device: str):
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    
    config = AttrDict(full_config.get("phase4", {}))
    proxy_recipe = config.proxy_recipe
    microbatch_size = max(1, min(getattr(config, "microbatch_size", 1), config.batch_size))
    ood_weight = float(getattr(config, "ood_weight", 0.2))
    clf_weight = float(getattr(config, "clf_weight", 0.25))
    
    model_config = _proxy_config(size, proxy_recipe)
    scaled_steps, scaled_lr, step_mult, lr_div = _scaled_training_hparams(config, size)
    
    print(f"Training production model: {size:,} params (proxy: {approximate_model_params(model_config):,})")
    print(f"Recipe: {proxy_recipe}")
    print(f"Hyperparams: steps={scaled_steps}, lr={scaled_lr:.8f} (step_mult={step_mult:.2f}, lr_div={lr_div:.2f})")
    print(f"Batching: batch_size={config.batch_size}, microbatch_size={microbatch_size}")
    print(f"Aux losses: ood_weight={ood_weight:.2f}, clf_weight={clf_weight:.2f}")
    print(f"Device: {device}")
    
    tokenizer = SimpleTokenizer(vocab_size=4096)
    import bitsandbytes as bnb
    model = JEPAModel(model_config).to(device).to(torch.bfloat16)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=scaled_lr)
    
    # Latent Buffer for stable regularization
    buffer = LatentBuffer(size=1024, dim=model_config.embed_dim, device=device)
    code_buffer = LatentBuffer(size=2048, dim=model_config.embed_dim, device=device)
    
    # We only use 1 split for production
    pairs = make_text_code_pairs(repeats=max(config.batch_size * 32, 512))
    
    model.train()
    pbar = tqdm(total=scaled_steps, desc="Training")
    
    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    start_time = time.time()
    
    for step in range(scaled_steps):
        # Sample batch
        batch_pairs = [pairs[(step * config.batch_size + offset) % len(pairs)] for offset in range(config.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        num_microbatches = 0

        for start in range(0, len(batch_pairs), microbatch_size):
            micro_pairs = batch_pairs[start : start + microbatch_size]
            texts = tokenizer.batch_encode([p[0] for p in micro_pairs], model_config.max_seq_len, device)
            codes = tokenizer.batch_encode([p[1] for p in micro_pairs], model_config.max_seq_len, device)
            ood = tokenizer.batch_encode(sample_ood_queries(len(micro_pairs)), model_config.max_seq_len, device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_text = model.encode(texts)
                z_code = model.encode(codes)
                z_ood = model.encode(ood)
                z_pred = model.predict(z_text, action_id=1)
                z_ood_pred = model.predict(z_ood, action_id=1)
                coding_logits = model.query_logits(z_text)
                ood_logits = model.query_logits(z_ood)

                pred_loss, _ = paired_alignment_loss(z_text, z_code, z_pred, negative_pool=code_buffer.get())
                code_candidates = torch.cat([z_code.detach(), code_buffer.get()], dim=0) if code_buffer.get().numel() else z_code.detach()
                ignorance_loss = ignorance_penalty(z_ood, code_candidates) + ignorance_penalty(z_ood_pred, code_candidates)
                clf_loss = F.binary_cross_entropy_with_logits(coding_logits, torch.ones_like(coding_logits))
                clf_loss = clf_loss + F.binary_cross_entropy_with_logits(ood_logits, torch.zeros_like(ood_logits))

                # Use buffer to get a broader estimate of the distribution.
                buffer.push(z_text)
                buffer.push(z_code)
                code_buffer.push(z_code)

                z_pool = buffer.get()
                if z_pool.shape[0] >= 128:
                    lambda_reg = 0.5
                    reg_loss = sigreg_loss(z_pool.unsqueeze(1), m=1024, lambda_reg=lambda_reg)
                    micro_loss = pred_loss + ood_weight * ignorance_loss + clf_weight * clf_loss + lambda_reg * reg_loss
                else:
                    micro_loss = pred_loss + ood_weight * ignorance_loss + clf_weight * clf_loss

            num_microbatches += 1
            loss_value += float(micro_loss.detach().cpu().item())
            (micro_loss / max((config.batch_size + microbatch_size - 1) // microbatch_size, 1)).backward()

        optimizer.step()
        loss = torch.tensor(loss_value / max(num_microbatches, 1), device=device)
        
        pbar.update(1)
        if step % 10 == 0:
            z_stat = buffer.get()
            iso = isotropic_score(z_stat) if z_stat.shape[0] > 4 else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.6f}", "iso": f"{iso:.2f}"})
        
        if (step + 1) % 500 == 0:
            # Save intermediate checkpoint
            torch.save(model.state_dict(), output_path + ".tmp")
            
    pbar.close()
    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0
    
    print(f"Training complete in {elapsed/60:.2f} minutes.")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    
    print(f"Saving final model to {output_path}...")
    torch.save(model.state_dict(), output_path)
    if Path(output_path + ".tmp").exists():
        Path(output_path + ".tmp").unlink()
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/overnight.yaml")
    parser.add_argument("--size", type=int, default=2700000000)
    parser.add_argument("--output", type=str, default="artifacts/ignorance_1_2.7b.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    train_production(args.config, args.size, args.output, args.device)
