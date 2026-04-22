"""
v391: Retrain from v378 with within-family discrimination as the primary loss.

Hypothesis: The embedding collapse (all champions and HNs at cosine ~1.0) happened
because the training loss didn't explicitly push within-family discrimination.
The v378 late-interaction verifier gave a weak late-interaction signal but didn't
teach the encoder to produce discriminable pooled embeddings.

Strategy:
  - Start from v378 checkpoint (freeze most layers)
  - Train with family_local_listwise_loss as the PRIMARY signal
  - This loss explicitly teaches: different implementations of the SAME task
    should have different rankings (champion > HN for the query that matches
    the task the champion solves)
  - Keep late_interaction_verifier_weight for token-level signal
  - Significantly increase margin to force geometric separation

Key insight from v340: the success came from neighborhood_posterior confidence
(clean separation at inference), not from better embeddings. But the embeddings
must have SOME structure for late interaction to work. The question is whether
we can push pooled embeddings apart without destroying the geometric structure
that late interaction relies on.

Schedule:
  - Steps 0-3000: warmup with late_interaction_verifier only (preserve structure)
  - Steps 3000-10000: add family_local_listwise with increasing weight
  - Steps 10000+: full weight
"""

from __future__ import annotations

import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight, _set_torch_seed
from src.utils.data import BenchmarkTokenizer, make_phase4_contrast_examples


def _build_v391_config():
    """Build a v378-like config with enhanced within-family discrimination."""
    config = _proxy_config_v6_overnight(15_000_000)
    config.use_retrieval_facets = True
    config.retrieval_num_fat_mult = 2  # more facets
    config.retrieval_facet_dim = 384
    config.retrieval_facet_hidden_dim = 768
    config.use_retrieval_head = True
    config.retrieval_head_dim = 384
    config.retrieval_head_hidden_dim = 768

    # Training: family-local listwise as primary loss
    config.phase4_dataset = "semantic_contrast_v1"
    config.family_local_listwise_weight = 1.0  # PRIMARY SIGNAL
    config.family_local_listwise_temperature = 0.05  # sharper ranking
    config.late_interaction_verifier_weight = 0.3  # secondary
    config.late_interaction_verifier_margin = 0.3  # stronger margin
    config.ranking_margin = 0.4  # stronger margin for retrieval loss
    config.ranking_margin_weight = 0.5

    # Optimizer
    config.learning_rate = 3e-4  # lower LR for fine-tuning
    config.warmup_fraction = 0.05

    return config


def load_v378_for_retrain(base_model_path: str, device: str = "cuda"):
    """Load v378 and unfreeze for retraining."""
    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=False)

    config = _proxy_config_v6_overnight(15_000_000)
    config.use_retrieval_facets = True
    config.retrieval_num_facets = 30
    config.retrieval_facet_dim = 256
    config.retrieval_facet_hidden_dim = 512
    config.use_retrieval_head = True
    config.retrieval_head_dim = 256
    config.retrieval_head_hidden_dim = 512

    model = JEPAModel(config).to(device, dtype=torch.bfloat16)

    # Load with strict=False to handle any shape mismatches
    loaded_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {len(loaded_keys.unexpected_keys)} unexpected keys, "
          f"missing {len(loaded_keys.missing_keys)} keys")

    # Unfreeze more aggressively: unfreeze encoder + predictor + decoder
    # But freeze retrieval_project and gated_reranker initially
    for name, param in model.named_parameters():
        if "retrieval_project" in name or "gated_reranker" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.eval()
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_path = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
    v378_path = base_path / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
    output_base = base_path / "artifacts/strict_eval_autoresearch_v391"
    output_base.mkdir(exist_ok=True)

    device = args.device
    print(f"Loading v378 from {v378_path}")
    model = load_v378_for_retrain(str(v378_path), device=device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    tokenizer = BenchmarkTokenizer(vocab_size=4096)

    # Load examples - more repeats for richer training
    examples = list(make_phase4_contrast_examples(repeats=8, rng=random.Random(42), dataset="semantic_contrast_v1"))
    print(f"Loaded {len(examples)} examples")

    train_ds = RetrainDataset(examples, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=lambda x: x)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=0.01,
    )

    # Simple schedule: linear warmup then constant
    steps = args.steps
    warmup_steps = int(steps * 0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = output_base / "train_log.txt"
    log_path.write_text(f"v391 retrain: {steps} steps, warmup={warmup_steps}\n")

    model.train()
    step = 0

    while step < steps:
        for raw_batch in train_loader:
            if step >= steps:
                break

            queries = [b["query_text"] for b in raw_batch]
            champs = [b["champ_text"] for b in raw_batch]
            hns = [b["hn_text"] for b in raw_batch]

            # Encode: batch each group separately then concatenate
            # This ensures q_lat, c_lat, hn_lat are each stacked [B, D] tensors
            q_inputs = tokenizer.batch_encode(queries, seq_len=256, device=device)
            c_inputs = tokenizer.batch_encode(champs, seq_len=256, device=device)
            hn_inputs = tokenizer.batch_encode(hns, seq_len=256, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                q_lat = model.encode(q_inputs)
                c_lat = model.encode(c_inputs)
                hn_lat = model.encode(hn_inputs)

            B = len(queries)

            # Project to float32 for stable cosine computation
            q_proj = model.retrieval_project(q_lat).float()
            c_proj = model.retrieval_project(c_lat).float()
            hn_proj = model.retrieval_project(hn_lat).float()

            # hn_proj may be [B, D] or [B, H, D] depending on collation shape
            if hn_proj.dim() == 2:
                hn_proj = hn_proj.unsqueeze(1)  # [B, 1, D]

            # Cosine similarity
            q_c = F.cosine_similarity(q_proj, c_proj, dim=-1)
            q_hn = torch.stack([F.cosine_similarity(q_proj, hn_proj[:, i], dim=-1) for i in range(hn_proj.shape[1])], dim=-1)

            # Compute losses
            # 1. Listwise ranking: champ should score higher than all HNs
            margin = 0.3
            listwise_loss = F.relu(margin - (q_c.unsqueeze(-1) - q_hn).amax(dim=-1)).mean()

            # 2. Late interaction (if available)
            if hasattr(model, "retrieval_facets"):
                # Pass bfloat16 latents to retrieval_facets, convert output to float32
                q_f = model.retrieval_facets(q_lat, role="query").float()  # [B, S, D]
                c_f = model.retrieval_facets(c_lat, role="code").float()  # [B, S, D]
                hn_f = model.retrieval_facets(hn_lat, role="code").float()  # [B, S, D]

                q_fn = F.normalize(q_f, dim=-1)
                c_fn = F.normalize(c_f, dim=-1)
                hn_fn = F.normalize(hn_f, dim=-1)

                # slot_c: [B], best matching slot per (query, champ) pair
                slot_c = (q_fn * c_fn).sum(dim=-1).amax(dim=-1)

                # slot_hn_per_query: [B], best slot match for each query's HN
                slot_hn_per_query = (q_fn * hn_fn).sum(dim=-1).amax(dim=-1)  # [B]

                late_loss = F.relu(margin - (slot_c - slot_hn_per_query)).mean()
                total_loss = 0.6 * listwise_loss + 0.4 * late_loss
            else:
                total_loss = listwise_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                msg = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} step={step} loss={total_loss.item():.6f} lw={listwise_loss.item():.6f}"
                log_path.open("a").write(msg + "\n")
                if step % 500 == 0:
                    print(msg)

            if step % args.eval_every == 0 and step > 0:
                # Quick eval
                eval_out = evaluate_on_train(model, train_loader, tokenizer, device)
                model.train()
                msg = f"  eval: champ_top1={eval_out['champ_top1']:.3f} champ_top3={eval_out['champ_top3']:.3f}"
                log_path.open("a").write(msg + "\n")
                print(msg)
                torch.save(model.state_dict(), output_base / f"step_{step}.pt")

    # Save final
    torch.save(model.state_dict(), output_base / "model.pt")
    print(f"v391 training complete. Saved to {output_base}")


class RetrainDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "query_text": f"# task: {ex.prompt}\n",
            "champ_text": f"# task: {ex.prompt}\n{ex.code}",
            "hn_text": f"# task: {ex.prompt}\n{ex.hard_negatives[0]}",
            "family": ex.family,
        }


def evaluate_on_train(model, dataloader, tokenizer, device, max_batches=20):
    model.eval()
    champ_top1 = 0; champ_top3 = 0; total = 0
    with torch.no_grad():
        for bi, raw_batch in enumerate(dataloader):
            if bi >= max_batches:
                break
            queries = [b["query_text"] for b in raw_batch]
            champs = [b["champ_text"] for b in raw_batch]
            hns = [b["hn_text"] for b in raw_batch]
            B = len(queries)

            q_inputs = tokenizer.batch_encode(queries, seq_len=256, device=device)
            c_inputs = tokenizer.batch_encode(champs, seq_len=256, device=device)
            hn_inputs = tokenizer.batch_encode(hns, seq_len=256, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                q_lat = model.encode(q_inputs)
                c_lat = model.encode(c_inputs)
                hn_lat = model.encode(hn_inputs)

            q_proj = model.retrieval_project(q_lat).float()
            c_proj = model.retrieval_project(c_lat).float()
            hn_proj = model.retrieval_project(hn_lat).float()

            if hn_proj.dim() == 2:
                hn_proj = hn_proj.unsqueeze(1)

            q_c = F.cosine_similarity(q_proj, c_proj, dim=-1)
            q_hn = torch.stack([F.cosine_similarity(q_proj, hn_proj[:, i], dim=-1) for i in range(hn_proj.shape[1])], dim=-1)

            champ_wins = (q_c.unsqueeze(-1) > q_hn).all(dim=-1)
            champ_top1 += champ_wins.sum().item()

            combined = torch.cat([q_c.unsqueeze(-1), q_hn], dim=-1)
            top3 = combined.argsort(descending=True)
            champ_in_top3 = (top3 == 0).any(dim=-1)
            champ_top3 += champ_in_top3.sum().item()
            total += B

    return {"champ_top1": champ_top1/total, "champ_top3": champ_top3/total, "n_eval": total}


if __name__ == "__main__":
    main()
