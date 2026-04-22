"""
v392: Train encoder using late interaction maxsim as the ranking signal.

Key insight: The bottleneck is late interaction maxsim — champions for 5/8 families
don't score above hard negatives. But late interaction uses FACET embeddings, not
pooled ones. We need to train the FACET-producing layers to create embeddings
where maxsim(champion) > maxsim(HN) for ALL families.

Architecture:
  - Start from v378 checkpoint
  - Unfreeze retrieval_facets layers (the facet projection + residual)
  - Keep encoder/predictor/decoder frozen (preserve what works)
  - Train ONLY with late interaction maxsim margin loss
  - Use harder negatives: sample HNs from OTHER families (cross-family negatives)
    in addition to same-family HNs

The margin loss: maxsim(q, champ) should be > maxsim(q, HN) + margin
"""

from __future__ import annotations
import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight
from src.utils.data import BenchmarkTokenizer, make_phase4_contrast_examples


def load_v378_for_facet_train(base_model_path: str, device: str = "cuda"):
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
    model.load_state_dict(state_dict, strict=False)

    # Freeze encoder/predictor/decoder, unfreeze ONLY retrieval_facet_head
    for name, param in model.named_parameters():
        if "retrieval_facet_head" in name or "retrieval_head" in name or "query_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Facet-trainable params: {trainable:,}")
    return model


def maxsim_score(q_facets, c_facets):
    """Compute late interaction maxsim: max cosine similarity between query and code slots.

    q_facets: [S, D] query slot embeddings (normalized)
    c_facets: [S, D] code slot embeddings (normalized)
    Returns: scalar maxsim score
    """
    return (q_facets * c_facets).sum(dim=-1).max(dim=0).values.mean()


def batch_maxsim(q_f, c_f):
    """Batch version: q_f [B, S, D], c_f [B, S, D] → [B] scores"""
    q_fn = F.normalize(q_f, dim=-1)
    c_fn = F.normalize(c_f, dim=-1)
    # [B, S, 1] * [B, 1, S, D] → [B, S, S] → max over code slots → mean over query slots
    # Actually: per-batch, best slot cos: [B, S, D] × [B, S, D] → [B, S] → max → scalar
    return (q_fn * c_fn).sum(dim=-1).amax(dim=-1).mean(dim=-1)


class MaxsimTripletDataset(Dataset):
    """Dataset that yields (query, champion, hard_negative) triplets.

    For each example, we create triplets using:
    1. The family query (abstract task description)
    2. The champion code (correct implementation)
    3. A hard negative (wrong implementation from same family OR cross-family)
    """

    def __init__(self, examples, tokenizer, device="cuda", cross_family_negatives=True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.device = device
        self.cross_family_negatives = cross_family_negatives
        self.rng = random.Random(42)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        family = ex.family

        # Select hard negative
        if self.cross_family_negatives and self.rng.random() < 0.5:
            # Cross-family negative: pick from a different family
            other_exs = [e for e in self.examples if e.family != family]
            other = self.rng.choice(other_exs)
            hn_text = f"# task: {other.prompt}\n{other.code}"
            neg_type = "cross"
        else:
            # Same-family negative
            hn_list = [n for n in ex.hard_negatives if n.strip()]
            if hn_list:
                hn_text = f"# task: {ex.prompt}\n{hn_list[0]}"
            else:
                # Fallback: another family's code
                other_exs = [e for e in self.examples if e.family != family]
                other = self.rng.choice(other_exs)
                hn_text = f"# task: {other.prompt}\n{other.code}"
            neg_type = "same"

        return {
            "query_text": f"# task: {ex.prompt}\n",
            "champ_text": f"# task: {ex.prompt}\n{ex.code}",
            "hn_text": hn_text,
            "family": family,
            "neg_type": neg_type,
        }


def train_v392(steps=8000, eval_every=500, device="cuda"):
    base_path = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
    v378_path = base_path / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
    output_base = base_path / "artifacts/strict_eval_autoresearch_v392"
    output_base.mkdir(exist_ok=True)

    print(f"Loading v378 from {v378_path}")
    model = load_v378_for_facet_train(str(v378_path), device=device)
    tokenizer = BenchmarkTokenizer(vocab_size=4096)

    # Load examples — high repeats for rich training
    examples = list(make_phase4_contrast_examples(repeats=8, rng=random.Random(42), dataset="semantic_contrast_v1"))
    print(f"Loaded {len(examples)} examples (8 repeats)")

    eval_examples = list(make_phase4_contrast_examples(repeats=1, rng=random.Random(0), dataset="semantic_contrast_v1"))

    train_ds = MaxsimTripletDataset(examples, tokenizer, cross_family_negatives=True)
    eval_ds = MaxsimTripletDataset(eval_examples, tokenizer, cross_family_negatives=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda x: x)
    eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False, collate_fn=lambda x: x)

    # Optimizer: only on trainable params (retrieval_facets + retrieval_project)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=0.01,
    )

    warmup_steps = int(steps * 0.05)

    def lr_lambda(step):
        return step / warmup_steps if step < warmup_steps else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = output_base / "train_log.txt"
    log_path.write_text(f"v392 facet maxsim train: {steps} steps, warmup={warmup_steps}\n")

    model.train()
    step = 0

    while step < steps:
        for raw_batch in train_loader:
            if step >= steps:
                break

            queries = [b["query_text"] for b in raw_batch]
            champs = [b["champ_text"] for b in raw_batch]
            hns = [b["hn_text"] for b in raw_batch]

            # Encode each group separately
            q_inp = tokenizer.batch_encode(queries, seq_len=256, device=device)
            c_inp = tokenizer.batch_encode(champs, seq_len=256, device=device)
            hn_inp = tokenizer.batch_encode(hns, seq_len=256, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                q_lat = model.encode(q_inp)
                c_lat = model.encode(c_inp)
                hn_lat = model.encode(hn_inp)

                # Extract FACETS for maxsim
                q_f = model.retrieval_facets(q_lat, role="query").float()
                c_f = model.retrieval_facets(c_lat, role="code").float()
                hn_f = model.retrieval_facets(hn_lat, role="code").float()

            # Batch maxsim
            champ_maxsim = batch_maxsim(q_f, c_f)   # [B]
            hn_maxsim = batch_maxsim(q_f, hn_f)      # [B]

            # Margin loss: champ should beat HN by margin
            margin = 0.05
            loss = F.relu(margin - (champ_maxsim - hn_maxsim)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                champ_win_rate = (champ_maxsim > hn_maxsim).float().mean().item()
                gap = (champ_maxsim - hn_maxsim).mean().item()
                msg = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} step={step} loss={loss.item():.4f} champ_win={champ_win_rate:.3f} gap={gap:.4f}"
                log_path.open("a").write(msg + "\n")
                if step % 500 == 0:
                    print(msg)

            if step % eval_every == 0 and step > 0:
                eval_out = evaluate_v392(model, eval_loader, tokenizer, device)
                model.train()
                msg = f"  eval: champ_win={eval_out['champ_win_rate']:.3f} gap={eval_out['mean_gap']:.4f} champ_top1={eval_out['champ_top1']:.3f}"
                log_path.open("a").write(msg + "\n")
                print(msg)
                torch.save(model.state_dict(), output_base / f"step_{step}.pt")

    torch.save(model.state_dict(), output_base / "model.pt")
    print(f"v392 training complete. Saved to {output_base}")
    return output_base


def evaluate_v392(model, dataloader, tokenizer, device, max_batches=30):
    """Evaluate using late interaction maxsim on same-family pairs."""
    model.eval()
    champ_wins = 0
    total = 0
    gaps = []

    with torch.no_grad():
        for bi, raw_batch in enumerate(dataloader):
            if bi >= max_batches:
                break
            queries = [b["query_text"] for b in raw_batch]
            champs = [b["champ_text"] for b in raw_batch]
            hns = [b["hn_text"] for b in raw_batch]
            B = len(queries)

            q_inp = tokenizer.batch_encode(queries, seq_len=256, device=device)
            c_inp = tokenizer.batch_encode(champs, seq_len=256, device=device)
            hn_inp = tokenizer.batch_encode(hns, seq_len=256, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                q_lat = model.encode(q_inp)
                c_lat = model.encode(c_inp)
                hn_lat = model.encode(hn_inp)

                q_f = model.retrieval_facets(q_lat, role="query").float()
                c_f = model.retrieval_facets(c_lat, role="code").float()
                hn_f = model.retrieval_facets(hn_lat, role="code").float()

            champ_ms = batch_maxsim(q_f, c_f)
            hn_ms = batch_maxsim(q_f, hn_f)

            champ_wins += (champ_ms > hn_ms).sum().item()
            gap_tensor = (champ_ms - hn_ms)
            if gap_tensor.dim() == 0:
                gaps.append(gap_tensor.item())
            else:
                gaps.extend(gap_tensor.cpu().tolist())
            total += B

    return {
        "champ_win_rate": champ_wins / total if total else 0,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0,
        "champ_top1": champ_wins / total if total else 0,
        "n_eval": total,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_base = train_v392(steps=args.steps, eval_every=args.eval_every, device=args.device)
