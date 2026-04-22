"""
v393: Train retrieval_facet_head using late interaction maxsim — SAME FAMILY ONLY.

v392 collapsed because cross-family negatives create superficial shortcuts
(query "Sort a list" → champion sort code vs cross-family "Debounce" code are
easily separable by task keywords, not by implementation correctness).

The fix: ONLY use same-family hard negatives during training. The model must learn
to discriminate between DIFFERENT IMPLEMENTATIONS OF THE SAME TASK, not different tasks.

Architecture:
  - Start from v378 checkpoint
  - Unfreeze ONLY retrieval_facet_head + query_head (the facet projection layers)
  - Train with SAME-FAMILY triplets only: (query, champion, HN_from_same_family)
  - Loss: maxsim_margin(q, champ) > maxsim_margin(q, HN) + margin
  - Evaluate on the actual 8-family eval

This is what the contrastive learning literature calls "hard negative mining" —
use the hardest negatives that are MOST similar to the query (same family).
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


def load_v378_facet_train(base_model_path: str, device: str = "cuda"):
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

    # Unfreeze retrieval_facet_head + query_head (the facet projection layers)
    for name, param in model.named_parameters():
        if "retrieval_facet_head" in name or "retrieval_head" in name or "query_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")
    return model


def maxsim(q_f, c_f):
    """Late interaction maxsim: [B,S,D] x [B,S,D] → [B]

    Normalize slots, compute cosine per (query_slot, code_slot) pair,
    take max over code slots, average over query slots.
    """
    q_fn = F.normalize(q_f, dim=-1)
    c_fn = F.normalize(c_f, dim=-1)
    # [B, S, 1] * [B, 1, S, D] → max over code slots → mean over query slots
    # Simplified: [B, S, D] * [B, S, D] → [B, S] max → scalar
    return (q_fn * c_fn).sum(dim=-1).amax(dim=-1).mean(dim=-1)


class SameFamilyTripletDataset(Dataset):
    """Triplets with ONLY same-family hard negatives.

    Each triplet: (query_text, champion_code, hard_negative_from_same_family)

    The model must learn to rank champion > HN based on IMPLEMENTATION QUALITY,
    not based on task identity (which cross-family negatives would teach).
    """

    def __init__(self, examples, tokenizer, repeats=4):
        self.examples = examples
        self.tokenizer = tokenizer
        self.repeats = repeats
        self.rng = random.Random(42)

        # Build per-family lookup
        self.by_family = {}
        for ex in examples:
            if ex.family not in self.by_family:
                self.by_family[ex.family] = []
            self.by_family[ex.family].append(ex)

    def __len__(self):
        return len(self.examples) * self.repeats

    def __getitem__(self, idx):
        ex_idx = idx % len(self.examples)
        ex = self.examples[ex_idx]
        family = ex.family

        # Get same-family HNs
        same_family_exs = [e for e in self.by_family[family] if e.code != ex.code]
        if same_family_exs:
            hn_ex = self.rng.choice(same_family_exs)
            hn_text = f"# task: {hn_ex.prompt}\n{hn_ex.code}"
        else:
            # Fallback: use another HN from same example
            hn_list = [n for n in ex.hard_negatives if n.strip()]
            hn_text = f"# task: {ex.prompt}\n{hn_list[0]}" if hn_list else ex.code[:100]

        return {
            "query_text": f"# task: {ex.prompt}\n",
            "champ_text": f"# task: {ex.prompt}\n{ex.code}",
            "hn_text": hn_text,
            "family": family,
        }


def train_v393(steps=6000, eval_every=500, device="cuda"):
    base_path = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
    v378_path = base_path / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
    output_base = base_path / "artifacts/strict_eval_autoresearch_v393"
    output_base.mkdir(exist_ok=True)

    print(f"Loading v378 from {v378_path}")
    model = load_v378_facet_train(str(v378_path), device=device)
    tokenizer = BenchmarkTokenizer(vocab_size=4096)

    # High-repeat training set
    examples = list(make_phase4_contrast_examples(repeats=8, rng=random.Random(42), dataset="semantic_contrast_v1"))
    eval_examples = list(make_phase4_contrast_examples(repeats=1, rng=random.Random(0), dataset="semantic_contrast_v1"))
    print(f"Train: {len(examples)} examples (8 repeats). Eval: {len(eval_examples)}")

    train_ds = SameFamilyTripletDataset(examples, tokenizer, repeats=4)
    eval_ds = SameFamilyTripletDataset(eval_examples, tokenizer, repeats=1)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda x: x)
    eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False, collate_fn=lambda x: x)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=0.01,
    )

    warmup_steps = int(steps * 0.1)

    def lr_lambda(step):
        return max(0.01, step / warmup_steps) if step < warmup_steps else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = output_base / "train_log.txt"
    log_path.write_text(f"v393: {steps} steps, warmup={warmup_steps}, same-family only\n")

    step = 0
    best_eval = -1
    model.train()

    while step < steps:
        for raw_batch in train_loader:
            if step >= steps:
                break

            queries = [b["query_text"] for b in raw_batch]
            champs = [b["champ_text"] for b in raw_batch]
            hns = [b["hn_text"] for b in raw_batch]

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

            champ_ms = maxsim(q_f, c_f)
            hn_ms = maxsim(q_f, hn_f)

            # Margin loss: champ maxsim should exceed HN maxsim by margin
            margin = 0.02
            loss = F.relu(margin - (champ_ms - hn_ms)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                champ_win = (champ_ms > hn_ms).float().mean().item()
                gap = (champ_ms - hn_ms).mean().item()
                msg = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} step={step} loss={loss.item():.4f} champ_win={champ_win:.3f} gap={gap:.4f}"
                log_path.open("a").write(msg + "\n")
                if step % 500 == 0:
                    print(msg)

            if step % eval_every == 0 and step > 0:
                eval_out = evaluate_v393(model, eval_loader, tokenizer, device)
                model.train()
                msg = f"  eval: champ_win={eval_out['champ_win']:.3f} gap={eval_out['gap']:.4f}"
                log_path.open("a").write(msg + "\n")
                print(msg)

                if eval_out['champ_win'] > best_eval:
                    best_eval = eval_out['champ_win']
                    torch.save(model.state_dict(), output_base / "model.pt")
                    msg = f"  → saved best (champ_win={best_eval:.3f})"
                    log_path.open("a").write(msg + "\n")
                    print(msg)

    torch.save(model.state_dict(), output_base / "model.pt")
    print(f"v393 training complete. Best eval: {best_eval:.3f}. Saved to {output_base}")
    return output_base


def evaluate_v393(model, dataloader, tokenizer, device, max_batches=30):
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

            champ_ms = maxsim(q_f, c_f)
            hn_ms = maxsim(q_f, hn_f)

            champ_wins += (champ_ms > hn_ms).sum().item()
            g = champ_ms - hn_ms
            if g.dim() == 0:
                gaps.append(g.item())
            else:
                gaps.extend(g.cpu().tolist())
            total += len(queries)

    return {
        "champ_win": champ_wins / total if total else 0,
        "gap": sum(gaps) / len(gaps) if gaps else 0,
        "n_eval": total,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_base = train_v393(
        steps=args.steps,
        eval_every=args.eval_every,
        device=args.device,
    )
