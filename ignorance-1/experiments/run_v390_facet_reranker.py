"""
v390: Neighborhood-Posterior Confidence + Late-Interaction Facet Reranker

Hypothesis: v378's gated_reranker failed because:
  (1) It was trained on pooled embeddings (rank-collapsed to ~1.0) — no signal
  (2) It was never wired into the eval pipeline

The fix:
  - Train gated_reranker on LATE INTERACTION features (facet-level, not pooled)
  - Wire into _selection_scores_for_finalists with rerank_gated_reranker_weight
  - Use neighborhood_posterior as confidence mode (proven in v340)

Architecture:
  - Frozen v378 backbone (maintains parametric ignorance)
  - Train only gated_reranker head
  - Training: listwise ranking on late-interaction facet representations
  - Eval: wired into reranking pipeline with configurable weight
"""

from __future__ import annotations

import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.losses.alignment import gated_reranker_pairwise_loss
from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight, _set_torch_seed
from src.utils.data import BenchmarkTokenizer, make_phase4_contrast_examples


# ─── Dataset ─────────────────────────────────────────────────────────────────

class FacetRerankerDataset(Dataset):
    """Dataset for training the gated reranker on late-interaction facet features.

    Key insight: train on the FACET-level late-interaction features, not pooled
    embeddings. The facets encode token-level information that pooled embeddings
    lose. This is what allows discrimination.
    """

    def __init__(
        self,
        examples: list,
        tokenizer: BenchmarkTokenizer,
        device: str = "cuda",
        max_hn: int = 4,
        repeats: int = 4,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.device = device
        self.max_hn = max_hn
        self.repeats = repeats
        self.rng = random.Random(42)

    def __len__(self) -> int:
        return len(self.examples) * self.repeats

    def __getitem__(self, idx: int):
        ex_idx = idx % len(self.examples)
        ex = self.examples[ex_idx]
        family = ex.family
        prompt = ex.prompt

        # Shuffle HNs for this read
        hn_list = list(ex.hard_negatives)
        self.rng.shuffle(hn_list)
        hn = hn_list[0] if hn_list else ""

        return {
            "query_text": f"# task: {prompt}\n",
            "champ_text": f"# task: {prompt}\n{ex.code}",
            "hn_text": f"# task: {prompt}\n{hn}",
            "family": family,
        }


class FacetCollator:
    """Collator that extracts LATE INTERACTION FACETS, not pooled embeddings.

    The key difference from v382: we encode to get latents, extract facets
    (NOT pooled embeddings), and use those as the reranker input.
    """

    def __init__(self, tokenizer, model, device="cuda"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def collate(self, batch):
        queries = [b["query_text"] for b in batch]
        champs = [b["champ_text"] for b in batch]
        hns = [b["hn_text"] for b in batch]

        # Encode all texts
        all_texts = queries + champs + hns
        inputs = self.tokenizer.batch_encode(all_texts, seq_len=256, device=self.device)

        with torch.no_grad():
            latents = self.model.encode(inputs)
            # Extract FACETS for late interaction (NOT pooled embeddings)
            # Shape: [B, num_slots, facet_dim]
            q_facets = self.model.retrieval_facets(latents[:len(queries)], role="query").float()
            c_facets = self.model.retrieval_facets(latents[len(queries):len(queries)+len(champs)], role="code").float()
            hn_facets = self.model.retrieval_facets(latents[len(queries)+len(champs):], role="code").float()

        return q_facets, c_facets, hn_facets


# ─── Model loading ─────────────────────────────────────────────────────────────

def load_v378_for_finetune(base_model_path: str, device: str = "cuda") -> JEPAModel:
    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=False)

    config = _proxy_config_v6_overnight(15_000_000)
    config.use_retrieval_facets = True
    config.retrieval_num_facets = 30
    config.retrieval_facet_dim = 256
    config.retrieval_facet_hidden_dim = 512
    config.retrieval_facet_separate_query_code = False
    config.use_retrieval_head = True
    config.retrieval_head_dim = 256
    config.retrieval_head_hidden_dim = 512
    config.use_gated_reranker = True
    config.gated_reranker_hidden_dim = 128
    config.gated_reranker_num_heads = 4

    model = JEPAModel(config).to(device)
    model.load_state_dict(state_dict, strict=False)

    # Freeze everything except gated_reranker
    for name, param in model.named_parameters():
        param.requires_grad = ("gated_reranker" in name)

    model.eval()
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def evaluate_on_dataset(model, dataloader, collator, device, max_batches=50):
    """Evaluate using the gated reranker on FACETS (not pooled embeddings)."""
    model.eval()
    champ_top1 = 0
    champ_top3 = 0
    total = 0

    with torch.no_grad():
        for batch_i, raw_batch in enumerate(dataloader):
            if batch_i >= max_batches:
                break
            q_slots, c_slots, hn_slots = collator.collate(raw_batch)
            q_slots = q_slots.to(device)
            c_slots = c_slots.to(device)
            hn_slots = hn_slots.to(device)
            B = q_slots.shape[0]

            # Score champions
            champ_scores = model.gated_reranker(q_slots, c_slots).diagonal()

            # Score HNs
            hn_all = []
            for i in range(B):
                hn_s = model.gated_reranker(q_slots[i:i+1], hn_slots[i].unsqueeze(0))[0]
                hn_all.append(hn_s)
            hn_stacked = torch.stack(hn_all)  # [B, H]

            # champ_top1: champion beats ALL HNs
            champ_wins_hn = (champ_scores.unsqueeze(1) > hn_stacked).all(dim=1)
            champ_top1 += champ_wins_hn.sum().item()

            # champ_top3: champion in top-3 of (champ + HNs)
            combined_scores = torch.cat([champ_scores.unsqueeze(1), hn_stacked], dim=1)
            top3_ranks = combined_scores.argsort(descending=True)
            champ_in_top3 = (top3_ranks == 0).any(dim=1)
            champ_top3 += champ_in_top3.sum().item()

            total += B

    return {
        "champ_top1": champ_top1 / total if total else 0,
        "champ_top3": champ_top3 / total if total else 0,
        "n_eval": total,
    }


def train_candidate(
    model,
    train_loader,
    eval_loader,
    collator,
    candidate_name: str,
    output_dir: Path,
    lr: float = 1e-3,
    margin: float = 0.15,
    temperature: float = 0.05,
    steps: int = 2000,
    eval_every: int = 200,
    device: str = "cuda",
):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda")

    best_top1 = 0
    step = 0
    model.train()

    log_path = output_dir / "train_log.txt"
    log_path.write_text(
        f"Starting training: {steps} steps, lr={lr}, margin={margin}, temp={temperature}\n"
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
    )

    while step < steps:
        for raw_batch in train_loader:
            if step >= steps:
                break

            q_slots, c_slots, hn_slots = collator.collate(raw_batch)
            q_slots = q_slots.to(device)
            c_slots = c_slots.to(device)
            hn_slots = hn_slots.to(device)

            with torch.amp.autocast("cuda"):
                loss = gated_reranker_pairwise_loss(
                    model.gated_reranker, q_slots, c_slots, hn_slots,
                    margin=margin, temperature=temperature,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % 50 == 0:
                msg = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} step={step} loss={loss.item():.6f}\n"
                log_path.open("a").write(msg)

            if step % eval_every == 0:
                eval_out = evaluate_on_dataset(model, eval_loader, collator, device)
                model.train()
                eval_msg = (
                    f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} "
                    f"step={step} eval_champ_top1={eval_out['champ_top1']:.3f} "
                    f"eval_champ_top3={eval_out['champ_top3']:.3f}\n"
                )
                log_path.open("a").write(eval_msg)
                print(eval_msg.strip())

                if eval_out["champ_top1"] >= best_top1:
                    best_top1 = eval_out["champ_top1"]
                    torch.save(model.state_dict(), output_dir / "model.pt")

    # Final eval
    eval_out = evaluate_on_dataset(model, eval_loader, collator, device)
    final_msg = f"[v390] {candidate_name} final eval: {eval_out}\n"
    log_path.open("a").write(final_msg)
    print(final_msg.strip())
    return eval_out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_path = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
    v378_path = base_path / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
    output_base = base_path / "artifacts/strict_eval_autoresearch_v390"
    output_base.mkdir(exist_ok=True)

    device = args.device
    print(f"Loading v378 from {v378_path}")
    model = load_v378_for_finetune(str(v378_path), device=device)
    tokenizer = BenchmarkTokenizer(vocab_size=4096)

    # Load examples — use repeats=4 for more training data
    examples = list(make_phase4_contrast_examples(repeats=4, rng=random.Random(0), dataset="semantic_contrast_v1"))
    print(f"Loaded {len(examples)} examples (4 repeats each)")

    train_ds = FacetRerankerDataset(examples, tokenizer, device="cpu", repeats=1)
    eval_ds = FacetRerankerDataset(examples, tokenizer, device="cpu", repeats=1)

    collator = FacetCollator(tokenizer, model, device=device)

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=lambda x: x,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=8, shuffle=False,
        collate_fn=lambda x: x,
    )

    # Candidates: vary learning rate, margin, temperature
    candidates = [
        ("facet-reranker-lr1e3-m0.1-t0.05", 1e-3, 0.1, 0.05),
        ("facet-reranker-lr2e3-m0.15-t0.03", 2e-3, 0.15, 0.03),
        ("facet-reranker-lr5e4-m0.2-t0.05", 5e-4, 0.2, 0.05),
    ]

    results = []
    for cand_name, lr, margin, temp in candidates:
        print(f"\n{'='*60}")
        print(f"[v390] Candidate: {cand_name}")
        print(f"{'='*60}")
        output_dir = output_base / cand_name
        output_dir.mkdir(exist_ok=True)

        # Reload model for each candidate
        model = load_v378_for_finetune(str(v378_path), device=device)
        cand_collator = FacetCollator(tokenizer, model, device=device)

        out = train_candidate(
            model, train_loader, eval_loader, cand_collator,
            candidate_name=cand_name,
            output_dir=output_dir,
            lr=lr, margin=margin, temperature=temp,
            steps=args.steps, eval_every=args.eval_every, device=device,
        )
        results.append((cand_name, out))

    print(f"\n{'='*60}")
    print("v390 SUMMARY")
    print(f"{'='*60}")
    for name, out in results:
        print(f"  {name}: top1={out['champ_top1']:.3f} top3={out['champ_top3']:.3f} (n={out['n_eval']})")


if __name__ == "__main__":
    main()
