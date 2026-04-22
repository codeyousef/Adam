"""
Research7 Experiment 1: Family-Local Gated Reranker - v382

Two-stage retrieve-then-verify:
  Stage 1 (frozen): v378 global cosine -> top-5 shortlist
  Stage 2 (trained): GatedRerankerHead -> champion over hard negatives

Diagnostic confirmed: champion in top-5 for ALL 8 families. Only discrimination needed.
"""
from __future__ import annotations

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


# ─── Model loading ────────────────────────────────────────────────────────────

def load_v378_with_reranker(base_model_path: str, device: str = "cuda") -> JEPAModel:
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


# ─── Dataset ──────────────────────────────────────────────────────────────────

class RerankerDataset(Dataset):
    """Per-example dataset: query (prompt only) vs champion vs hard negatives.

    Each item returns:
      query_text: str  (prompt only, no code)
      champ_text: str  (prompt + champion code)
      hn_texts: list[str]  (prompt + each hard negative)
    """

    def __init__(
        self,
        examples: list,
        tokenizer: BenchmarkTokenizer,
        device: str = "cuda",
        max_hn: int = 4,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.device = device
        self.max_hn = max_hn

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        query_text = f"# task: {ex.prompt}\n"
        champ_text = f"# task: {ex.prompt}\n{ex.code}"
        hn_texts = [
            f"# task: {ex.prompt}\n{hn}" for hn in ex.hard_negatives[:self.max_hn]
        ]
        return {
            "query_text": query_text,
            "champ_text": champ_text,
            "hn_texts": hn_texts,
            "family": ex.family,
        }


def encode_batch_texts(
    texts: list[str],
    tokenizer: BenchmarkTokenizer,
    model: JEPAModel,
    seq_len: int = 256,
    device: str = "cuda",
):
    """Encode texts and extract retrieval facets (query role for prompt, code role for code)."""
    input_ids = tokenizer.batch_encode(texts, seq_len=seq_len, device=device)
    with torch.no_grad():
        latents = model.encode(input_ids)
    # For query (prompt-only), use role=query for facets
    # For code texts, use role=code for facets
    return latents


class RerankerCollator:
    """Collate a batch of examples into training tensors.

    For each example in the batch:
      - Encode query_text with role=query  -> query_slots [S, D]
      - Encode champ_text with role=code   -> champ_slots [S, D]
      - Encode each hn_text with role=code -> hn_slots [K, S, D]

    Returns:
      query_slots:    [B, S, D]
      champ_slots:    [B, S, D]
      hn_slots:       [B*K, S, D]
    """

    def __init__(
        self,
        tokenizer: BenchmarkTokenizer,
        model: JEPAModel,
        num_slots: int = 30,
        facet_dim: int = 256,
        device: str = "cuda",
        max_hn: int = 4,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.num_slots = num_slots
        self.facet_dim = facet_dim
        self.device = device
        self.max_hn = max_hn

    def collate(self, batch):
        B = len(batch)
        K = self.max_hn

        all_query_texts = [b["query_text"] for b in batch]
        all_champ_texts = [b["champ_text"] for b in batch]
        all_hn_texts = [hn for b in batch for hn in b["hn_texts"]]

        # Pad HNs to K per example
        # After flattening: B groups of K
        # For examples with <K HNs, pad with zeros
        padded_hn_texts = []
        for b in batch:
            hns = b["hn_texts"]
            padded_hn_texts.extend(hns)
            # Already extended; padding handled below via masking

        # Encode queries: role=query
        q_input_ids = self.tokenizer.batch_encode(
            all_query_texts, seq_len=256, device=self.device
        )
        with torch.no_grad():
            q_latents = self.model.encode(q_input_ids)
            q_slots_raw = self.model.retrieval_facets(q_latents, role="query").float()
        # q_slots_raw: [B, S, D] (if role=query returns shared) or [B, S, D]
        # Actually retrieval_facets with role=query uses the facet head with query role
        # Ensure shape: [B, num_slots, facet_dim]
        if q_slots_raw.ndim == 2:
            q_slots_raw = q_slots_raw.unsqueeze(1)
        query_slots = q_slots_raw

        # Encode champions: role=code
        c_input_ids = self.tokenizer.batch_encode(
            all_champ_texts, seq_len=256, device=self.device
        )
        with torch.no_grad():
            c_latents = self.model.encode(c_input_ids)
            c_slots_raw = self.model.retrieval_facets(c_latents, role="code").float()
        if c_slots_raw.ndim == 2:
            c_slots_raw = c_slots_raw.unsqueeze(1)
        champ_slots = c_slots_raw

        # Encode HNs: role=code
        if all_hn_texts:
            hn_input_ids = self.tokenizer.batch_encode(
                all_hn_texts, seq_len=256, device=self.device
            )
            with torch.no_grad():
                hn_latents = self.model.encode(hn_input_ids)
                hn_slots_raw = self.model.retrieval_facets(hn_latents, role="code").float()
            # hn_slots_raw: [B*K, S, D] or [B*K, D]
            if hn_slots_raw.ndim == 2:
                hn_slots_raw = hn_slots_raw.unsqueeze(1)
            # Reshape to [B, K, S, D] then flatten to [B*K, S, D]
            hn_slots = hn_slots_raw.view(B, K, self.num_slots, self.facet_dim)
            # Handle variable HNs: some examples have fewer than K HNs
            # We packed them sequentially, so we need to handle per-example
            # Simpler: just use the packed version; padding zeros will score ~0
            hn_slots = hn_slots.reshape(B * K, self.num_slots, self.facet_dim)
        else:
            hn_slots = torch.zeros(
                B * K, self.num_slots, self.facet_dim, device=self.device
            )

        return query_slots, champ_slots, hn_slots


# ─── Training ─────────────────────────────────────────────────────────────────

def compute_reranker_loss(reranker, query_slots, champ_slots, hn_slots, margin, temperature):
    """Wrapper for the pairwise loss."""
    return gated_reranker_pairwise_loss(
        reranker,
        query_slots,
        champ_slots,
        hn_slots,
        margin=margin,
        temperature=temperature,
    )


def evaluate_on_dataset(model, eval_loader, collator, device, max_batches=50):
    """Compute champion-in-top-1 and top-3 on eval set."""
    model.eval()
    top1_correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= max_batches:
                break
            query_slots, champ_slots, hn_slots = collator.collate(batch)
            query_slots = query_slots.to(device)
            champ_slots = champ_slots.to(device)
            hn_slots = hn_slots.to(device)

            B = query_slots.shape[0]
            # Scores: champion vs each HN
            champ_scores = model.gated_reranker(query_slots, champ_slots).diagonal()
            # hn_slots: [B*K, S, D] -> group per example
            K = hn_slots.shape[0] // B
            hn_slots_per_ex = hn_slots.view(B, K, -1, hn_slots.shape[-1])
            hn_scores_list = []
            for i in range(B):
                hn_s = model.gated_reranker(
                    query_slots[i : i + 1], hn_slots_per_ex[i]
                )[0]
                hn_scores_list.append(hn_s)
            hn_scores = torch.stack(hn_scores_list, dim=0)  # [B, K]

            # Check ranking
            all_scores = torch.cat(
                [champ_scores.unsqueeze(-1), hn_scores], dim=-1
            )  # [B, 1+K]
            sorted_idx = all_scores.argsort(dim=-1, descending=True)  # [B, 1+K]
            top1_is_champ = (sorted_idx[:, 0] == 0).cpu()
            top3_has_champ = (sorted_idx[:, :3] == 0).any(dim=-1).cpu()

            top1_correct += top1_is_champ.sum().item()
            top3_correct += top3_has_champ.sum().item()
            total += B

    model.train()
    return {
        "champ_top1": top1_correct / max(total, 1),
        "champ_top3": top3_correct / max(total, 1),
        "n_eval": total,
    }


def train_reranker(
    model,
    train_loader,
    eval_loader,
    collator,
    *,
    device: str,
    num_steps: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 200,
    margin: float = 0.1,
    temperature: float = 0.05,
    log_path=None,
):
    reranker_params = list(model.gated_reranker.parameters())
    optimizer = torch.optim.AdamW(reranker_params, lr=lr, weight_decay=weight_decay)

    step = 0
    eval_results = []

    def log(msg):
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        line = f"{ts} {msg}"
        print(line, flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(line + "\n")

    log(f"Starting training: {num_steps} steps, lr={lr}, margin={margin}, temp={temperature}")
    log(f"Trainable params: {sum(p.numel() for p in reranker_params):,}")

    while step < num_steps:
        model.train()
        for batch in train_loader:
            query_slots, champ_slots, hn_slots = collator.collate(batch)
            query_slots = query_slots.to(device)
            champ_slots = champ_slots.to(device)
            hn_slots = hn_slots.to(device)

            optimizer.zero_grad()
            loss = compute_reranker_loss(
                model.gated_reranker,
                query_slots, champ_slots, hn_slots,
                margin=margin, temperature=temperature,
            )
            loss.backward()
            optimizer.step()

            step += 1
            if step % 50 == 0:
                log(f"step={step} loss={loss.item():.6f}")

            if step % eval_every == 0:
                eval_out = evaluate_on_dataset(model, eval_loader, collator, device)
                eval_results.append({"step": step, **eval_out})
                log(
                    f"step={step} eval_champ_top1={eval_out['champ_top1']:.3f} "
                    f"eval_champ_top3={eval_out['champ_top3']:.3f}"
                )

            if step >= num_steps:
                break

    return eval_results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    base_path = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
    base_model = (
        base_path / "artifacts/strict_eval_autoresearch_v378"
                  "/v378-late-inter-high-weight-seed511-seed514/model.pt"
    )
    output_base = base_path / "artifacts/strict_eval_autoresearch_v382"
    output_base.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v382] Device: {device}")

    # Load model once (all candidates share the same frozen backbone)
    print(f"[v382] Loading v378 from {base_model}")
    model = load_v378_with_reranker(str(base_model), device=device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[v382] Total params: {total:,} | Trainable (reranker): {trainable:,}")

    # Build dataset
    print("[v382] Building training/eval datasets...")
    train_examples = list(
        make_phase4_contrast_examples(
            repeats=8, rng=random.Random(42), dataset="semantic_contrast_v1"
        )
    )
    eval_examples = list(
        make_phase4_contrast_examples(
            repeats=2, rng=random.Random(0), dataset="semantic_contrast_v1"
        )
    )

    tokenizer = BenchmarkTokenizer(vocab_size=4096)
    train_dataset = RerankerDataset(train_examples, tokenizer, device=device, max_hn=4)
    eval_dataset = RerankerDataset(eval_examples, tokenizer, device=device, max_hn=4)

    # Check that retrieval_facets works correctly
    dummy_batch = [train_dataset[0]]
    dummy_collator = RerankerCollator(
        tokenizer=tokenizer, model=model, device=device
    )
    qs, cs, hs = dummy_collator.collate(dummy_batch)
    print(f"[v382] query_slots shape: {qs.shape}")
    print(f"[v382] champ_slots shape: {cs.shape}")
    print(f"[v382] hn_slots shape: {hs.shape}")

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: x,
        num_workers=0,
    )

    candidates = [
        {
            "name": "gated-reranker-lr1e3-m0.1-t0.05",
            "num_steps": 2000,
            "lr": 1e-3,
            "margin": 0.1,
            "temperature": 0.05,
        },
        {
            "name": "gated-reranker-lr5e4-m0.2-t0.05",
            "num_steps": 2000,
            "lr": 5e-4,
            "margin": 0.2,
            "temperature": 0.05,
        },
        {
            "name": "gated-reranker-lr2e3-m0.15-t0.03",
            "num_steps": 2000,
            "lr": 2e-3,
            "margin": 0.15,
            "temperature": 0.03,
        },
    ]

    all_results = []
    for cand in candidates:
        name = cand["name"]
        print(f"\n{'='*60}")
        print(f"[v382] Candidate: {name}")
        print(f"{'='*60}")

        # Fresh optimizer for each candidate
        cand_dir = output_base / name
        cand_dir.mkdir(exist_ok=True)
        log_path = cand_dir / "train_log.txt"

        # Reload model weights for each candidate (reset reranker to random init)
        state_dict = torch.load(str(base_model), map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        # Freeze everything except gated_reranker
        for name_p, param in model.named_parameters():
            param.requires_grad = ("gated_reranker" in name_p)

        eval_results = train_reranker(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            collator=dummy_collator,
            device=device,
            num_steps=cand["num_steps"],
            lr=cand["lr"],
            weight_decay=1e-4,
            eval_every=200,
            margin=cand["margin"],
            temperature=cand["temperature"],
            log_path=log_path,
        )

        # Save
        model_path = cand_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        result = {
            "candidate": name,
            "base_model": str(base_model),
            **cand,
            "final_eval": eval_results[-1] if eval_results else None,
            "all_eval": eval_results,
        }
        with open(cand_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"[v382] {name} final eval: {result['final_eval']}")
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("v382 SUMMARY")
    print("=" * 60)
    for r in all_results:
        ev = r["final_eval"] or {}
        top1 = ev.get("champ_top1", None)
        top3 = ev.get("champ_top3", None)
        top1_str = f"{top1:.3f}" if top1 is not None else "?"
        top3_str = f"{top3:.3f}" if top3 is not None else "?"
        print(
            f"  {r['candidate']}: "
            f"top1={top1_str} "
            f"top3={top3_str} "
            f"(lr={r['lr']}, m={r['margin']}, t={r['temperature']})"
        )


if __name__ == "__main__":
    main()
