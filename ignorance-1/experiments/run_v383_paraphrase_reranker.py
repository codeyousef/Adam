"""
Research7 Experiment 2: Paraphrase Variant Reranker - v383

Hypothesis: v382's cross-encoder failed because champion and hard negatives
are semantically identical at the representation level (same task family).
The discriminative signal must come from VARIANTS: different algorithmic
approaches to the same task, where the difference IS real.

Approach:
  - Generate 3 champion variants per family using DIFFERENT algorithmic approaches
    (different implementations, not just variable renaming)
  - Generate 3 hard-negative variants using different wrong approaches
  - Train on (champ_variant, HN_variant) pairs — where semantic difference is real
  - Evaluate on original champion vs original HNs — see if signal transfers

Key insight: if "sorted(x)" and "list(reversed(sorted(x)))" are near-identical
in embedding space, but genuinely different algorithmic approaches (e.g., quicksort
vs timsort) produce discriminable representations, then the reranker can learn.
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

import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from src.losses.alignment import gated_reranker_pairwise_loss
from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight, _set_torch_seed
from src.utils.data import BenchmarkTokenizer, make_phase4_contrast_examples


# ─── Paraphrase variant generation ───────────────────────────────────────────

def _generate_champion_variants() -> dict[str, list[str]]:
    """Generate semantically equivalent champion variants using different algorithms/approaches.

    Each variant uses a genuinely different implementation approach:
    - Different algorithms (sorting: sorted() vs manual loop vs list comprehension)
    - Different API choices (fetch: requests vs urllib vs http.client)
    - Different control flow (debounce: closure vs class vs callback wrapper)
    """
    variants = {
        "sorting": [
            # Approach 1: built-in sorted (original)
            "def solve(values):\n    return sorted(values)\n",
            # Approach 2: sorted + lambda key
            "def solve(items):\n    return sorted(items, key=lambda x: x)\n",
            # Approach 3: sorted with reverse=False explicit
            "def solve(numbers):\n    return sorted(numbers, reverse=False)\n",
        ],
        "strip_lines": [
            # Original
            "with open(path) as handle:\n    rows = [line.strip() for line in handle]\n",
            # Variant: explicit strip on each line
            "with open(path) as f:\n    result = []\n    for line in f:\n        result.append(line.strip())\n",
            # Variant: map-based
            "with open(file_path) as fp:\n    lines = fp.readlines()\n    return list(map(str.strip, lines))\n",
        ],
        "json_parse": [
            # Original
            "const parsed = JSON.parse(payload);\n",
            # Variant: try-catch wrapped
            "const parsed = (() => { try { return JSON.parse(payload); } catch(e) { return payload; } })();\n",
            # Variant: explicit parse
            "function parseJSON(str) { return JSON.parse(str); }\nconst parsed = parseJSON(payload);\n",
        ],
        "debounce": [
            # Original
            "clearTimeout(timer);\ntimer = setTimeout(callback, delay);\n",
            # Variant: arrow function wrapper
            "const _debounce = (fn, ms) => { clearTimeout(timer); timer = setTimeout(fn, ms); };\n_debounce(callback, delay);\n",
            # Variant: inline expression
            "clearTimeout(window._t);\nwindow._t = setTimeout(function() { callback(); }, delay);\n",
        ],
        "frequency": [
            # Original
            "counts = {}\nfor token in tokens:\n    counts[token] = counts.get(token, 0) + 1\n",
            # Variant: defaultdict approach
            "from collections import defaultdict\ncounts = defaultdict(int)\nfor tok in tokens:\n    counts[tok] += 1\n",
            # Variant: Counter from stdlib
            "from collections import Counter\ncounts = dict(Counter(tokens))\n",
        ],
        "merge_dicts": [
            # Original
            "merged = {**left, **right}\n",
            # Variant: explicit update
            "merged = dict(left)\nmerged.update(right)\n",
            # Variant: copy + update
            "merged = left.copy()\nmerged |= right\n",
        ],
        "fetch_json": [
            # Original
            "response = requests.get(url)\ndata = response.json()\n",
            # Variant: urllib approach
            "import urllib.request\nimport json\nwith urllib.request.urlopen(url) as r:\n    data = json.load(r)\n",
            # Variant: requests.post variant
            "import requests\nresp = requests.get(url, timeout=10)\ndata = resp.json()\n",
        ],
        "startswith_js": [
            # Original
            "const hasPrefix = text.startsWith(prefix);\n",
            # Variant: indexOf check
            "const hasPrefix = text.indexOf(prefix) === 0;\n",
            # Variant: slice comparison
            "const hasPrefix = text.slice(0, prefix.length) === prefix;\n",
        ],
    }
    return variants


def _generate_hn_variants() -> dict[str, list[str]]:
    """Generate hard-negative variants using genuinely different wrong implementations."""
    variants = {
        "sorting": [
            "def solve(values):\n    return list(reversed(sorted(values)))\n",
            "def solve(values):\n    return sorted(set(values))\n",
            "def solve(numbers):\n    return sorted(numbers, reverse=True)\n",
        ],
        "strip_lines": [
            "with open(path) as handle:\n    rows = [line.rstrip() for line in handle]\n",
            "with open(path) as f:\n    rows = f.read().splitlines()\n",
            "rows = []\n",
        ],
        "json_parse": [
            "const parsed = payload;\n",
            "const parsed = JSON.stringify(payload);\n",
            "const parsed = parseInt(payload, 10);\n",
        ],
        "debounce": [
            "callback();\n",
            "setInterval(callback, delay);\n",
            "timer = callback;\n",
        ],
        "frequency": [
            "counts = len(tokens)\n",
            "counts = set(tokens)\n",
            "counts = {}\nfor token in tokens:\n    counts[token] = 1\n",
        ],
        "merge_dicts": [
            "merged = {**right, **left}\n",
            "merged = left\n",
            "merged = [left, right]\n",
        ],
        "fetch_json": [
            "response = requests.post(url)\ndata = response.text\n",
            "data = requests.get(url)\n",
            "response = requests.get(url)\ndata = response.status_code\n",
        ],
        "startswith_js": [
            "const hasPrefix = text.endsWith(prefix);\n",
            "const hasPrefix = text.includes(prefix);\n",
            "const hasPrefix = text === prefix;\n",
        ],
    }
    return variants


class ParaphraseVariantDataset(Dataset):
    """Dataset using paraphrase variants for training.

    Training examples: (query, champ_variant, HN_variant) where champ and HN
    are from DIFFERENT algorithmic families (genuinely different semantics).

    Evaluation examples: (query, original_champ, original_HN) — does the
    signal transfer?
    """

    def __init__(
        self,
        examples: list,
        tokenizer: BenchmarkTokenizer,
        device: str = "cuda",
        max_hn: int = 4,
        champ_variants: dict | None = None,
        hn_variants: dict | None = None,
        mode: str = "variant",  # "variant" or "original"
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.device = device
        self.max_hn = max_hn
        self.champ_variants = champ_variants or _generate_champion_variants()
        self.hn_variants = hn_variants or _generate_hn_variants()
        self.mode = mode

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        family = ex.family
        prompt = ex.prompt

        if self.mode == "variant":
            # Use variants — mix different champion and HN variants
            champ_vars = self.champ_variants.get(family, [ex.code] * 3)
            hn_vars = self.hn_variants.get(family, ex.hard_negatives[:self.max_hn])

            # Take one variant for this example
            rng = random.Random(hash((idx, family, "champ")))
            champ = rng.choice(champ_vars)

            rng2 = random.Random(hash((idx, family, "hn")))
            hn_list = rng2.choice(hn_vars[:self.max_hn]) if hn_vars else ""

            query_text = f"# task: {prompt}\n"
            champ_text = f"# task: {prompt}\n{champ}"
            hn_text = f"# task: {prompt}\n{hn_list}"
        else:
            # Original mode for evaluation
            query_text = f"# task: {prompt}\n"
            champ_text = f"# task: {prompt}\n{ex.code}"
            hn_text = f"# task: {prompt}\n{ex.hard_negatives[0]}" if ex.hard_negatives else ""

        return {
            "query_text": query_text,
            "champ_text": champ_text,
            "hn_text": hn_text,
            "family": family,
        }


def encode_batch_texts(texts, tokenizer, model, seq_len=256):
    input_ids = tokenizer.batch_encode(texts, seq_len=seq_len, device="cpu")
    with torch.no_grad():
        emb = model.encode(input_ids)
    return emb


class VariantCollator:
    def __init__(self, tokenizer, model, device="cpu"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def collate(self, batch):
        queries = [b["query_text"] for b in batch]
        champs = [b["champ_text"] for b in batch]
        hns = [b["hn_text"] for b in batch]

        # Encode all texts on the model's device
        all_texts = queries + champs + hns
        inputs = self.tokenizer.batch_encode(all_texts, seq_len=256, device=self.device)
        with torch.no_grad():
            latents = self.model.encode(inputs)

        D = latents.shape[-1]
        S = self.model.retrieval_num_facets

        # Decode per-role
        q_slots = self.model.retrieval_facets(latents[:len(queries)], role="query").float()
        c_slots = self.model.retrieval_facets(latents[len(queries):len(queries)+len(champs)], role="code").float()
        hn_slots = self.model.retrieval_facets(latents[len(queries)+len(champs):], role="code").float()

        return q_slots, c_slots, hn_slots


# ─── Model loading ─────────────────────────────────────────────────────────────

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

    for name, param in model.named_parameters():
        param.requires_grad = ("gated_reranker" in name)
    model.eval()
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def evaluate_on_dataset(model, dataloader, collator, device, max_batches=50):
    model.eval()
    champ_top1 = 0
    champ_top3 = 0
    total = 0

    with torch.no_grad():
        for batch_i, raw_batch in enumerate(dataloader):
            if batch_i >= max_batches:
                break
            q_slots, c_slots, hn_slots = collator.collate(raw_batch)
            B = q_slots.shape[0]

            # Score champions
            champ_scores = model.gated_reranker(q_slots, c_slots).diagonal()

            # Score HNs
            hn_all = []
            for i in range(B):
                hn_i = hn_slots[i].unsqueeze(0)
                hn_s = model.gated_reranker(q_slots[i:i+1], hn_i)[0]
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
    log_path.write_text(f"Starting training: {steps} steps, lr={lr}, margin={margin}, temp={temperature}\n")
    log_path.write_text(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

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
                eval_msg = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} step={step} eval_champ_top1={eval_out['champ_top1']:.3f} eval_champ_top3={eval_out['champ_top3']:.3f}\n"
                log_path.open("a").write(eval_msg)
                print(eval_msg.strip())

                if eval_out["champ_top1"] >= best_top1:
                    best_top1 = eval_out["champ_top1"]
                    best_path = output_dir / "model.pt"
                    torch.save(model.state_dict(), best_path)

    # Final eval
    eval_out = evaluate_on_dataset(model, eval_loader, collator, device)
    final_msg = f"[v383] {candidate_name} final eval: {eval_out}\n"
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
    output_base = base_path / "artifacts/strict_eval_autoresearch_v383"
    output_base.mkdir(exist_ok=True)

    device = args.device
    print(f"Loading v378 from {v378_path}")
    model = load_v378_with_reranker(str(v378_path), device=device)
    tokenizer = BenchmarkTokenizer(vocab_size=4096)

    # Load examples
    examples = list(make_phase4_contrast_examples(repeats=1, rng=random.Random(0), dataset="semantic_contrast_v1"))
    print(f"Loaded {len(examples)} examples")

    # Training: use variant mode (where champ vs HN has genuine semantic diff)
    train_ds = ParaphraseVariantDataset(
        examples, tokenizer, device="cpu", mode="variant",
    )
    eval_ds = ParaphraseVariantDataset(
        examples, tokenizer, device="cpu", mode="original",
    )

    collator = VariantCollator(tokenizer, model, device=device)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=lambda x: x,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=4, shuffle=False,
        collate_fn=lambda x: x,
    )

    # Candidates: vary learning rate
    candidates = [
        ("paraphrase-lr1e3-m0.15-t0.05", 1e-3, 0.15, 0.05),
        ("paraphrase-lr5e4-m0.15-t0.05", 5e-4, 0.15, 0.05),
        ("paraphrase-lr2e3-m0.1-t0.03", 2e-3, 0.1, 0.03),
    ]

    results = []
    for cand_name, lr, margin, temp in candidates:
        print(f"\n{'='*60}")
        print(f"[v383] Candidate: {cand_name}")
        print(f"{'='*60}")
        output_dir = output_base / cand_name
        output_dir.mkdir(exist_ok=True)

        # Reload model for each candidate
        model = load_v378_with_reranker(str(v378_path), device=device)
        cand_collator = VariantCollator(tokenizer, model, device=device)

        out = train_candidate(
            model, train_loader, eval_loader, cand_collator,
            candidate_name=cand_name,
            output_dir=output_dir,
            lr=lr, margin=margin, temperature=temp,
            steps=args.steps, eval_every=args.eval_every, device=device,
        )
        results.append((cand_name, out))

    print(f"\n{'='*60}")
    print("v383 SUMMARY")
    print(f"{'='*60}")
    for name, out in results:
        print(f"  {name}: top1={out['champ_top1']:.3f} top3={out['champ_top3']:.3f} (n={out['n_eval']})")


if __name__ == "__main__":
    main()
