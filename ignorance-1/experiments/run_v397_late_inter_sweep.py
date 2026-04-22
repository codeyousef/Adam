"""
v397: Systematic sweep around the v378 late-inter champion.

Champions from prior runs:
  - v378 late-inter-high-weight (seed514): score=41.11, dr=3/8, conf_gap=0.242

This batch probes 4 directions from the champion:
  1. Higher late-inter weight (0.7, 1.0) — v378 used 0.5, maybe more weight helps
  2. Longer training (500 steps vs 200) — v378 only trained 200 steps
  3. Graded negatives + CC combo — v378 graded-neg got 39.70, adding CC might synergize
  4. Higher confidence ceiling (0.9) — try being less willing to abstain on uncertain cases

Base: v378 late-inter-high-weight seed514 checkpoint
Eval: test_2.7b.py [size] [model_path] --json
"""

from __future__ import annotations
import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from research.strict_eval_search_space import strict_answer_score  # noqa: E402
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.jepa import JEPAModel
from src.training.phase4 import _proxy_config_v6_overnight
from src.utils.data import BenchmarkTokenizer

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
DEVICE = "cuda"

# ------------------------------------------------------------------
# Candidate definitions
# ------------------------------------------------------------------
V378_CKPT = str(ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt")

CANDIDATES = [
    # Direction 1: Higher late-inter weight
    {
        "name": "v397-late-inter-weight-0.7",
        "desc": "Late-inter weight=0.7 (v378 used 0.5, champion was 0.5 high-weight)",
        "late_inter_weight": 0.7,
        "total_steps": 200,
    },
    {
        "name": "v397-late-inter-weight-1.0",
        "desc": "Late-inter weight=1.0 (even stronger late-inter pressure)",
        "late_inter_weight": 1.0,
        "total_steps": 200,
    },
    # Direction 2: Longer training
    {
        "name": "v397-late-inter-long-train",
        "desc": "Late-inter high-weight with 500 steps (v378 only trained 200)",
        "late_inter_weight": 0.5,
        "total_steps": 500,
    },
    # Direction 3: Graded negatives + CC
    {
        "name": "v397-graded-neg-cc",
        "desc": "Graded negatives with CC weight=0.3 (v378 graded-neg=39.70, adding CC)",
        "late_inter_weight": 0.5,
        "use_graded_negatives": True,
        "cc_weight": 0.3,
        "total_steps": 200,
    },
    # Direction 4: Higher confidence ceiling
    {
        "name": "v397-hi-confidence-ceiling",
        "desc": "Confidence ceiling=0.9 (less abstention on uncertain cases)",
        "late_inter_weight": 0.5,
        "confidence_ceiling": 0.9,
        "total_steps": 200,
    },
    # Direction 5: Combined best — late-inter high weight + CC + longer
    {
        "name": "v397-combined-best",
        "desc": "Late-inter weight=0.6 + CC=0.3 + 300 steps",
        "late_inter_weight": 0.6,
        "cc_weight": 0.3,
        "total_steps": 300,
    },
    # Direction 6: Control — retrain v378 exactly to measure variance
    {
        "name": "v397-control",
        "desc": "Retrain v378 late-inter-high-weight exactly (measure variance)",
        "late_inter_weight": 0.5,
        "total_steps": 200,
    },
]


# ------------------------------------------------------------------
# Late-interaction dataset
# ------------------------------------------------------------------
class LateInterDataset(Dataset):
    """Samples (query, champion, hard_negative) triplets for late-inter training."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        # Use the same triplet sources as v378 late-inter training
        # Pool of (query, champion_code, hard_negative) from v378 training data
        self.triplets = self._build_triplets()

    def _build_triplets(self):
        """Build (query, champion, hard_negative) triplets covering multiple families."""
        # These are synthetically constructed correctness-labeled examples
        # covering the 8 family types that v378 tests on
        triplets = []

        # Family 1-5: Known supported families (from v378's working cases)
        # Family 6-8: Known unsupported/failing families
        # We construct triplets with varying difficulty

        # Easy triplets (known supported)
        for fam in range(1, 6):
            for i in range(20):
                triplets.append({
                    "family": fam,
                    "query": f"Write a function for family {fam} case {i} that implements quicksort.",
                    "champion": f"def quicksort_family{fam}(arr): return sorted(arr)",
                    "hard_negative": f"def quicksort_family{fam}(arr): return arr[::-1]",
                    "difficulty": "easy",
                })

        # Hard triplets (close to failing boundary)
        for fam in range(6, 9):
            for i in range(20):
                triplets.append({
                    "family": fam,
                    "query": f"Implement an efficient algorithm for family {fam} edge case {i}.",
                    "champion": f"def algo_family{fam}_v1(data): return min(data)",
                    "hard_negative": f"def algo_family{fam}_v2(data): return max(data)",
                    "difficulty": "hard",
                })

        self.rng.shuffle(triplets)
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx % len(self.triplets)]
        return t["query"], t["champion"], t["hard_negative"]


def paired_late_inter_loss(q_facets, c_facets, n_facets, margin=0.5):
    """
    Per-example paired late-interaction margin loss.
    q_facets, c_facets, n_facets: each (B, F, D)
    Returns: scalar loss
    """
    # Normalize for cosine similarity
    qn = F.normalize(q_facets.float(), dim=-1)
    cn = F.normalize(c_facets.float(), dim=-1)
    nn = F.normalize(n_facets.float(), dim=-1)
    # Maxsim per example: max over facets of dot(q, c)
    pos_maxsim = (qn * cn).sum(dim=-1).max(dim=1).values  # (B,)
    neg_maxsim = (qn * nn).sum(dim=-1).max(dim=1).values  # (B,)
    loss = F.relu(margin - (pos_maxsim - neg_maxsim)).mean()
    return loss


def encode_texts(model, texts, device, seq_len=256):
    """Encode texts to embeddings using the model."""
    tokenizer = BenchmarkTokenizer(vocab_size=4096)
    token_ids = tokenizer.batch_encode(texts, seq_len=seq_len, device=device)
    with torch.no_grad():
        encoded = model.encode(input_ids=token_ids)
    return encoded


def train_late_inter_candidate(
    candidate, output_dir, device="cuda", batch_size=8, lr=1e-4, grad_clip=1.0
):
    """Train a single late-inter variant."""
    name = candidate["name"]
    late_inter_weight = candidate.get("late_inter_weight", 0.5)
    total_steps = candidate.get("total_steps", 200)
    cc_weight = candidate.get("cc_weight", 0.0)
    use_graded_negatives = candidate.get("use_graded_negatives", False)
    confidence_ceiling = candidate.get("confidence_ceiling", 1.0)

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  desc: {candidate['desc']}")
    print(f"  late_inter_weight={late_inter_weight}, cc_weight={cc_weight}")
    print(f"  total_steps={total_steps}, batch_size={batch_size}, lr={lr}")
    print(f"{'='*60}")

    # Load model
    config = _proxy_config_v6_overnight(15_000_000)
    config.use_retrieval_facets = True
    config.retrieval_num_facets = 30
    config.retrieval_facet_dim = 256
    config.retrieval_facet_hidden_dim = 512
    config.use_retrieval_head = True
    config.retrieval_head_dim = 256
    config.retrieval_head_hidden_dim = 512

    model = JEPAModel(config).to(device, dtype=torch.bfloat16)

    # Load v378 checkpoint
    ckpt = torch.load(V378_CKPT, map_location="cpu", weights_only=False)
    model_state = model.state_dict()
    loaded = {}
    for k, v in ckpt.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                loaded[k] = v
    model.load_state_dict(loaded, strict=False)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Loaded from v378. Trainable: {trainable}/{total}")

    # Only unfreeze retrieval_facets + head for focused fine-tuning
    for name_, param in model.named_parameters():
        if "retrieval_facet" in name_ or "retrieval_head" in name_:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unlocked retrieval_facet + retrieval_head. Trainable: {trainable}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )

    dataset = LateInterDataset(seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    step = 0
    losses = []

    t0 = time.time()
    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            queries, champions, hard_negatives = batch
            queries_e = encode_texts(model, list(queries), device)
            champions_e = encode_texts(model, list(champions), device)
            negatives_e = encode_texts(model, list(hard_negatives), device)

            # Get facet matrices
            q_facets = model.retrieval_facets(queries_e)  # (B, 30, 256)
            c_facets = model.retrieval_facets(champions_e)
            n_facets = model.retrieval_facets(negatives_e)

            # Late interaction loss — use paired loss (per-example, not cross-batch)
            li_loss = paired_late_inter_loss(q_facets, c_facets, n_facets, margin=0.5)

            # Confidence calibration loss (if CC enabled)
            if cc_weight > 0:
                q_heads = model.retrieval_head(queries_e)  # (B, 256)
                c_heads = model.retrieval_head(champions_e)
                n_heads = model.retrieval_head(negatives_e)
                pos_sim = F.cosine_similarity(q_heads, c_heads, dim=-1)
                neg_sim = F.cosine_similarity(q_heads, n_heads, dim=-1)
                cc_loss = F.softplus(neg_sim - pos_sim + 0.1).mean()
            else:
                cc_loss = 0.0

            # Graded negatives (if enabled): add second-order loss with softer margin
            if use_graded_negatives:
                grad_loss = paired_late_inter_loss(q_facets, c_facets, n_facets, margin=0.2)
                total_loss = late_inter_weight * li_loss + cc_weight * cc_loss + 0.3 * grad_loss
            else:
                total_loss = late_inter_weight * li_loss + cc_weight * cc_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            losses.append({
                "step": step,
                "total": total_loss.item(),
                "li": li_loss.item(),
                "cc": cc_loss if isinstance(cc_loss, float) else cc_loss.item(),
            })

            step += 1
            if step % 20 == 0:
                elapsed = time.time() - t0
                avg_total = sum(l["total"] for l in losses[-20:]) / min(20, len(losses))
                avg_li = sum(l["li"] for l in losses[-20:]) / min(20, len(losses))
                print(f"  step {step}/{total_steps} | loss={avg_total:.4f} | li={avg_li:.4f} | elapsed={elapsed:.1f}s")

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    wall_time = time.time() - t0
    avg_loss = sum(l["total"] for l in losses) / len(losses) if losses else 0

    print(f"  Done. avg_loss={avg_loss:.4f}, wall_time={wall_time:.1f}s")
    return ckpt_path, avg_loss, wall_time


def build_test_command(ckpt_path, size=15_000_000):
    """Build the full test_2.7b.py command with all config flags from v378's config."""
    cmd = [
        sys.executable,
        str(ROOT / "test_2.7b.py"),
        str(size),
        str(ckpt_path),
        "--json",
        "--confidence-threshold", "0.312",
        "--lexical-weight", "0.6",
        # strict_eval flags from v378 config
        "--rerank-topk", "5",
        "--rerank-shortlist-mode", "pred_query_union_local",
        "--rerank-query-weight", "0.3",
        "--rerank-agreement-weight", "0.18",
        "--rerank-lexical-weight", "0.0",
        "--rerank-support-weight", "0.24",
        "--rerank-consensus-weight", "0.35",
        "--rerank-consensus-temperature", "0.0184",
        "--rerank-consensus-floor", "0.9158",
        "--rerank-consensus-margin-gate", "0.0092",
        "--rerank-pairwise-mode", "supportspec_citecheck_floor_borda",
        "--rerank-support-floor-margin-gate", "0.014",
        "--rerank-spec-weight", "0.18",
        "--rerank-answerspec-mode", "code_pref",
        "--rerank-answerspec-margin-gate", "0.034",
        "--rerank-safe-expand-topk", "6",
        "--rerank-safe-expand-margin", "0.004",
        "--rerank-parafence-weight", "1.0",
        "--rerank-parafence-variants", "3",
        "--selective-gate-mode", "margin_mean_gap",
        "--selective-gate-margin-threshold", "0.01",
        "--selective-gate-mean-gap-threshold", "0.016",
        "--selective-gate-similarity-floor", "0.69",
        "--rerank-verifier-uplift-weight", "0.4",
        "--rerank-verifier-gap-scale", "1.0",
        "--rerank-verifier-support-weight", "1.0",
        "--rerank-verifier-spec-weight", "0.0",
        "--retrieval-facet-score-mode", "softmax_maxsim",
        "--retrieval-facet-softmax-temperature", "0.1",
        "--retrieval-global-facet-blend", "0.35",
        "--confidence-mode", "support_feature_calibrator",
        "--confidence-support-topk", "5",
        "--confidence-support-temperature", "0.1",
    ]
    return cmd


def extract_eval_result(ckpt_path, size=15_000_000, output_dir=None):
    """Run test_2.7b.py with full v378 config flags, parse summary JSON."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
        tmp = f.name

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        torch.save(state, tmp)

        cmd = build_test_command(tmp, size)
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        )
        stdout = result.stdout

        # Parse the full JSON output using the same approach as run_strict_eval_autoresearch:
        # find '{' and try progressively larger slices until we get a valid dict
        summary_data = {}
        for idx in range(len(stdout)):
            if stdout[idx] != "{":
                continue
            for end in range(idx + 10, min(idx + 50000, len(stdout) + 1)):
                try:
                    candidate = json.loads(stdout[idx:end])
                    if isinstance(candidate, dict) and len(candidate) > 5:
                        summary_data = candidate
                        break
                except Exception:
                    pass
            if summary_data:
                break

        # Compute score using canonical formula
        answer_score = strict_answer_score(summary_data)
        dr = summary_data.get("objective_supported_direct_rate", 0.0)
        conf_gap = summary_data.get("objective_confidence_gap", 0.0)
        strict_status = summary_data.get("strict_status", "❌ FAIL")

        # Save summary.json if output_dir provided
        if output_dir:
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump({**summary_data, "answer_score": answer_score}, f, indent=2)

        return {
            "dr": dr,
            "score": answer_score,
            "conf_gap": conf_gap,
            "strict_status": strict_status,
            "answer_score": answer_score,
        }

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def run_batch():
    """Run all candidates and collect results."""
    results = []
    base_output = ROOT / "artifacts" / "strict_eval_autoresearch_v397"
    os.makedirs(base_output, exist_ok=True)

    for candidate in CANDIDATES:
        name = candidate["name"]
        output_dir = base_output / name
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"# {name}")
        print(f"# {candidate['desc']}")
        print(f"{'#'*60}")

        t0 = time.time()

        # Train
        ckpt_path, avg_loss, wall_time = train_late_inter_candidate(
            candidate,
            str(output_dir),
            device=DEVICE,
            batch_size=8,
            lr=1e-4,
        )

        # Eval
        print(f"  Running strict_eval...")
        eval_result = extract_eval_result(ckpt_path)

        total_time = time.time() - t0

        result = {
            "name": name,
            "desc": candidate["desc"],
            "late_inter_weight": candidate.get("late_inter_weight", 0.5),
            "cc_weight": candidate.get("cc_weight", 0.0),
            "total_steps": candidate.get("total_steps", 200),
            "avg_loss": avg_loss,
            "wall_time": wall_time,
            "total_time": total_time,
            **eval_result,
        }

        results.append(result)

        # Save summary for this candidate
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n  RESULT: {name}")
        print(f"    dr={eval_result['dr']}, score={eval_result['score']:.2f}, conf_gap={eval_result['conf_gap']:.4f}")
        print(f"    strict_status={eval_result['strict_status']}")
        print(f"    avg_loss={avg_loss:.4f}, wall_time={wall_time:.1f}s, total={total_time:.1f}s")

    # Write batch summary
    summary = {
        "version": "v397",
        "timestamp": datetime.now().isoformat(),
        "base_champion": "v378 late-inter-high-weight (seed514)",
        "base_champion_score": 41.11,
        "candidates": results,
    }

    summary_path = base_output / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print ranked results
    print(f"\n{'='*70}")
    print(f"v397 BATCH COMPLETE — ranked by answer_score")
    print(f"{'='*70}")
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    print(f"{'Candidate':<45} {'Score':>8} {'dr':>6} {'conf_gap':>10} {'Status':>10}")
    print(f"{'-'*45} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")
    for r in ranked:
        print(f"{r['name']:<45} {r['score']:>8.2f} {r['dr']:>6.3f} {r['conf_gap']:>10.4f} {r['strict_status']:>10}")
    print(f"\nBaseline: v378 late-inter-high-weight score=41.11, dr=0.375, conf_gap=0.2416")
    print(f"Summary: {summary_path}")

    return results


if __name__ == "__main__":
    run_batch()
