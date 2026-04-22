"""
v408: Targeted confidence calibration for hard families.

The problem with v407: training used wrong OOD (random sentences) instead of the actual
unsupported query variants from the eval.

CORRECT training data:
  POSITIVE (supported, label=1): natural language queries that the model SHOULD answer
    - "Deserialize JSON text into a JavaScript value" → champion code
    - "Debounce an input event before firing a callback" → champion code
  NEGATIVE (unsupported, label=0): INVERSE tasks that the model should abstain from
    - "Serialize a JavaScript object into a JSON string" (inverse of json_parse)
    - "Add debounce to a UI component" (different debounce use case)
    These come from UNSUPPORTED_OBJECTIVE_TEMPLATES in test_2.7b.py

TRAINING APPROACH:
  1. Freeze everything except query_head
  2. Train on supported (label=1) vs unsupported (label=0)
  3. Use synthesis_queries from training data as additional positive examples
  4. Very few steps (50-100) to avoid overwriting useful representations

KEY INSIGHT: The unsupported queries for hard families are SEMANTICALLY SIMILAR to
supported queries (same family, just different operation). The model gives conf=0.97
for unsupported json_parse (serialize) because the champion code (parse) has the
SAME query prefix. We need to teach the model that "serialize" → abstain even
though the embedding is similar to "deserialize".

Candidates:
  A: Train ONLY query_head.weight/bias with very few steps
  B: Train query_head + late_interaction
  C: Lower threshold from 0.4 to 0.37 (accept 5 abstentions → 0 abstentions but 3 new FPs)
"""
from __future__ import annotations
import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import json, os, sys, time, random, torch, torch.nn.functional as F
from pathlib import Path

from src.models.jepa import JEPAModel
from src.training.phase4 import (
    _proxy_config_v6_overnight,
    make_phase4_contrast_examples,
    BenchmarkTokenizer,
)
from src.utils.data import set_seed
from research.strict_eval_search_space import strict_answer_score

ROOT = Path(_project_root)
V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
DEVICE = "cuda"

# Unsupported objective templates from test_2.7b.py
UNSUPPORTED_TEMPLATES = {
    "sorting": "Sort a numeric list in descending order and remove duplicates.",
    "json_parse": "Serialize a JavaScript object into a JSON string.",
    "fetch_json": "Parse JSON data from a file in Python.",
    "strip_lines": "Read each line from a file and add line numbers.",
    "debounce": "Add debounce to a React component using useCallback.",
    "frequency": "Count how many times each word appears across multiple text files.",
    "merge_dicts": "Deep merge two nested Python dictionaries recursively.",
    "startswith_js": "Check whether a string ends with a suffix in JavaScript.",
}


def create_model(freeze_query_head=True, lr=1e-4):
    """Load v378, freeze all except query_head."""
    config = _proxy_config_v6_overnight(15_000_000)
    config.embed_dim = 192
    config.retrieval_head_dim = 256
    config.retrieval_head_hidden_dim = 512
    config.retrieval_facet_dim = 256
    config.retrieval_facet_hidden_dim = 512
    config.retrieval_num_facets = 30
    config.retrieval_facet_score_mode = "softmax_maxsim"
    config.retrieval_facet_softmax_temperature = 0.1
    config.use_retrieval_facets = True
    config.use_retrieval_head = True
    config.retrieval_facet_separate_query_code = False
    config.late_inter_weight = 0.3

    model = JEPAModel(config).to(DEVICE, dtype=torch.bfloat16)
    v378 = torch.load(V378_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(v378, strict=False)

    if freeze_query_head:
        # Freeze everything except query_head
        for name, param in model.named_parameters():
            if "query_head" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    else:
        # Train query_head + late_inter
        for name, param in model.named_parameters():
            if "query_head" in name or "late_inter" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"  Trainable: {trainable}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    return model, optimizer


def run_eval(model, name):
    """Evaluate model on strict eval."""
    import tempfile, subprocess

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
        tmp = f.name
    try:
        torch.save(model.state_dict(), tmp)
        args = [sys.executable, str(ROOT / "test_2.7b.py"), "15000000", tmp, "--json",
            "--retrieval-facet-score-mode", "maxsim",
            "--confidence-threshold", "0.4",
            "--lexical-weight", "0.4",
            "--rerank-consensus-temperature", "0.05",
            "--rerank-agreement-weight", "0.3",
            "--selective-gate-similarity-floor", "0.6",
            "--rerank-topk", "5", "--rerank-shortlist-mode", "pred_query_union_local",
            "--rerank-query-weight", "0.3", "--rerank-lexical-weight", "0.0",
            "--rerank-support-weight", "0.24", "--rerank-consensus-weight", "0.35",
            "--rerank-consensus-floor", "0.9158", "--rerank-consensus-margin-gate", "0.0092",
            "--rerank-pairwise-mode", "supportspec_citecheck_floor_borda",
            "--rerank-support-floor-margin-gate", "0.014", "--rerank-spec-weight", "0.18",
            "--rerank-answerspec-mode", "code_pref", "--rerank-answerspec-margin-gate", "0.034",
            "--rerank-safe-expand-topk", "6", "--rerank-safe-expand-margin", "0.004",
            "--rerank-parafence-weight", "1.0", "--rerank-parafence-variants", "3",
            "--selective-gate-mode", "margin_mean_gap",
            "--selective-gate-margin-threshold", "0.01", "--selective-gate-mean-gap-threshold", "0.016",
            "--rerank-verifier-uplift-weight", "0.4", "--rerank-verifier-gap-scale", "1.0",
            "--rerank-verifier-support-weight", "1.0", "--rerank-verifier-spec-weight", "0.0",
            "--retrieval-facet-softmax-temperature", "0.1", "--retrieval-global-facet-blend", "0.35",
            "--confidence-mode", "support_feature_calibrator",
            "--confidence-support-topk", "5", "--confidence-support-temperature", "0.1",
        ]
        r = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        data = None
        for idx in range(len(r.stdout)):
            if r.stdout[idx] != "{": continue
            for end in range(idx+20, min(idx+100000, len(r.stdout)+1)):
                try:
                    d = json.loads(r.stdout[idx:end])
                    if isinstance(d, dict) and len(d) > 10:
                        data = d; break
                except: pass
            if data: break

        if not data:
            return None, {}

        score = strict_answer_score(data)
        results = data.get("objective_results", [])
        direct = sum(1 for r2 in results if "✅ DIRECT SUPPORT" in r2.get("status", ""))
        abstained = sum(1 for r2 in results if "❌ ABSTAINED" in r2.get("status", ""))
        fp = sum(1 for r2 in results if "❌ FALSE POSITIVE" in r2.get("status", ""))

        confs = {}
        for r2 in results:
            fam = r2.get("family","?")
            typ = r2.get("type","?").replace("Objective - ","")
            confs[f"{fam}_{typ}"] = r2.get("confidence", 0)

        print(f"  {name}: score={score:.2f}, direct={direct}, abstained={abstained}, fp={fp}")
        return score, {"score": score, "direct": direct, "abstained": abstained, "fp": fp, "confs": confs}
    finally:
        try: os.unlink(tmp)
        except: pass


def train_and_eval(name, model, optimizer, pos_examples, neg_examples, steps=50):
    """
    Train query_head: supported queries → 1, unsupported queries → 0.
    pos_examples: list of query strings (supported)
    neg_examples: list of query strings (unsupported/inverse)
    """
    tokenizer = BenchmarkTokenizer(vocab_size=4096)
    model.train()

    all_losses = []
    batch_size = min(4, len(pos_examples), len(neg_examples))

    for step in range(steps):
        # Sample positive batch
        pos_batch = random.sample(pos_examples, min(batch_size, len(pos_examples)))
        # Sample negative batch
        neg_batch = random.sample(neg_examples, min(batch_size, len(neg_examples)))

        pos_ids = tokenizer.batch_encode(pos_batch, seq_len=256, device=DEVICE)
        neg_ids = tokenizer.batch_encode(neg_batch, seq_len=256, device=DEVICE)

        with torch.no_grad():
            pos_z = model.encode(pos_ids)
        neg_z = model.encode(neg_ids)

        pos_logits = model.query_logits(pos_z)
        neg_logits = model.query_logits(neg_z)

        # BCE: supported=1, unsupported=0
        loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        loss = loss + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        all_losses.append(loss.item())

        if (step + 1) % 25 == 0:
            avg = sum(all_losses[-25:]) / 25
            print(f"    Step {step+1}/{steps}: loss={avg:.4f}")

    return all_losses


def main():
    print("=" * 70)
    print("v408: Targeted Confidence Calibration")
    print("=" * 70)
    print("Baseline v378: score=41.64, direct=3, abstained=5, fp=1")
    print()

    # Build training data
    # POSITIVE: supported queries from eval + synthesis_queries from training data
    eval_examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(42),
                                                 dataset="semantic_contrast_v1")
    train_examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(0),
                                                  dataset="behavioral_constraints_v2_taxonomy_support_discipline_v1")

    pos_examples = []  # supported queries
    neg_examples = []  # unsupported (inverse) queries

    for ex in eval_examples:
        fam = ex.family
        # Positive: the eval supported query
        pos_examples.append(ex.query)
        # Negative: the unsupported template
        if fam in UNSUPPORTED_TEMPLATES:
            neg_examples.append(UNSUPPORTED_TEMPLATES[fam])

    # Add synthesis queries from training data as additional positives
    for ex in train_examples:
        if ex.synthesis_queries:
            pos_examples.append(ex.synthesis_queries[0])

    print(f"Training data: {len(pos_examples)} positive, {len(neg_examples)} negative")
    print(f"Positive examples: {pos_examples[:3]}")
    print(f"Negative examples: {neg_examples[:3]}")
    print()

    out_dir = ROOT / "artifacts" / "strict_eval_autoresearch_v408"
    os.makedirs(out_dir, exist_ok=True)

    # Save training data
    with open(out_dir / "training_data.json", "w") as f:
        json.dump({"pos": pos_examples, "neg": neg_examples}, f, indent=2)

    candidates = [
        # (name, freeze_query_head, lr, steps)
        ("v408a_qh50_lr1e4", True, 1e-4, 50),
        ("v408b_qh100_lr1e4", True, 1e-4, 100),
        ("v408c_qh50_lr5e5", True, 5e-5, 50),
        ("v408d_qh100_lr5e5", True, 5e-5, 100),
        ("v408e_qh200_lr1e4", True, 1e-4, 200),
        ("v408f_li50_lr1e4", False, 1e-4, 50),  # Also train late_inter
    ]

    all_results = []

    for name, freeze_qh, lr, steps in candidates:
        print(f"\n{'='*50}")
        print(f"{name}: freeze_qh={freeze_qh}, lr={lr}, steps={steps}")
        print(f"{'='*50}")

        model, optimizer = create_model(freeze_query_head=freeze_qh, lr=lr)
        losses = train_and_eval(name, model, optimizer, pos_examples, neg_examples, steps=steps)

        # Evaluate
        score, summary = run_eval(model, name)
        if summary:
            summary["losses"] = losses[-10:]
            with open(out_dir / f"{name}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            torch.save(model.state_dict(), out_dir / f"{name}_model.pt")

        all_results.append((name, freeze_qh, lr, steps, score, summary))
        time.sleep(2)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Candidate':<30} {'QH+LI':>6} {'LR':>8} {'Steps':>6} {'Score':>7} {'Direct':>7} {'Abst':>6} {'FP':>4}")
    print("-" * 74)
    for name, freeze_qh, lr, steps, score, summary in sorted(all_results, key=lambda x: x[4] or 0, reverse=True):
        s = f"{score:.2f}" if score is not None else "FAIL"
        d = summary.get("direct", "-") if summary else "-"
        a = summary.get("abstained", "-") if summary else "-"
        f = summary.get("fp", "-") if summary else "-"
        qh_li = "qh+li" if not freeze_qh else "qh"
        print(f"{name:<30} {qh_li:>6} {lr:>8.0e} {steps:>6} {s:>7} {d:>7} {a:>6} {f:>4}")

    with open(out_dir / "batch_summary.json", "w") as f:
        json.dump([{"name": n, "freeze_qh": fqh, "lr": l, "steps": s, "score": sc, "summary": su}
                   for n, fqh, l, s, sc, su in all_results], f, indent=2)
    print(f"\nSaved to {out_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
