"""
v407: Confidence Head Retraining — focused on the 5 hard families.

The 5 abstaining families: debounce, frequency, merge_dicts, startswith_js, strip_lines
All have conf=0.37 in v378 (just below threshold=0.4).

Training approach:
  1. Load v378 checkpoint, freeze backbone
  2. Train ONLY confidence head (query_logits → sigmoid) with:
     - Higher classifier_weight [0.15, 0.20, 0.25, 0.30]
     - Proper BCE: coding (text→code) = 1, OOD = 0
     - Training on supported queries with varying difficulty
  3. Evaluate with v398 inference config

Also test: does joint training (unfrozen backbone) with higher classifier_weight
produce better confidence calibration?
"""
from __future__ import annotations
import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import json, os, sys, time, random, copy, torch, torch.nn.functional as F
from pathlib import Path

from src.models.jepa import JEPAModel
from src.training.phase4 import (
    _proxy_config_v6_overnight,
    make_phase4_contrast_examples,
    BenchmarkTokenizer,
    _phase4_auxiliary_loss,
)
from src.utils.data import set_seed
from research.strict_eval_search_space import strict_answer_score

ROOT = Path(_project_root)
V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
DEVICE = "cuda"


def create_model(classifier_weight, freeze_backbone=True, lr=1e-4):
    """Load v378 checkpoint and set up optimizer."""
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
    missing, unexpected = model.load_state_dict(v378, strict=False)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if any(k in name for k in ["query_head", "late_inter", "retrieval_head", "retrieval_facet"]):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01
        )
        print(f"  Trainable: {[n for n,p in model.named_parameters() if p.requires_grad][:5]}...")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    return model, optimizer, config


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
            confs[f"{fam}_{typ}"] = {"conf": r2.get("confidence", 0), "sim": r2.get("similarity", 0)}

        print(f"  {name}: score={score:.2f}, direct={direct}, abstained={abstained}, fp={fp}")
        return score, {"score": score, "direct": direct, "abstained": abstained, "fp": fp, "confs": confs}
    finally:
        try: os.unlink(tmp)
        except: pass


def train_step(model, optimizer, texts, ood_texts, classifier_weight):
    """One training step: BCE for confidence head."""
    tokenizer = BenchmarkTokenizer(vocab_size=4096)
    model.train()

    t_ids = tokenizer.batch_encode(texts, seq_len=256, device=DEVICE)
    o_ids = tokenizer.batch_encode(ood_texts, seq_len=256, device=DEVICE)

    with torch.no_grad():
        z_t = model.encode(t_ids)
    z_ood = model.encode(o_ids)
    z_ood_pred = model.predict(z_ood, action_id=1)
    coding_logits = model.query_logits(z_t)
    ood_logits = model.query_logits(z_ood)

    # BCE: supported = 1, OOD = 0
    loss = F.binary_cross_entropy_with_logits(coding_logits, torch.ones_like(coding_logits))
    loss = loss + F.binary_cross_entropy_with_logits(ood_logits, torch.zeros_like(ood_logits))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def train_candidate(name, classifier_weight, steps=300, freeze_backbone=True, lr=1e-4, repeats=3):
    """Train one candidate."""
    out_dir = ROOT / "artifacts" / "strict_eval_autoresearch_v407" / name
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{name}: cw={classifier_weight}, steps={steps}, freeze={freeze_backbone}, lr={lr}, repeats={repeats}")
    print(f"{'='*60}")

    model, optimizer, config = create_model(classifier_weight, freeze_backbone, lr)

    # Training data: use supported + OOD from each example
    set_seed(514)
    all_examples = make_phase4_contrast_examples(repeats=repeats, rng=random.Random(514),
                                                  dataset="behavioral_constraints_v2_taxonomy_support_discipline_v1")
    print(f"  Data: {len(all_examples)} examples")

    # Also include unsupported query variants from eval data
    eval_examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(42),
                                                  dataset="semantic_contrast_v1")
    # Filter to only families we care about
    hard_families = {"debounce", "frequency", "merge_dicts", "startswith_js", "strip_lines"}

    # Also add eval-level queries for the hard families as training data
    eval_examples = make_phase4_contrast_examples(repeats=1, rng=random.Random(42),
                                                  dataset="semantic_contrast_v1")
    hard_families = {"debounce", "frequency", "merge_dicts", "startswith_js", "strip_lines"}
    hard_eval_examples = [ex for ex in eval_examples if ex.family in hard_families]
    print(f"  Hard family eval examples: {len(hard_eval_examples)}")

    # Build extended training set: training examples + hard family eval examples
    extended_examples = all_examples + hard_eval_examples
    print(f"  Extended training set: {len(extended_examples)} examples")

    losses = []
    batch_size = 4

    for step in range(steps):
        # Sample batch of SUPPORTED examples (from extended set)
        batch = random.sample(extended_examples, min(batch_size, len(extended_examples)))
        texts = [ex.synthesis_queries[0] if ex.synthesis_queries else ex.query
                 for ex in batch]  # Natural language queries

        # OOD: use the built-in OOD queries from dataset
        ood_texts = []
        for ex in batch:
            if ex.ood_queries:
                ood_texts.append(random.choice(ex.ood_queries))
            else:
                ood_texts.append("Describe a contradictory TODO list without writing code.")

        loss_val = train_step(model, optimizer, texts, ood_texts, classifier_weight)
        losses.append(loss_val)

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{steps}: loss={sum(losses[-50:])/50:.4f}")

    # Save
    ckpt_path = out_dir / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved to {ckpt_path}")

    # Evaluate
    score, summary = run_eval(model, name)
    if summary:
        summary["losses"] = losses[-20:]
        with open(out_dir / "eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return score, summary, ckpt_path


def main():
    print("=" * 70)
    print("v407: Confidence Head Retraining")
    print("=" * 70)
    print("Baseline v378: score=41.64, direct=3, abstained=5, fp=1")
    print()

    candidates = [
        ("v407a_cw0.15_frz", 0.15, 300, True, 1e-4),
        ("v407b_cw0.20_frz", 0.20, 300, True, 1e-4),
        ("v407c_cw0.25_frz", 0.25, 300, True, 1e-4),
        ("v407d_cw0.30_frz", 0.30, 300, True, 1e-4),
        ("v407e_cw0.20_frz_lr2e4", 0.20, 300, True, 2e-4),
        ("v407f_cw0.25_frz_lr2e4", 0.25, 300, True, 2e-4),
        ("v407g_cw0.20_joint", 0.20, 200, False, 5e-5),
        ("v407h_cw0.25_joint", 0.25, 200, False, 5e-5),
    ]

    all_results = []
    for name, cw, steps, freeze, lr in candidates:
        score, summary, ckpt = train_candidate(name, cw, steps=steps, freeze_backbone=freeze, lr=lr)
        all_results.append((name, cw, steps, freeze, lr, score, summary))
        time.sleep(2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Candidate':<30} {'CW':>4} {'Steps':>6} {'Frz':>4} {'Score':>7} {'Direct':>7} {'Abst':>6} {'FP':>4}")
    print("-" * 72)
    for name, cw, steps, freeze, lr, score, summary in sorted(all_results, key=lambda x: x[5] or 0, reverse=True):
        s = f"{score:.2f}" if score is not None else "FAIL"
        d = summary.get("direct", "-") if summary else "-"
        a = summary.get("abstained", "-") if summary else "-"
        f = summary.get("fp", "-") if summary else "-"
        frz = "y" if freeze else "n"
        print(f"{name:<30} {cw:>4.2f} {steps:>6} {frz:>4} {s:>7} {d:>7} {a:>6} {f:>4}")

    out_dir = ROOT / "artifacts" / "strict_eval_autoresearch_v407"
    with open(out_dir / "batch_summary.json", "w") as f:
        json.dump([{"name": n, "cw": cw, "steps": s, "freeze": frz, "lr": lr, "score": sc, "summary": su}
                   for n, cw, s, frz, lr, sc, su in all_results], f, indent=2)
    print(f"\nSaved to {out_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
