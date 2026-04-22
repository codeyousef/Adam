#!/usr/bin/env python3
"""
v413: Counterfactual Multi-View CC Training

Experiment 1 from research9.md:
- Resume from v378 checkpoint
- Multi-view query alignment: use synthesis_queries as alternate views
- Semantic-inverse negatives: hard negatives from unsupported variants of SAME family
- Anti-collapse: spread loss on embeddings + VICReg variance term
- CC weight = 0.3 (active ingredient from v376)
- Late inter weight = 0.3 (from v378)

The key insight from research9:
  "multi-view alignment loss uses the synthesis_queries as alternative views of the
  same semantic content, pushing all views toward the same code embedding while
  the anti-collapse term prevents the code embeddings from collapsing to a point."
"""
from __future__ import annotations

import sys, os, random, json, tempfile, subprocess, torch, copy
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
sys.path.insert(0, str(ROOT))
PY = sys.executable

# Checkpoint from v378 (best so far at 41.64)
V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V413_DIR = ROOT / "artifacts/strict_eval_autoresearch_v413"
V413_DIR.mkdir(exist_ok=True, parents=True)

# Training config
NUM_STEPS = 3000
BATCH_SIZE = 16
LR = 3e-5  # Conservative LR for fine-tuning
CC_WEIGHT = 0.3  # Active ingredient from v376
LATE_INTER_WEIGHT = 0.3  # From v378
QUERY_MULTIVIEW_WEIGHT = 0.5  # New: multi-view alignment
SEMANTIC_INVERSE_WEIGHT = 0.5  # New: push away unsupported variants
ANTI_COLLAPSE_WEIGHT = 0.3  # New: anti-collapse spread
RANKING_MARGIN = 0.3  # From v378
SEED = 700

print("=" * 70)
print("v413: Counterfactual Multi-View CC Training")
print("=" * 70)
print(f"Resume from: {V378_CKPT}")
print(f"CC weight: {CC_WEIGHT}")
print(f"Late inter weight: {LATE_INTER_WEIGHT}")
print(f"Query multiview weight: {QUERY_MULTIVIEW_WEIGHT}")
print(f"Semantic-inverse weight: {SEMANTIC_INVERSE_WEIGHT}")
print(f"Anti-collapse weight: {ANTI_COLLAPSE_WEIGHT}")
print(f"Steps: {NUM_STEPS}, Batch: {BATCH_SIZE}, LR: {LR}")

# Create temporary symlink to v378 checkpoint (to avoid copying)
tmp_ckpt = f"/tmp/v413_base_{SEED}.pt"
torch.save(torch.load(V378_CKPT, map_location="cpu", weights_only=False), tmp_ckpt)
print(f"Loaded checkpoint to {tmp_ckpt}")

# Build the training command
# Key new flags:
#   --late-inter-weight 0.3 (from v378)
#   --classifier-weight 0.3 (CC, from v376)
#   --use-query-multiview (new: use synthesis_queries as alternate views)
#   --query-multiview-weight 0.5 (new)
#   --semantic-inverse-negative-weight 0.5 (new: use unsupported queries as hard negatives)
#   --anti-collapse-weight 0.3 (new: spread loss)
#   --ranking-margin 0.3 (from v378)
#   --use-phase4-contrast-data (already in v378)
#   --freeze-backbone (keep encoder frozen, train late_inter + query_head)

train_cmd = [
    PY, str(ROOT / "train_production.py"),
    "--size", "15000000",
    "--train-steps", str(NUM_STEPS),
    "--batch-size", str(BATCH_SIZE),
    "--learning-rate", str(LR),
    "--warmup-steps", "100",
    "--late-inter-weight", str(LATE_INTER_WEIGHT),
    "--classifier-weight", str(CC_WEIGHT),
    "--use-query-multiview",
    "--query-multiview-weight", str(QUERY_MULTIVIEW_WEIGHT),
    "--semantic-inverse-negative-weight", str(SEMANTIC_INVERSE_WEIGHT),
    "--anti-collapse-weight", str(ANTI_COLLAPSE_WEIGHT),
    "--ranking-margin", str(RANKING_MARGIN),
    "--use-phase4-contrast-data",
    "--dataset", "semantic_contrast_v1",
    "--prompt-template", "default",
    "--phase4-balance-families",
    "--freeze-backbone",
    "--resume-from", tmp_ckpt,
    "--seed", str(SEED),
    "--log-interval", "100",
    "--save-interval", "500",
    "--output-dir", str(V413_DIR),
]

print()
print("Training command:")
print(" ".join(train_cmd))
print()

result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=36000)

# Save output
with open(V413_DIR / "train_output.log", "w") as f:
    f.write(result.stdout)
with open(V413_DIR / "train_error.log", "w") as f:
    f.write(result.stderr)

if result.returncode != 0:
    print(f"Training FAILED with exit code {result.returncode}")
    print("STDERR:")
    print(result.stderr[-3000:])
    sys.exit(1)

print("Training completed!")
print()

# Find the final checkpoint
checkpoints = sorted((V413_DIR).glob("step_*.pt"))
if checkpoints:
    final_ckpt = checkpoints[-1]
    print(f"Final checkpoint: {final_ckpt}")
else:
    # Look for the main checkpoint
    main_ckpt = V413_DIR / "model.pt"
    if main_ckpt.exists():
        final_ckpt = main_ckpt
        print(f"Final checkpoint: {final_ckpt}")
    else:
        print("ERROR: No checkpoint found!")
        sys.exit(1)

# Evaluate on strict eval
print()
print("=" * 70)
print("Evaluating on strict eval...")
print("=" * 70)

eval_cmd = [
    PY, str(ROOT / "test_2.7b.py"), "15000000", str(final_ckpt), "--json",
    "--retrieval-facet-score-mode", "maxsim",
    "--confidence-threshold", "0.38",
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

r = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=300)
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

if data:
    from research.strict_eval_search_space import strict_answer_score
    score = strict_answer_score(data)
    results = data.get("objective_results", [])
    print(f"\nV413 Strict Eval Score: {score:.2f}")
    print()
    print(f"{'Family':<15} {'Type':<12} {'Conf':>7} {'Sim':>7} {'Status':<40}")
    print("-" * 85)
    for r2 in sorted(results, key=lambda x: (x.get("family",""), x.get("type",""))):
        fam = r2.get("family","?").replace("Objective - ","")
        typ = r2.get("type","?").replace("Objective - ","")
        conf = r2.get("confidence", 0)
        sim = r2.get("similarity", 0)
        status = r2.get("status","?")
        print(f"{fam:<15} {typ:<12} conf={conf:.4f} sim={sim:.4f} {status}")

    direct = sum(1 for r2 in results if "✅ DIRECT SUPPORT" in r2.get("status",""))
    fp = sum(1 for r2 in results if "❌ FALSE POSITIVE" in r2.get("status",""))
    ci = sum(1 for r2 in results if "CORRECTLY IGNORANT" in r2.get("status",""))
    abstain = sum(1 for r2 in results if "❌ ABSTAINED" in r2.get("status",""))
    sf = sum(1 for r2 in results if "SAME-FAMILY" in r2.get("status",""))
    print(f"\nDIRECT={direct}, FP={fp}, CI={ci}, ABSTAIN={abstain}, SAME_FAM={sf}")
    print(f"Score: {score:.2f} (baseline v378=41.64)")

    # Save results
    with open(V413_DIR / "eval_results.json", "w") as f:
        json.dump({"score": score, "direct": direct, "fp": fp, "ci": ci, "abstain": abstain, "same_family": sf, "results": results}, f, indent=2)
else:
    print("EVALUATION FAILED")
    print(r.stdout[-1000:])
    sys.exit(1)

# Cleanup
os.unlink(tmp_ckpt)
print(f"\nDone! Results saved to {V413_DIR}")
