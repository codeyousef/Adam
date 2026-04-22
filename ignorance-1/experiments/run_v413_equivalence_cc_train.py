#!/usr/bin/env python3
"""
v413 Launch Script — Equivalence Alignment + Higher CC Training

Key changes from v378:
  - equivalence_alignment_weight: 0.5 (was 0.0)
  - equivalence_prediction_weight: 0.5 (was 0.0)
  - equivalence_include_synthesis_views: true (was false)
  - classifier_weight: 0.15 (was 0.09)
  - phase4_steps: 500 (was 300)
  - warm_start_model_path: v378 checkpoint (resumes from v378)
"""
import sys, os, json, subprocess, shutil, yaml
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
PY = str(ROOT / "../.venv/bin/python")
RUN_DIR = ROOT / "artifacts/strict_eval_autoresearch_v413/v413-equivalence-cc-seed700"
RUN_DIR.mkdir(parents=True, exist_ok=True)

V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V338_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"

# Build config from v378 + key changes
v378_config = yaml.safe_load((V378_CKPT.parent / "config.yaml").read_text())

# Deep copy and apply v413 changes
config = yaml.safe_load((V378_CKPT.parent / "config.yaml").read_text())
phase4 = config.setdefault("phase4", {})

config["seed"] = 700
config["profile"] = "strict-eval-autoresearch-v4-v413-equivalence-cc"
config["warm_start_phase3_only"] = False
config["warm_start_model_path"] = str(V378_CKPT)
config["base_model_path"] = str(V338_CKPT)

phase4["seed"] = 700
phase4["steps"] = 500  # Controls actual step count via _scaled_training_hparams
phase4["phase4_steps"] = 500  # Reference only (train_production.py uses phase4.steps)
phase4["classifier_weight"] = 0.15
phase4["clf_weight"] = 0.15
phase4["equivalence_alignment_weight"] = 0.5
phase4["equivalence_prediction_weight"] = 0.5
phase4["equivalence_include_synthesis_views"] = True
phase4["warm_start_phase3_only"] = False
phase4["warm_start_model_path"] = str(V378_CKPT)
phase4["production_mode"] = False
phase4["production_steps"] = 0
phase4["production_phase4_repeats"] = 0

# Save config
config_path = RUN_DIR / "config.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print(f"Config saved to {config_path}")

# Model output path
model_path = RUN_DIR / "model.pt"
tmp_ckpt = str(model_path) + ".tmp"

# Copy warm_start to tmp for resume
print(f"Copying warm_start from {V378_CKPT} -> {tmp_ckpt}")
shutil.copy2(V378_CKPT, tmp_ckpt)

# Build training command (matches run_strict_eval_autoresearch.py pattern)
# Note: NO --resume flag — train_production.py auto-detects {output}.tmp
train_cmd = [
    PY,
    str(ROOT / "train_production.py"),
    "--config", str(config_path),
    "--size", str(int((config.get("sizes") or config.get("phase4", {}).get("sizes", [15_000_000]))[0])),
    "--output", str(model_path),
    "--device", str(config.get("device", "cuda")),
]

print("\nTraining command:")
print(" ".join(train_cmd))
print()

result = subprocess.run(train_cmd, cwd=ROOT, timeout=43200)

with open(RUN_DIR / "train_output.log", "w") as f:
    f.write(result.stdout)
with open(RUN_DIR / "train_error.log", "w") as f:
    f.write(result.stderr)

if result.returncode != 0:
    print(f"Training FAILED: {result.returncode}")
    print(result.stderr[-3000:])
    sys.exit(1)

print("Training complete!")

# Find checkpoint
checkpoints = sorted(RUN_DIR.glob("step_*.pt"))
if checkpoints:
    final_ckpt = checkpoints[-1]
    print(f"Final checkpoint: {final_ckpt}")
else:
    final_ckpt = model_path
    print(f"Using model.pt: {final_ckpt}")

# Evaluate
print("\nEvaluating...")
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

if not data:
    print("EVAL FAILED"); print(r.stdout[-1000:]); sys.exit(1)

from research.strict_eval_search_space import strict_answer_score
score = strict_answer_score(data)
obj = data.get("objective_results", [])

print(f"\nV413 Score: {score:.2f} (baseline v378=41.64)")
print()
for r2 in sorted(obj, key=lambda x: (x.get("family",""), x.get("type",""))):
    fam = r2.get("family","?").replace("Objective - ","")
    typ = r2.get("type","?").replace("Objective - ","")
    conf = r2.get("confidence", 0)
    sim = r2.get("similarity", 0)
    status = r2.get("status","?")
    print(f"  {fam:<15} {typ:<12} conf={conf:.4f} sim={sim:.4f} {status}")

direct = sum(1 for r2 in obj if "✅ DIRECT SUPPORT" in r2.get("status",""))
fp = sum(1 for r2 in obj if "❌ FALSE POSITIVE" in r2.get("status",""))
ci = sum(1 for r2 in obj if "CORRECTLY IGNORANT" in r2.get("status",""))
abstain = sum(1 for r2 in obj if "❌ ABSTAINED" in r2.get("status",""))
sf = sum(1 for r2 in obj if "SAME-FAMILY" in r2.get("status",""))
print(f"\nD={direct} FP={fp} CI={ci} A={abstain} SF={sf} | Score={score:.2f}")

with open(RUN_DIR / "eval_results.json", "w") as f:
    json.dump({"score": score, "direct": direct, "fp": fp, "ci": ci, "abstain": abstain, "sf": sf, "results": obj}, f, indent=2)

print(f"\nDone: {RUN_DIR}")
