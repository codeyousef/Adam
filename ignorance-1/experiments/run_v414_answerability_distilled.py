#!/usr/bin/env python3
"""
v414 Launch Script — Answerability-Distilled Encoder Geometry Training

Research9 hypothesis: The wall is encoder supervision geometry, not reranker calibration.
The bi-encoder learns coarse family similarity but not the direct-support boundary.

Key changes from v378:
  - phase4_dataset: "answerability_distilled_v1" (was "benchmark_v1" implicit)
  - answerability_distilled_weight: 1.0 (loss term fires only when > 0)
  - Starting hyperparams from memo:
      margin = 0.20
      unsupported_ceiling = 0.45
      pairwise_weight = 1.0
      pushaway_weight = 0.5
      uniformity_weight = 0.05
      temperature = 0.07
  - classifier_weight: 0.09 (unchanged from v378 — keep proven config)
  - query_multiview_weight: 1.0 (unchanged from v378)
  - freeze_backbone: true (unchanged from v378)
  - phase4_steps: 300 (unchanged from v378)
  - warm_start_model_path: v378 checkpoint

Primary success metric: PRE-GATE encoder geometry, not final strict-eval score.
  Target: sim(q_supported, champion) - sim(q_unsupported, champion) widens
  without both rising together.

Sequencing for AutoResearcher:
  1. Run v414 (this) — answerability-distilled family-local contrast training
  2. If hard-family geometry improves: scale training
  3. If supported and unsupported both rise: strengthen push-away, relabel ambiguous negatives
  4. If geometry does not move: try multi-vector or operator-aware first-stage retrieval
  5. If only inverse-operation FPs remain: add operator-aware residual channel
  6. If ambiguity still dominates: move to atomic-support indexing

What NOT to spend cycles on:
  - Reranker pooler toggles
  - Confidence threshold sweeps
  - Generic classifier_weight bumps outside answerability context
  - Broad paraphrase/multiview augmentation
"""
import sys, os, json, subprocess, shutil, yaml
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
PY = str(ROOT / "../.venv/bin/python")
RUN_DIR = ROOT / "artifacts/strict_eval_autoresearch_v4/v414-answerability-distilled-seed702"
RUN_DIR.mkdir(parents=True, exist_ok=True)

V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V338_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"

# Build config from v378 proven config
config = yaml.safe_load((V378_CKPT.parent / "config.yaml").read_text())
phase4 = config.setdefault("phase4", {})

# === v414 Identity ===
config["seed"] = 702
config["profile"] = "strict-eval-autoresearch-v4-v414-answerability-distilled"
config["warm_start_phase3_only"] = False
config["warm_start_model_path"] = str(V378_CKPT)
config["base_model_path"] = str(V338_CKPT)

phase4["seed"] = 702
phase4["steps"] = 300  # Phase4 steps (same as v378 proven config)
phase4["phase4_steps"] = 300  # Reference only
phase4["classifier_weight"] = 0.09  # Keep v378 proven value
phase4["clf_weight"] = 0.09  # Keep v378 proven value

# === Research9: Answerability-Distilled Encoder Geometry ===
# These params are read by _run_phase4_joint_training in phase4.py
phase4["phase4_dataset"] = "answerability_distilled_v1"  # Hard families + unsupported_queries
phase4["answerability_distilled_weight"] = 1.0  # Fire the new loss term
phase4["answerability_margin"] = 0.20
phase4["answerability_unsupported_ceiling"] = 0.45
phase4["answerability_pairwise_weight"] = 1.0
phase4["answerability_pushaway_weight"] = 0.5
phase4["answerability_uniformity_weight"] = 0.05
phase4["answerability_temperature"] = 0.07

# Keep v378 proven training params
phase4["query_multiview_weight"] = 1.0  # Already set in v378 but be explicit
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

# Copy warm_start to tmp for resume (train_production.py auto-detects {output}.tmp)
print(f"Copying warm_start from {V378_CKPT} -> {tmp_ckpt}")
shutil.copy2(V378_CKPT, tmp_ckpt)

# Build training command (matches run_strict_eval_autoresearch.py pattern)
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

# Run with 10-hour timeout (long enough for 300 Phase4 steps on 2.7B model)
result = subprocess.run(train_cmd, cwd=ROOT, timeout=36000)

print(f"\nReturn code: {result.returncode}")
if result.returncode != 0:
    print("STDERR:")
    print(result.stderr[-3000:] if result.stderr else "(none)")
    sys.exit(result.returncode)

print(f"\nModel saved to {model_path}")
print("Run strict eval next:")
print(f"  python test_2.7b.py --strict-eval 15000000 {model_path}")
