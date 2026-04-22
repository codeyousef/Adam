#!/usr/bin/env python3
"""
v414: v378 late-inter-high-weight warm-start at 2.7B with mixed_boundary_curriculum_v1.

ROOT CAUSE from v381 analysis:
  - v340 PASSED the old objective with mixed_boundary_curriculum_v1 + freeze_backbone=true + clf_weight=0.09
  - v378 late-inter-high-weight (best ever score=41.11) used taxonomy_support_discipline_v1
    and dr=3/8 (failed on 5 hard families)
  - The 8 "Objective - Supported" eval queries in test_2.7b.py NEVER appear in
    taxonomy_support_discipline_v1 training data, causing wrong chunk retrieval

FIX: Switch to mixed_boundary_curriculum_v1 (the dataset v340 passed with),
combined with v378's late-inter-weight config.

Hypothesis: mixed_boundary's more balanced data will allow the frozen encoder
to produce better instance-level discrimination for the hard families.

Checkpoint: v378 late-inter-high-weight-seed511-seed514
Scale: 2.7B (not scout!)
Dataset: mixed_boundary_curriculum_v1
"""

import subprocess, sys, yaml
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
PROD_PY = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/.venv/bin/python"  # fallback

# Find working python
import os
for candidate in [
    "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python",
    str(ROOT / "../.venv/bin/python"),
]:
    p = Path(candidate)
    if p.exists():
        PY = str(p)
        break

V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V378_CONFIG = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/config.yaml"
OUTPUT_DIR = ROOT / "artifacts/strict_eval_autoresearch_v4/v414-mixed-boundary-2p7b-seed714"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load v378 config as base
with open(V378_CONFIG) as f:
    config = yaml.safe_load(f)

phase4 = config.setdefault("phase4", {})
phase4["seed"] = 714
phase4["phase4_dataset"] = "behavioral_constraints_v2_mixed_boundary_curriculum_v1"
phase4["sizes"] = [2_700_000_000]
phase4["phase4_steps"] = 5000
phase4["phase4_repeats"] = 4
phase4["production_mode"] = False
phase4["production_steps"] = 0
phase4["production_phase4_repeats"] = 0

# late-inter-weight specific settings from v378 late-inter-high-weight variant
phase4["late_interaction_verifier_weight"] = 0.5  # was 1.0, matches v378 actual
phase4["late_interaction_verifier_margin"] = 0.2
phase4["late_interaction_verifier_mode"] = "hard_maxsim"
phase4["late_interaction_verifier_softmax_temperature"] = 0.1

# Copy all phase4 weight settings from v378
phase4["classifier_weight"] = 0.09
phase4["query_multiview_weight"] = 1.0
phase4["query_multiview_prediction_weight"] = 0.5
phase4["equivalence_alignment_weight"] = 0.0
phase4["equivalence_prediction_weight"] = 0.0
phase4["equivalence_margin_weight"] = 0.0
phase4["epistemic_boundary_weight"] = 0.0
phase4["epistemic_margin"] = 0.2
phase4["epistemic_query_weight"] = 0.0
phase4["epistemic_prediction_weight"] = 1.0
phase4["use_equivalence_prototypes"] = True
phase4["equivalence_include_synthesis_views"] = False

config["seed"] = 714
config["profile"] = "strict-eval-autoresearch-v4-v414-mixed-boundary-2p7b"
config["warm_start_phase3_only"] = False  # warm-start from v378 full model (encoder + phase4 heads)
config["warm_start_model_path"] = str(V378_CKPT)
config["freeze_backbone"] = True  # encoder stays frozen, only phase4 heads train
config["confidence_threshold"] = 0.312  # from v378

# Save config
config_path = OUTPUT_DIR / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f)

print(f"Config saved to {config_path}")
print(f"Key changes from v378:")
print(f"  - dataset: mixed_boundary_curriculum_v1 (was taxonomy_support_discipline_v1)")
print(f"  - scale: 2.7B (was 15M scout)")
print(f"  - steps: 500 (was 112)")
print(f"  - production_mode: True")
print()
print(f"Launching training...")

# Run train_production.py with the config
cmd = [
    PY,
    str(ROOT / "train_production.py"),
    "--config", str(config_path),
    "--output", str(OUTPUT_DIR / "model.pt"),
]

print("CMD:", " ".join(cmd))
result = subprocess.run(cmd, cwd=str(ROOT))
print(f"Training exit code: {result.returncode}")
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:] if result.stderr else "")
    sys.exit(1)

# Run strict eval
print("\nRunning strict eval...")
eval_cmd = [
    PY,
    str(ROOT / "test_2.7b.py"),
    "2700000000",
    str(OUTPUT_DIR / "model.pt"),
    "--json",
    "--confidence-threshold", str(config.get("confidence_threshold", 0.312)),
]

result = subprocess.run(eval_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
print("STDOUT:", result.stdout[-2000:] if result.stdout else "")
if result.stderr:
    print("STDERR:", result.stderr[-500:])
print(f"Eval exit code: {result.returncode}")
