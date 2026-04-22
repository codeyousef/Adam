#!/usr/bin/env python3
"""
v415 Launch Script — Unsupported Queries Diagnostic on v378 Proven Baseline

AIM: Test whether unsupported_queries (per-family same-family negatives that differ
only in operator/modifier) add signal when added to the PROVEN behavioral_constraints_v2
dataset WITHOUT changing training loss or swapping datasets.

Key difference from v414: v414 swapped to answerability_distilled_v1 dataset
(+ new answerability_distilled_loss). v415 keeps v378's proven dataset and just
extends it with unsupported_queries as additional cross-family negatives.

v378 baseline: score=41.64 (D=3, A=5, FP=1, SF=0)
  - json_parse-u FP: serialize retrieves parse code (corpus gap)
  - fetch_json-u FP: POST retrieves GET code (corpus gap)
  - Hard families abstain: strip_lines, debounce, frequency, merge_dicts, startswith_js
    (encode_sim ~0.41 between supported/unsupported, below 0.6 gate floor)

v415 hypothesis: Adding unsupported_queries as additional cross_family_negatives
inside the proven behavioral_constraints_v2_taxonomy_support_discipline_v1 dataset
will improve abstention WITHOUT diluting the working signal.

- phase4_dataset: behavioral_constraints_v2_taxonomy_support_discipline_v1 (UNCHANGED from v378)
- answerability_distilled_weight: 0.0 (no new loss, just extra negatives)
- All other params: v378 proven values
- warm_start_model_path: v378 checkpoint (warm start)

If v415 ≥ v378: unsupported_queries are useful negatives, proceed to v416
If v415 < v378: dataset dilution effect, stick with v378proven
If v415 >> v378: unsupported_queries were blocking progress, keep both

What NOT to spend time on:
  - Longer training (test the signal, not the step count)
  - Loss changes (this is a data-only diagnostic)
  - Classifier weight changes
"""
import sys, os, json, subprocess, shutil, yaml
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
PY = str(ROOT / "../.venv/bin/python")
RUN_DIR = ROOT / "artifacts/strict_eval_autoresearch_v4/v415-unsupported-diagnostic-seed703"
RUN_DIR.mkdir(parents=True, exist_ok=True)

V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V338_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"

# Build config from v378 proven config
config = yaml.safe_load((V378_CKPT.parent / "config.yaml").read_text())
phase4 = config.setdefault("phase4", {})

# === v415 Identity ===
config["seed"] = 703
config["profile"] = "strict-eval-autoresearch-v4-v415-unsupported-diagnostic"
config["warm_start_phase3_only"] = False
config["warm_start_model_path"] = str(V378_CKPT)
config["base_model_path"] = str(V338_CKPT)

phase4["seed"] = 703
phase4["steps"] = 300
phase4["phase4_steps"] = 300
phase4["classifier_weight"] = 0.09
phase4["clf_weight"] = 0.09
phase4["query_multiview_weight"] = 1.0
phase4["warm_start_phase3_only"] = False
phase4["warm_start_model_path"] = str(V378_CKPT)
phase4["production_mode"] = False
phase4["production_steps"] = 0
phase4["production_phase4_repeats"] = 0

# === Research9 Diagnostic: Add unsupported_queries as extra cross-family negatives ===
# KEEP v378's proven dataset (behavioral_constraints_v2_taxonomy_support_discipline_v1)
# DO NOT swap to answerability_distilled_v1
# DO NOT enable answerability_distilled_loss
phase4["phase4_dataset"] = "behavioral_constraints_v2_taxonomy_support_discipline_v1"
phase4["answerability_distilled_weight"] = 0.0  # Diagnostic only — no new loss

# Unsupported queries ARE added to _PHASE4_CONTRAST_FAMILIES in data.py
# so they appear as cross_family_negatives in the example.
# This is the only change from v378proven config.

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

# Build training command
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

result = subprocess.run(train_cmd, cwd=ROOT, timeout=36000)

print(f"\nReturn code: {result.returncode}")
if result.returncode != 0:
    print("STDERR:")
    print(result.stderr[-3000:] if result.stderr else "(none)")
    sys.exit(result.returncode)

print(f"\nModel saved to {model_path}")
print("Run strict eval next:")
print(f"  python test_2.7b.py --strict-eval 15000000 {model_path}")
