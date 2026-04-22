#!/usr/bin/env python3
"""
v419 Launch Script — Reranker-Only Focus (late_interaction_verifier_weight scaling)

AIM: The bi-encoder (first-stage) has hit its ceiling — v378/v414/v415/v416 all show
identical margins. The reranker (late-interaction verifier) gets per-token late
interaction scores that can discriminate debounce from throttle IF trained on the right pairs.

v378 already has late_interaction_verifier_weight=0.5, but v419 scales this up
significantly to test whether more reranker signal can fix the same-family problem.

SAFE: Only modifies config. Does NOT touch data.py or loss functions.

Key changes:
  - late_interaction_verifier_weight: 2.0 (was 0.5 in v378 — 4x increase)
  - late_interaction_verifier_margin: 0.15 (was 0.2 — tighter margin)
  - late_interaction_verifier_softmax_temperature: 0.05 (was 0.1 — lower temp = harder)
  - warm_start_model_path: v378 checkpoint
  - All other params: v378 proven values

Decision:
  - v419 > v378: reranker is the path forward for same-family discrimination
  - v419 = v378: reranker can't fix the problem (needs correct champion code)
  - v419 < v378: too much reranker weight destabilizes training
"""
import sys, subprocess, shutil, yaml
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
PY = str(ROOT / "../.venv/bin/python")
RUN_DIR = ROOT / "artifacts/strict_eval_autoresearch_v4/v419-reranker-only-seed707"
RUN_DIR.mkdir(parents=True, exist_ok=True)

V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V338_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"

# === Build v419 config (pure config change, no data.py patching) ===
config = yaml.safe_load((V378_CKPT.parent / "config.yaml").read_text())
phase4 = config.setdefault("phase4", {})

config["seed"] = 707
config["profile"] = "strict-eval-autoresearch-v4-v419-reranker-only"
config["warm_start_phase3_only"] = False
config["warm_start_model_path"] = str(V378_CKPT)
config["base_model_path"] = str(V338_CKPT)

phase4["seed"] = 707
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
phase4["phase4_dataset"] = "behavioral_constraints_v2_taxonomy_support_discipline_v1"

# === v419 key changes: Reranker-only focus ===
phase4["late_interaction_verifier_weight"] = 2.0  # Was 0.5 in v378 — 4x increase
phase4["late_interaction_verifier_margin"] = 0.15  # Was 0.2 — tighter margin
phase4["late_interaction_verifier_mode"] = "hard_maxsim"
phase4["late_interaction_verifier_softmax_temperature"] = 0.05  # Was 0.1 — lower temp

config_path = RUN_DIR / "config.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print(f"Config saved to {config_path}")

# === Train v419 ===
model_path = RUN_DIR / "model.pt"
tmp_ckpt = str(model_path) + ".tmp"
shutil.copy2(V378_CKPT, tmp_ckpt)

train_cmd = [
    PY, str(ROOT / "train_production.py"),
    "--config", str(config_path),
    "--size", str(int((config.get("sizes") or config.get("phase4", {}).get("sizes", [15_000_000]))[0])),
    "--output", str(model_path),
    "--device", str(config.get("device", "cuda")),
]

print("\nTraining command:", " ".join(train_cmd))
result = subprocess.run(train_cmd, cwd=ROOT, timeout=36000)
print(f"\nReturn code: {result.returncode}")
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:] if result.stderr else "(none)")
    sys.exit(result.returncode)

print(f"\nModel saved to {model_path}")
print("Run strict eval:")
print(f"  python test_2.7b.py 15000000 {model_path} 2>&1 | grep -E 'Objective|D=|A=|FP=|SF=|score='")
