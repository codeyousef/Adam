#!/usr/bin/env python3
"""
v416 Launch Script — ColBERT Multi-Vector First-Stage Retrieval

AIM: Replace single-vector first-stage retrieval with ColBERT-style multi-vector
retrieval using the already-trained facets from v378's architecture.

v378 already has use_retrieval_facets=true with num_facets=30, facet_dim=256.
But retrieval_encode() averages all facets into a single vector before indexing.
The facets are only used by the late-interaction reranker, NOT first-stage.

v416 hypothesis: Using max-sim between query facets and code facets for first-stage
retrieval (instead of single-vector dot product) will better discriminate
debounce/throttle, startswith/endswith, and parse/serialize.

Key changes from v378:
  - use_multi_vector_first_stage: true
  - multi_vector_aggregation: max (not mean/CLS)
  - Retrieval index rebuilt with facet matrices
  - late_inter_weight: 0.0 (pure multi-vector first-stage, no late-interaction)
  - All other params: v378 proven values
  - warm_start_model_path: v378 checkpoint

This is NOT just enabling use_retrieval_facets — that only affects the reranker.
v416 requires changing how retrieval_project() and the index work.

Corpus gap context:
  - The corpus doesn't contain debounce vs throttle implementations
  - The corpus doesn't distinguish startswith from endswith
  - json_parse and serialize have the same embedding under the current model
  - Multi-vector can capture these differences even with partial corpus coverage
  - because "debounce" tokens and "throttle" tokens get different facet slots

What NOT to spend time on:
  - Reranker changes (facets already trained)
  - Loss function changes
  - Confidence threshold tuning
"""
import sys, os, json, subprocess, shutil, yaml
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
PY = str(ROOT / "../.venv/bin/python")
RUN_DIR = ROOT / "artifacts/strict_eval_autoresearch_v4/v416-multivector-firststage-seed704"
RUN_DIR.mkdir(parents=True, exist_ok=True)

V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
V338_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"

# Build config from v378 proven config
config = yaml.safe_load((V378_CKPT.parent / "config.yaml").read_text())
phase4 = config.setdefault("phase4", {})

# === v416 Identity ===
config["seed"] = 704
config["profile"] = "strict-eval-autoresearch-v4-v416-multivector-firststage"
config["warm_start_phase3_only"] = False
config["warm_start_model_path"] = str(V378_CKPT)
config["base_model_path"] = str(V338_CKPT)

phase4["seed"] = 704
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

# === v416: Multi-Vector First-Stage Retrieval ===
# The facets (num_facets=30, facet_dim=256) are already trained in v378.
# v416 changes HOW facets are used for first-stage retrieval by setting
# global_facet_blend=0.0 (instead of default 1.0).
# This makes first-stage use max-sim between query/code facets (ColBERT-style)
# instead of single-vector dot product.
# The facets were stored in the index in v378 but global_facet_blend=1.0
# meant only global embedding was used. This flips that.
phase4["phase4_dataset"] = "behavioral_constraints_v2_taxonomy_support_discipline_v1"  # Keep proven
phase4["use_retrieval_facets"] = True  # Already true in v378 but be explicit
phase4["retrieval_num_facets"] = 30  # Already set in v378 but be explicit
phase4["retrieval_facet_dim"] = 256  # Already set in v378 but be explicit
phase4["retrieval_facet_separate_query_code"] = False  # Shared facet head (already set)
phase4["retrieval_facet_score_mode"] = "hard_maxsim"  # Max-sim aggregation (already default)
phase4["retrieval_global_facet_blend"] = 0.0  # KEY: 0=facets-only, 1.0=global-only; v378 default is 1.0
phase4["retrieval_facet_softmax_temperature"] = 0.10  # Softmax temp for facet scoring

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
