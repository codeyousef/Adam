#!/usr/bin/env python3
"""
v422: Train 2.7B with v378's recipe + freeze_backbone: false

HYPOTHESIS: Every tested 2.7B model used freeze_backbone: true and ALL collapsed
identically. The frozen encoder may not adapt properly to the query head at 2.7B scale.
Letting the encoder train freely should produce healthy geometry.

Key changes from v421:
  - freeze_backbone: false (was true)
  - All other params identical to v378 recipe
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
EXP  = ROOT / "experiments"

# Base model = same as v421 (= v338 promoted tiny mixed bridge, seed 504)
BASE_MODEL = (
    ROOT / "artifacts/strict_eval_autoresearch_v338/"
         "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"
)

TRAIN_SCRIPT = ROOT / "train_production.py"
PROFILE      = "strict-eval-autoresearch-v4-v422-freeze-backbone-false-2p7b"

config = {
    "profile": PROFILE,
    "device": "cuda",
    "seed": 711,
    "sizes": [2_700_000_000],
    "embed_dim": 384,
    "encoder_layers": 10,
    "decoder_layers": 4,
    "decoder_heads": 8,
    "decoder_hidden_dim": 1536,
    "encoder_heads": 8,
    "predictor_layers": 4,
    "predictor_heads": 8,
    "predictor_hidden_dim": 1536,
    "vocab_size": 4096,
    "max_seq_len": 256,
    "patch_size": 32,
    "batch_size": 4,
    "warm_start_phase3_only": False,
    "warm_start_model_path": None,
    "base_model_path": str(BASE_MODEL),
    "phase4": {
        "seed": 711,
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "sizes": [2_700_000_000],
        "phase4_steps": 500,
        "steps": 500,
        "lr": 5e-5,
        "batch_size": 4,
        "microbatch_size": 1,
        "num_splits": 1,
        "phase4_repeats": 1,
        "production_mode": False,
        "production_steps": 0,
        "production_phase4_repeats": 0,
        "proxy_recipe": "v5_distinct",
        "reference_size": 300_000_000,
        "step_scale_power": 0.55,
        "max_step_multiplier": 5.0,
        "lr_scale_power": 0.2,
        "max_lr_divisor": 2.5,
        "scheduler": "constant",
        "warmup_fraction": 0.0,
        "min_lr_ratio": 1.0,
        "grad_accum_steps": 4,
        "ema_target_decay": 0.0,
        "proxy_disable_batchnorm": False,
        "common_random_numbers": True,
        "validation_eval_mode": True,
        "classifier_weight": 0.09,
        "clf_weight": 0.09,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "equivalence_include_synthesis_views": False,
        "use_equivalence_prototypes": True,
        "alignment_embedding_weight": 0.0,
        "alignment_prediction_weight": 0.0,
        "alignment_mse_weight": 0.0,
        "alignment_decoupled": False,
        "rank_reg_weight": 0.08,
        "rank_reg_eps": 0.0001,
        "rank_reg_target": "code+query",
        "freeze_backbone": False,          # <--- THE CHANGE
        "late_interaction_verifier_weight": 0.5,
        "late_interaction_verifier_margin": 0.2,
        "late_interaction_verifier_mode": "hard_maxsim",
        "late_interaction_verifier_softmax_temperature": 0.1,
        "epistemic_boundary_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        "ignorance_ood_weight": 0.0,
        "ignorance_pred_weight": 0.0,
    },
}

import yaml, json, tempfile

with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    yaml.dump(config, f)
    cfg_path = f.name

out_path = ROOT / "artifacts" / "strict_eval_autoresearch_v4" / f"{PROFILE}-seed{config['seed']}.pt"
out_path.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, str(TRAIN_SCRIPT),
    "--config", cfg_path,
    "--output", str(out_path),
]
print("RUNNING:", " ".join(cmd))
subprocess.check_call(cmd)
print("DONE. Model →", out_path)
