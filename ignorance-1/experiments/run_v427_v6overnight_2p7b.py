#!/usr/bin/env python3
"""Run v427: v6_overnight recipe at 2.7B size class.
v378 used v6_overnight @ 30M (13.6M params) and passed.
All our 2.7B attempts used v5_distinct which is a DIFFERENT recipe.
This is the first attempt with the CORRECT recipe matching v378."""
import subprocess, sys, os, tempfile, yaml

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT = f"{ROOT}/artifacts/strict_eval_autoresearch_v4/v427-v6overnight-2p7b-seed711"

os.makedirs(OUT, exist_ok=True)

# v378 key config: v6_overnight recipe, classifier_weight=0.25, 
# alignment_embedding_weight=0.5, rank_reg_weight=0.05, ignorance_ood_weight=0.2
# phase4_steps=112 at 30M, scaled by step_scale_power=0.55

config = {
    "seed": 711,
    "device": "cuda",
    "phase4": {
        "sizes": [2700000000],
        "steps": 500,
        "batch_size": 4,
        "microbatch_size": 1,
        "lr": 5e-5,
        "proxy_recipe": "v6_overnight",
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "ignorance_ood_weight": 0.2,
        "ignorance_pred_weight": 0.2,
        "classifier_weight": 0.25,
        "rank_reg_weight": 0.05,
        "alignment_embedding_weight": 0.5,
        "alignment_prediction_weight": 1.0,
        "champion_challenger_weight": 0.5,
        "champion_challenger_margin": 0.05,
        "champion_challenger_temperature": 0.1,
        "freeze_backbone": False,
    }
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(config, f)
    tmp = f.name

try:
    cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", f"{OUT}/model.pt"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    sys.exit(result.returncode)
finally:
    os.unlink(tmp)
