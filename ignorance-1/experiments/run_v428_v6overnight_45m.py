#!/usr/bin/env python3
"""Run v428: v6_overnight @ 45M (26.8M params) — test if recipe survives scale-up.

v378 (v6_overnight @ 30M, 13.6M) PASSED legacy but strict FAIL.
v427 (v6_overnight @ 2.7B, 2.7B) COMPLETE COLLAPSE.
Test the middle: v6_overnight @ 45M (26.8M) — does recipe survive at larger scale?
If 45M also collapses, collapse threshold is between 13.6M and 45M.
If 45M passes legacy, collapse happens between 45M and 2.7B.
"""
import subprocess, sys, os, tempfile, yaml

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT = f"{ROOT}/artifacts/strict_eval_autoresearch_v4/v428-v6overnight-45m-seed711"

os.makedirs(OUT, exist_ok=True)

# v6_overnight @ 45M = 8L/320-dim/26.8M actual params
# proxy_recipe MUST be inside phase4 section (train_production.py reads it from there)
config = {
        "phase4": {
        "seed": 711,
        "proxy_recipe": "v6_overnight",
        "sizes": [45000000],
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "steps": 500,
        "batch_size": 4,
        "microbatch_size": 1,
        "lr": 5e-5,
        "phase4_balance_families": True,
        "ignorance_ood_weight": 0.2,
        "ignorance_pred_weight": 0.2,
        "classifier_weight": 0.25,
        "clf_weight": 0.25,
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
    cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", f"{OUT}/model.pt", "--size", "45000000", "--device", "cuda"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    sys.exit(result.returncode)
finally:
    os.unlink(tmp)
