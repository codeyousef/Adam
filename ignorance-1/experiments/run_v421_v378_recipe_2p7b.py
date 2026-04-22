#!/usr/bin/env python3
"""
v421: Train 2.7B with v378's recipe (no alignment_embedding_weight)

HYPOTHESIS: The embedding collapse in ALL 2.7B models (winning, v340, v413)
is caused by alignment_embedding_weight=0.5 in the Phase 4 recipe.
v378 (healthy embeddings at proxy) uses alignment_embedding_weight=0.0.

v378 recipe:
  - classifier_weight: 0.09
  - query_multiview_weight: 1.0
  - equivalence_alignment_weight: 0.0
  - alignment_embedding_weight: 0.0 (key!)
  - late_interaction_verifier_weight: 0.5
  - freeze_backbone: true
  - phase4_dataset: taxonomy_support_discipline_v1
  - phase4_steps: 300

The winning recipe (collapsed at 2.7B) uses:
  - alignment_embedding_weight: 0.5
  - classifier_weight: 0.25
  - behavioral_constraints_v2_rigorous

If v421's 2.7B training produces healthy embeddings (code_sim < 0.85, rank > 0.10),
then the alignment_embedding_weight is the cause of universal collapse.
"""

from __future__ import annotations
import subprocess, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
RUN_DIR = ROOT / "artifacts" / "strict_eval_autoresearch_v4" / "v421-v378-recipe-2p7b-seed711"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Use the v338 base model (the actual base for v378's training)
V338_BASE = ROOT / "artifacts" / "strict_eval_autoresearch_v338" / "v338-promoted-earlier-onset-tiny-mixed-bridge-seed504" / "model.pt"

# v378's proven recipe, adapted for 2.7B scale
config = {
    "profile": "strict-eval-autoresearch-v4-v421-v378-recipe-2p7b",
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
    "warm_start_model_path": None,  # Fresh 2.7B init from v338 base recipe
    "base_model_path": str(V338_BASE),
    "phase4": {
        "seed": 711,
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "sizes": [2_700_000_000],
        "phase4_steps": 500,
        "steps": 500,
        "lr": 5e-05,
        "batch_size": 4,
        "microbatch_size": 1,
        "num_splits": 1,
        "phase4_repeats": 1,
        "production_mode": False,
        "production_steps": 0,
        "production_phase4_repeats": 0,
        # Proxy recipe fields (required by train_production.py)
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
        # v378's recipe — key differences from winning
        "classifier_weight": 0.09,           # v378 value (winning: 0.25)
        "clf_weight": 0.09,                # v378 value
        "query_multiview_weight": 1.0,      # v378 value
        "query_multiview_prediction_weight": 0.5,
        "equivalence_alignment_weight": 0.0,    # v378 value (winning: not set)
        "equivalence_prediction_weight": 0.0,
        "equivalence_margin_weight": 0.0,
        "equivalence_include_synthesis_views": False,
        "use_equivalence_prototypes": True,
        # KEY: 0.0 vs winning's 0.5 — THIS is the suspected cause of collapse
        "alignment_embedding_weight": 0.0,
        "alignment_prediction_weight": 0.0,
        "alignment_mse_weight": 0.0,
        "alignment_decoupled": False,
        "rank_reg_weight": 0.08,
        "rank_reg_eps": 0.0001,
        "rank_reg_target": "code+query",
        "freeze_backbone": True,
        "late_interaction_verifier_weight": 0.5,   # v378 value
        "late_interaction_verifier_margin": 0.2,   # v378 value
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

import yaml
config_path = RUN_DIR / "config.yaml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print(f"Config saved to {config_path}")

# Save the config as a JSON too for reference
with open(RUN_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# Train
output_path = RUN_DIR / "model.pt"
train_cmd = [
    str(PYTHON), str(ROOT / "train_production.py"),
    "--config", str(config_path),
    "--size", "2700000000",
    "--output", str(output_path),
    "--device", "cuda",
]

print(f"\nTraining command: {' '.join(train_cmd)}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
result = subprocess.run(train_cmd, cwd=str(ROOT), timeout=43200)  # 12h timeout
print(f"\nReturn code: {result.returncode}")
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:] if result.stderr else "(none)")
    sys.exit(result.returncode)

print(f"\nModel saved to {output_path}")
print(f"Train complete at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Run strict eval immediately after training
print("\nRunning strict_eval on trained model...")
eval_cmd = [
    str(PYTHON), str(ROOT / "test_2.7b.py"),
    "--json",
    "--confidence-threshold", "0.312",
    "--lexical-weight", "0.4",
    "--",
    "2700000000",
    str(output_path),
]
eval_result = subprocess.run(eval_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
print(eval_result.stdout[-3000:] if eval_result.stdout else "(no stdout)")
if eval_result.returncode != 0:
    print("EVAL STDERR:", eval_result.stderr[-1000:] if eval_result.stderr else "")

# Save results
results_path = RUN_DIR / "eval_output.json"
with open(results_path, "w") as f:
    f.write(eval_result.stdout)

# Extract and save summary
try:
    import re
    stdout = eval_result.stdout
    json_start = stdout.find('{')
    json_text = stdout[json_start:]
    depth = 0
    end = 0
    for i, c in enumerate(json_text):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    eval_data = json.loads(json_text[:end])
    summary = {
        "version": 421,
        "model_path": str(output_path),
        "size": 2_700_000_000,
        "score": eval_data.get("score"),
        "strict_status": eval_data.get("strict_status"),
        "strict_results": eval_data.get("strict_results"),
        "avg_known_similarity": eval_data.get("avg_known_similarity"),
        "avg_known_exact_similarity": eval_data.get("avg_known_exact_similarity"),
        "avg_known_paraphrase_similarity": eval_data.get("avg_known_paraphrase_similarity"),
        "avg_ignorant_similarity": eval_data.get("avg_ignorant_similarity"),
        "ignorance_gap": eval_data.get("ignorance_gap"),
        "synthesis_similarity": eval_data.get("synthesis_similarity"),
        "code_diagnostics": eval_data.get("code_diagnostics"),
        "query_diagnostics": eval_data.get("query_diagnostics"),
        "strict_failures": eval_data.get("strict_failures", []),
    }
    with open(RUN_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {RUN_DIR / 'summary.json'}")
    print(f"strict_status: {summary['strict_status']}")
    print(f"score: {summary['score']}")
    if summary.get('code_diagnostics'):
        cd = summary['code_diagnostics']
        print(f"code sim: {cd.get('avg_offdiag_similarity')}, rank: {cd.get('participation_ratio_fraction')}")
except Exception as e:
    print(f"Failed to parse eval output: {e}")
    print("Raw stdout:", eval_result.stdout[-1000:])
