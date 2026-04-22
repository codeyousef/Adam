#!/usr/bin/env python3
"""
v443: v378 config replication WITH FULL LOSS FIX.
The key fix: train_production.py was NOT computing prediction_margin_loss
and embedding_margin_loss despite them being in config. Fixed now.

v378 achieved: offdiag=0.030, known_conf=0.966, ood_conf=0.075
v441 (broken loss) collapsed: offdiag=0.999

FIXES APPLIED to train_production.py:
  1. sigreg_weight now reads from config (was hardcoded 0.5)
  2. prediction_margin_weight: local var + added to micro_loss
  3. embedding_margin_weight: local var + added to micro_loss
  4. Both losses now call retrieval_margin_loss() with proper args
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

V443_CONFIG = {
    "seed": 511,
    "device": "cuda",
    "profile": "4090",
    "phase4": {
        "seed": 511,
        "proxy_recipe": "v6_overnight",
        "sizes": [15000000],
        "num_splits": 1,
        "steps": 112,
        "batch_size": 4,
        "microbatch_size": 1,
        "lr": 5e-5,
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
        "phase4_balance_families": True,
        "phase4_joint_training": True,
        "phase4_factorized_hard_negatives": True,
        "use_phase4_contrast_data": True,
        "step_scale_power": 0.55,
        "max_step_multiplier": 5.0,
        "lr_scale_power": 0.2,
        "max_lr_divisor": 2.5,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ema_target_decay": 0.995,
        # THE FULL LOSS SUITE — these are now properly computed in train_production.py
        "sigreg_weight": 0.50,
        "prediction_margin_weight": 1.00,   # <-- NEW (was missing from micro_loss)
        "embedding_margin_weight": 0.50,   # <-- NEW (was missing from micro_loss)
        "retrieval_margin": 0.25,
        "retrieval_margin_weight": 0.25,
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        "reset_query_head_on_resume": False,
        "production_mode": False,
        "production_steps": 0,
        "production_phase4_repeats": 0,
        "freeze_backbone": True,
        "optimizer_name": "paged_adamw32bit",
        # Basic losses
        "ood_weight": 0.0,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "classifier_query_weight": 1.0,
        "classifier_prediction_weight": 0.0,
        "pred_ood_weight": 0.0,
        # Alignment
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "alignment_temperature": 0.1,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        # Margin + ranking
        "ranking_margin_weight": 0.2,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        "same_family_only_ranking": True,
        "paraphrase_batch_probability": 0.0,
        "query_margin_weight": 0.0,
        "query_margin": 0.0,
        # Support-slate localization
        "family_local_listwise_weight": 0.0,
        "champion_challenger_weight": 0.0,
        "graded_negative_weight": 0.0,
        # VICReg
        "use_vicreg_retrieval": True,
        "vicreg_weight": 0.02,
        "vicreg_covariance_weight": 0.05,
        "vicreg_variance_target": 0.75,
        "vicreg_queue_samples": 128,
        "vicreg_prediction_weight": 0.0,
        "vicreg_invariance_weight": 1.0,
        "vicreg_variance_weight": 1.0,
        # Rank reg
        "rank_reg_weight": 0.08,
        "rank_reg_eps": 0.0001,
        "rank_reg_target": "code+query",
        # Retrieval heads
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "use_gated_reranker": False,
        # Late inter verifier
        "late_interaction_verifier_weight": 0.3,
        "late_interaction_verifier_margin": 0.2,
        "late_interaction_verifier_mode": "hard_maxsim",
        "late_interaction_verifier_softmax_temperature": 0.1,
        # Query multiview
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        # Momentum queue
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "momentum_queue_temperature": 0.1,
        # Epistemic
        "epistemic_boundary_weight": 0.0,
        "epistemic_margin": 0.2,
        "epistemic_query_weight": 0.0,
        "epistemic_prediction_weight": 1.0,
        # Retrieval data strategy
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "prototype_target": "family",
        "prototype_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "prototype_query_weight": 0.0,
        "prototype_temperature": 0.1,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "max_hard_negatives_per_example": 4,
        "use_surface_code_variants": False,
    },
    "confidence_threshold": 0.312,
    "lexical_weight": 0.4,
    "strict_eval": {"enabled": True},
    "sizes": [15000000],
    "reference_size": 15000000,
}

def train_one(seed):
    label = "v443lossfix"
    out_dir = f"{OUT_BASE}/{label}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    if os.path.exists(model_path):
        print(f"  Already exists, skipping")
        return True, model_path, out_dir

    cfg = {"seed": seed, "device": "cuda", "profile": "4090",
           "phase4": dict(V443_CONFIG["phase4"], seed=seed),
           "confidence_threshold": 0.312, "lexical_weight": 0.4,
           "strict_eval": {"enabled": True},
           "sizes": [15000000], "reference_size": 15000000}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", "15000000", "--device", "cuda"]
        print(f"  TRAIN v443 seed={seed}")
        with open(f"{out_dir}/train.log", 'w') as lf:
            result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=600)
        return result.returncode == 0, model_path, out_dir
    finally:
        os.unlink(tmp)

def eval_one(model_path, seed):
    eval_path = f"{OUT_BASE}/v443lossfix-seed{seed}/eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data = json.load(f)
    else:
        cmd = [PY, f"{ROOT}/test_2.7b.py",
               "--json", "--confidence-threshold", "0.312", "--lexical-weight", "0.4",
               "--proxy-recipe", "v6_overnight", "--", "15000000", model_path]
        print(f"  EVAL v443 seed={seed}")
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=120)
        text = result.stdout.decode() + result.stderr.decode()
        marker = text.rfind('-------------------')
        json_start = text.find('{', marker)
        if json_start == -1:
            return None
        try:
            data = json.loads(text[json_start:])
            with open(eval_path, 'w') as f:
                json.dump(data, f)
        except:
            return None
    cd = data.get("code_diagnostics", {})
    return {
        "code_offdiag": cd.get("avg_offdiag_similarity"),
        "code_rank": cd.get("participation_ratio"),
        "known_conf": data.get("avg_known_confidence"),
        "ood_conf": data.get("avg_ood_confidence"),
        "strict_status": data.get("strict_status", ""),
        "strict_failures": data.get("strict_failures", []),
    }

def main():
    print("="*60)
    print("v443: v378 config WITH FULL LOSS FIX")
    print("  prediction_margin_weight=1.00, embedding_margin_weight=0.50")
    print("  sigreg_weight=0.50 (was hardcoded, now from config)")
    print("  All losses now properly computed in train_production.py")
    print("="*60)

    seeds = [511, 509]
    for seed in seeds:
        out_dir = f"{OUT_BASE}/v443lossfix-seed{seed}"
        for f in [f"{out_dir}/model.pt", f"{out_dir}/eval.json"]:
            if os.path.exists(f):
                os.remove(f)

        ok, model_path, _ = train_one(seed)
        if not ok:
            print(f"  FAILED seed={seed}")
            continue
        time.sleep(2)

        r = eval_one(model_path, seed)
        if r:
            off = r.get("code_offdiag")
            kc = r.get("known_conf")
            ooc = r.get("ood_conf")
            status = "COLLAPSED" if off and off > 0.5 else "PRESERVED"
            strict = "PASS" if "PASS" in r.get("strict_status", "") else "FAIL"
            print(f"  seed={seed}: offdiag={off:.4f} known_conf={kc:.4f} ood_conf={ooc:.4f} [{status}] strict={strict}")
            if r.get("strict_failures"):
                print(f"  strict_failures: {r['strict_failures'][:3]}")

    print(f"\nReference: v378-late-inter-only offdiag=0.030, known_conf=0.966, ood_conf=0.075")
    print(f"Broken:    v441 collapsed offdiag=0.999")

if __name__ == "__main__":
    main()
