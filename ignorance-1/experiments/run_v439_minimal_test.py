#!/usr/bin/env python3
"""
v439: Minimal test — v378 config but with only the SIMPLE losses.
Same seed=509, same architecture, but strip out margin/retrieval_facets/vicreg.
Goal: isolate which component (if any) prevents collapse.
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

V439_CONFIG = {
    "seed": 509,
    "device": "cuda",
    "profile": "4090",
    "phase4": {
        "seed": 509,
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
        "step_scale_power": 0.55,
        "max_step_multiplier": 5.0,
        "lr_scale_power": 0.2,
        "max_lr_divisor": 2.5,
        "warmup_fraction": 0.15,
        "min_lr_ratio": 0.2,
        "ramp_regularizers": True,
        "regularizer_ramp_fraction": 0.2,
        "ema_target_decay": 0.995,
        "sigreg_weight": 0.5,
        "reset_query_head_on_resume": False,
        "production_mode": False,
        "production_steps": 0,
        "production_phase4_repeats": 0,
        # NO freeze backbone for this test
        "freeze_backbone": False,
        # NO warm-start
        "warm_start_model_path": None,
        # ONLY basic losses
        "ood_weight": 0.0,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        # Alignment
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "alignment_temperature": 0.1,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        # NO margin losses
        "retrieval_margin_weight": 0.0,
        "retrieval_margin": 0.0,
        "ranking_margin_weight": 0.0,
        "ranking_margin": 0.0,
        "ranking_focal_gamma": 0.0,
        "same_family_only_ranking": False,
        "paraphrase_batch_probability": 0.0,
        "query_margin_weight": 0.0,
        "query_margin": 0.0,
        # NO spread
        "spread_weight": 0.0,
        "query_spread_weight": 0.0,
        "pred_spread_weight": 0.0,
        # NO vicreg
        "use_vicreg_retrieval": False,
        "vicreg_weight": 0.0,
        "vicreg_covariance_weight": 0.0,
        "vicreg_variance_target": 0.75,
        "vicreg_queue_samples": 0,
        "vicreg_prediction_weight": 0.0,
        "vicreg_invariance_weight": 0.0,
        "vicreg_variance_weight": 0.0,
        # NO rank reg
        "rank_reg_weight": 0.0,
        "rank_reg_eps": 0.0001,
        "rank_reg_target": "code+query",
        # NO retrieval facets
        "use_retrieval_facets": False,
        "retrieval_num_facets": 0,
        "retrieval_facet_loss_weight": 0.0,
        # NO late inter verifier
        "late_interaction_verifier_weight": 0.0,
        "late_interaction_verifier_margin": 0.0,
        "late_interaction_verifier_mode": "hard_maxsim",
        "late_interaction_verifier_softmax_temperature": 0.1,
        "late_interaction_verifier_start_step": 9999,
        # NO query multiview
        "use_query_multiview": False,
        "query_multiview_weight": 0.0,
        "query_multiview_prediction_weight": 0.0,
        # NO momentum queue
        "use_momentum_queue": False,
        "momentum_queue_weight": 0.0,
        "momentum_queue_prediction_weight": 0.0,
        "momentum_queue_temperature": 0.1,
        # NO retrieval data strategy
        "use_retrieval_data_strategy": False,
        "use_family_prototypes": False,
        "prototype_target": "family",
        "prototype_weight": 0.0,
        "prototype_code_weight": 0.0,
        "prototype_prediction_weight": 0.0,
        "prototype_repulsion_weight": 0.0,
        "prototype_temperature": 0.1,
        "use_equivalence_prototypes": False,
        "equivalence_include_synthesis_views": False,
        "max_hard_negatives_per_example": 0,
        "use_surface_code_variants": False,
    },
    "confidence_threshold": 0.312,
    "lexical_weight": 0.4,
    "strict_eval": {"enabled": True},
    "freeze_backbone": False,
    "sizes": [15000000],
    "reference_size": 15000000,
}

def train_one(seed, label, cfg_override=None):
    out_dir = f"{OUT_BASE}/v439-{label}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    if os.path.exists(model_path):
        return True, model_path, out_dir

    cfg = dict(V439_CONFIG)
    cfg["seed"] = seed
    cfg["phase4"]["seed"] = seed
    if cfg_override:
        for k, v in cfg_override.items():
            cfg["phase4"][k] = v

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", "15000000", "--device", "cuda"]
        with open(f"{out_dir}/train.log", 'w') as lf:
            result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=600)
        return result.returncode == 0, model_path, out_dir
    finally:
        os.unlink(tmp)

def eval_one(model_path, seed, label):
    eval_path = f"{OUT_BASE}/v439-{label}-seed{seed}/eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data = json.load(f)
    else:
        cmd = [PY, f"{ROOT}/test_2.7b.py",
               "--json", "--confidence-threshold", "0.312", "--lexical-weight", "0.4",
               "--proxy-recipe", "v6_overnight", "--", "15000000", model_path]
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

    code_diag = data.get("code_diagnostics", {})
    return {
        "code_offdiag": code_diag.get("avg_offdiag_similarity"),
        "code_rank": code_diag.get("participation_ratio"),
        "known_conf": data.get("avg_known_confidence"),
        "ood_conf": data.get("avg_ood_confidence"),
        "strict_status": data.get("strict_status", ""),
    }

def main():
    print("="*60)
    print("v439: Minimal losses test — isolate what prevents collapse")
    print("="*60)

    configs = [
        ("BASIC", {}, "Basic: pred+ood+clf+sigreg+align only"),
        ("SIGREG0.5", {"sigreg_weight": 0.5}, "Same as BASIC but explicit sigreg=0.5"),
        ("SIGREG1.0", {"sigreg_weight": 1.0}, "sigreg=1.0 to test anti-collapse"),
        ("SIGREG2.0", {"sigreg_weight": 2.0}, "sigreg=2.0 stronger anti-collapse"),
        ("MARGIN", {"retrieval_margin_weight": 0.25, "retrieval_margin": 0.25}, "Add margin loss"),
    ]

    results = {}
    for label, override, desc in configs:
        print(f"\n--- {label}: {desc} ---")
        seed = 509
        out_dir = f"{OUT_BASE}/v439-{label}-seed{seed}"

        # Clean
        for f in [f"{out_dir}/model.pt", f"{out_dir}/eval.json"]:
            if os.path.exists(f):
                os.remove(f)

        success, model_path, _ = train_one(seed, label, override)
        if not success:
            print(f"  FAILED")
            continue
        time.sleep(1)
        r = eval_one(model_path, seed, label)
        if r:
            results[label] = r
            off = r.get("code_offdiag")
            kc = r.get("known_conf")
            print(f"  offdiag={off:.4f} known_conf={kc:.4f} (v378: offdiag=0.028)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, r in results.items():
        off = r.get("code_offdiag", 0)
        kc = r.get("known_conf", 0)
        status = "COLLAPSED" if off > 0.5 else "PRESERVED"
        print(f"  {label}: offdiag={off:.4f} known_conf={kc:.4f} [{status}]")

    with open(f"{OUT_BASE}/v439_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
