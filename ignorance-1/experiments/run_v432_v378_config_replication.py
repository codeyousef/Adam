#!/usr/bin/env python3
"""
v432: Replicate v378 incumbent config at 13.6M.

CRITICAL FINDING from v431: ALL 30 seeds collapsed at 13.6M with freeze_backbone=false.
v378 incumbent used:
  1. warm_start from v338 (freeze_backbone=true, backbone already trained)
  2. late_interaction_verifier_weight=0.5 (verifier head only)
  3. query_multiview=true
  4. vicreg_weight=0.02, rank_reg_weight=0.08 (anti-collapse regularizers)

This batch tests: 10 seeds with exact v378 incumbent config at 13.6M.
If any pass → the config is replicable with multi-seed selection
If all fail → the v378 PASS was a lucky seed even within this config
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

WARM_START = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"
SEEDS = list(range(600, 610))  # 10 seeds
SIZE = 30000000
PROXY = "v6_overnight"

def make_config(seed):
    return {
        "phase4": {
            # Exact v378 incumbent config
            "seed": seed,
            "proxy_recipe": PROXY,
            "sizes": [SIZE],
            "num_splits": 1,
            "steps": 300,  # v378 incumbent used 300
            "batch_size": 4,
            "microbatch_size": 1,
            "lr": 5e-5,
            "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
            "phase4_balance_families": True,
            "phase4_joint_training": True,
            "phase4_factorized_hard_negatives": True,
            "warm_start_model_path": WARM_START,
            "warm_start_phase3_only": True,
            "freeze_backbone": True,
            "warmup_fraction": 0.15,
            "min_lr_ratio": 0.2,
            # Late interaction verifier (key component)
            "late_interaction_verifier_mode": "hard_maxsim",
            "late_interaction_verifier_margin": 0.2,
            "late_interaction_verifier_weight": 0.5,
            "late_interaction_verifier_softmax_temperature": 0.1,
            # Alignment
            "alignment_decoupled": True,
            "alignment_embedding_weight": 0.0,
            "alignment_mse_weight": 0.04,
            "alignment_prediction_weight": 0.5,
            "alignment_temperature": 0.1,
            "alignment_symmetric": False,
            # Classifier
            "classifier_weight": 0.09,
            "clf_weight": 0.09,
            "classifier_query_weight": 1.0,
            "classifier_prediction_weight": 0.0,
            # Ignorance — DISABLED (was 0.0 in incumbent)
            "ignorance_ood_weight": 0.0,
            "ignorance_pred_weight": 0.0,
            "ood_weight": 0.0,
            "pred_ood_weight": 0.0,
            # Rank regularization
            "rank_reg_weight": 0.08,
            "rank_reg_target": "code+query",
            "rank_reg_eps": 0.0001,
            # Retrieval margins
            "retrieval_margin": 0.25,
            "retrieval_margin_weight": 0.25,
            "ranking_margin": 0.25,
            "ranking_margin_weight": 0.2,
            "ranking_focal_gamma": 2.0,
            # Multiview
            "use_query_multiview": True,
            "query_multiview_weight": 1.0,
            "query_multiview_prediction_weight": 0.5,
            # VICReg
            "use_vicreg_retrieval": True,
            "vicreg_weight": 0.02,
            "vicreg_covariance_weight": 0.05,
            "vicreg_prediction_weight": 0.0,
            "vicreg_queue_samples": 128,
            # Spread
            "spread_weight": 0.02,
            "query_spread_weight": 0.02,
            "pred_spread_weight": 0.02,
            # Retrieval facets
            "use_retrieval_facets": True,
            "use_retrieval_head": True,
            "use_retrieval_data_strategy": True,
            "retrieval_facet_loss_weight": 0.35,
            "retrieval_facet_dim": 256,
            "retrieval_facet_hidden_dim": 512,
            "retrieval_facet_score_mode": "softmax_maxsim",
            "retrieval_facet_softmax_temperature": 0.1,
            "retrieval_head_dim": 256,
            "retrieval_head_hidden_dim": 512,
            "retrieval_num_facets": 30,
            "retrieval_facet_separate_query_code": False,
            # Family prototypes
            "use_family_prototypes": True,
            "use_equivalence_prototypes": True,
            "prototype_weight": 0.0,
            "prototype_repulsion_weight": 0.0,
            "prototype_query_weight": 0.0,
            "prototype_code_weight": 0.0,
            "prototype_prediction_weight": 0.0,
            # Champion challenger
            "champion_challenger_weight": 0.0,
            "champion_challenger_margin": 0.05,
            "champion_challenger_temperature": 0.1,
            # EMA
            "ema_target_decay": 0.995,
            # Momentum queue — DISABLED
            "use_momentum_queue": False,
            "momentum_queue_weight": 0.0,
            "momentum_queue_prediction_weight": 0.0,
            # Equivalence — DISABLED
            "equivalence_alignment_weight": 0.0,
            "equivalence_prediction_weight": 0.0,
            "equivalence_margin_weight": 0.0,
            "equivalence_include_synthesis_views": False,
            # Misc
            "same_family_only_ranking": True,
            "max_hard_negatives_per_example": 4,
            "sigreg_weight": 0.5,
            "ramp_regularizers": True,
            "regularizer_ramp_fraction": 0.2,
            "reset_query_head_on_resume": False,
            # Epistemic — DISABLED
            "epistemic_boundary_weight": 0.0,
            "epistemic_margin": 0.2,
            "epistemic_prediction_weight": 1.0,
            "epistemic_query_weight": 0.0,
            # Family local listwise
            "family_local_listwise_weight": 1.0,
            "family_local_listwise_temperature": 0.07,
            # Graded negative — DISABLED
            "graded_negative_weight": 0.0,
            # Production
            "production_mode": False,
            "production_steps": 300,
        }
    }

def run_one(seed):
    out_dir = f"{OUT_BASE}/v432-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    log_path = f"{out_dir}/train.log"
    if os.path.exists(model_path):
        print(f"  SKIP seed={seed}")
        return out_dir, model_path
    config = make_config(seed)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", str(SIZE), "--device", "cuda"]
        print(f"  TRAIN seed={seed}")
        with open(log_path, 'w') as lf:
            subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
    finally:
        os.unlink(tmp)
    return out_dir, model_path

def eval_one(out_dir, model_path, seed):
    eval_path = f"{out_dir}/eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            return json.load(f)
    cmd = [PY, f"{ROOT}/test_2.7b.py",
           "--json", "--confidence-threshold", "0.312", "--lexical-weight", "0.4",
           "--proxy-recipe", PROXY, "--", str(SIZE), model_path]
    print(f"  EVAL seed={seed}")
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
        return data
    except Exception:
        return None

def extract_summary(data):
    if not data:
        return None
    code_diag = data.get("code_diagnostics", {})
    query_diag = data.get("query_diagnostics", {})
    return {
        "strict_status": data.get("strict_status", ""),
        "code_offdiag": code_diag.get("avg_offdiag_similarity"),
        "code_rank": code_diag.get("participation_ratio"),
        "query_offdiag": query_diag.get("avg_offdiag_similarity"),
        "query_rank": query_diag.get("participation_ratio"),
        "known_conf": data.get("avg_known_confidence"),
        "ood_conf": data.get("avg_ood_confidence"),
        "known_sim": data.get("avg_known_similarity"),
        "ignorant_sim": data.get("avg_ignorant_similarity"),
        "gap": data.get("ignorance_gap"),
        "legacy_status": data.get("legacy_status", ""),
    }

def main():
    print("="*70)
    print("v432: Replicate v378 incumbent config (warm-start + frozen backbone)")
    print(f"  {len(SEEDS)} seeds × 300 steps @ 13.6M")
    print("="*70)

    results = []
    for seed in SEEDS:
        out_dir, model_path = run_one(seed)
        if os.path.exists(model_path):
            data = eval_one(out_dir, model_path, seed)
            summary = extract_summary(data)
            if summary:
                results.append({"seed": seed, **summary})
                is_pass = "PASS" in summary.get("strict_status", "")
                off = summary.get("code_offdiag")
                print(f"  seed={seed}: offdiag={off:.4f} {'✓ PASS' if is_pass else ''}")

    # Summary
    n = len(results)
    passes = [r for r in results if "PASS" in r.get("strict_status", "")]
    offs = [r["code_offdiag"] for r in results if r.get("code_offdiag") is not None]
    ranks = [r["code_rank"] for r in results if r.get("code_rank") is not None]

    print("\n" + "="*70)
    print("v432 RESULTS")
    print("="*70)
    print(f"Seeds tested: {n}")
    print(f"Strict PASS: {len(passes)}/{n}")
    if offs:
        print(f"Avg code_offdiag: {sum(offs)/len(offs):.4f}")
        print(f"Avg code_rank: {sum(ranks)/len(ranks):.4f}")

    if passes:
        print(f"\nPASSING SEEDS:")
        for r in passes:
            print(f"  seed={r['seed']}: offdiag={r['code_offdiag']:.4f} rank={r['code_rank']:.4f} "
                  f"gap={r.get('gap',0):.4f} known_conf={r['known_conf']:.4f} ood_conf={r['ood_conf']:.4f}")
    else:
        best = min(results, key=lambda r: r.get("code_offdiag", 1.0), default=None)
        if best:
            print(f"\nBest (lowest offdiag): seed={best['seed']} offdiag={best['code_offdiag']:.4f} "
                  f"rank={best['code_rank']:.4f} gap={best.get('gap',0):.4f}")

    with open(f"{OUT_BASE}/v432_summary.json", 'w') as f:
        json.dump({"results": results, "n_tested": n, "n_pass": len(passes)}, f, indent=2)

if __name__ == "__main__":
    main()
