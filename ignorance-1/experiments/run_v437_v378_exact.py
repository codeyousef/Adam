#!/usr/bin/env python3
"""
v437: Replicate v378 cycle-0 EXACTLY using restored full train_production.py.

CRITICAL FINDING: The FULL 771-line train_production.py was committed March 30
(AFTER v378 completed). The v378-era code (356419a, March 27) is the SAME
simplified 4-loss version I was using. v378 got offdiag=0.028 but I'm getting
offdiag=0.999 — same code, different outcome.

v378 cycle-0 config (embed=192, hidden=576, 4L):
  - sigreg_weight=0.5
  - ood_weight=0.0
  - clf_weight=0.09
  - retrieval_margin_weight=0.25, margin=0.25
  - ranking_margin_weight=0.2, margin=0.25
  - alignment_prediction_weight=0.5, embedding=0.0, mse=0.04
  - spread_weight=0.02
  - query_spread_weight=0.02
  - pred_spread_weight=0.02
  - vicreg_weight=0.02
  - rank_reg_weight=0.08
  - late_interaction_verifier_weight=0.3
  - freeze_backbone=True

v378 used ONLY 112 steps (scaled). The restored train_production.py has ALL these components.

Strategy: Use the restored full train_production.py with v378's exact config
to see if the full loss components prevent collapse.
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"
PROXY = "v6_overnight"

# Exact v378 cycle-0 config
V378_CONFIG = {
    "seed": 509,
    "device": "cuda",
    "profile": "4090",
    "phase4": {
        "seed": 509,
        "proxy_recipe": PROXY,
        "sizes": [15000000],
        "num_splits": 1,
        "steps": 112,  # Exact v378 step count
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
        # Freeze backbone
        "freeze_backbone": True,
        # Warm-start (broken but v378 also had broken warm-start)
        "warm_start_model_path": "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt",
        "warm_start_phase3_only": True,
        # Classification
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
        # Margin losses
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        "ranking_margin_weight": 0.2,
        "ranking_margin": 0.25,
        "ranking_focal_gamma": 2.0,
        # Spread
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
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
        # Retrieval facets
        "use_retrieval_facets": True,
        "retrieval_num_facets": 30,
        "retrieval_facet_dim": 256,
        "retrieval_facet_hidden_dim": 512,
        "retrieval_facet_separate_query_code": False,
        "retrieval_facet_score_mode": "softmax_maxsim",
        "retrieval_facet_softmax_temperature": 0.1,
        "retrieval_facet_loss_weight": 0.35,
        "same_family_only_ranking": True,
        "use_retrieval_head": True,
        "retrieval_head_dim": 256,
        "retrieval_head_hidden_dim": 512,
        # Late interaction verifier
        "late_interaction_verifier_weight": 0.3,
        "late_interaction_verifier_margin": 0.2,
        "late_interaction_verifier_mode": "hard_maxsim",
        "late_interaction_verifier_softmax_temperature": 0.1,
        # Query multiview
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
        # Other
        "use_retrieval_data_strategy": True,
        "use_family_prototypes": True,
        "use_equivalence_prototypes": True,
        "equivalence_include_synthesis_views": False,
        "max_hard_negatives_per_example": 4,
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
    },
    "confidence_threshold": 0.312,
    "lexical_weight": 0.4,
    "strict_eval": {"enabled": True},
    "freeze_backbone": True,
    "sizes": [15000000],
    "reference_size": 15000000,
}

def train_one(seed, config_label):
    out_dir = f"{OUT_BASE}/v437-{config_label}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    if os.path.exists(model_path):
        return True, model_path, out_dir

    cfg = dict(V378_CONFIG)
    cfg["seed"] = seed
    cfg["phase4"]["seed"] = seed

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", "15000000", "--device", "cuda"]
        print(f"  TRAIN seed={seed} ({config_label})")
        with open(f"{out_dir}/train.log", 'w') as lf:
            result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
        return result.returncode == 0, model_path, out_dir
    finally:
        os.unlink(tmp)

def eval_one(model_path, seed, label):
    eval_path = f"{OUT_BASE}/v437-{label}-seed{seed}/eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data = json.load(f)
    else:
        cmd = [PY, f"{ROOT}/test_2.7b.py",
               "--json", "--confidence-threshold", "0.312", "--lexical-weight", "0.4",
               "--proxy-recipe", PROXY, "--", "15000000", model_path]
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
        except:
            return None

    code_diag = data.get("code_diagnostics", {})
    query_diag = data.get("query_diagnostics", {})
    return {
        "code_offdiag": code_diag.get("avg_offdiag_similarity"),
        "code_rank": code_diag.get("participation_ratio"),
        "known_conf": data.get("avg_known_confidence"),
        "ood_conf": data.get("avg_ood_confidence"),
        "strict_status": data.get("strict_status", ""),
        "strict_failures": data.get("strict_failures", []),
        "known_exact": data.get("avg_known_exact_similarity"),
        "gap": data.get("ignorance_gap"),
    }

def main():
    print("="*60)
    print("v437: Exact v378 config + full train_production.py")
    print("  Testing if full loss components (margin, vicreg, rank_reg,")
    print("  retrieval_facets, late_inter_verifier) prevent collapse")
    print("="*60)

    seeds = [509, 510, 511, 512, 513, 514, 515, 516]
    label = "v378exact"

    for seed in seeds:
        success, model_path, out_dir = train_one(seed, label)
        if not success:
            print(f"  FAILED seed={seed}")
            continue
        time.sleep(1)

    results = []
    for seed in seeds:
        out_dir = f"{OUT_BASE}/v437-{label}-seed{seed}"
        model_path = f"{out_dir}/model.pt"
        if not os.path.exists(model_path):
            continue
        r = eval_one(model_path, seed, label)
        if r:
            results.append({"seed": seed, **r})
            is_pass = "PASS" in r.get("strict_status", "")
            off = r.get("code_offdiag")
            kc = r.get("known_conf", 0)
            ooc = r.get("ood_conf", 0)
            print(f"  seed={seed}: offdiag={off:.4f} known_conf={kc:.4f} ood_conf={ooc:.4f} {'✓ PASS' if is_pass else ''}")

    n = len(results)
    passes = [r for r in results if "PASS" in r.get("strict_status", "")]
    offs = [r["code_offdiag"] for r in results if r.get("code_offdiag") is not None]
    kcs = [r["known_conf"] for r in results if r.get("known_conf") is not None]

    print(f"\n{'='*60}")
    print("v437 RESULTS — v378 exact config + full train_production.py")
    print(f"{'='*60}")
    print(f"Seeds tested: {n}, Strict PASS: {len(passes)}/{n}")
    if offs:
        print(f"Avg code_offdiag: {sum(offs)/len(offs):.4f} (v378: 0.028)")
        print(f"Avg known_conf: {sum(kcs)/len(kcs):.4f} (v378: 0.966)")

    if passes:
        print(f"\nPASSING SEEDS:")
        for r in passes:
            print(f"  seed={r['seed']}: offdiag={r['code_offdiag']:.4f} known_conf={r['known_conf']:.4f}")
    else:
        best = max(results, key=lambda r: r.get("known_conf", 0), default=None) if results else None
        if best:
            print(f"\nBest: seed={best['seed']} offdiag={best['code_offdiag']:.4f} known_conf={best['known_conf']:.4f}")

    with open(f"{OUT_BASE}/v437_summary.json", 'w') as f:
        json.dump({"results": results, "n_tested": n, "n_pass": len(passes)}, f, indent=2)

if __name__ == "__main__":
    main()
