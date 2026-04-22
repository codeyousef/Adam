#!/usr/bin/env python3
"""
v440: Test with corrected config propagation (retrieval heads now wired into model).
Also test with phase4_dataset=behavioral_constraints_v2_rigorous (the dataset with
better coverage of hard families).

Key fix: model_config now gets use_retrieval_head, use_retrieval_facets, etc.
This means the model will have retrieval_facet_head and retrieval_head modules.
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

V440_CONFIGS = [
    ("RET_HEADS", {
        "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
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
        "use_gated_reranker": True,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
    }),
    ("RIGOROUS", {
        "phase4_dataset": "behavioral_constraints_v2_rigorous",
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
        "use_gated_reranker": True,
        "use_query_multiview": True,
        "query_multiview_weight": 1.0,
        "query_multiview_prediction_weight": 0.5,
    }),
]

BASE_CONFIG = {
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
        # Basic losses
        "ood_weight": 0.0,
        "clf_weight": 0.09,
        "classifier_weight": 0.09,
        "alignment_prediction_weight": 0.5,
        "alignment_embedding_weight": 0.0,
        "alignment_mse_weight": 0.04,
        "alignment_temperature": 0.1,
        "alignment_decoupled": True,
        "alignment_symmetric": False,
        # Margin
        "retrieval_margin_weight": 0.25,
        "retrieval_margin": 0.25,
        # Spread
        "spread_weight": 0.02,
        "query_spread_weight": 0.02,
        "pred_spread_weight": 0.02,
        # No warm-start (start fresh)
    },
    "confidence_threshold": 0.312,
    "lexical_weight": 0.4,
    "strict_eval": {"enabled": True},
    "sizes": [15000000],
    "reference_size": 15000000,
}

def train_one(seed, label, extra_phase4):
    out_dir = f"{OUT_BASE}/v440-{label}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    if os.path.exists(model_path):
        print(f"  Already exists: v440-{label} seed={seed}")
        return True, model_path, out_dir

    cfg = {"seed": seed, "device": "cuda", "profile": "4090",
           "phase4": dict(BASE_CONFIG["phase4"], seed=seed, **extra_phase4),
           "confidence_threshold": 0.312, "lexical_weight": 0.4,
           "strict_eval": {"enabled": True},
           "sizes": [15000000], "reference_size": 15000000}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(cfg, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", "15000000", "--device", "cuda"]
        print(f"  TRAIN v440-{label} seed={seed}")
        with open(f"{out_dir}/train.log", 'w') as lf:
            result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=600)
        return result.returncode == 0, model_path, out_dir
    finally:
        os.unlink(tmp)

def eval_one(model_path, seed, label):
    eval_path = f"{OUT_BASE}/v440-{label}-seed{seed}/eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data = json.load(f)
    else:
        cmd = [PY, f"{ROOT}/test_2.7b.py",
               "--json", "--confidence-threshold", "0.312", "--lexical-weight", "0.4",
               "--proxy-recipe", "v6_overnight", "--", "15000000", model_path]
        print(f"  EVAL v440-{label} seed={seed}")
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
    }

def main():
    print("="*60)
    print("v440: Retrieval heads now wired into model (config fix)")
    print("="*60)

    all_results = {}
    seeds = [509, 510, 511]

    for label, extra in V440_CONFIGS:
        print(f"\n--- Config: {label} ---")
        for seed in seeds:
            out_dir = f"{OUT_BASE}/v440-{label}-seed{seed}"
            for f in [f"{out_dir}/model.pt", f"{out_dir}/eval.json"]:
                if os.path.exists(f):
                    os.remove(f)

        for seed in seeds:
            ok, _, _ = train_one(seed, label, extra)
            if not ok:
                print(f"  FAILED seed={seed}")
            time.sleep(2)

        time.sleep(5)
        for seed in seeds:
            out_dir = f"{OUT_BASE}/v440-{label}-seed{seed}"
            model_path = f"{out_dir}/model.pt"
            if not os.path.exists(model_path):
                continue
            r = eval_one(model_path, seed, label)
            if r:
                all_results.setdefault(label, []).append({"seed": seed, **r})
                off = r.get("code_offdiag")
                kc = r.get("known_conf")
                status = "COLLAPSED" if off and off > 0.5 else "PRESERVED"
                print(f"  seed={seed}: offdiag={off:.4f} known_conf={kc:.4f} [{status}]")
            time.sleep(1)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, reses in all_results.items():
        offs = [r["code_offdiag"] for r in reses if r.get("code_offdiag") is not None]
        avg_off = sum(offs)/len(offs) if offs else float('nan')
        n_preserved = sum(1 for r in reses if r.get("code_offdiag", 1) < 0.5)
        print(f"  {label}: {n_preserved}/{len(reses)} preserved geometry, avg offdiag={avg_off:.4f}")
        for r in reses:
            off = r.get("code_offdiag", 0)
            kc = r.get("known_conf", 0)
            strict = r.get("strict_status", "")
            print(f"    seed={r['seed']}: offdiag={off:.4f} known_conf={kc:.4f} strict={strict[:30]}")

    with open(f"{OUT_BASE}/v440_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_BASE}/v440_summary.json")

if __name__ == "__main__":
    main()
