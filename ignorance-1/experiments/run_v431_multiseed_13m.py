#!/usr/bin/env python3
"""
v431: Multi-seed × multi-step selection at 13.6M (v6_overnight @ 30M → 6L/256-dim).

CRITICAL FINDING from v430: ALL 9 runs collapsed at 13.6M — v378 was a lucky seed.

v378 used exactly 112 steps. All v430 runs used 500 steps. The training length
may be the critical variable: longer training → collapse.

This batch tests:
  - 15 seeds @ 112 steps (matching v378 exactly)
  - 10 seeds @ 200 steps (intermediate)
  - 5 seeds @ 500 steps (for comparison)

If 112-step seeds pass → v378's result is replicable with correct step count
If all collapse → the passing regime requires multi-cycle selection, not single-shot
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

SIZE = 30000000
PROXY = "v6_overnight"
BATCH_SIZE = 4

STEP_CONFIGS = [
    ("112", list(range(100, 115)), 112),   # 15 seeds — matches v378
    ("200", list(range(200, 210)), 200),   # 10 seeds — intermediate
    ("500", list(range(500, 505)), 500),   # 5 seeds — for comparison
]

def make_config(seed, steps):
    return {
        "phase4": {
            "seed": seed,
            "proxy_recipe": PROXY,
            "sizes": [SIZE],
            "num_splits": 1,
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "microbatch_size": 1,
            "lr": 5e-5,
            "phase4_dataset": "behavioral_constraints_v2_taxonomy_support_discipline_v1",
            "phase4_balance_families": True,
            "freeze_backbone": False,
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
        }
    }

def run_one(label, seed, steps):
    out_dir = f"{OUT_BASE}/v431-{label}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    log_path = f"{out_dir}/train.log"
    if os.path.exists(model_path):
        return out_dir, model_path, "skipped"
    config = make_config(seed, steps)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", str(SIZE), "--device", "cuda"]
        with open(log_path, 'w') as lf:
            result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
        return out_dir, model_path, "done" if result.returncode == 0 else f"fail({result.returncode})"
    finally:
        os.unlink(tmp)

def eval_one(out_dir, model_path, label, seed):
    eval_path = f"{out_dir}/eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            return json.load(f)
    cmd = [PY, f"{ROOT}/test_2.7b.py",
           "--json", "--confidence-threshold", "0.312", "--lexical-weight", "0.4",
           "--proxy-recipe", PROXY, "--", str(SIZE), model_path]
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
    print("v431: Multi-seed × Multi-step @ 13.6M")
    print(f"  112-step: {len(list(range(100,115)))} seeds")
    print(f"  200-step: {len(list(range(200,210)))} seeds")
    print(f"  500-step: {len(list(range(500,505)))} seeds")
    print("="*70)

    all_results = {}

    for label, seeds, steps in STEP_CONFIGS:
        print(f"\n--- {label}-step seeds ---")
        step_results = []

        for seed in seeds:
            out_dir, model_path, status = run_one(label, seed, steps)
            if status == "skipped":
                print(f"  SKIP seed={seed}")
            else:
                print(f"  TRAIN seed={seed} [{status}]")

        # Eval all
        for seed in seeds:
            out_dir = f"{OUT_BASE}/v431-{label}-seed{seed}"
            model_path = f"{out_dir}/model.pt"
            if os.path.exists(model_path):
                data = eval_one(out_dir, model_path, label, seed)
                summary = extract_summary(data)
                if summary:
                    step_results.append({"seed": seed, **summary})
                    is_pass = "PASS" in summary.get("strict_status", "")
                    off = summary.get("code_offdiag")
                    print(f"  EVAL seed={seed}: offdiag={off:.4f} {'✓ PASS' if is_pass else ''}")

        # Summarize step config
        n = len(step_results)
        passes = [r for r in step_results if "PASS" in r.get("strict_status", "")]
        offs = [r["code_offdiag"] for r in step_results if r.get("code_offdiag") is not None]
        ranks = [r["code_rank"] for r in step_results if r.get("code_rank") is not None]
        all_results[label] = {
            "n_tested": n,
            "n_pass": len(passes),
            "passing_seeds": [{"seed": r["seed"], "offdiag": r["code_offdiag"], "rank": r["code_rank"]} for r in passes],
            "avg_offdiag": sum(offs)/len(offs) if offs else -1,
            "avg_rank": sum(ranks)/len(ranks) if ranks else -1,
            "all": step_results,
        }

    # Print final table
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Steps':<8} {'Tested':<10} {'PASS':<8} {'AvgOffDiag':<12} {'AvgRank':<10}")
    print("-"*70)
    for label, res in all_results.items():
        print(f"{label:<8} {res['n_tested']:<10} {res['n_pass']:<8} {res['avg_offdiag']:<12.4f} {res['avg_rank']:<10.4f}")

    # Show passing seeds
    for label, res in all_results.items():
        if res["passing_seeds"]:
            print(f"\n{label}-step PASSING SEEDS:")
            for s in res["passing_seeds"]:
                r = next((r for r in res["all"] if r["seed"] == s["seed"]), {})
                print(f"  seed={s['seed']}: offdiag={s['offdiag']:.4f} rank={s['rank']:.4f} "
                      f"gap={r.get('gap',0):.4f} known_conf={r.get('known_conf',0):.4f} ood_conf={r.get('ood_conf',0):.4f}")

    if not any(res["passing_seeds"] for res in all_results.values()):
        print("\nNO PASSING SEEDS — all collapsed at all step counts")
        # Show best by offdiag
        for label, res in all_results.items():
            best = min(res["all"], key=lambda r: r.get("code_offdiag", 1.0), default=None)
            if best:
                print(f"  Best {label}-step: seed={best['seed']} offdiag={best['code_offdiag']:.4f} "
                      f"rank={best['code_rank']:.4f} gap={best.get('gap',0):.4f}")

    with open(f"{OUT_BASE}/v431_multiseed_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
