#!/usr/bin/env python3
"""
v432: freeze_backbone=true at 13.6M — THE critical test.

All prior batches (v428-v431) used freeze_backbone=false (default).
v431: 30 seeds, all collapsed at 13.6M.
v430: 9 seeds, all collapsed at 13.6M.
v429: 12 models, all collapsed at 26.8M.

The v378 incumbent used freeze_backbone=true.
This batch tests ONLY the freeze_backbone variable:
- 15 seeds at 13.6M, freeze_backbone=true, all else minimal
- 5 seeds at 13.6M, freeze_backbone=false (baseline, should collapse)

If frozen seeds pass → backbone freezing is the key to preventing collapse
If frozen seeds still collapse → the v378 result came from the 8-cycle selection
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

SIZE = 30000000
PROXY = "v6_overnight"
FROZEN_SEEDS = list(range(600, 615))  # 15 seeds with freeze_backbone=true
THAWED_SEEDS = list(range(700, 705))  # 5 seeds with freeze_backbone=false (baseline)

def make_config(seed, freeze_backbone):
    return {
        "phase4": {
            "seed": seed,
            "proxy_recipe": PROXY,
            "sizes": [SIZE],
            "num_splits": 1,
            "steps": 300,
            "batch_size": 4,
            "microbatch_size": 1,
            "lr": 5e-5,
            "phase4_dataset": "behavioral_constraints_v2_rigorous",
            "phase4_balance_families": True,
            "freeze_backbone": freeze_backbone,
            "warmup_fraction": 0.15,
            "min_lr_ratio": 0.2,
        }
    }

def run_one(seed, freeze_backbone):
    label = "frozen" if freeze_backbone else "thawed"
    out_dir = f"{OUT_BASE}/v432-{label}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    log_path = f"{out_dir}/train.log"
    if os.path.exists(model_path):
        return out_dir, model_path, "skipped"
    config = make_config(seed, freeze_backbone)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", str(SIZE), "--device", "cuda"]
        print(f"  TRAIN seed={seed} freeze={freeze_backbone}")
        with open(log_path, 'w') as lf:
            result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
        return out_dir, model_path, "done" if result.returncode == 0 else f"fail({result.returncode})"
    finally:
        os.unlink(tmp)

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
    print("v432: freeze_backbone=true vs false @ 13.6M")
    print(f"  Frozen: {len(FROZEN_SEEDS)} seeds | Thawed: {len(THAWED_SEEDS)} seeds")
    print("="*70)

    all_results = {}

    # Train frozen seeds
    print("\n--- FROZEN (freeze_backbone=true) ---")
    frozen_results = []
    for seed in FROZEN_SEEDS:
        out_dir, model_path, status = run_one(seed, True)
        if status != "skipped":
            time.sleep(0.5)
    for seed in FROZEN_SEEDS:
        out_dir = f"{OUT_BASE}/v432-frozen-seed{seed}"
        model_path = f"{out_dir}/model.pt"
        if os.path.exists(model_path):
            data = eval_one(out_dir, model_path, seed)
            summary = extract_summary(data)
            if summary:
                frozen_results.append({"seed": seed, **summary})
                is_pass = "PASS" in summary.get("strict_status", "")
                off = summary.get("code_offdiag")
                print(f"  seed={seed}: offdiag={off:.4f} {'✓ PASS' if is_pass else ''}")

    # Train thawed seeds
    print("\n--- THAWED (freeze_backbone=false) ---")
    thawed_results = []
    for seed in THAWED_SEEDS:
        out_dir, model_path, status = run_one(seed, False)
        if status != "skipped":
            time.sleep(0.5)
    for seed in THAWED_SEEDS:
        out_dir = f"{OUT_BASE}/v432-thawed-seed{seed}"
        model_path = f"{out_dir}/model.pt"
        if os.path.exists(model_path):
            data = eval_one(out_dir, model_path, seed)
            summary = extract_summary(data)
            if summary:
                thawed_results.append({"seed": seed, **summary})
                is_pass = "PASS" in summary.get("strict_status", "")
                off = summary.get("code_offdiag")
                print(f"  seed={seed}: offdiag={off:.4f} {'✓ PASS' if is_pass else ''}")

    all_results["frozen"] = frozen_results
    all_results["thawed"] = thawed_results

    # Summarize
    print("\n" + "="*70)
    print("v432 RESULTS — freeze_backbone @ 13.6M")
    print("="*70)
    for group, res_list in [("FROZEN", frozen_results), ("THAWED", thawed_results)]:
        n = len(res_list)
        passes = [r for r in res_list if "PASS" in r.get("strict_status", "")]
        offs = [r["code_offdiag"] for r in res_list if r.get("code_offdiag") is not None]
        ranks = [r["code_rank"] for r in res_list if r.get("code_rank") is not None]
        avg_off = sum(offs)/len(offs) if offs else -1
        avg_rank = sum(ranks)/len(ranks) if ranks else -1
        print(f"{group}: {len(passes)}/{n} PASS | avg_offdiag={avg_off:.4f} | avg_rank={avg_rank:.4f}")
        if passes:
            for r in passes:
                print(f"  seed={r['seed']}: offdiag={r['code_offdiag']:.4f} rank={r['code_rank']:.4f} "
                      f"gap={r.get('gap',0):.4f}")

    with open(f"{OUT_BASE}/v432_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Interpretation
    frozen_passes = len([r for r in frozen_results if "PASS" in r.get("strict_status", "")])
    thawed_passes = len([r for r in thawed_results if "PASS" in r.get("strict_status", "")])

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    if frozen_passes > 0 and thawed_passes == 0:
        print("RESULT: freeze_backbone=true PREVENTS collapse at 13.6M!")
        print("  Next: test freeze_backbone=true at 26.8M and 2.7B")
    elif frozen_passes > 0 and thawed_passes > 0:
        print("RESULT: Both frozen and thawed pass — collapse not prevented by freezing alone")
    elif frozen_passes == 0 and thawed_passes == 0:
        print("RESULT: Both frozen and thawed collapse — v378 result came from multi-cycle selection")
        print("  Next: need multi-cycle adaptive selection, not single-shot training")
    else:
        print(f"RESULT: Mixed. Frozen={frozen_passes} thawed={thawed_passes}")

if __name__ == "__main__":
    main()
