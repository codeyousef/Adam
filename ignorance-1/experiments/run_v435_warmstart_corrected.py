#!/usr/bin/env python3
"""
v435: Warm-start from v338 + freeze_backbone + strong classifier @ 13.6M.

Same as v434 but patch is now in train_production.py (which is the ACTUAL training code).
v434 failed because warm-start loading was in phase4.py (dead code) — train_production.py
has its own training loop that does NOT call run_phase4.

This time the warm-start + freeze_backbone should actually work.
10 seeds, 300 steps.
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

WARM_START = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v338/v338-promoted-earlier-onset-tiny-mixed-bridge-seed504/model.pt"
SEEDS = list(range(810, 820))  # 10 seeds
SIZE = 15000000
PROXY = "v6_overnight"

def make_config(seed):
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
            # KEY: warm-start from v338 + freeze backbone
            "warm_start_model_path": WARM_START,
            "freeze_backbone": True,
            # Stronger classifier head
            "ood_weight": 0.0,
            "clf_weight": 0.15,
            "warmup_fraction": 0.15,
            "min_lr_ratio": 0.2,
        }
    }

def run_one(seed):
    out_dir = f"{OUT_BASE}/v435-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    log_path = f"{out_dir}/train.log"
    if os.path.exists(model_path):
        return out_dir, model_path, "skipped"
    config = make_config(seed)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", str(SIZE), "--device", "cuda"]
        print(f"  TRAIN seed={seed}")
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
    except:
        return None

def extract_summary(data):
    if not data:
        return None
    code_diag = data.get("code_diagnostics", {})
    query_diag = data.get("query_diagnostics", {})
    return {
        "strict_status": data.get("strict_status", ""),
        "strict_failures": data.get("strict_failures", []),
        "code_offdiag": code_diag.get("avg_offdiag_similarity"),
        "code_rank": code_diag.get("participation_ratio"),
        "query_offdiag": query_diag.get("avg_offdiag_similarity"),
        "query_rank": query_diag.get("participation_ratio"),
        "known_conf": data.get("avg_known_confidence"),
        "ood_conf": data.get("avg_ood_confidence"),
        "known_sim": data.get("avg_known_similarity"),
        "ignorant_sim": data.get("avg_ignorant_similarity"),
        "gap": data.get("ignorance_gap"),
    }

def main():
    print("="*70)
    print("v435: Warm-start + freeze_backbone @ 13.6M (CORRECTED)")
    print(f"  {len(SEEDS)} seeds × 300 steps")
    print(f"  warm_start_model_path={WARM_START}")
    print("="*70)

    results = []
    for seed in SEEDS:
        out_dir, model_path, status = run_one(seed)
        if status != "skipped":
            time.sleep(0.5)

    for seed in SEEDS:
        out_dir = f"{OUT_BASE}/v435-seed{seed}"
        model_path = f"{out_dir}/model.pt"
        if os.path.exists(model_path):
            data = eval_one(out_dir, model_path, seed)
            summary = extract_summary(data)
            if summary:
                results.append({"seed": seed, **summary})
                is_pass = "PASS" in summary.get("strict_status", "")
                off = summary.get("code_offdiag")
                kc = summary.get("known_conf", 0)
                ooc = summary.get("ood_conf", 0)
                print(f"  seed={seed}: offdiag={off:.4f} known_conf={kc:.4f} ood_conf={ooc:.4f} {'✓ PASS' if is_pass else ''}")

    n = len(results)
    passes = [r for r in results if "PASS" in r.get("strict_status", "")]
    offs = [r["code_offdiag"] for r in results if r.get("code_offdiag") is not None]
    kcs = [r["known_conf"] for r in results if r.get("known_conf") is not None]

    print("\n" + "="*70)
    print("v435 RESULTS — Warm-start + freeze_backbone @ 13.6M")
    print("="*70)
    print(f"Seeds tested: {n}")
    print(f"Strict PASS: {len(passes)}/{n}")
    if offs:
        print(f"Avg code_offdiag: {sum(offs)/len(offs):.4f}")
        print(f"Avg known_conf: {sum(kcs)/len(kcs):.4f}")

    if passes:
        print(f"\nPASSING SEEDS:")
        for r in passes:
            print(f"  seed={r['seed']}: offdiag={r['code_offdiag']:.4f} rank={r['code_rank']:.4f} "
                  f"gap={r.get('gap',0):.4f} known_conf={r['known_conf']:.4f} ood_conf={r['ood_conf']:.4f}")
    else:
        best = max(results, key=lambda r: r.get("known_conf", 0), default=None)
        if best:
            print(f"\nBest (highest known_conf): seed={best['seed']} "
                  f"known_conf={best['known_conf']:.4f} offdiag={best['code_offdiag']:.4f}")
            print(f"  Failures: {best.get('strict_failures', [])[:3]}")

    with open(f"{OUT_BASE}/v435_summary.json", 'w') as f:
        json.dump({"results": results, "n_tested": n, "n_pass": len(passes)}, f, indent=2)

if __name__ == "__main__":
    main()
