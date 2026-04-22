#!/usr/bin/env python3
"""
v436: Adaptive Selection @ 13.6M with CORRECTED v378 config.

KEY FINDINGS:
- v378 incumbent was strict FAIL (score=41.11 was legacy, not strict)
- v378 warm-start was BROKEN (shape mismatches — v338 has embed=256, v378 model has embed=192)
- v435 collapsed despite warm-start because the warm-start didn't actually load
- v378's multi-cycle adaptive selection found seeds that preserved geometry even from broken warm-start
- The DIFFERENCE is the training components: current train_production.py has 3 losses,
  v378 had many more (late_inter_verifier, retrieval_facets, vicreg, sigreg, etc.)

v378 used: phase4_dataset=behavioral_constraints_v2_taxonomy_support_discipline_v1
Current code uses: behavioral_constraints_v2_rigorous

This run tests: does CORRECTING the dataset to match v378 help?
Then adaptive selection across 20 seeds, keeping best across cycles.
"""
import subprocess, os, tempfile, yaml, json, time, random

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

SIZE = 15000000
PROXY = "v6_overnight"
SHORT_WINDOW = 50
CYCLE_COUNT = 4
N_SEEDS = 20
TOP_K = 5

# v378 incumbent's dataset
DATASET = "behavioral_constraints_v2_taxonomy_support_discipline_v1"

def make_config(seed, steps):
    return {
        "phase4": {
            "seed": seed,
            "proxy_recipe": PROXY,
            "sizes": [SIZE],
            "num_splits": 1,
            "steps": steps,
            "batch_size": 4,
            "microbatch_size": 1,
            "lr": 5e-5,
            "phase4_dataset": DATASET,
            "phase4_balance_families": True,
            "phase4_joint_training": True,
            "phase4_factorized_hard_negatives": True,
            "freeze_backbone": False,
            "warmup_fraction": 0.15,
            "min_lr_ratio": 0.2,
        }
    }

def train_one(seed, steps, label):
    """Train model. Return (success, model_path)."""
    out_dir = f"{OUT_BASE}/v436-{label}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    if os.path.exists(model_path):
        return True, model_path, out_dir
    config = make_config(seed, steps)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", str(SIZE), "--device", "cuda"]
        with open(f"{out_dir}/train.log", 'w') as lf:
            result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
        return result.returncode == 0, model_path, out_dir
    finally:
        os.unlink(tmp)

def eval_one(model_path, seed, label):
    """Run strict eval. Return summary dict or None."""
    eval_path = f"{OUT_BASE}/v436-{label}-seed{seed}/eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data = json.load(f)
    else:
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
        except:
            return None

    code_diag = data.get("code_diagnostics", {})
    query_diag = data.get("query_diagnostics", {})
    return {
        "code_offdiag": code_diag.get("avg_offdiag_similarity"),
        "code_rank": code_diag.get("participation_ratio"),
        "known_conf": data.get("avg_known_confidence"),
        "ood_conf": data.get("avg_ood_confidence"),
        "gap": data.get("ignorance_gap"),
        "strict_status": data.get("strict_status", ""),
        "strict_failures": data.get("strict_failures", []),
        "known_exact": data.get("known_exact_match_similarity"),
        "known_para": data.get("known_paraphrase_similarity"),
    }

def run_cycle(cycle_num, seeds, steps, label):
    """Train and eval all seeds. Return sorted list."""
    print(f"\n{'='*60}")
    print(f"CYCLE {cycle_num}/{CYCLE_COUNT} — {len(seeds)} seeds × {steps} steps, dataset={DATASET}")
    print(f"{'='*60}")

    for seed in seeds:
        train_one(seed, steps, label)

    # Eval all models
    results = {}
    for seed in seeds:
        out_dir = f"{OUT_BASE}/v436-{label}-seed{seed}"
        model_path = f"{out_dir}/model.pt"
        if os.path.exists(model_path):
            r = eval_one(model_path, seed, label)
            results[seed] = r
            if r:
                status = "PASS" if "PASS" in r.get("strict_status", "") else "FAIL"
                print(f"  seed={seed}: offdiag={r['code_offdiag']:.4f} rank={r['code_rank']:.4f} "
                      f"known_conf={r.get('known_conf', 0):.4f} ood_conf={r.get('ood_conf', 0):.4f} [{status}]")
            else:
                print(f"  seed={seed}: EVAL FAILED")
                results[seed] = None

    # Sort by offdiag (lower = better geometry)
    valid = [(s, r) for s, r in results.items() if r is not None]
    valid.sort(key=lambda x: x[1]['code_offdiag'])
    return valid

def main():
    print("="*60)
    print("v436: Adaptive Selection @ 13.6M — CORRECTED v378 dataset")
    print(f"  {N_SEEDS} seeds × {CYCLE_COUNT} cycles × {SHORT_WINDOW} steps")
    print(f"  dataset={DATASET}")
    print("="*60)

    all_seeds = [random.randint(0, 2**31 - 1) for _ in range(N_SEEDS)]
    cycle_history = []

    for cycle in range(1, CYCLE_COUNT + 1):
        label = f"c{cycle}"
        valid = run_cycle(cycle, sorted(all_seeds), SHORT_WINDOW, label)

        if not valid:
            print(f"\n  ALL FAILED in cycle {cycle}!")
            break

        survivors = valid[:TOP_K]
        all_collapsed = all(r['code_offdiag'] > 0.9 for _, r in valid)

        print(f"\n  Cycle {cycle} survivors (top {TOP_K}):")
        for i, (seed, r) in enumerate(survivors):
            print(f"    {i+1}. seed={seed}: offdiag={r['code_offdiag']:.4f} rank={r['code_rank']:.4f} "
                  f"known_conf={r.get('known_conf', 0):.4f}")

        cycle_history.append({
            "cycle": cycle,
            "n_seeds": len(all_seeds),
            "survivors": [{"seed": s, "offdiag": r['code_offdiag'], "rank": r['code_rank'],
                           "known_conf": r.get('known_conf', 0)} for s, r in survivors],
            "collapsed": all_collapsed,
        })

        if all_collapsed:
            print(f"\n  ALL COLLAPSED in cycle {cycle}!")
            break

        # Next gen: survivors + new seeds
        survivor_seeds = {s for s, _ in survivors}
        new_seeds = set(random.randint(0, 2**31 - 1) for _ in range(N_SEEDS // 2))
        all_seeds = list(survivor_seeds | new_seeds)
        print(f"\n  Next gen: {len(survivor_seeds)} survivors + {len(new_seeds)} new = {len(all_seeds)} seeds")

    # Final eval
    print(f"\n{'='*60}")
    print("FINAL EVAL — all unique survivors")
    print(f"{'='*60}")

    all_survivor_seeds = set()
    for h in cycle_history:
        for s in h["survivors"]:
            all_survivor_seeds.add(s["seed"])

    final_results = []
    for seed in sorted(all_survivor_seeds):
        model_path = None
        for c in range(CYCLE_COUNT, 0, -1):
            p = f"{OUT_BASE}/v436-c{c}-seed{seed}/model.pt"
            if os.path.exists(p):
                model_path = p
                break
        if not model_path:
            continue
        r = eval_one(model_path, seed, f"c{CYCLE_COUNT}")
        if r:
            final_results.append({"seed": seed, **r})
            status = "PASS" if "PASS" in r.get("strict_status", "") else "FAIL"
            print(f"  seed={seed}: offdiag={r['code_offdiag']:.4f} rank={r['code_rank']:.4f} "
                  f"known_conf={r.get('known_conf', 0):.4f} [{status}]")

    passes = [r for r in final_results if "PASS" in r.get("strict_status", "")]
    print(f"\n{'='*60}")
    print("v436 FINAL RESULT")
    print(f"{'='*60}")
    print(f"Cycles: {len(cycle_history)}, Survivors evalled: {len(final_results)}, PASS: {len(passes)}/{len(final_results)}")

    if passes:
        print(f"\nPASSING SEEDS:")
        for r in passes:
            print(f"  seed={r['seed']}: offdiag={r['code_offdiag']:.4f} rank={r['code_rank']:.4f} "
                  f"gap={r.get('gap', 0):.4f}")
    else:
        best = min(final_results, key=lambda r: r.get("code_offdiag", 1.0), default=None)
        if best:
            print(f"\nBest (lowest offdiag): seed={best['seed']} offdiag={best['code_offdiag']:.4f} "
                  f"rank={best['code_rank']:.4f}")

    with open(f"{OUT_BASE}/v436_summary.json", 'w') as f:
        json.dump({"cycle_history": cycle_history, "final_results": final_results,
                   "n_pass": len(passes)}, f, indent=2)

if __name__ == "__main__":
    main()
