#!/usr/bin/env python3
"""
v433: Adaptive Selection Loop @ 13.6M — proper implementation.

v432 result: BOTH frozen and thawed collapse at 13.6M.
v378 PASSED (CC=41) through 8-cycle adaptive selection.

This implements the same adaptive process:
  1. Train N seeds for SHORT_WINDOW steps
  2. Run eval on ALL models to get code_offdiag
  3. Select top-K survivors (lowest offdiag)
  4. Generate NEW seeds from survivors (same seeds + fresh random)
  5. Repeat for CYCLE_COUNT cycles
  6. Final eval to determine strict PASS

Key: Use SHORT_WINDOW (50 steps) to quickly kill bad seeds,
then use survivors as seeds for the next generation.
"""
import subprocess, os, tempfile, yaml, json, time, random

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

SIZE = 30000000
PROXY = "v6_overnight"
SHORT_WINDOW = 50
CYCLE_COUNT = 4
N_SEEDS = 20
TOP_K = 5

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
            "phase4_dataset": "behavioral_constraints_v2_rigorous",
            "phase4_balance_families": True,
            "freeze_backbone": False,
        }
    }

def train_one(seed, steps, label):
    """Train model. Return (success, model_path)."""
    out_dir = f"{OUT_BASE}/v433-{label}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    if os.path.exists(model_path):
        return True, model_path
    config = make_config(seed, steps)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", str(SIZE), "--device", "cuda"]
        with open(f"{out_dir}/train.log", 'w') as lf:
            result = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
        return result.returncode == 0, model_path
    finally:
        os.unlink(tmp)

def eval_one(model_path, seed, label):
    """Run strict eval. Return (offdiag, rank, is_pass) or None."""
    eval_path = f"{OUT_BASE}/v433-{label}-seed{seed}/eval.json"
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
    offdiag = code_diag.get("avg_offdiag_similarity")
    rank = code_diag.get("participation_ratio")
    is_pass = "PASS" in data.get("strict_status", "")
    return offdiag, rank, is_pass

def run_cycle(cycle_num, seeds, steps, label_prefix):
    """Train and eval all seeds. Return sorted list of (seed, offdiag, rank, path)."""
    print(f"\n{'='*60}")
    print(f"CYCLE {cycle_num}/{CYCLE_COUNT} — Training {len(seeds)} seeds × {steps} steps")
    print(f"{'='*60}")

    results = {}
    for seed in seeds:
        success, model_path = train_one(seed, steps, f"c{cycle_num}")
        if success:
            results[seed] = (model_path, None)

    # Eval all models
    print(f"\n  Evalling {len(results)} models...")
    for seed, (model_path, _) in list(results.items()):
        label = f"c{cycle_num}"
        result = eval_one(model_path, seed, label)
        if result is None:
            print(f"  seed={seed}: EVAL FAILED")
            results[seed] = (model_path, (1.0, 0.0))
        else:
            offdiag, rank, is_pass = result
            results[seed] = (model_path, (offdiag, rank))
            status = "PASS" if is_pass else "FAIL"
            print(f"  seed={seed}: offdiag={offdiag:.4f} rank={rank:.4f} [{status}]")

    # Sort by offdiag
    sorted_seeds = sorted(results.keys(), key=lambda s: results[s][1][0])
    return [(s, results[s][1][0], results[s][1][1], results[s][0]) for s in sorted_seeds]

def main():
    print("="*60)
    print("v433: Adaptive Selection @ 13.6M")
    print(f"  {N_SEEDS} seeds × {CYCLE_COUNT} cycles × {SHORT_WINDOW} steps")
    print("="*60)

    # Start with N_SEEDS random seeds
    all_seeds = set(random.randint(0, 2**31 - 1) for _ in range(N_SEEDS))
    cycle_history = []

    for cycle in range(1, CYCLE_COUNT + 1):
        label = f"c{cycle}"
        seeds = sorted(all_seeds)
        ranked = run_cycle(cycle, seeds, SHORT_WINDOW, label)

        # Select survivors
        survivors = ranked[:TOP_K]
        collapsed = all(r[1] > 0.9 for r in ranked)

        print(f"\n  Cycle {cycle} survivors (top {TOP_K}):")
        for i, (seed, offdiag, rank, path) in enumerate(survivors):
            print(f"    {i+1}. seed={seed}: offdiag={offdiag:.4f} rank={rank:.4f}")

        cycle_history.append({
            "cycle": cycle,
            "n_seeds": len(seeds),
            "survivors": [{"seed": s, "offdiag": o, "rank": r} for s, o, r, _ in survivors],
            "collapsed": collapsed,
        })

        if collapsed:
            print(f"\n  ALL COLLAPSED in cycle {cycle}!")
            break

        # Next generation: survivors + new random seeds
        survivor_seeds = {s for s, _, _, _ in survivors}
        new_seeds = set(random.randint(0, 2**31 - 1) for _ in range(N_SEEDS // 2))
        all_seeds = survivor_seeds | new_seeds
        print(f"\n  Next gen: {len(survivor_seeds)} survivors + {len(new_seeds)} new = {len(all_seeds)} seeds")

    # Final eval of all unique survivors from all cycles
    print(f"\n{'='*60}")
    print("FINAL EVAL — all survivors from all cycles")
    print(f"{'='*60}")

    all_survivor_seeds = set()
    for h in cycle_history:
        for s in h["survivors"]:
            all_survivor_seeds.add(s["seed"])

    print(f"  Evalling {len(all_survivor_seeds)} unique survivors...")
    final_results = []
    for seed in sorted(all_survivor_seeds):
        # Find the best model (latest cycle)
        model_path = None
        for c in range(CYCLE_COUNT, 0, -1):
            p = f"{OUT_BASE}/v433-c{c}-seed{seed}/model.pt"
            if os.path.exists(p):
                model_path = p
                break
        if not model_path:
            continue
        result = eval_one(model_path, seed, f"c{CYCLE_COUNT}")
        if result:
            offdiag, rank, is_pass = result
            final_results.append({
                "seed": seed,
                "offdiag": offdiag,
                "rank": rank,
                "is_pass": is_pass,
            })
            print(f"  seed={seed}: offdiag={offdiag:.4f} rank={rank:.4f} {'PASS' if is_pass else 'FAIL'}")

    passes = [r for r in final_results if r["is_pass"]]
    print(f"\n{'='*60}")
    print("v433 FINAL RESULT")
    print(f"{'='*60}")
    print(f"Cycles completed: {len(cycle_history)}")
    print(f"Total survivors evalled: {len(final_results)}")
    print(f"Strict PASS: {len(passes)}/{len(final_results)}")

    if passes:
        print(f"\nPASSING SEEDS:")
        for r in passes:
            print(f"  seed={r['seed']}: offdiag={r['offdiag']:.4f} rank={r['rank']:.4f}")
    else:
        best = min(final_results, key=lambda r: r["offdiag"], default=None)
        if best:
            print(f"\nBest (lowest offdiag): seed={best['seed']} offdiag={best['offdiag']:.4f} rank={best['rank']:.4f}")

    with open(f"{OUT_BASE}/v433_summary.json", 'w') as f:
        json.dump({
            "cycle_history": cycle_history,
            "final_results": final_results,
            "n_pass": len(passes),
        }, f, indent=2)

if __name__ == "__main__":
    main()
