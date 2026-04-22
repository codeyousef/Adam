#!/usr/bin/env python3
"""
Loss-Isolation Matrix at 26.8M (v6_overnight @ 45M).

Research2 hypothesis: collapse is caused by objective coupling — the ignorance/
classifier losses push OOD toward a low-info attractor simultaneously with retrieval
alignment, and larger models can "satisfy" both by collapsing to a cheap shared
manifold. The 13.6M model survives because limited capacity prevents reaching
the collapse attractor.

4 candidates × 3 seeds = 12 runs at 26.8M:
  A — Retrieval + negatives ONLY (ood_weight=0, clf_weight=0, ood_pred_weight=0)
  B — Full stack (ood_weight=0.2, clf_weight=0.25, ood_pred_weight=0.2) = v428 baseline
  C — Retrieval + OOD only (clf_weight=0, but ood active)
  D — Retrieval + negatives ONLY + stronger negatives (queue size ×4)

All use v6_overnight @ 45M → 8L/320-dim, ~26.8M actual params.
"""
import subprocess, sys, os, tempfile, yaml, json, time, glob

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

CANDIDATES = {
    "A_retrieval_only": dict(loss_ood_weight=0.0, loss_ood_pred_weight=0.0, loss_clf_weight=0.0),
    "B_full_stack":     dict(loss_ood_weight=0.2, loss_ood_pred_weight=0.2, loss_clf_weight=0.25),
    "C_retrieval_ood":  dict(loss_ood_weight=0.2, loss_ood_pred_weight=0.2, loss_clf_weight=0.0),
    "D_retrieval_strong_neg": dict(loss_ood_weight=0.0, loss_ood_pred_weight=0.0, loss_clf_weight=0.0),
}

SEEDS = [711, 2025, 1337]
SIZE = 45000000  # 45M → 8L/320-dim, 26.8M actual params
PROXY = "v6_overnight"
NUM_SPLITS = 1  # single split per seed for speed
STEPS = 500
BATCH_SIZE = 4

def make_config(seed, candidate_name, loss_weights, num_splits=NUM_SPLITS, steps=STEPS, batch_size=BATCH_SIZE):
    cfg = {
        "phase4": {
            "seed": seed,
            "proxy_recipe": PROXY,
            "sizes": [SIZE],
            "num_splits": num_splits,
            "steps": steps,
            "batch_size": batch_size,
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
    cfg["phase4"].update(loss_weights)
    return cfg

def run_one(cand_name, seed, loss_weights):
    out_dir = f"{OUT_BASE}/v429-{cand_name}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    log_path = f"{out_dir}/train.log"

    # Skip if already done
    if os.path.exists(model_path):
        print(f"  SKIP {cand_name} seed={seed} — already exists")
        return out_dir, model_path, log_path

    config = make_config(seed, cand_name, loss_weights)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name

    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", str(SIZE), "--device", "cuda"]
        print(f"  RUN  {cand_name} seed={seed} -> {out_dir}")
        with open(log_path, 'w') as logf:
            result = subprocess.run(cmd, cwd=ROOT, capture_output=False, stdout=logf, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"  FAIL {cand_name} seed={seed} — exit {result.returncode}")
            with open(log_path) as lf:
                print(lf.read()[-500:])
    finally:
        os.unlink(tmp)

    return out_dir, model_path, log_path

def eval_one(out_dir, model_path, cand_name, seed):
    eval_path = f"{out_dir}/eval.json"
    if os.path.exists(eval_path):
        print(f"  SKIP EVAL {cand_name} seed={seed}")
        with open(eval_path) as f:
            return json.load(f)

    cmd = [PY, f"{ROOT}/test_2.7b.py",
           "--json",
           "--confidence-threshold", "0.312",
           "--lexical-weight", "0.4",
           "--proxy-recipe", PROXY,
           "--", str(SIZE), model_path]
    print(f"  EVAL {cand_name} seed={seed}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True)
    try:
        data = json.loads(result.stdout.decode())
    except Exception:
        print(f"  EVAL ERROR: {result.stdout.decode()[-500:]}")
        return None
    with open(eval_path, 'w') as f:
        json.dump(data, f)
    return data

def summarize(cand_name, results):
    statuses = [r.get("strict_status","?") for r in results if r]
    codes = [r.get("code_offdiag", -1) for r in results if r]
    ranks = [r.get("code_rank", -1) for r in results if r]
    ood_confs = [r.get("ood_conf", -1) for r in results if r]
    known_confs = [r.get("known_conf", -1) for r in results if r]
    return {
        "candidate": cand_name,
        "n_runs": len(results),
        "strict_passes": statuses.count("PASS"),
        "strict_fails": statuses.count("FAIL"),
        "avg_code_offdiag": sum(codes)/len(codes) if codes else -1,
        "avg_code_rank": sum(ranks)/len(ranks) if ranks else -1,
        "avg_ood_conf": sum(ood_confs)/len(ood_confs) if ood_confs else -1,
        "avg_known_conf": sum(known_confs)/len(known_confs) if known_confs else -1,
        "all_results": results,
    }

def main():
    print("=" * 70)
    print("LOSS-ISOLATION MATRIX — v429 batch")
    print("=" * 70)

    all_results = {}

    for cand_name, loss_weights in CANDIDATES.items():
        print(f"\n{'='*60}")
        print(f"CANDIDATE: {cand_name}")
        print(f"  ood_weight={loss_weights.get('loss_ood_weight')}, "
              f"ood_pred_weight={loss_weights.get('loss_ood_pred_weight')}, "
              f"clf_weight={loss_weights.get('loss_clf_weight')}")
        print(f"{'='*60}")
        cand_results = []

        for seed in SEEDS:
            out_dir, model_path, log_path = run_one(cand_name, seed, loss_weights)
            if os.path.exists(model_path):
                eval_data = eval_one(out_dir, model_path, cand_name, seed)
                cand_results.append(eval_data)
            time.sleep(1)

        summary = summarize(cand_name, cand_results)
        all_results[cand_name] = summary

        # Print interim summary
        print(f"\n  >>> {cand_name} SUMMARY:")
        print(f"      strict_passes={summary['strict_passes']}/{summary['n_runs']}")
        print(f"      avg_code_offdiag={summary['avg_code_offdiag']:.4f}")
        print(f"      avg_code_rank={summary['avg_code_rank']:.4f}")
        print(f"      avg_ood_conf={summary['avg_ood_conf']:.4f}")
        print(f"      avg_known_conf={summary['avg_known_conf']:.4f}")

    # Final summary table
    print("\n" + "=" * 70)
    print("FINAL BATCH SUMMARY — Loss-Isolation Matrix")
    print("=" * 70)
    print(f"{'Candidate':<30} {'Pass/Total':<12} {'CodeOffDiag':<12} {'CodeRank':<10} {'OODConf':<10}")
    print("-" * 70)
    for cand_name, summary in all_results.items():
        print(f"{cand_name:<30} {summary['strict_passes']}/{summary['n_runs']:<12} "
              f"{summary['avg_code_offdiag']:<12.4f} {summary['avg_code_rank']:<10.4f} "
              f"{summary['avg_ood_conf']:<10.4f}")

    # Save summary
    summary_path = f"{OUT_BASE}/v429_loss_isolation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    a = all_results.get("A_retrieval_only", {})
    b = all_results.get("B_full_stack", {})
    c = all_results.get("C_retrieval_ood", {})
    d = all_results.get("D_retrieval_strong_neg", {})

    a_coll = a.get("avg_code_offdiag", 0) > 0.95
    b_coll = b.get("avg_code_offdiag", 0) > 0.95
    c_coll = c.get("avg_code_offdiag", 0) > 0.95

    if a_coll and not b_coll:
        print("UNEXPECTED: Full stack survives but retrieval-only collapses")
    elif b_coll and not a_coll:
        print("KEY FINDING: Retrieval-only SURVIVES — objective coupling confirmed!")
        print("  Next: staged training (retrieve first, then attach OOD/calibration)")
    elif b_coll and a_coll and not c_coll:
        print("KEY FINDING: OOD/clf losses are the collapse trigger!")
        print("  Next: disable OOD loss, use evidence-based abstention instead")
    elif all([a_coll, b_coll, c_coll]):
        print("ALL collapsed — collapse is in retrieval/alignment itself or parameterization")
        print("  Next: μP/u-μP parameterization test")
    else:
        print("Mixed results — see table above")

if __name__ == "__main__":
    main()
