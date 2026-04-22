#!/usr/bin/env python3
"""
v430: Loss-Isolation Matrix at 13.6M scale (v6_overnight @ 30M → 6L/256-dim).

Purpose: Isolate whether the 13.6M passing result (v378) is:
  (a) Reproducible with any seed — the scale itself is the cure
  (b) Lucky seed/ininitialization — only specific seeds work
  (c) Specific to certain loss combinations — some recipes work, some don't

4 candidates × 3 seeds = 12 runs at 13.6M:
  A — Retrieval + negatives ONLY (ood_weight=0, clf_weight=0, ood_pred_weight=0)
  B — Full stack (ood_weight=0.2, clf_weight=0.25, ood_pred_weight=0.2)
  C — Retrieval + OOD only (clf_weight=0)
  D — Retrieval + stronger negatives (queue size ×4, ood_weight=0)

Same as v429 but at 13.6M (30M size class) instead of 26.8M (45M size class).
If A passes → 13.6M scale itself prevents collapse, no specific recipe needed.
If B passes but A fails → OOD/clf losses help at small scale
If all fail → v378 was a lucky seed — need multi-seed selection
If all pass → any recipe works at 13.6M — reproducibility confirmed
"""
import subprocess, os, tempfile, yaml, json, time

PY = "/mnt/Storage/Projects/catbelly_studio/.venv/bin/python"
ROOT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
OUT_BASE = f"{ROOT}/artifacts/strict_eval_autoresearch_v4"

CANDIDATES = {
    "A_retrieval_only": dict(loss_ood_weight=0.0, loss_ood_pred_weight=0.0, loss_clf_weight=0.0),
    "B_full_stack":     dict(loss_ood_weight=0.2, loss_ood_pred_weight=0.2, loss_clf_weight=0.25),
    "C_retrieval_ood":  dict(loss_ood_weight=0.2, loss_ood_pred_weight=0.2, loss_clf_weight=0.0),
}
SEEDS = [711, 2025, 1337]
SIZE = 30000000   # 30M → 6L/256-dim, 13.6M actual params
PROXY = "v6_overnight"
STEPS = 500
BATCH_SIZE = 4

def make_config(seed, loss_weights):
    cfg = {
        "phase4": {
            "seed": seed,
            "proxy_recipe": PROXY,
            "sizes": [SIZE],
            "num_splits": 1,
            "steps": STEPS,
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
    cfg["phase4"].update(loss_weights)
    return cfg

def run_one(cand_name, seed, loss_weights):
    out_dir = f"{OUT_BASE}/v430-{cand_name}-seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/model.pt"
    log_path = f"{out_dir}/train.log"
    if os.path.exists(model_path):
        print(f"  SKIP train {cand_name} seed={seed}")
        return out_dir, model_path
    config = make_config(seed, loss_weights)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp = f.name
    try:
        cmd = [PY, f"{ROOT}/train_production.py", "--config", tmp, "--output", model_path,
               "--size", str(SIZE), "--device", "cuda"]
        print(f"  RUN  {cand_name} seed={seed}")
        with open(log_path, 'w') as lf:
            subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
    finally:
        os.unlink(tmp)
    return out_dir, model_path

def eval_one(out_dir, model_path, cand_name, seed):
    eval_path = f"{out_dir}/eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            return json.load(f)
    cmd = [PY, f"{ROOT}/test_2.7b.py",
           "--json", "--confidence-threshold", "0.312", "--lexical-weight", "0.4",
           "--proxy-recipe", PROXY, "--", str(SIZE), model_path]
    print(f"  EVAL {cand_name} seed={seed}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=120)
    # Parse from mixed stdout/stderr
    text = result.stdout.decode() + result.stderr.decode()
    data = extract_eval(text)
    if data:
        with open(eval_path, 'w') as f:
            json.dump(data, f)
    return data

def extract_eval(text):
    """Parse key metrics from test_2.7b.py raw output."""
    import re
    def get(key):
        m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', text)
        return float(m.group(1)) if m else None
    def get_str(key):
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
        return m.group(1) if m else None
    data = {
        "strict_status": get_str("strict_status"),
        "code_offdiag": get("code_offdiag"),
        "query_offdiag": get("query_offdiag"),
        "code_rank": get("code_rank"),
        "query_rank": get("query_rank"),
        "known_conf": get("known_conf"),
        "ood_conf": get("ood_conf"),
        "known_sim": get("known_sim"),
        "ignorant_sim": get("ignorant_sim"),
        "gap": get("gap"),
    }
    return data if any(v is not None for v in data.values()) else None

def summarize_all():
    results = {}
    for cand_name in CANDIDATES:
        runs = []
        for seed in SEEDS:
            out_dir = f"{OUT_BASE}/v430-{cand_name}-seed{seed}"
            eval_path = f"{out_dir}/eval.json"
            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    runs.append(json.load(f))
        n = len(runs)
        passes = sum(1 for r in runs if r and r.get('strict_status') == 'PASS')
        def avg(key): return sum(r[key] for r in runs if r and r.get(key) is not None) / n if n else -1
        results[cand_name] = {
            "n_runs": n, "strict_passes": passes,
            "avg_code_offdiag": avg("code_offdiag"),
            "avg_code_rank": avg("code_rank"),
            "avg_ood_conf": avg("ood_conf"),
            "avg_known_conf": avg("known_conf"),
            "all_results": runs,
        }
    return results

def main():
    print("="*70)
    print("LOSS-ISOLATION MATRIX at 13.6M — v430 batch")
    print("="*70)

    # Train all 12 models
    for cand_name, loss_weights in CANDIDATES.items():
        print(f"\nCANDIDATE: {cand_name}")
        for seed in SEEDS:
            run_one(cand_name, seed, loss_weights)
            time.sleep(0.5)

    # Eval all 12 models
    for cand_name in CANDIDATES:
        for seed in SEEDS:
            out_dir = f"{OUT_BASE}/v430-{cand_name}-seed{seed}"
            model_path = f"{out_dir}/model.pt"
            if os.path.exists(model_path):
                eval_one(out_dir, model_path, cand_name, seed)

    # Summarize
    results = summarize_all()
    print("\n" + "="*70)
    print("FINAL RESULTS — v430 Loss-Isolation @ 13.6M")
    print("="*70)
    print(f"{'Candidate':<30} {'Pass/Total':<12} {'CodeOffDiag':<12} {'CodeRank':<10} {'OODConf':<10}")
    print("-"*70)
    for cand, s in results.items():
        print(f"{cand:<30} {s['strict_passes']}/{s['n_runs']:<12} "
              f"{s['avg_code_offdiag']:<12.4f} {s['avg_code_rank']:<10.4f} {s['avg_ood_conf']:<10.4f}")

    with open(f"{OUT_BASE}/v430_loss_isolation_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    a = results.get("A_retrieval_only", {})
    b = results.get("B_full_stack", {})
    c = results.get("C_retrieval_ood", {})

    a_off = a.get("avg_code_offdiag", -1)
    b_off = b.get("avg_code_offdiag", -1)
    c_off = c.get("avg_code_offdiag", -1)
    a_passes = a.get("strict_passes", 0)
    b_passes = b.get("strict_passes", 0)

    print(f"A_retrieval_only:  passes={a_passes}/3 offdiag={a_off:.4f} rank={a.get('avg_code_rank',0):.4f}")
    print(f"B_full_stack:      passes={b_passes}/3 offdiag={b_off:.4f} rank={b.get('avg_code_rank',0):.4f}")
    print(f"C_retrieval_ood:   offdiag={c_off:.4f}")

    if a_passes >= 2 and a_off < 0.9:
        print("\nRESULT: Retrieval-only SURVIVES at 13.6M — scale itself prevents collapse!")
        print("  Next: test intermediate scales (20M, 25M) to find the collapse boundary precisely")
    elif b_passes >= 2 and b_off < 0.9:
        print("\nRESULT: Full stack needed at small scale — OOD helps, not hurts")
        print("  Next: staged training (retrieve-only phase then attach OOD)")
    elif a_passes == 0 and b_passes == 0:
        print("\nRESULT: ALL collapsed at 13.6M — v378 was a lucky seed!")
        print("  Next: multi-seed selection (>5 seeds) to find lucky seeds at 13.6M")
    else:
        print("\nRESULT: Mixed — need 5+ seeds for statistical robustness")

if __name__ == "__main__":
    main()
