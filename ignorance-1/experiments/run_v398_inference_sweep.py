"""
v398: Inference hyperparameter sweep on frozen v378 late-inter-high-weight checkpoint.
Keeps model weights frozen; only varies inference parameters.
Finds best answer_score config.
"""
import json, os, random, subprocess, sys, tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/Storage/Projects/catbelly_studio/ignorance-1")
CKPT = str(ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt")
PYTHON = str(ROOT.parent / ".venv/bin/python")
SIZE = "15000000"

# V378's fixed config (from config.yaml)
FIXED_FLAGS = [
    "--rerank-topk", "5",
    "--rerank-shortlist-mode", "pred_query_union_local",
    "--rerank-query-weight", "0.3",
    "--rerank-agreement-weight", "0.18",
    "--rerank-lexical-weight", "0.0",
    "--rerank-support-weight", "0.24",
    "--rerank-consensus-weight", "0.35",
    "--rerank-consensus-temperature", "0.0184",
    "--rerank-consensus-floor", "0.9158",
    "--rerank-consensus-margin-gate", "0.0092",
    "--rerank-pairwise-mode", "supportspec_citecheck_floor_borda",
    "--rerank-support-floor-margin-gate", "0.014",
    "--rerank-spec-weight", "0.18",
    "--rerank-answerspec-mode", "code_pref",
    "--rerank-answerspec-margin-gate", "0.034",
    "--rerank-safe-expand-topk", "6",
    "--rerank-safe-expand-margin", "0.004",
    "--rerank-parafence-weight", "1.0",
    "--rerank-parafence-variants", "3",
    "--selective-gate-mode", "margin_mean_gap",
    "--selective-gate-margin-threshold", "0.01",
    "--selective-gate-mean-gap-threshold", "0.016",
    "--selective-gate-similarity-floor", "0.69",
    "--rerank-verifier-uplift-weight", "0.4",
    "--rerank-verifier-gap-scale", "1.0",
    "--rerank-verifier-support-weight", "1.0",
    "--rerank-verifier-spec-weight", "0.0",
    "--retrieval-facet-score-mode", "softmax_maxsim",
    "--retrieval-facet-softmax-temperature", "0.1",
    "--retrieval-global-facet-blend", "0.35",
    "--confidence-mode", "support_feature_calibrator",
    "--confidence-support-topk", "5",
    "--confidence-support-temperature", "0.1",
]

# Sweep space
SWEEP_SPACE = {
    "--confidence-threshold": [0.1, 0.15, 0.2, 0.25, 0.312, 0.35, 0.4, 0.45, 0.5],
    "--lexical-weight": [0.0, 0.2, 0.4, 0.6, 0.8],
    "--retrieval-facet-score-mode": ["softmax_maxsim", "maxsim", "mean_pool"],
    "--rerank-consensus-temperature": [0.01, 0.0184, 0.03, 0.05],
    "--rerank-agreement-weight": [0.1, 0.18, 0.3],
    "--selective-gate-similarity-floor": [0.5, 0.6, 0.69, 0.8, 0.9],
}


def parse_summary_json(stdout):
    """Find the summary JSON dict in stdout using incremental parsing."""
    for idx in range(len(stdout)):
        if stdout[idx] != "{":
            continue
        for end in range(idx + 20, min(idx + 100000, len(stdout) + 1)):
            try:
                data = json.loads(stdout[idx:end])
                if isinstance(data, dict) and len(data) > 5:
                    return data
            except Exception:
                pass
    return {}


def run_eval(flags):
    """Run test_2.7b.py with given flags, return parsed summary."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
        tmp = f.name
    try:
        state = __import__("torch").load(CKPT, map_location="cpu", weights_only=False)
        __import__("torch").save(state, tmp)

        cmd = [PYTHON, str(ROOT / "test_2.7b.py"), SIZE, tmp, "--json"] + FIXED_FLAGS + flags
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        return parse_summary_json(result.stdout)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def random_config():
    """Generate a random config from sweep space."""
    cfg = []
    for param, values in SWEEP_SPACE.items():
        val = random.choice(values)
        cfg.extend([param, str(val) if not isinstance(val, str) else val])
    return cfg


def main():
    out_dir = ROOT / "artifacts" / "strict_eval_autoresearch_v398"
    os.makedirs(out_dir, exist_ok=True)

    # 60 random configs
    configs = []
    for i in range(60):
        cfg = random_config()
        # Deduplicate: skip if exact same config
        if cfg not in configs:
            configs.append(cfg)

    print(f"Running {len(configs)} inference configs...")
    results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {' '.join(cfg)}")
        summary = run_eval(cfg)
        print(f"  summary keys: {list(summary.keys())[:8]}")

        sys.path.insert(0, str(ROOT))
        from research.strict_eval_search_space import strict_answer_score
        score = strict_answer_score(summary)
        results.append({"config": cfg, "summary": summary, "answer_score": score})

        # Quick save
        with open(out_dir / "partial_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Rank by answer_score
    results.sort(key=lambda x: x["answer_score"], reverse=True)

    print(f"\n{'='*70}")
    print("TOP 10 INFERENCE CONFIGS")
    print(f"{'='*70}")
    for i, r in enumerate(results[:10]):
        cfg_str = " ".join(r["config"])
        print(f"{i+1}. score={r['answer_score']:.2f} | {cfg_str}")

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save top config
    top = results[0]
    with open(out_dir / "top_config.json", "w") as f:
        json.dump({"config": top["config"], "score": top["answer_score"], "summary": top["summary"]}, f, indent=2)

    print(f"\nDone. Top score: {results[0]['answer_score']:.2f}")
    print(f"Baseline v378: 41.11")


if __name__ == "__main__":
    main()
