from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_run_test():
    module_path = ROOT / "test_2.7b.py"
    spec = importlib.util.spec_from_file_location("test_2_7b_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load verifier module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_test


def main() -> int:
    run_test = _load_run_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths to diagnose")
    parser.add_argument("--size", type=int, default=2700000000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--lexical-weight", type=float, default=0.7)
    args = parser.parse_args()

    reports = []
    for checkpoint in args.checkpoints:
        summary = run_test(
            args.size,
            checkpoint,
            force_cpu=args.cpu,
            confidence_threshold=args.confidence_threshold,
            lexical_weight=args.lexical_weight,
            verbose=False,
        )
        reports.append(
            {
                "checkpoint": checkpoint,
                "legacy_status": summary["legacy_status"],
                "strict_status": summary["strict_status"],
                "strict_failures": summary["strict_failures"],
                "avg_known_similarity": summary["avg_known_similarity"],
                "avg_known_paraphrase_similarity": summary["avg_known_paraphrase_similarity"],
                "avg_ignorant_similarity": summary["avg_ignorant_similarity"],
                "ignorance_gap": summary["ignorance_gap"],
                "avg_known_margin": summary["avg_known_margin"],
                "code_avg_offdiag_similarity": summary["code_diagnostics"]["avg_offdiag_similarity"],
                "code_rank_fraction": summary["code_diagnostics"]["participation_ratio_fraction"],
                "query_avg_offdiag_similarity": summary["query_diagnostics"]["avg_offdiag_similarity"],
                "query_rank_fraction": summary["query_diagnostics"]["participation_ratio_fraction"],
            }
        )

    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())