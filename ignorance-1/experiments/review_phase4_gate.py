from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase4 import evaluate_phase4_gate


def _load_artifact(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _review_artifact(path: Path) -> dict:
    artifact = _load_artifact(path)
    legacy_gate = evaluate_phase4_gate(artifact, gate_version="legacy_v1")
    relative_gate = evaluate_phase4_gate(artifact, gate_version="relative_v2")
    return {
        "artifact": str(path),
        "best_size": artifact["best_size"],
        "largest_wins": artifact["largest_wins"],
        "largest_margin_ratio": artifact["largest_margin_ratio"],
        "pairwise_win_rate": artifact["pairwise_win_rate"],
        "worst_pairwise_margin_ratio": artifact["worst_pairwise_margin_ratio"],
        "legacy_gate": legacy_gate,
        "recalibrated_gate": relative_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", help="Phase 4 result JSON files to review")
    args = parser.parse_args()

    reviews = [_review_artifact(Path(artifact)) for artifact in args.artifacts]
    print(json.dumps(reviews, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())