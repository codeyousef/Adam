from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.phase4_evidence import load_evidence


def main() -> int:
    grouped = defaultdict(list)
    for row in load_evidence():
        grouped[row.get("candidate_name", "")].append(row)
    summary = []
    for name, rows in sorted(grouped.items()):
        summary.append(
            {
                "candidate_name": name,
                "events": len(rows),
                "stages": sorted({row.get("stage", "") for row in rows}),
                "latest_decision": rows[-1].get("decision"),
                "latest_answer_score": rows[-1].get("answer_score"),
            }
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
