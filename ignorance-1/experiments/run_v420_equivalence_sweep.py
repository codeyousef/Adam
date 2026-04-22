#!/usr/bin/env python3
"""
v420 — Equivalence Alignment Sweep

Based on v413's successful training run (model.pt exists, training completed).
This script evaluates v413 AND tests whether varying equivalence_alignment_weight
can push json_parse-u and the hard families across the gate.

Strategy:
  - Use v413's trained model (equivalence_alignment_weight=0.5) as baseline
  - Run strict eval on v413
  - Compare against v378 baseline

If v413 > v378: try higher equivalence weight (0.7, 1.0)
If v413 ≤ v378: the equivalence approach isn't working, pivot
"""
from __future__ import annotations
import subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / ".venv" / "bin" / "python"

V413_MODEL = ROOT / "artifacts" / "strict_eval_autoresearch_v413" / "v413-equivalence-cc-seed700" / "model.pt"
V378_MODEL = ROOT / "artifacts" / "strict_eval_autoresearch_v378" / "v378-late-inter-high-weight-seed511-seed514" / "model.pt"

MODELS = {
    "v413-equivalence-cc": V413_MODEL,
    "v378-baseline": V378_MODEL,
}

for name, path in MODELS.items():
    if not path.exists():
        print(f"SKIP {name}: {path} not found")
        continue
    print(f"\n=== Evaluating {name} ===")
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "test_2.7b.py"), "--model", str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout[-2000:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:] if result.stderr else "")
