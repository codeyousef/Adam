"""
v409: Threshold sweep to find optimal confidence threshold.

Key insight from v406 diagnostic:
- With gate DISABLED: score=28.26 (8 FPs, 3 correct, 5 same-family wrong)
- With gate ENABLED (v398 baseline, thresh=0.4): score=41.64 (1 FP, 5 abstentions, 3 correct)

The 5 abstentions have conf=0.37 (just below 0.4).
If we lower threshold to 0.35-0.38, do we get more correct answers or more FPs?

Sweep thresholds: 0.30, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/mnt/Storage/Projects/catbelly_studio/ignorance-1")

import json, tempfile, subprocess, torch
from research.strict_eval_search_space import strict_answer_score
from collections import defaultdict

PY = sys.executable
CKPT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
    tmp = f.name
torch.save(torch.load(CKPT, map_location="cpu", weights_only=False), tmp)

BASE_ARGS = [
    "--retrieval-facet-score-mode", "maxsim",
    "--lexical-weight", "0.4",
    "--rerank-consensus-temperature", "0.05",
    "--rerank-agreement-weight", "0.3",
    "--selective-gate-similarity-floor", "0.6",
    "--rerank-topk", "5", "--rerank-shortlist-mode", "pred_query_union_local",
    "--rerank-query-weight", "0.3", "--rerank-lexical-weight", "0.0",
    "--rerank-support-weight", "0.24", "--rerank-consensus-weight", "0.35",
    "--rerank-consensus-floor", "0.9158", "--rerank-consensus-margin-gate", "0.0092",
    "--rerank-pairwise-mode", "supportspec_citecheck_floor_borda",
    "--rerank-support-floor-margin-gate", "0.014", "--rerank-spec-weight", "0.18",
    "--rerank-answerspec-mode", "code_pref", "--rerank-answerspec-margin-gate", "0.034",
    "--rerank-safe-expand-topk", "6", "--rerank-safe-expand-margin", "0.004",
    "--rerank-parafence-weight", "1.0", "--rerank-parafence-variants", "3",
    "--selective-gate-mode", "margin_mean_gap",
    "--selective-gate-margin-threshold", "0.01", "--selective-gate-mean-gap-threshold", "0.016",
    "--rerank-verifier-uplift-weight", "0.4", "--rerank-verifier-gap-scale", "1.0",
    "--rerank-verifier-support-weight", "1.0", "--rerank-verifier-spec-weight", "0.0",
    "--retrieval-facet-softmax-temperature", "0.1", "--retrieval-global-facet-blend", "0.35",
    "--confidence-mode", "support_feature_calibrator",
    "--confidence-support-topk", "5", "--confidence-support-temperature", "0.1",
]

thresholds = [0.30, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
results = []

for thresh in thresholds:
    args = [PY, "/mnt/Storage/Projects/catbelly_studio/ignorance-1/test_2.7b.py", "15000000", tmp, "--json",
            "--confidence-threshold", str(thresh)] + BASE_ARGS

    r = subprocess.run(args, capture_output=True, text=True, timeout=300)
    data = None
    for idx in range(len(r.stdout)):
        if r.stdout[idx] != "{": continue
        for end in range(idx+20, min(idx+100000, len(r.stdout)+1)):
            try:
                d = json.loads(r.stdout[idx:end])
                if isinstance(d, dict) and len(d) > 10:
                    data = d; break
            except: pass
        if data: break

    if not data:
        print(f"  thresh={thresh}: ERROR")
        continue

    score = strict_answer_score(data)
    obj_results = data.get("objective_results", [])

    direct = sum(1 for r2 in obj_results if "✅ DIRECT SUPPORT" in r2.get("status", ""))
    abstained = sum(1 for r2 in obj_results if "❌ ABSTAINED" in r2.get("status", ""))
    fp = sum(1 for r2 in obj_results if "❌ FALSE POSITIVE" in r2.get("status", ""))
    same_fam = sum(1 for r2 in obj_results if "SAME-FAMILY" in r2.get("status", ""))

    results.append({
        "threshold": thresh,
        "score": score,
        "direct": direct,
        "abstained": abstained,
        "fp": fp,
        "same_family_wrong": same_fam,
        "score": score,
    })

    print(f"  thresh={thresh:.2f}: score={score:.2f}, direct={direct}, abstained={abstained}, fp={fp}, same_fam={same_fam}")

import os; os.unlink(tmp)

print("\n=== SUMMARY ===")
print(f"{'Thresh':>7} {'Score':>7} {'Direct':>7} {'Abst':>6} {'FP':>4} {'SameFam':>8}")
print("-" * 42)
for r2 in sorted(results, key=lambda x: x["score"], reverse=True):
    print(f"{r2['threshold']:>7.2f} {r2['score']:>7.2f} {r2['direct']:>7} {r2['abstained']:>6} {r2['fp']:>4} {r2['same_family_wrong']:>8}")

# Save
import os
os.makedirs("/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v409", exist_ok=True)
with open("/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v409/threshold_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
