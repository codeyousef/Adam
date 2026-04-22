"""
v409b: Selective gate parameter sweep.

Key finding: threshold doesn't matter for abstentions (they come from gate).
Need to find if there's a gate config that lets hard families through
without opening FPs.

Gate parameters to sweep:
  - mean_gap_threshold: 0.010, 0.012, 0.014, 0.016, 0.018, 0.020
  - margin_threshold: 0.005, 0.008, 0.010, 0.012, 0.015
  - similarity_floor: 0.5, 0.55, 0.6, 0.65, 0.7

Fixed: threshold=0.38 (optimal from v398)
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/mnt/Storage/Projects/catbelly_studio/ignorance-1")

import json, tempfile, subprocess, torch
from research.strict_eval_search_space import strict_answer_score
from itertools import product

PY = sys.executable
CKPT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
    tmp = f.name
torch.save(torch.load(CKPT, map_location="cpu", weights_only=False), tmp)

BASE_ARGS = [
    "--retrieval-facet-score-mode", "maxsim",
    "--confidence-threshold", "0.38",
    "--lexical-weight", "0.4",
    "--rerank-consensus-temperature", "0.05",
    "--rerank-agreement-weight", "0.3",
    "--rerank-topk", "5", "--rerank-shortlist-mode", "pred_query_union_local",
    "--rerank-query-weight", "0.3", "--rerank-lexical-weight", "0.0",
    "--rerank-support-weight", "0.24", "--rerank-consensus-weight", "0.35",
    "--rerank-consensus-floor", "0.9158", "--rerank-consensus-margin-gate", "0.0092",
    "--rerank-pairwise-mode", "supportspec_citecheck_floor_borda",
    "--rerank-support-floor-margin-gate", "0.014", "--rerank-spec-weight", "0.18",
    "--rerank-answerspec-mode", "code_pref", "--rerank-answerspec-margin-gate", "0.034",
    "--rerank-safe-expand-topk", "6", "--rerank-safe-expand-margin", "0.004",
    "--rerank-parafence-weight", "1.0", "--rerank-parafence-variants", "3",
    "--rerank-verifier-uplift-weight", "0.4", "--rerank-verifier-gap-scale", "1.0",
    "--rerank-verifier-support-weight", "1.0", "--rerank-verifier-spec-weight", "0.0",
    "--retrieval-facet-softmax-temperature", "0.1", "--retrieval-global-facet-blend", "0.35",
    "--confidence-mode", "support_feature_calibrator",
    "--confidence-support-topk", "5", "--confidence-support-temperature", "0.1",
    "--selective-gate-mode", "margin_mean_gap",
]

# Sweep: mean_gap_threshold × margin_threshold × similarity_floor
gate_params = [
    # (mean_gap, margin, sim_floor)
    (0.008, 0.005, 0.55),
    (0.008, 0.005, 0.50),
    (0.010, 0.005, 0.55),
    (0.010, 0.008, 0.55),
    (0.010, 0.010, 0.55),
    (0.012, 0.008, 0.55),
    (0.012, 0.010, 0.55),
    (0.014, 0.008, 0.55),
    (0.014, 0.010, 0.55),
    (0.016, 0.010, 0.55),
    (0.016, 0.012, 0.55),
    (0.018, 0.010, 0.55),
    (0.020, 0.010, 0.55),
    # Also try with similarity_floor 0.50
    (0.010, 0.008, 0.50),
    (0.012, 0.008, 0.50),
    (0.014, 0.008, 0.50),
    (0.016, 0.010, 0.50),
    (0.018, 0.010, 0.50),
    # Lower sim_floor to 0.45
    (0.010, 0.008, 0.45),
    (0.012, 0.008, 0.45),
    (0.014, 0.008, 0.45),
    (0.016, 0.008, 0.45),
    (0.018, 0.010, 0.45),
]

all_results = []

for mean_gap, margin, sim_floor in gate_params:
    args = [PY, "/mnt/Storage/Projects/catbelly_studio/ignorance-1/test_2.7b.py", "15000000", tmp, "--json",
            "--selective-gate-mean-gap-threshold", str(mean_gap),
            "--selective-gate-margin-threshold", str(margin),
            "--selective-gate-similarity-floor", str(sim_floor)] + BASE_ARGS

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
        print(f"  mg={mean_gap}, m={margin}, sf={sim_floor}: ERROR")
        continue

    score = strict_answer_score(data)
    obj = data.get("objective_results", [])
    direct = sum(1 for r2 in obj if "✅ DIRECT SUPPORT" in r2.get("status",""))
    abstained = sum(1 for r2 in obj if "❌ ABSTAINED" in r2.get("status",""))
    fp = sum(1 for r2 in obj if "❌ FALSE POSITIVE" in r2.get("status",""))

    # Per-case details
    confs = {}
    for r2 in obj:
        fam = r2.get("family","?")
        typ = r2.get("type","?").replace("Objective - ","")
        confs[f"{fam}_{typ}"] = {"conf": r2.get("confidence",0), "sim": r2.get("similarity",0), "status": r2.get("status","")}

    result = {
        "mean_gap": mean_gap, "margin": margin, "sim_floor": sim_floor,
        "score": score, "direct": direct, "abstained": abstained, "fp": fp,
        "confs": confs,
    }
    all_results.append(result)
    print(f"  mg={mean_gap}, m={margin}, sf={sim_floor}: score={score:.2f}, D={direct}, A={abstained}, FP={fp}")

import os; os.unlink(tmp)

# Find best
best = max(all_results, key=lambda x: x["score"])
print(f"\nBEST: mg={best['mean_gap']}, m={best['margin']}, sf={best['sim_floor']} → score={best['score']:.2f}")

# Save
import os
os.makedirs("/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v409", exist_ok=True)
with open("/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v409/gate_sweep.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Print top 5
print("\n=== TOP 5 ===")
for r2 in sorted(all_results, key=lambda x: x["score"], reverse=True)[:5]:
    print(f"  mg={r2['mean_gap']}, m={r2['margin']}, sf={r2['sim_floor']}: score={r2['score']:.2f}, D={r2['direct']}, A={r2['abstained']}, FP={r2['fp']}")
