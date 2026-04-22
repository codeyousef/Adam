"""
v406: Diagnostic — what does the reranker actually produce for hard families?

We need to understand: does the reranker produce non-zero maxsim for the 5 abstaining families?
And what's the exact selective gate computation?
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/mnt/Storage/Projects/catbelly_studio/ignorance-1")

import json, torch, tempfile, subprocess
from research.strict_eval_search_space import strict_answer_score

PY = sys.executable
CKPT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"

# Run eval with selective gate DISABLED to see raw reranker output
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
    tmp = f.name
torch.save(torch.load(CKPT, map_location="cpu", weights_only=False), tmp)

# Run with selective gate DISABLED and very low threshold
r = subprocess.run([PY, "/mnt/Storage/Projects/catbelly_studio/ignorance-1/test_2.7b.py", "15000000", tmp, "--json",
    "--retrieval-facet-score-mode", "maxsim",
    "--confidence-threshold", "0.0",  # No confidence threshold
    "--lexical-weight", "0.4",
    "--rerank-consensus-temperature", "0.05",
    "--rerank-agreement-weight", "0.3",
    "--selective-gate-similarity-floor", "0.0",  # DISABLED
    "--selective-gate-mode", "none",  # DISABLED
    "--selective-gate-margin-threshold", "0.0",
    "--selective-gate-mean-gap-threshold", "0.0",
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
], capture_output=True, text=True, timeout=300)

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

import os; os.unlink(tmp)

if not data:
    print("ERROR: no JSON output")
    print(r.stdout[:500])
    print(r.stderr[:500])
    sys.exit(1)

results = data.get("objective_results", [])
score = strict_answer_score(data)
print(f"Score (gate=disabled, thresh=0.0): {score:.2f}")
print()
print(f"{'Family':<15} {'Type':<12} {'Conf':>7} {'Top1':>7} {'Margin':>7} {'MeanK':>7} {'Sim':>7} {'Status':<30}")
print("-" * 100)

for r2 in sorted(results, key=lambda x: x.get("family","")):
    fam = r2.get("family","?")
    typ = r2.get("type","?").replace("Objective - ","")
    conf = r2.get("confidence", 0)
    det = r2.get("confidence_details", {})
    top1 = det.get("top1", 0)
    margin = det.get("margin", 0)
    mean_topk = det.get("mean_topk", 0)
    sim = r2.get("similarity", 0)
    status = r2.get("status","?")
    print(f"{fam:<15} {typ:<12} {conf:>7.4f} {top1:>7.4f} {margin:>7.4f} {mean_topk:>7.4f} {sim:>7.4f} {status:<30}")
