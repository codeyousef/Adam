"""
v411: Family-local ranking diagnostic (Eval Guardrail 2 from research9).
For every hard family, track: is champion absent from top-k, present but below sibling, or ranked first but rejected by confidence?
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/mnt/Storage/Projects/catbelly_studio/ignorance-1")

import json, tempfile, subprocess, torch, os
from research.strict_eval_search_space import strict_answer_score

PY = sys.executable
CKPT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
OUT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v411"

os.makedirs(OUT, exist_ok=True)

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
    tmp = f.name
torch.save(torch.load(CKPT, map_location="cpu", weights_only=False), tmp)

# Run with gate disabled (similarity_floor=0.0) and thresh=0.0 so the reranker fires for ALL cases
args = [PY, "/mnt/Storage/Projects/catbelly_studio/ignorance-1/test_2.7b.py", "15000000", tmp, "--json",
    "--retrieval-facet-score-mode", "maxsim",
    "--confidence-threshold", "0.0",
    "--lexical-weight", "0.4",
    "--rerank-consensus-temperature", "0.05",
    "--rerank-agreement-weight", "0.3",
    "--selective-gate-similarity-floor", "0.0",  # DISABLED
    "--selective-gate-mode", "none",
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
]

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
os.unlink(tmp)

if not data:
    print("ERROR"); sys.exit(1)

score = strict_answer_score(data)
results = data.get("objective_results", [])

print(f"v411 Family-Local Ranking Diagnostic")
print(f"Score (gate=off, thresh=0.0): {score:.2f}")
print()
print(f"{'Family':<15} {'Type':<12} {'Conf':>7} {'Sim':>7} {'Top1':>7} {'Rerank_Score':>13} {'Status':<40}")
print("-" * 110)

# Also run with gate ENABLED (sim_floor=0.6) to compare
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
    tmp2 = f.name
torch.save(torch.load(CKPT, map_location="cpu", weights_only=False), tmp2)
args2 = [PY, "/mnt/Storage/Projects/catbelly_studio/ignorance-1/test_2.7b.py", "15000000", tmp2, "--json",
    "--retrieval-facet-score-mode", "maxsim",
    "--confidence-threshold", "0.38",
    "--lexical-weight", "0.4",
    "--rerank-consensus-temperature", "0.05",
    "--rerank-agreement-weight", "0.3",
    "--selective-gate-similarity-floor", "0.6",
    "--selective-gate-mode", "margin_mean_gap",
    "--selective-gate-margin-threshold", "0.01",
    "--selective-gate-mean-gap-threshold", "0.016",
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
]
r2 = subprocess.run(args2, capture_output=True, text=True, timeout=300)
data2 = None
for idx in range(len(r2.stdout)):
    if r2.stdout[idx] != "{": continue
    for end in range(idx+20, min(idx+100000, len(r2.stdout)+1)):
        try:
            d = json.loads(r2.stdout[idx:end])
            if isinstance(d, dict) and len(d) > 10:
                data2 = d; break
        except: pass
    if data2: break
os.unlink(tmp2)

if data2:
    results2 = data2.get("objective_results", [])
    confs_gate_on = {r2.get("family","?")+"_"+r2.get("type","?").replace("Objective - ",""): r2.get("confidence",0) for r2 in results2}
else:
    confs_gate_on = {}

for r2 in sorted(results, key=lambda x: (x.get("family",""), x.get("type",""))):
    fam = r2.get("family","?").replace("Objective - ","")
    typ = r2.get("type","?").replace("Objective - ","")
    conf = r2.get("confidence", 0)
    sim = r2.get("similarity", 0)
    det = r2.get("confidence_details", {})
    top1 = det.get("top1", 0)
    margin = det.get("margin", 0)
    rerank_score = r2.get("score", 0)
    status = r2.get("status","?")

    # Gate-on conf
    key = fam + "_" + typ
    conf_on = confs_gate_on.get(key, conf)

    # Diagnosis
    if "DIRECT" in status:
        diag = "✅ RERANKER CORRECT (champion passed)"
    elif "SAME-FAMILY" in status:
        if sim < 0.5:
            diag = "❌ CANDIDATE GEN FAIL (champion not in top-k)"
        else:
            diag = "❌ LOCAL ORDERING FAIL (champion in top-k but ranked below sibling)"
    elif "FALSE POSITIVE" in status:
        diag = "❌ FALSE POSITIVE (reranker retrieved wrong champion)"
    elif "ABSTAINED" in status:
        if conf_on >= 0.38:
            diag = f"⚠️  CONFIDENCE THRESHOLD (conf={conf_on:.4f} >= 0.38 but < 0.4)"
        elif sim < 0.5:
            diag = "❌ CANDIDATE GEN FAIL (champion not in top-k)"
        else:
            diag = "❌ LOCAL ORDERING FAIL (champion retrieved but conf too low)"
    elif "CORRECTLY IGNORANT" in status:
        diag = "✅ CORRECTLY IGNORANT"
    else:
        diag = status[:40]

    print(f"{fam:<15} {typ:<12} conf={conf:.4f} sim={sim:.4f} top1={top1:.4f} | conf_on={conf_on:.4f} score={rerank_score:.4f}")
    print(f"  → {diag}")

print()
print("=== DIAGNOSIS SUMMARY ===")
print("CANDIDATE GEN FAIL: champion not in top-k retrieval set (encoder embedding issue)")
print("LOCAL ORDERING FAIL: champion retrieved but ranked below sibling (ranking issue)")
print("CONFIDENCE THRESHOLD: champion retrieved & ranked well, but conf < threshold")
print("RERANKER CORRECT: reranker correctly ranked champion first")
