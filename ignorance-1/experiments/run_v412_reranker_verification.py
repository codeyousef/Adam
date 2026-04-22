"""
v412: Verify reranker behavior with gate completely disabled.
Run selective_prediction directly with gate off and thresh=0.0 to see reranker output.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/mnt/Storage/Projects/catbelly_studio/ignorance-1")

import json, tempfile, subprocess, torch, os
from research.strict_eval_search_space import strict_answer_score

PY = sys.executable
CKPT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
    tmp = f.name
torch.save(torch.load(CKPT, map_location="cpu", weights_only=False), tmp)

# Run with: gate DISABLED (similarity_floor=0.0, mode=none), thresh=0.0
# This should invoke reranker for ALL cases
args = [PY, "/mnt/Storage/Projects/catbelly_studio/ignorance-1/test_2.7b.py", "15000000", tmp, "--json",
    "--retrieval-facet-score-mode", "maxsim",
    "--confidence-threshold", "0.0",
    "--lexical-weight", "0.4",
    "--rerank-consensus-temperature", "0.05",
    "--rerank-agreement-weight", "0.3",
    "--selective-gate-similarity-floor", "0.0",
    "--selective-gate-mode", "none",
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

print(f"v412: Gate=OFF, Thresh=0.0 — Score={score:.2f}")
print()
print(f"{'Family':<15} {'Type':<12} {'Conf':>7} {'Sim':>7} {'Rerank_Score':>13} {'Champion_Fam':>12} {'Status':<35}")
print("-" * 115)

for r2 in sorted(results, key=lambda x: (x.get("family",""), x.get("type",""))):
    fam = r2.get("family","?").replace("Objective - ","")
    typ = r2.get("type","?").replace("Objective - ","")
    conf = r2.get("confidence", 0)
    sim = r2.get("similarity", 0)
    rerank = r2.get("score", 0)
    champ_fam = r2.get("champion_family", "?")
    status = r2.get("status","?")

    # What SHOULD the outcome be?
    if champ_fam == fam:
        if "DIRECT" in status:
            outcome = "✅ DIRECT SUPPORT"
        elif "SAME-FAMILY" in status:
            outcome = "❌ SAME-FAMILY WRONG"
        else:
            outcome = status[:35]
    else:
        if "FALSE POSITIVE" in status:
            outcome = "❌ FALSE POSITIVE"
        elif "CORRECTLY IGNORANT" in status:
            outcome = "✅ CORRECTLY IGNORANT"
        else:
            outcome = status[:35]

    print(f"{fam:<15} {typ:<12} conf={conf:.4f} sim={sim:.4f} rerank={rerank:.4f} champ_fam={champ_fam:<12} {outcome}")

print()
# Summary
direct = sum(1 for r2 in results if "✅ DIRECT SUPPORT" in r2.get("status",""))
fp = sum(1 for r2 in results if "❌ FALSE POSITIVE" in r2.get("status",""))
sf = sum(1 for r2 in results if "SAME-FAMILY" in r2.get("status",""))
ci = sum(1 for r2 in results if "CORRECTLY IGNORANT" in r2.get("status",""))
abstain = sum(1 for r2 in results if "❌ ABSTAINED" in r2.get("status",""))

print(f"Summary: DIRECT={direct}, FP={fp}, SAME_FAM={sf}, CORRECTLY_IGN={ci}, ABSTAIN={abstain}")
print(f"Score: {score:.2f}")
