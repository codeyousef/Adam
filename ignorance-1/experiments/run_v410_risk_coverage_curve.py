"""
v410: Risk-coverage curve for supported and unsupported separately.
Evaluates v378 at many confidence thresholds to get the full tradeoff curve.

This is Evaluation Guardrail 1 from research9: "stop relying on the single final score."
The memo says: "If a confidence redesign is real, you should see improved coverage
at the same risk, not just one threshold that happens to look better."
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/mnt/Storage/Projects/catbelly_studio/ignorance-1")

import json, tempfile, subprocess, torch, os
from research.strict_eval_search_space import strict_answer_score

PY = sys.executable
CKPT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"
OUT = "/mnt/Storage/Projects/catbelly_studio/ignorance-1/artifacts/strict_eval_autoresearch_v410"

os.makedirs(OUT, exist_ok=True)

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
    tmp = f.name
torch.save(torch.load(CKPT, map_location="cpu", weights_only=False), tmp)

BASE_ARGS = [
    "--retrieval-facet-score-mode", "maxsim",
    "--lexical-weight", "0.4",
    "--rerank-consensus-temperature", "0.05",
    "--rerank-agreement-weight", "0.3",
    "--selective-gate-similarity-floor", "0.0",  # DISABLED for pure threshold sweep
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

# Sweep confidence threshold from 0.0 to 0.95 in steps of 0.05
thresholds = [round(x * 0.05, 2) for x in range(21)]  # 0.0 to 1.0
all_results = []

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
        print(f"  thresh={thresh:.2f}: ERROR"); continue

    score = strict_answer_score(data)
    obj = data.get("objective_results", [])

    # Per-case breakdown
    direct_s = [r for r in obj if "✅ DIRECT SUPPORT" in r.get("status","") and "unsupported" not in r.get("type","").lower()]
    direct_u = [r for r in obj if "✅ DIRECT SUPPORT" in r.get("status","") and "unsupported" in r.get("type","").lower()]
    abstain_s = [r for r in obj if "❌ ABSTAINED" in r.get("status","") and "unsupported" not in r.get("type","").lower()]
    abstain_u = [r for r in obj if "❌ ABSTAINED" in r.get("status","") and "unsupported" in r.get("type","").lower()]
    fp_s = [r for r in obj if "❌ FALSE POSITIVE" in r.get("status","") and "unsupported" not in r.get("type","").lower()]
    fp_u = [r for r in obj if "❌ FALSE POSITIVE" in r.get("status","") and "unsupported" in r.get("type","").lower()]
    sf_s = [r for r in obj if "SAME-FAMILY" in r.get("status","") and "unsupported" not in r.get("type","").lower()]
    sf_u = [r for r in obj if "SAME-FAMILY" in r.get("status","") and "unsupported" in r.get("type","").lower()]
    ci_u = [r for r in obj if "CORRECTLY IGNORANT" in r.get("status","")]

    n_supported = 8
    n_unsupported = 8
    n_correct_supported = len(direct_s)
    n_correct_unsupported = len(direct_u) + len(ci_u)
    n_abstain_supported = len(abstain_s)
    n_abstain_unsupported = len(abstain_u)
    n_fp_supported = len(fp_s)
    n_fp_unsupported = len(fp_u)
    n_sf_supported = len(sf_s)
    n_sf_unsupported = len(sf_u)

    # Coverage: fraction of supported queries that get answered
    coverage_supported = n_correct_supported / n_supported
    # Risk: fraction of answers that are wrong (FP or same-family)
    n_answered = n_correct_supported + n_fp_supported + n_sf_supported
    risk_supported = (n_fp_supported + n_sf_supported) / n_answered if n_answered > 0 else 0.0

    # Coverage for unsupported: fraction correctly abstained or correctly answered
    coverage_unsupported = n_correct_unsupported / n_unsupported
    # Risk for unsupported: fraction of answers that are wrong
    n_answered_u = n_fp_unsupported + n_sf_unsupported
    risk_unsupported = n_answered_u / n_answered_u if n_answered_u > 0 else 0.0

    result = {
        "threshold": thresh,
        "score": score,
        "direct_s": n_correct_supported, "direct_u": n_correct_unsupported,
        "ci_u": len(ci_u),
        "abstain_s": n_abstain_supported, "abstain_u": n_abstain_unsupported,
        "fp_s": n_fp_supported, "fp_u": n_fp_unsupported,
        "sf_s": n_sf_supported, "sf_u": n_sf_unsupported,
        "coverage_supported": coverage_supported,
        "risk_supported": risk_supported,
        "coverage_unsupported": coverage_unsupported,
        "risk_unsupported": risk_unsupported,
        "n_answered": n_answered,
    }
    all_results.append(result)

    print(f"  thresh={thresh:.2f}: score={score:.2f} D_s={n_correct_supported} D_u={n_correct_unsupported} "
          f"A_s={n_abstain_supported} A_u={n_abstain_unsupported} "
          f"FP_s={n_fp_supported} FP_u={n_fp_unsupported} "
          f"SF_s={n_sf_supported} SF_u={n_sf_unsupported} "
          f"| cov_s={coverage_supported:.2f} risk_s={risk_supported:.2f}")

import os as _os; _os.unlink(tmp)

# Save
with open(f"{OUT}/risk_coverage_curve.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved to {OUT}/risk_coverage_curve.json")

# Print key thresholds
print("\n=== KEY THRESHOLDS ===")
print(f"{'Thresh':>7} {'Score':>7} {'Cov_S':>7} {'Risk_S':>7} {'Cov_U':>7} {'Risk_U':>7} {'D_S':>5} {'D_U':>5} {'FP_U':>5} {'SF_S':>5} {'Abst_S':>7}")
for r in sorted(all_results, key=lambda x: x["score"], reverse=True)[:10]:
    print(f"{r['threshold']:>7.2f} {r['score']:>7.2f} {r['coverage_supported']:>7.2f} {r['risk_supported']:>7.2f} "
          f"{r['coverage_unsupported']:>7.2f} {r['risk_unsupported']:>7.2f} "
          f"{r['direct_s']:>5} {r['direct_u']:>5} {r['fp_u']:>5} {r['sf_s']:>5} {r['abstain_s']:>7}")

# Find the "knee" — where coverage_supported is highest while risk_supported is 0
print("\n=== ZERO-RISK THRESHOLDS (risk_supported=0) ===")
zero_risk = [r for r in all_results if r["risk_supported"] == 0 and r["n_answered"] > 0]
for r in sorted(zero_risk, key=lambda x: x["score"], reverse=True):
    print(f"  thresh={r['threshold']:.2f}: score={r['score']:.2f}, cov_s={r['coverage_supported']:.2f}, D_s={r['direct_s']}, Abst_s={r['abstain_s']}")
