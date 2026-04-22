"""
v404: Targeted fixes for the two real failure modes in v378+v398.

FAILURE MODE 1 — json_parse false positive:
  Unsupported query (serialize JSON) returns champion code (parse JSON).
  The re-ranker (code_pref) gives the same-family wrong chunk a score boost.
  Fix: lower confidence threshold (so unsupported abstains instead of FP) OR
       add semantic override for known unsupported families.

FAILURE MODE 2 — 5 abstaining families (conf=0.37 vs threshold=0.4):
  debounce, startswith_js, strip_lines, frequency, merge_dicts
  These are 0.03 below threshold. Small calibration shift converts them.
  Fix options:
    A) Lower threshold from 0.4 to 0.35 (risky — might cause other FPs)
    B) Use a different confidence mode for supported queries
    C) Use parafang variants to boost agreement for correctly-supported families

DIAGNOSTIC APPROACH:
  For each candidate config, run full eval and check:
    - Does json_parse unsupported abstain?
    - Do the 5 abstaining families become direct?
    - Do any previously correct cases flip to FP?
"""
from __future__ import annotations
import sys as _sys
_project_root = "/mnt/Storage/Projects/catbelly_studio/ignorance-1"
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import json, os, sys, tempfile, subprocess, torch
from pathlib import Path
from research.strict_eval_search_space import strict_answer_score

ROOT = Path(_project_root)
PY = sys.executable
V378_CKPT = ROOT / "artifacts/strict_eval_autoresearch_v378/v378-late-inter-high-weight-seed511-seed514/model.pt"

# V398 best inference config (baseline)
V398_BASE = [
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


def run_eval(name, extra_args=None):
    """Run eval with v378 checkpoint and given args. Returns (score, summary)."""
    out_dir = ROOT / "artifacts" / "strict_eval_autoresearch_v404"
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as f:
        tmp = f.name
    try:
        torch.save(torch.load(V378_CKPT, map_location="cpu", weights_only=False), tmp)

        args = [PY, str(ROOT / "test_2.7b.py"), "15000000", tmp, "--json"] + V398_BASE
        if extra_args:
            args.extend(extra_args)

        r = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
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
            return None, {"error": r.stdout[:200] + r.stderr[:200]}

        score = strict_answer_score(data)
        results = data.get("objective_results", [])

        # Per-family breakdown
        by_family = {}
        for r2 in results:
            fam = r2.get("family", "?")
            if fam not in by_family:
                by_family[fam] = []
            by_family[fam].append(r2)

        # Key metrics
        direct = sum(1 for r in results if "✅ DIRECT SUPPORT" in r.get("status", ""))
        abstained = sum(1 for r in results if "❌ ABSTAINED" in r.get("status", ""))
        fp = sum(1 for r in results if "❌ FALSE POSITIVE" in r.get("status", ""))
        same_fam = sum(1 for r in results if "❌ SAME-FAMILY" in r.get("status", ""))

        # json_parse unsupported: should abstain
        jp_unsup = [r for r in results if r.get("family") == "json_parse" and "Unsupported" in r.get("type", "")]
        jp_unsup_status = jp_unsup[0].get("status", "?") if jp_unsup else "?"

        # The 5 abstaining families
        abstaining = ["debounce", "startswith_js", "strip_lines", "frequency", "merge_dicts"]
        abst_converted = 0
        for fam in abstaining:
            if fam in by_family:
                for r in by_family[fam]:
                    if "Supported" in r.get("type", "") and "✅ DIRECT SUPPORT" in r.get("status", ""):
                        abst_converted += 1

        summary = {
            "name": name,
            "score": score,
            "direct": direct,
            "abstained": abstained,
            "fp": fp,
            "same_family": same_fam,
            "json_parse_unsup": jp_unsup_status,
            "abst_converted": abst_converted,
            "abstaining_total": sum(1 for r in results if r.get("family") in abstaining and "Supported" in r.get("type", "")),
        }

        print(f"  {name:<50} score={score:>7.2f} direct={direct:>2} abst={abstained:>2} fp={fp:>2} jp_unsup={jp_unsup_status:<25} abst_conv={abst_converted}")

        with open(out_dir / f"{name}.json", "w") as f:
            json.dump({**data, **summary}, f, indent=2)

        return score, summary

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def main():
    os.makedirs(ROOT / "artifacts" / "strict_eval_autoresearch_v404", exist_ok=True)

    print("=" * 80)
    print("v404: Confidence Calibration Fixes")
    print("=" * 80)
    print(f"\nBaseline v378+v398: score=41.11")
    print()

    # Baseline
    _, base = run_eval("baseline-v398", extra_args=["--confidence-threshold", "0.4"])

    # ── Fix 1: Lower threshold to capture the 5 near-threshold cases ──────────
    # conf=0.37 vs threshold=0.4, margin=0.03
    print("\n--- Fix 1: Threshold sweep ---")
    results = [("baseline-v398", base)]

    for thresh in [0.39, 0.38, 0.37, 0.36, 0.35, 0.30]:
        _, r = run_eval(f"thresh-{thresh}", extra_args=["--confidence-threshold", str(thresh)])
        results.append((f"thresh-{thresh}", r))

    # ── Fix 2: confidence mode alternatives ─────────────────────────────────────
    print("\n--- Fix 2: Confidence mode alternatives ---")
    for mode in ["model_head", "query_head", "agreement_augmented"]:
        if mode == "agreement_augmented":
            extra = ["--confidence-parafence-variants", "3"]
        else:
            extra = []
        _, r = run_eval(f"conf-{mode}", extra_args=["--confidence-mode", mode] + extra)
        results.append((f"conf-{mode}", r))

    # ── Fix 3: Combined fixes ───────────────────────────────────────────────────
    print("\n--- Fix 3: Combined ---")
    for thresh in [0.37, 0.35]:
        for mode in ["model_head", "support_feature_calibrator"]:
            if mode == "agreement_augmented":
                extra = ["--confidence-parafence-variants", "3"]
            else:
                extra = []
            _, r = run_eval(f"combined-t{thresh}-{mode}", extra_args=["--confidence-threshold", str(thresh), "--confidence-mode", mode] + extra)
            results.append((f"combined-t{thresh}-{mode}", r))

    # ── Fix 4: Try answerspec disable (may reduce json_parse FP) ───────────────
    print("\n--- Fix 4: Disable answerspec ---")
    for mode in ["none", "soft"]:
        for t in ["0.37", "0.35"]:
            _, r = run_eval(f"noanswerspec-t{t}-{mode}", extra_args=[
                "--rerank-answerspec-mode", mode,
                "--confidence-threshold", t,
            ])
            results.append((f"noanswerspec-t{t}-{mode}", r))

    # ── Fix 5: parafang-only (no consensus/reranker) ──────────────────────────
    print("\n--- Fix 5: Minimal reranker ---")
    minimal = [
        "--retrieval-facet-score-mode", "maxsim",
        "--lexical-weight", "0.4",
        "--confidence-threshold", "0.35",
        "--rerank-topk", "1",
        "--rerank-parafence-weight", "0.0",
        "--rerank-consensus-weight", "0.0",
        "--selective-gate-mode", "none",
    ]
    _, r = run_eval("minimal", extra_args=minimal)
    results.append(("minimal", r))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS — sorted by score")
    print("=" * 80)
    results.sort(key=lambda x: x[1].get("score", 0) or 0, reverse=True)

    print(f"\n{'Candidate':<45} {'Score':>7} {'Direct':>7} {'Abst':>5} {'FP':>4} {'JP-Unsup':>25} {'AbstConv':>9}")
    print(f"{'-'*45} {'-'*7} {'-'*7} {'-'*5} {'-'*4} {'-'*25} {'-'*9}")
    for name, r in results:
        print(f"{name:<45} {r.get('score', 0):>7.2f} {r.get('direct', 0):>7} {r.get('abstained', 0):>5} {r.get('fp', 0):>4} {r.get('json_parse_unsup', '?'):>25} {r.get('abst_converted', 0):>9}")

    print(f"\nBaseline v378+v398: score=41.11, direct=3, abstained=5, fp=0, jp_unsup=❌ FALSE POSITIVE")
    print(f"Best candidate: {results[0][0]} with score={results[0][1].get('score', 0):.2f}")

    # Save summary
    out_dir = ROOT / "artifacts" / "strict_eval_autoresearch_v404"
    with open(out_dir / "batch_summary.json", "w") as f:
        json.dump([{"name": n, **r} for n, r in results], f, indent=2)


if __name__ == "__main__":
    main()
