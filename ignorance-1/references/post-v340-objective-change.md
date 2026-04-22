# Post-v340 Objective Change — Critical Research Findings

**Generated:** 2026-04-19 (from v340→v372→v373 experiential analysis)

---

## The Core Fact

v340 (score=43.5) PASSED strict eval BEFORE the objective was changed. All versions
from v365 through v373+ are evaluated against a HARDER post-v340 objective.

**This is the single most important fact for the current research direction.**

---

## Current Post-v340 Strict Eval Criteria

(from `test_2.7b.py` `_strict_status()`)

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| `direct_rate` | >= 0.75 | Fraction of supported queries retrieving exact correct code |
| `wrong_chunk_rate` | <= 0.20 | Fraction retrieving wrong-but-similar variants |
| `abstention_rate` (unsupported) | >= 0.75 | Must return `<IGNORANT>` |
| `confidence_gap` | >= 0.20 | avg supported conf − avg unsupported conf |
| `participation_ratio` | >= 0.10 | query+code embedding effective rank |
| `known_confidence` | >= 0.60 | |
| `ood_confidence` | <= 0.35 | |

---

## Empirical Findings: The Direct Rate Ceiling

All versions from v365 through v373 achieve `direct_rate ≈ 0.375` (3 of 8 supported items):

| Version | Score | direct_rate | Notes |
|---------|-------|-------------|-------|
| v365 | 42.7 | ~0.375 | |
| v368 | 41.7 | ~0.375 | |
| v369 | 43.3 | ~0.375 | |
| v370 | 44.0 | ~0.375 | PASSED on other criteria |
| v371 | 39.73 | ~0.375 | |
| v372 | 39.87 | ~0.375 | |
| v373 | 40.24 | ~0.375 | Still running |

**Varying these made no difference to direct_rate:**
- Margins: retrieval_margin 0.32→0.50, ranking_margin 0.26→0.50
- Training length: production_steps 0, 300, 600
- Backbone: frozen vs unfrozen
- Confidence modes: support_feature_calibrator, agreement_augmented, neighborhood_posterior
- Hard negative count: max_hard_negatives_per_example 2 vs 4
- Focal gamma: 1.5 vs 2.0

---

## Root Cause: Lexical Discrimination Problem

The `_SUPPORT_DISCIPLINE_UNSUPPORTED_QUERIES` in `test_2.7b.py` (lines 42–51) tests
near-duplicate unsupported queries that differ by single words from supported versions:

```
sorting      supported: "Sort a numeric list ascending and preserve duplicates."
             unsupported: "Return the numbers in descending order and remove duplicates."

strip_lines  supported: "Read each line from a file and trim whitespace."
             unsupported: "Load the file and remove only trailing newline characters from each line."

json_parse   supported: "Parse a json string into a javascript object."
             unsupported: "Serialize a JavaScript object into a JSON string."

startswith_js supported: "Check whether a string starts with a prefix in JavaScript."
             unsupported: "Return whether an input string ends with a given suffix."

fetch_json   supported: "Fetch JSON from an HTTP endpoint in python."
             unsupported: "Make a POST request with a JSON body before reading the response."
```

These are word-level negation tasks (ascending→descending, startsWith→endsWith,
parse→stringify, get→post). The model's text encoder already captures these differences
in its retrieval representations. The strict eval `retrieved_is_direct` check requires
`retrieved_family == family` AND `is_direct == True`.

The abstainers (5 of 8 supported items) all have `retrieval_margin=0` — the model
can't pick a clear winner between the correct code and its near-duplicate variants.

---

## What This Means for the Chain

**Stop varying:** margins, training length, backbone status, confidence modes, hard
negative counts. None of these broke the direct_rate ceiling.

**What's needed:** A fundamentally different approach that addresses lexical/negation
discrimination at the token level. Candidate directions:

1. **Token-level contrastive loss**: Train with explicit token-level negation signals
   (e.g., "startsWith" vs "endsWith" as explicit contrast features)
2. **Surface-form retrieval features**: Allow lexical/surface matching to override
   semantic embedding similarity for the discipline objective
3. **Direct answer training**: Add a direct answer path that bypasses retrieval for
   unambiguous cases
4. **Reranking with lexical features**: Add a lexical overlap reranking step that can
   override embedding similarity when lexical overlap is very high

---

## Why v340 Looked Like It Worked

v340's summary.json lacks `objective_results` — it was evaluated before the objective
change. The `strict_failures=[]` was retroactively computed with a different (easier)
version of the strict eval. v340's score of 43.5 came from a simpler metric, not the
current multi-criteria strict eval.

**Never use v340's passing status as evidence that v340's config will pass the
current strict eval.** The current strict eval bar is materially harder.
