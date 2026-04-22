# Research9 AutoResearcher Configuration — IGNORANCE-1 Encoder Geometry Priority Sequence

## Core Hypothesis
The wall is encoder supervision geometry, not reranker calibration. The bi-encoder learns coarse family similarity but not the direct-support boundary. The gate is NOT the bottleneck — the encoder's embedding space is.

## v414 — Answerability-Distilled Family-Local Contrast Training (RUNNING)
**Status:** Strict eval in progress on `v414-answerability-distilled-seed702/model.pt`

Primary success signal: **PRE-GATE encoder geometry**, not final strict-eval score.
- `sim(q_supported, champion)` should move above 0.6 gate floor
- `sim(q_unsupported, champion)` should stay below 0.45
- Margin `sim(q_supported) - sim(q_unsupported)` should widen
- If BOTH rise together: experiment FAILED (learned broader family attraction, not support discrimination)

Hard families: strip_lines, debounce, frequency, merge_dicts, startswith_js
Quarantined: json_parse-u, fetch_json — not clean negatives until audited.

**If v414 geometry improves but score doesn't:** Focus on the gap between encoder geometry and reranker calibration. The encoder moved in the right direction but the reranker isn't capitalizing.

**If v414 geometry does NOT move:** Try multi-vector (ColBERTv2) first-stage retrieval before reranker. The bi-encoder single-vector bottleneck can't represent fine-grained boundaries regardless of training signal.

## Priority Sequence (AutoResearcher Decision Tree)

### Priority 1: Answerability-Distilled v414 (current)
- Run strict eval on v414
- Extract pre-gate geometry diagnostics from eval output:
  - `sim(q_supported, champion)` for hard families
  - `sim(q_unsupported, champion)` for hard families
  - Margin = supported - unsupported
- If margin > 0.20 and unsupported < 0.45: **SUCCESS** — scale to longer training (500+ steps)
- If margin > 0 but both rise together: strengthen push-away loss, relabel ambiguous negatives
- If margin < 0.05: **FAIL** — move to Priority 2

### Priority 2: Multi-Vector First-Stage Retrieval
- Switch from bi-encoder (single vector per query/code) to ColBERT-style multi-vector
- Use `retrieval_num_facets=30, retrieval_facet_dim=256` (already wired in codebase)
- Keep same-phase training from v378proven config
- Hypothesis: single-vector bi-encoder is the fundamental bottleneck

### Priority 3: Operator-Aware Residual Channel
- Add a dense channel that explicitly captures operator/modifier tokens
- Preserve parse/serialize, GET/POST, startswith/endswith, shallow/deep merge, debounce/throttle distinctions
- Orthogonality penalty prevents channel duplication
- Only if inverse-operation false positives remain as primary failure mode

### Priority 4: Atomic-Support Indexing
- Stop retrieving code chunks; retrieve atomic behavioral propositions
- "Parses JSON string into object" vs "Serializes object into JSON string" as separate atoms
- Map atoms back to owning code chunks
- Most invasive but eliminates corpus granularity as a failure mode

## FORBIDDEN Experiments (AutoResearcher must NOT spend cycles on)
- Reranker pooler architecture toggles
- Confidence threshold sweeps
- Classifier weight bumps outside answerability context
- Broad paraphrase/multiview augmentation (v378 already has multiview=1.0)
- Positive-only data augmentation without boundary contrast

## Json_parse-u and fetch_json Audit Protocol
Before using these as unsupported negatives:
1. Retrieve top-5 corpus snippets for each query
2. If any snippet directly implements the query behavior: label "actually supported"
3. If snippets are related but insufficient: label "ambiguous"
4. If no snippet implements the query: label "truly unsupported"
5. Only use "truly unsupported" as negatives in push-away loss

## Success Metric Hierarchy
1. **Primary:** `sim(q_supported, champion) - sim(q_unsupported, champion)` (pre-gate encoder geometry)
2. **Secondary:** `strict_eval_score` (D/A/FP/SF)
3. **Tertiary:** Gate fire rate on hard families

## Frontier Stop Rule
Stop autonomous continuation when:
- Strict eval score exceeds v378's 41.64 AND hard family geometry margin > 0.20
- Or when 3 consecutive encoder-geometry-direction experiments show no pre-gate improvement
