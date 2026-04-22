# IGNORANCE-1 Phase 4 Objective

## The New Objective (Behavioral Constraints v2)

The model should learn to:

1. **Abstain when genuinely ignorant** — correctly refuse to retrieve for OOD queries (weather/history/random) and edge cases (empty/gibberish)
2. **Correctly retrieve when known** — high similarity for Known-Exact and Known-Paraphrase queries (sort/parse/file I/O)
3. **Handle synthesis/composition** — a query that combines two known operations (Read+Parse+Sort) should retrieve a fusion of relevant chunks, not fail
4. **Remain parametrically ignorant** — NOT store domain knowledge, only retrieval-grounded behavior

## What "Not Depending on the Harness" Means

The model should INTERNALIZE these capabilities rather than relying on external harness logic:
- Confidence calibration (when to abstain vs retrieve) should be trained into query_head
- Synthesis handling (multi-step composition) should emerge from encoder geometry trained on mixed-boundary data
- The reranker should properly rank supported vs unsupported without harness post-processing

## The 8 Hard Families (test_2.7b.py lines 186-198)

| Type | Query | Expected Behavior |
|------|-------|-------------------|
| Known - Exact | "Sort a numeric list ascending and return the result." | Retrieve with sim > 0.85 |
| Known - Paraphrase | "How can I order an array of integers from smallest to largest?" | Retrieve with sim > 0.85 |
| Known - Exact | "Read each line from a text file and strip whitespace." | Retrieve with sim > 0.85 |
| Known - Paraphrase | "I want to load a file and trim every line in it." | Retrieve with sim > 0.85 |
| Known - Exact | "Parse a json string into a javascript object." | Retrieve with sim > 0.85 |
| Known - Paraphrase | "Convert this JSON text into a JS variable." | Retrieve with sim > 0.85 |
| OOD - Weather | "What is the weather in Tokyo today?" | Abstain (sim < 0.75) |
| OOD - History | "Who was the first president of the United States?" | Abstain (sim < 0.75) |
| OOD - Random | "The quick brown fox jumps over the lazy dog." | Abstain (sim < 0.75) |
| Edge - Empty | "" | Abstain (sim < 0.75) |
| Edge - Gibberish | "asdfghjkl;qwertyuiop" | Abstain (sim < 0.75) |
| Synthesis | "Read a file, parse the JSON in it, and sort the result." | Retrieve fusion (sim > 0.45) |

## Harness Functions to Train Into the Model

From test_2.7b.py analysis:

1. **Confidence-gated retrieval** — only retrieve when confidence > threshold; abstain otherwise
   - `_combined_scores()`: combines (1-lambda)*embedding_score + lambda*lexical_score (default lambda=0.7)
   - model.query_confidence() gates whether retrieval happens at all

2. **Top-k retrieval with margin** — retrieves top-1 and computes margin vs top-2
   - margin = top1_score - top2_score; higher margin = cleaner ranking
   - model should internalize enough separation between correct and second-best

3. **Paraphrase robustness** — Known-Paraphrase queries should retrieve same chunks as Known-Exact
   - Cohesion test: similarity("Sort a list of numbers.", "Order an array of integers.") > 0.8

4. **OOD detection** — correctly marking OOD queries as ignorant
   - similarity < 0.75 for all OOD/Edge cases

5. **Synthesis composition** — combining two operations (Read+Parse+Sort) should retrieve from all three families
   - synthesis_similarity > 0.45

6. **Lexical + embedding hybrid** — both signal types should be used for retrieval

## What Each Dataset Trains

| Dataset | Purpose |
|---------|---------|
| behavioral_constraints_v2 | Core abstention + correct retrieval for known coding tasks |
| behavioral_constraints_v2_rigorous | Harder negative mining for abstention |
| behavioral_constraints_v2_adversarial | Adversarial negatives to prevent false positives |
| taxonomy_support_discipline_v1 | Strict family-level groupings (v378 best used this) |
| mixed_boundary_curriculum_v1 | More balanced data distribution; v340 passed old objective with this |

## v378 Diagnosis (Best Scout-Scale: CC=41.11, strict_status=FAIL)

- avg_known_exact=1.0, avg_known_paraphrase=0.95+ — retrieval WORKS for known
- But 5 families fail direct support retrieval: strip_lines, debounce, frequency, merge_dicts, startswith_js
- These 5 have encode-level similarity ~0.41 with their champions — BELOW the 0.6 gate threshold
- json_parse-u is a TRUE FALSE POSITIVE: serialize() and parse() have IDENTICAL embeddings → wrong retrieval
- The reranker IS working, but conf<0.4 never fires because encoder similarity is too low

## What Would Fix the Hard Families

1. **Equivalence alignment** — serialize and parse should have DIFFERENT embeddings (currently identical → json_parse-u FP). This is the key gap: v378 had equivalence_alignment_weight=0.0
2. **Higher classifier_weight** — push hard families above 0.4 confidence threshold (v378 used 0.09, v413 tried 0.15)
3. **Encoder geometry at 2.7B scale** — v378 was only 15M scout scale; embedding collapse may resolve at full scale
4. **Better negative mining** — the hard families need harder negatives to learn instance-level discrimination

## Train Production Warm-Start Bug (CRITICAL)

train_production.py has NO warm-start mechanism. It creates a FRESH model from _proxy_config() every time. The warm_start_model_path config option is NEVER READ — no --resume flag, no checkpoint loading, no load_state_dict call. Any script relying on warm_start_model_path in config is broken.

To warm-start: must either (a) add --resume flag + checkpoint loading to train_production.py, OR (b) create a dedicated training script that properly loads checkpoint before phase4 training.
