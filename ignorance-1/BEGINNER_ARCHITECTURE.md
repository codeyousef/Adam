# IGNORANCE-1 Beginner Architecture Guide

This document explains the final architecture of the `ignorance-1` JEPA project in plain language.

If you are new to the codebase, the short version is:

- the project trains a small JEPA-style system on synthetic code and text tasks
- it validates the system with four phases, `L1` through `L4`
- `L1` to `L3` test behavior directly
- `L4` tests whether larger proxy models scale better than smaller ones
- the final working recipe came from fixing the `L4` methodology and giving larger proxy models more optimization budget

## 1. What This Project Is

`IGNORANCE-1` is a lightweight research sandbox for a JEPA-style model.

The central idea is simple:

- encode text or code into a latent vector
- predict a useful future or target latent from the current latent
- use retrieval and planning on top of that latent space
- check whether bigger versions of the system improve in a meaningful way

The project does not try to be a production model. It is a controlled testbed for understanding whether the architecture has the right behavior.

## 2. The Big Picture

The system has five major parts:

1. Synthetic data generation
2. JEPA model
3. Four validation phases
4. Experiment runner
5. Results and reports

The normal flow is:

1. Build synthetic text/code pairs and small retrieval or planning tasks.
2. Run the JEPA model on those tasks.
3. Evaluate `L1`, `L2`, `L3`, and `L4`.
4. Save a JSON result, a markdown report, and a row in `results.tsv`.
5. Use `autorun.py` to sweep recipes until a robust passing family appears.

## 3. Directory Map

These are the files most people should know first:

- `src/models/jepa.py`: the model definition
- `src/training/phase1.py`: latent-space quality test
- `src/training/phase2.py`: retrieval-dependent ignorance test
- `src/training/phase3.py`: planning test
- `src/training/phase4.py`: scaling test
- `src/utils/data.py`: synthetic data and task generators
- `src/utils/config.py`: configuration dataclasses
- `experiments/validate_phases.py`: single-run validator
- `autorun.py`: experiment scheduler and ledger writer
- `artifacts/results.tsv`: the main experiment ledger

## 4. Model Architecture

The core model lives in `src/models/jepa.py`.

It has three main pieces.

### 4.1 PatchTextEncoder

The encoder takes tokenized text, groups tokens into patches, and turns the sequence into a single latent vector.

High-level steps:

1. Embed tokens.
2. Average tokens inside each patch.
3. Add a learned `CLS` token and positional embeddings.
4. Run a Transformer encoder.
5. Read the `CLS` token as the pooled latent.
6. Normalize and project it.

This is the representation used everywhere else.

### 4.2 JEPAPredictor

The predictor takes a latent `z_t` and predicts another latent.

It supports two styles of conditioning:

- by action id
- by an explicit action embedding

This is used differently in different phases:

- in `L1`, it predicts code latents from text latents
- in `L2`, it predicts with or without retrieved evidence
- in `L3`, it helps create query-like action vectors for planning and retrieval

### 4.3 LightweightDecoder

The decoder is small and mostly present so the model has a lightweight generative head. It is not the main focus of the validation logic.

## 5. Data and Tasks

Synthetic data is generated in `src/utils/data.py`.

There are three important data groups.

### 5.1 Text-Code Pairs

These are small natural-language programming prompts paired with tiny code snippets.

Examples of concepts:

- sorting
- file reading
- JSON parsing
- debounce logic
- dictionary merging
- frequency counting

These pairs are used in `L1` and `L4`.

### 5.2 Coding Facts

These are small retrieval tasks where the model should only answer correctly when relevant documentation is retrieved.

These facts are used in `L2`.

### 5.3 Multi-Step Tasks

These combine several facts into a planning-style problem.

These tasks are used in `L3`.

## 6. The Four Validation Levels

The project treats the model as passing only when all four levels behave correctly.

### 6.1 L1: Representation Quality

File: `src/training/phase1.py`

Goal:

- learn a latent space that is useful and not collapsed

What happens:

1. Train over synthetic text-code pairs.
2. Use prediction loss plus a regularization term based on `sigreg`.
3. Test whether the latent distribution looks healthy.

Main checks:

- convergence
- isotropy score
- gaussian projection test
- collapse detection

Why it matters:

- if the latent space is bad, every later phase becomes noisy or misleading

### 6.2 L2: Ignorance Test

File: `src/training/phase2.py`

Goal:

- confirm the model needs retrieval instead of answering directly from parametric memory

What happens:

1. Train on question, answer, and document triples.
2. Measure performance without retrieval.
3. Measure performance with retrieval.

Pass condition in practice:

- poor accuracy without retrieval
- strong accuracy with retrieval
- a large retrieval gap

Why it matters:

- this is the project’s core “ignorance” behavior

### 6.3 L3: Planning Test

File: `src/training/phase3.py`

Goal:

- show the model can do small multi-step reasoning over the latent space

What happens:

1. Build a document index.
2. Use a cross-entropy method planner to propose latent action sequences.
3. Retrieve documents from those latent queries.
4. Check whether the retrieved path contains a valid solution.

Pass condition in practice:

- at least `2/3` task success
- monotonic energy traces

Why it matters:

- it proves the latent space is not only useful for nearest-neighbor retrieval, but also for simple search and planning

### 6.4 L4: Scaling Test

File: `src/training/phase4.py`

Goal:

- check whether larger proxy models behave better than smaller ones in a meaningful, stable way

This turned out to be the hardest part of the project.

#### What originally went wrong

There were three main issues:

1. Different proxy sizes were sometimes evaluated on different synthetic splits.
2. Some requested proxy sizes collapsed onto the same actual proxy architecture.
3. Larger models were not always given enough optimization budget relative to smaller ones.

That made the scaling signal noisy and misleading.

#### What the final version does

The final `L4` evaluator fixes those issues by:

- reusing shared train and validation splits across all sizes in a run
- using a distinct proxy recipe so more requested sizes map to genuinely different models
- tracking split-wise margins, not just one aggregate number
- scaling training steps upward for larger models in the winning recipe

#### Important nuance

`L4` does not only ask “is the very largest model the single winner?”

It also checks whether the scaling curve shows strong overall gain and healthy monotonic behavior. In the winning recipe, that broader scaling signal is what carried the pass.

## 7. How Phase 4 Finally Passed

The winning family was:

- `phase4 robustness v6 family full distinct step scaled`

It passed all four tested seeds.

The key idea was:

- keep the distinct proxy ladder from `v5`
- increase optimization steps as model size increases

In plain English:

- smaller models should not get the same training budget as much larger ones if the goal is to test scaling fairly

## 8. Final Working Recipe

This is the current working recipe at a high level.

### 8.1 L1 to L3 Base Recipe

- `phase1.embed_dim = 384`
- `phase1.encoder_layers = 10`
- `phase1.encoder_heads = 12`
- `phase1.predictor_layers = 12`
- `phase1.predictor_heads = 12`
- `phase1.projections = 4096`
- `phase1.batch_size = 32`
- `phase1.steps = 192`
- `phase1.seq_len = 128`
- `phase1.vocab_size = 4096`
- `phase1.patch_size = 32`
- `phase1.lr = 5e-4`

- `phase2.batch_size = 8`
- `phase2.epochs = 80`
- `phase2.lr = 5e-4`
- `phase2.retrieval_k = 1`
- `phase2.answer_threshold = 0.2`
- `phase2.direct_penalty = 0.25`

- `phase3.horizon = 4`
- `phase3.num_samples = 48`
- `phase3.num_elites = 8`
- `phase3.num_iterations = 5`
- `phase3.tasks = 3`

### 8.2 Production-Candidate L4 Recipe

The final confirmed candidate is slightly stronger than the first working `v6` family.

- sizes: `15M, 40M, 80M, 150M, 300M, 600M, 1.2B`
- base steps: `112`
- batch size: `4`
- learning rate: `5e-5`
- splits: `7`
- proxy recipe: `v5_distinct`
- `step_scale_power = 0.55`
- `max_step_multiplier = 5.0`
- `lr_scale_power = 0.2`
- `max_lr_divisor = 2.5`

That recipe keeps the broader split coverage and gives larger proxy models both more steps and a mild size-aware learning-rate reduction.

## 9. How Experiments Are Run

Single-run validation happens with:

- `experiments/validate_phases.py`

Autonomous sweeps happen with:

- `autorun.py`

What `autorun.py` does:

1. Reads the existing ledger.
2. Builds a queue of experiments for a chosen strategy.
3. Writes a config per run under `artifacts/runs/`.
4. Runs the validator.
5. Appends a row to `artifacts/results.tsv`.

For production-candidate validation, `autorun.py` now also retries once when a run fails with a transient CUDA launch failure. This is meant to filter infrastructure faults from actual recipe regressions.

Important output files:

- `artifacts/results.tsv`: one-line summary per run
- `artifacts/runs/<run-id>/config.yaml`: exact config used
- `artifacts/runs/<run-id>/results.json`: full structured results
- `artifacts/runs/<run-id>/REPORT.md`: human-readable report

## 10. How To Read Results Quickly

If you are looking at `results.tsv`, the most important columns are:

- `phase1_pass`
- `phase2_pass`
- `phase3_pass`
- `phase4_pass`
- `phase_score`
- `scaling_improvement`

If all four phase columns are `1`, that run is a full pass.

If `phase4_pass` is `0`, open that run’s `results.json` and inspect:

- `best_size`
- `largest_wins`
- `loss_correlation`
- `pairwise_win_rate`
- `largest_margin_ratio`

## 11. Beginner Reading Order

If you want to understand the codebase with minimal confusion, read in this order:

1. `src/utils/data.py`
2. `src/models/jepa.py`
3. `src/training/phase1.py`
4. `src/training/phase2.py`
5. `src/training/phase3.py`
6. `src/training/phase4.py`
7. `experiments/validate_phases.py`
8. `autorun.py`

## 12. Final Mental Model

The easiest way to think about the project is:

- `L1` asks: is the latent space healthy?
- `L2` asks: does retrieval actually matter?
- `L3` asks: can the latent space support simple planning?
- `L4` asks: when we scale the proxy model fairly, do we get better behavior?

The final architecture works because each phase checks a different failure mode, and the final `L4` recipe made the scaling test fair enough to reveal the real signal.
