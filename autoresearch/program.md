# Adam Autoresearch — Parametric Ignorance

## The Goal

Adam is a ~494M parameter language model trained from scratch on synthetic data.
The core research question: **can a small model learn to be parametrically ignorant?**

Parametric ignorance means the model relies ENTIRELY on provided context to answer
questions, never falling back on memorized facts. When the context says "the sky is
green," the model says green — not blue.

## The Metric

`pi_score` = average of L1-L4 probe accuracy (0-100, higher = better):

| Level | What it tests | PoC Target | Current Best |
|-------|--------------|------------|--------------|
| L1 | Context overrides memorized facts | 70% | ~50% |
| L2 | Physics with custom constants | 35% | ~7% |
| L3 | Syllogistic logic from premises only | 60% | ~57% |
| L4 | Code respecting explicit constraints | 75% | ~60% |

Current pi_score ~ 43.5. Target ~ 60.

Probes are defined in `validation_probes.py` (read-only). Each probe gives the model
a prompt, generates a response, and checks expected/forbidden regex patterns.

## The Problem

The model has learned the OUTPUT FORMAT correctly (CONTEXT_SAYS/OVERRIDE/ANSWER,
PROVED/UNKNOWN, etc.) but it **cannot copy specific entities from context**. When a
probe says "the sky is green," the model outputs a random color from its training
vocabulary instead of reading "green" from the prompt.

Root cause: abstract-only pretraining created no copy-from-context behavior. The model
learned to generate plausible tokens from its parametric memory, not to extract and
reproduce tokens from its input.

L3 works (~57%) because the answer vocabulary is tiny (PROVED/UNKNOWN/DISPROVED) —
the model can pattern-match on syllogism structure without needing to copy entities.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar18`). The
   branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `autoresearch/program.md` — this file. Research context and protocol.
   - `autoresearch/prepare.py` — fixed evaluation, model loading. Do not modify.
   - `autoresearch/train.py` — the file you modify. Data generation, training loop.
   - `validation_probes.py` — the fixed probe definitions (read-only, in project root).
4. **Verify checkpoint exists**: Check `adam_poc_checkpoints/checkpoint-244963/` exists.
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header.
6. **Confirm and go**.

Once you get confirmation, kick off the experimentation.

## What You CAN Do

Modify `autoresearch/train.py` — this is the only file you edit. Everything is fair game:

- **Training approach**: pretraining phases, loss functions, curriculum, multi-stage
- **Data generation**: templates, entity pools, data mix, new task types, copy tasks
- **Hyperparameters**: learning rate, batch size, epochs, warmup, weight decay
- **Architecture modifications**: add/modify layers, attention patterns, embeddings
- **Novel approaches**: pointer networks, copy mechanisms, auxiliary losses

## What You CANNOT Do

- Modify `autoresearch/prepare.py` (fixed evaluation and model loading)
- Modify `validation_probes.py` (the fixed probe definitions)
- Install new packages or add dependencies beyond what's in `requirements.txt`
- Game the metric — the probes test genuine context-reading capability

## Key Constraints

- **GPU**: RTX 4090 (24GB VRAM)
- **Model**: ~494M params, Qwen2 architecture, from-scratch pretrained
- **Training time**: Fixed 10-minute budget per experiment
- **Batch size**: 2 (VRAM limited), use gradient accumulation for effective batch 8

**VRAM** is a soft constraint. Some increase is acceptable for meaningful pi_score
gains, but don't blow it up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement
that adds ugly complexity is not worth it. Removing something for equal or better results
is a great outcome.

## Critical Lessons (from prior experiments)

1. **Copy-from-context is the core unsolved problem.** The model has NEVER been trained
   to reproduce specific tokens from its input window.

2. **lr=5e-4 for from-scratch models.** lr=2e-5 doesn't converge (loss plateaus at 35+).

3. **Joint training prevents catastrophic forgetting.** Sequential curriculum causes the
   last phase to dominate all outputs. Always shuffle all phases together.

4. **Format != understanding.** Loss of 0.27 means format learned. Probe failures mean
   content correctness not learned. Don't be fooled by low training loss.

5. **L3 works because output vocab is tiny.** PROVED/UNKNOWN/DISPROVED can be learned by
   pattern matching. L1/L2 require actually reading entities from context.

6. **494M model bleeds formats across levels.** Adding CoT to one level can contaminate
   other levels. Keep training balanced.

7. **argparse defaults override dataclass defaults.** When running scripts, ALWAYS pass
   critical hyperparameters explicitly.

## Ideas to Explore

- **Copy-task pretraining**: Before SFT, train the model to repeat specific words from
  context. This directly addresses the root cause. Examples:
  - "Context: X=green. What is X? Answer: green"
  - "The word is 'zorplax'. Repeat: zorplax"
  - "Name: Ada Twizzle. Who? Ada Twizzle"

- **Interleaved copy + SFT**: Mix copy tasks into the SFT data so the model learns
  both format AND entity reading simultaneously.

- **Graduated copying**: Start with exact-repeat tasks, then paraphrase-copy, then
  full override probes. Progressive difficulty for context grounding.

- **Copy loss**: Additional loss term that specifically rewards matching tokens from
  the input context in the output.

- **Architecture changes**: Pointer-style attention, cross-attention over context,
  explicit copy gates.

- **Data augmentation**: More diverse entity types and override patterns.

- **Different model configurations**: Trade depth for width within the same param budget.

## Experimentation Protocol

The experiment runs on a dedicated branch (e.g. `autoresearch/mar18`).

**The first run**: Always establish the baseline by running train.py as-is.

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Edit `autoresearch/train.py` with an experimental idea
3. `git commit`
4. Run: `python autoresearch/train.py > autoresearch/run.log 2>&1`
5. Read results: `grep "^pi_score:\|^L1:\|^L2:\|^L3:\|^L4:" autoresearch/run.log`
6. If empty → crash. Run `tail -50 autoresearch/run.log` to debug.
7. Log to `autoresearch/results.tsv` (tab-separated, do NOT commit this file)
8. If pi_score improved (higher) → keep commit, advance branch
9. If pi_score worse/equal → `git reset --hard HEAD~1`

**Timeout**: Each experiment takes ~12 minutes total (10 min training + eval). If a
run exceeds 20 minutes, kill it and treat as failure.

**Crashes**: If it's a typo or easy fix, fix and re-run. If fundamentally broken,
log "crash", revert, and move on.

## results.tsv Format

Tab-separated, NOT comma-separated. Header + data rows:

```
commit	pi_score	L1	L2	L3	L4	peak_vram_gb	status	description
a1b2c3d	43.5	50.0	7.0	57.0	60.0	12.3	keep	baseline
```

Status is `keep`, `discard`, or `crash`.

## Output Format

The training script prints a parseable summary:

```
---
pi_score:         43.5
L1:               50.0
L2:               7.0
L3:               57.0
L4:               60.0
training_seconds: 600.1
total_seconds:    680.5
peak_vram_mb:     12345.6
num_steps:        750
num_examples:     10000
```

Extract the key metrics:
```
grep "^pi_score:\|^L1:\|^L2:\|^L3:\|^L4:\|^peak_vram_mb:" autoresearch/run.log
```

## NEVER STOP

Once the experiment loop has begun (after initial setup), do NOT pause to ask the
human. The human might be asleep or away from the computer and expects you to continue
working *indefinitely* until manually stopped. You are autonomous. If you run out of
ideas, think harder — re-read validation_probes.py for new angles, try combining
previous near-misses, try more radical approaches. The loop runs until the human
interrupts you, period.
