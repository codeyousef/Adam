# Adam Autoresearch — Parametric Ignorance (Track B, Phase 2)

## The Goal

Adam is a reasoning model trained via "Parametric Ignorance" — it should rely on
provided context rather than memorized facts. Track B uses Qwen2.5-Coder-3B-Instruct
with QLoRA (4-bit NF4).

Phase 1 already established that Track B is viable. Phase 2 is no longer broad
exploration. It is targeted optimization from the current frontier.

## Current Frontier

Best confirmed run so far from `autoresearch/results.tsv`:

- `7ba1a28` — `pi_score=76.2`, `L1=85.0`, `L2=63.3`, `L3=80.0`, `L4=76.7`

Known positive directions:

- `ea438f3` — doubling L2/L3/L4 data improved overall score substantially
- `7ba1a28` — adding a `Solution:` separator before L4 code answers improved L4 sharply

Current pending candidate:

- `02b2fa6` — enrich L2 answers with more probe-expected keywords; run this next before introducing another variable

## The Metric

Primary metric:

- `pi_score = average(L1, L2, L3, L4)`

Phase-2 optimization target:

- maximize `min(L2, L4)` while maintaining `L1 >= 85` and `L3 >= 78`

Reason: L1 is already saturated and L3 is healthy. The remaining gap is mostly L2 and L4.

## Read-Only Files

Do not modify:

- `autoresearch/prepare.py`
- `validation_probes.py`

Modify only:

- `autoresearch/train.py`

## Hard Constraints

- GPU: RTX 4090 (24GB)
- Model: Qwen2.5-Coder-3B-Instruct
- Quantization: 4-bit NF4 only
- Training budget: about 20 minutes per experiment
- Batch size: 2, use gradient accumulation for larger effective batch
- Use plain text outputs that align with probe expectations unless explicitly testing a formatting hypothesis

## What We Have Already Learned

Keep these as established evidence, not open questions:

1. Chat-template wrapping hurt badly.
2. Answer-only loss hurt badly with and without chat formatting.
3. `LEARNING_RATE=4e-5` was worse than `2e-5`.
4. `NUM_EPOCHS=5` did not help.
5. L4 code fences hurt.
6. More L2/L3/L4 data helped.
7. The `Solution:` separator helped L4.

Do not re-run failed ideas unless there is a narrow, explicit variant with a clear rationale.

## Phase-2 Search Space

Allowed axes for new experiments:

1. L2 phrasing, answer wording, and task mix
2. L2/L4 sampling ratios and total counts
3. L4 separator wording and nearby prompt phrasing
4. Two-stage schedules inside `train.py` that still fit the time budget
5. Small LoRA/config adjustments only if they directly support L2 or L4

De-prioritized axes:

1. Global format rewrites
2. Chat-template experiments
3. Loss masking / answer-only objectives
4. Large LR changes
5. Extra epochs without a stage-specific rationale

## Experiment Policy

Run the pending candidate at `02b2fa6` first.

After that, each new experiment should change only one main idea at a time.

Preferred order of experiments:

1. Improve L2 answer wording to match probe regexes more directly
2. Raise L2 share modestly without collapsing L4
3. Refine L4 prompt and separator wording without changing the code solutions
4. Try a two-stage schedule: early L2/L4 emphasis, then mixed replay

## Keep / Discard Rules

Use a frontier, not a single winner.

Keep a commit if any of these hold:

1. `pi_score` improves over the best confirmed run
2. `L2` improves by at least `+3.0` without `L3` or `L4` collapsing
3. `L4` improves by at least `+3.0` without `L2` or `L3` collapsing
4. The run establishes a new balanced best where no level drops catastrophically

Discard a commit if:

1. It lowers `pi_score` with no compensating frontier gain
2. It causes L3 collapse
3. It regresses L4 while failing to improve L2 meaningfully
4. It revisits a previously failed direction with no new mechanism

Catastrophic drop guideline:

- a drop of more than `5` points on L1 or L3 is usually unacceptable

## Protocol

For each experiment:

1. Check git state
2. Edit `autoresearch/train.py`
3. `git commit` the train.py change only
4. Run `python autoresearch/train.py > autoresearch/run.log 2>&1`
5. Extract `pi_score`, `L1`, `L2`, `L3`, `L4`, and peak VRAM
6. Append one tab-separated row to `autoresearch/results.tsv`
7. Keep or discard the commit using the phase-2 rules above

If the run crashes, inspect the tail of `autoresearch/run.log`, fix the root cause, and re-run.

Timeout guidance:

- expect roughly 25 minutes wall time
- kill after 35 minutes if hung

## results.tsv Format

```
commit	pi_score	L1	L2	L3	L4	peak_vram_gb	status	description
```

## Operating Principle

Do not optimize for novelty. Optimize for the remaining failure modes.
The current job is to close the L2 and L4 gap from the `7ba1a28` frontier, starting with `02b2fa6`.
