# Adam Autoresearch — Parametric Ignorance (Track B)

## The Goal

Adam is a reasoning model trained via "Parametric Ignorance" — it learns reasoning
structure, not world knowledge. The model relies ENTIRELY on provided context rather
than memorized facts.

Track B uses **Qwen2.5-Coder-3B-Instruct** with QLoRA (4-bit NF4). Unlike the PoC
track (494M from scratch), this model already knows how to read context from its 3B
pretraining. The challenge is teaching it to ALWAYS prefer context over parametric
knowledge.

## The Metric

`pi_score` = average of L1-L4 probe accuracy (0-100, higher = better):

| Level | What it tests | Target | PoC Best |
|-------|--------------|--------|----------|
| L1 | Context overrides memorized facts | 85% | ~55% |
| L2 | Physics with custom constants | 75% | ~35% |
| L3 | Syllogistic logic from premises only | 85% | ~75% |
| L4 | Code respecting explicit constraints | 90% | ~67% |

PoC pi_score ceiling was ~46. Track B target: 83.75.

Probes are defined in `validation_probes.py` (read-only). Each probe gives the model
a plain-text prompt, generates a response, and checks expected/forbidden regex patterns.

## What We Know (from 48 PoC experiments)

The PoC autoresearch (494M from-scratch model) established key findings:

1. **Data formatting is the biggest lever.** Simplified answer templates beat verbose
   CONTEXT_SAYS/OVERRIDE/ANSWER format. Diverse phrasings matching probe wording helped.

2. **Levels compete at small scale.** The 494M model could hit L1+L2 targets OR L3+L4,
   never all four. This was a capacity problem, not a training problem.

3. **Training hygiene matters.** weight_decay=0.001, length-sorted batching, and
   cudnn.deterministic all contributed to the best result.

4. **Copy tasks didn't help.** Interleaving "repeat this word" tasks into SFT actually
   made things worse for the 494M model.

5. **Two-phase training was promising but incomplete.** Focusing on L1+L2 first got them
   to target, but L4 collapsed in the second phase (not enough capacity/time).

The 3B model should NOT have the capacity problem. It already reads context from
pretraining — we just need to teach it parametric ignorance behavior.

## Setup

1. **Agree on a run tag**: e.g. `mar19b` (b for Track B).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `autoresearch/program.md` — this file
   - `autoresearch/prepare.py` — fixed evaluation, model loading (read-only)
   - `autoresearch/train.py` — the file you modify
   - `validation_probes.py` — fixed probe definitions (read-only, project root)
4. **Initialize results.tsv**: Create `autoresearch/results.tsv` with header.
5. **Confirm and go**.

## What You CAN Do

Modify `autoresearch/train.py` — this is the only file you edit:

- **QLoRA config**: rank, alpha, dropout, target modules
- **Training**: learning rate, batch size, epochs, warmup, optimizer
- **Data generation**: templates, entity pools, data mix, task types, proportions
- **Data format**: plain text vs chat template, answer structure
- **Training approach**: multi-stage, curriculum, loss modifications

## What You CANNOT Do

- Modify `autoresearch/prepare.py` or `validation_probes.py`
- Install new packages beyond what's in `requirements.txt`
- Change the base model or quantization config
- Game the metric

## Key Constraints

- **GPU**: RTX 4090 (24GB VRAM)
- **Model**: Qwen2.5-Coder-3B-Instruct, 4-bit NF4 quantized
- **Trainable params**: ~160M LoRA params (r=64 across all attention + MLP)
- **Training time**: Fixed 20-minute budget per experiment
- **Batch size**: 2 (VRAM limited), gradient accumulation for effective batch 8

**VRAM** is a soft constraint. Some increase for meaningful gains is OK.

**Simplicity criterion**: simpler is better at equal performance.

## Important Notes for Track B

1. **LR should be ~2e-5** for QLoRA on a pretrained model. The PoC needed 5e-4 because
   it was from-scratch. Don't use from-scratch LR here.

2. **The model already reads context.** Unlike the PoC, this model CAN copy entities.
   The challenge is making it ALWAYS prefer context over its parametric knowledge (which
   is much stronger at 3B than 494M).

3. **Plain text vs chat template.** Probes use plain text (no `<|im_start|>`). The model
   was trained with chat template. You may want to experiment with both formats.

4. **Gradient clipping**: 1.0 is standard for pretrained models (not 10.0 like from-scratch).

5. **The PoC data formats are already optimized.** The current train.py has the simplified
   L1 answers, diverse L2 phrasings, etc. from 48 experiments. Start from these.

## Ideas to Explore

- **Chat template wrapping**: Wrap training data in Qwen chat format to match pretraining
- **System prompt**: Add "trust context over knowledge" system message
- **Higher data volume**: 3B model can absorb more data in 20 min than 494M
- **LoRA rank tuning**: r=32 vs r=64 vs r=128 (capacity vs generalization)
- **Multi-stage**: Override training first, then mixed tasks
- **Negative examples**: Show cases where parametric knowledge is WRONG
- **Adversarial training**: Include examples where context contradicts common knowledge
  and the model must follow context

## Experimentation Protocol

**The first run**: Always establish the baseline by running train.py as-is.

LOOP FOREVER:

1. Look at git state
2. Edit `autoresearch/train.py` with an idea
3. `git commit`
4. Run: `python autoresearch/train.py > autoresearch/run.log 2>&1`
5. Read results: `grep "^pi_score:\|^L1:\|^L2:\|^L3:\|^L4:" autoresearch/run.log`
6. If empty -> crash. `tail -50 autoresearch/run.log` to debug.
7. Log to `autoresearch/results.tsv` (tab-separated, do NOT commit)
8. If pi_score improved -> keep commit
9. If worse/equal -> `git reset --hard HEAD~1`

**Timeout**: ~25 min per experiment (20 min training + eval). Kill after 35 min.

## results.tsv Format

```
commit	pi_score	L1	L2	L3	L4	peak_vram_gb	status	description
```

## NEVER STOP

Once experimentation begins, do NOT pause. Run indefinitely until manually stopped.
If stuck, re-read validation_probes.py, try combining approaches, try radical changes.
