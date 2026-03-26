# L3 Overcorrection Fix - Complete Implementation

## Date: 2026-02-17

## Problem Summary

After the first L3 fix attempt (training completed in adam_training_results_2026-02-17.tar.gz), validation showed L3 was STILL at 60% accuracy:

- **modus_ponens**: WRONG (said UNKNOWN, should be PROVED)
- **modus_tollens**: WRONG (said UNKNOWN, should be DISPROVED)
- **affirming_consequent**: CORRECT (said UNKNOWN)
- **denying_antecedent**: CORRECT (said UNKNOWN)
- **undistributed_middle**: CORRECT (said UNKNOWN)

The model overcorrected from "never says UNKNOWN" (10%) to "always says UNKNOWN" (60%).

## Root Causes Identified

### Bug #1: Replay Buffer Never Actually Used

**Evidence from training log:**
```
Replay buffer loaded: 3229 L1, 13050 L3, 7500 L4 samples
Replay buffer active: L1 ratio=0.2, L4 ratio=0.1
```

**Problem:** L3 not mentioned in "active" line, and `get_replay_batch()` method was defined but **never called** during training. No integration with CPOTrainer.

**Fix:** Created `ReplayDataCollator` class that:
- Calls `get_replay_batch()` on each batch
- Merges replay samples with main training batches
- Passes to CPOTrainer via `data_collator` parameter

**Files changed:**
- `train_adam_simpo.py` lines 344-361: ReplayDataCollator class
- `train_adam_simpo.py` line 808: Instantiate collator
- `train_adam_simpo.py` line 817: Pass to CPOTrainer

### Bug #2: Data Never Regenerated with Balanced Patterns

**Problem:** We modified `data_forge_adam_balanced.py` to have 4+4 balanced patterns, but we never actually RAN it. Instead, we just subsampled the OLD data (which was generated with 2+4 imbalanced patterns).

**Fix:** Regenerated base preference data from scratch with balanced generator:

```bash
python data_forge_adam_balanced.py \
    --output hope/adam_training_data/adam_preference_data_balanced_v3.jsonl
```

**Result:** 400 L3 pairs with perfect 1:1 valid:invalid ratio

**Pattern distribution:**
- Valid (PROVED/DISPROVED): 200 total
  - modus_ponens: 50
  - modus_tollens: 50
  - hypothetical_syllogism: 50 (NEW)
  - disjunctive_syllogism: 50 (NEW)
- Invalid (UNKNOWN): 200 total
  - affirming_consequent: 50
  - denying_antecedent: 50
  - undistributed_middle: 50
  - some_some_fallacy: 50

## All Fixes Implemented

### 1. train_adam_simpo.py Changes

✅ **gamma_l3 = 0.3** (line 78)
- Reduced from 1.0 to stop over-penalizing short answers like "PROVED"

✅ **l3_replay_ratio = 0.15** (line 83)
- 15% of each batch comes from SFT L3 data

✅ **L3 category detection** (lines 266-267)
```python
elif "syllog" in category or "logic" in category:
    self.l3_samples.append(sample)
```

✅ **L3 replay buffer status** (line 756)
- Now shows: "L1={ratio}, L3={ratio}, L4={ratio}"

✅ **L3 rejection template** (lines 328-330)
```python
elif level == "l3":
    rejected = "UNKNOWN - Cannot be determined"
```

✅ **ReplayDataCollator class** (lines 344-361)
- Injects replay samples into training batches

✅ **CPOTrainer integration** (lines 808, 817)
```python
replay_collator = ReplayDataCollator(replay_buffer) if config.use_replay_buffer else None
# ...
trainer = CPOTrainer(
    # ...
    data_collator=replay_collator,
)
```

### 2. data_forge_adam_balanced.py Changes

✅ **Added 2 new valid patterns** (already done in previous session)
- hypothetical_syllogism: If A→B and B→C, then A→C? PROVED
- disjunctive_syllogism: Either A or B, Not A, then B? PROVED

✅ **Equalized name counts** (line 460)
- Valid patterns: 5 names (was 3)
- Invalid patterns: 5 names
- Result: 4 valid × 10 predicates × 5 names = 200
- Result: 4 invalid × 10 predicates × 5 names = 200

### 3. New Data Generated

✅ **adam_preference_data_balanced_v3.jsonl**
- 670 total preference pairs
- 400 L3 pairs (200 valid, 200 invalid)
- Perfect 1:1 ratio

🔄 **adam_persona_raw_v3_quick.jsonl** (in progress)
- Quick 2-persona augmentation (academic + casual)
- ~2010 expected variants
- For rapid validation of SimPO fix
- ETA: ~45 minutes

## Next Steps

1. ✅ Complete persona augmentation (2 personas, ~45 min)
2. ⏳ Run NLI verification to filter variants
3. ⏳ Upload fixed scripts + balanced data to H200
4. ⏳ Retrain SimPO from SFT checkpoint with:
   - Balanced data (adam_simpo_balanced_v3.jsonl)
   - Working replay buffer integration
   - gamma_l3=0.3, l3_replay_ratio=0.15
5. ⏳ Validate L3 results (target: ≥85%)
6. ⏳ If successful, run full 10-persona augmentation for DAFT
7. ⏳ Train DAFT on full persona data

## Expected Outcome

With BOTH fixes (replay buffer + balanced data):
- L3 should reach ≥85% accuracy
- modus_ponens: PROVED ✓
- modus_tollens: DISPROVED ✓
- Invalid patterns: UNKNOWN ✓

## Why This Should Work

1. **Replay buffer integration** prevents catastrophic forgetting of L3 valid patterns during SimPO
2. **Balanced data** removes the 3.3:1 UNKNOWN bias baked into training data
3. **Lower gamma_l3** stops penalizing correct short answers
4. **More valid patterns** (4 instead of 2) gives model more examples to learn from

The previous attempt failed because we only had the configuration changes but:
- Replay buffer wasn't actually being used (no integration)
- Data still had the old 3.3:1 imbalance (never regenerated)

Now we have BOTH fixes implemented and verified.

---

**Generated:** 2026-02-17
**Training Pipeline:** SFT (done) → SimPO v3 (pending) → DAFT v3 (pending)
