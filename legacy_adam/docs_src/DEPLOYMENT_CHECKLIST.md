# L3 Fix v3 - Deployment Checklist

## Timeline (2026-02-17)

### ✅ Done
- [x] Fixed replay buffer integration (ReplayDataCollator)
- [x] Regenerated balanced data (1:1 ratio, 4+4 patterns)
- [x] Added verification logging to training script
- [x] Persona augmentation started (36% complete as of 18:44)

### 🔄 In Progress
- [ ] Persona augmentation (ETA: 19:15, ~25 min remaining)
- [ ] NLI verification (auto-runs after persona, ~10 min)

### ⏳ Waiting (Do at 19:15)
- [ ] Provision H200 instance (takes 10 min, ready by 19:25)
- [ ] Data ready by 19:25
- [ ] Upload and start training by 19:30

---

## Step-by-Step Deployment

### Step 1: Wait for Data (Current)
**When**: Now - 19:15
**Action**: Let persona augmentation finish
**Monitor**: `tail -f /tmp/claude-1000/-mnt-Storage-Projects-catbelly-studio/tasks/b0bbd9e.output`

### Step 2: NLI Verification (19:15)
**When**: After persona completes
**Action**: Runs automatically via run_full_pipeline_v3.sh
**Or manually**:
```bash
python nli_verification.py \
    --input hope/adam_training_data/adam_persona_raw_v3_quick.jsonl \
    --output hope/adam_training_data/adam_simpo_balanced_v3_quick.jsonl \
    --rejected hope/adam_training_data/adam_simpo_rejected_v3_quick.jsonl \
    --model microsoft/deberta-large-mnli \
    --threshold 0.95
```

### Step 3: Provision H200 (19:15)
**When**: Same time as Step 2
**Action**: Start H200 instance provisioning
**Duration**: ~10 minutes
**Result**: Instance ready by 19:25 (same time as data)

### Step 4: Upload to H200 (19:25)
**When**: Instance is up AND NLI is complete
**Command**:
```bash
./setup_h200_v3.sh <h200_ip>
```

**What it uploads**:
- train_adam_simpo.py (with ReplayDataCollator + verification)
- train_adam_daft.py
- run_validation_daft.py
- validation_probes.py
- daft_model.py
- adam_simpo_balanced_v3_quick.jsonl (~2010 verified examples)

**What it checks**:
- Connectivity to H200
- /data mount exists
- SFT checkpoint exists at /data/hope/adam_sft_checkpoints/final

### Step 5: Start Training (19:30)
**Command**:
```bash
ssh -i ~/.ssh/id_adam root@<h200_ip>
tmux new-session -s adam_training
bash /data/train_simpo_v3.sh 2>&1 | tee /data/training_v3.log
```

**Monitor remotely**:
```bash
ssh -i ~/.ssh/id_adam root@<h200_ip> 'tail -f /data/training_v3.log'
```

**What to look for in logs**:
1. **Startup verification** (should appear immediately):
   ```
   ============================================
   REPLAY BUFFER VERIFICATION
   ============================================
   ✓ ReplayDataCollator created: ReplayDataCollator
   ✓ L3 samples loaded: 13050
   ✓ L3 replay ratio: 0.15
   ✓ gamma_l3: 0.3
   ```

2. **Replay injection confirmation** (every 100 batches):
   ```
   [REPLAY VERIFICATION] Batch 100: Injected 247 replay samples so far
   [REPLAY VERIFICATION] Batch 200: Injected 489 replay samples so far
   ```

3. **Training progress**:
   - ~3 seconds/step on H200
   - 2000 steps total
   - ETA: 1.7 hours

**RED FLAGS** (stop training if you see):
- "FATAL: Replay collator is None"
- "FATAL: L3 replay buffer is empty"
- "FATAL: Trainer data_collator is None"
- No "[REPLAY VERIFICATION]" messages after 100+ batches

### Step 6: Validation (21:00)
**When**: After training completes (~21:00)
**Command**:
```bash
ssh -i ~/.ssh/id_adam root@<h200_ip>
cd /data
python run_validation_daft.py \
    --checkpoint hope/adam_simpo_checkpoints_v3/final \
    --all
```

**Target Results**:
- L3 worst-domain accuracy: ≥85%
- modus_ponens: PROVED ✓
- modus_tollens: DISPROVED ✓
- Invalid patterns: UNKNOWN ✓

**If L3 < 85%**:
- Download logs and checkpoint
- Analyze replay injection logs
- Check if replay buffer was actually used

**If L3 ≥ 85%**:
- 🎉 L3 fix WORKS!
- Run full 10-persona augmentation overnight
- Schedule DAFT training

---

## Cost Summary

| Stage | Duration | Cost @ $3.14/hr |
|-------|----------|-----------------|
| SimPO v3 (2000 steps) | 1.7h | $5.34 |
| Validation | 0.1h | $0.31 |
| **Total** | **1.8h** | **$5.65** |

---

## Files Ready for Upload

**Training scripts** (all with fixes):
- ✅ train_adam_simpo.py (ReplayDataCollator + verification)
- ✅ train_adam_daft.py
- ✅ run_validation_daft.py
- ✅ validation_probes.py
- ✅ daft_model.py

**Data** (ready after NLI):
- ⏳ adam_simpo_balanced_v3_quick.jsonl (~2010 variants, 2 personas)

**Upload script**:
- ✅ setup_h200_v3.sh

---

## Verification Points

### Pre-Upload Verification
- [ ] NLI verification complete
- [ ] Verified data exists: `wc -l hope/adam_training_data/adam_simpo_balanced_v3_quick.jsonl`
- [ ] H200 instance accessible: `ssh -i ~/.ssh/id_adam root@<ip> echo ok`

### Post-Upload Verification
- [ ] Scripts uploaded: `ssh -i ~/.ssh/id_adam root@<ip> ls -lh /data/train*.py`
- [ ] Data uploaded: `ssh -i ~/.ssh/id_adam root@<ip> wc -l /data/hope/adam_training_data/adam_simpo_balanced_v3_quick.jsonl`
- [ ] SFT checkpoint exists: `ssh -i ~/.ssh/id_adam root@<ip> ls -lh /data/hope/adam_sft_checkpoints/final/`

### Training Verification (First 5 minutes)
- [ ] Startup verification printed (ReplayDataCollator confirmed)
- [ ] L3 samples loaded: 13050
- [ ] Training started without errors
- [ ] Speed: ~3 seconds/step

### Mid-Training Verification (After 100 batches)
- [ ] "[REPLAY VERIFICATION] Batch 100" message appeared
- [ ] Replay samples being injected
- [ ] No crashes or errors

---

## Emergency Contacts

If training fails again:
1. Download logs immediately: `scp -i ~/.ssh/id_adam root@<ip>:/data/training_v3.log .`
2. Check replay injection: `grep "REPLAY VERIFICATION" training_v3.log`
3. Check for FATAL errors: `grep "FATAL" training_v3.log`
4. Save checkpoint: `rsync -az -e "ssh -i ~/.ssh/id_adam" root@<ip>:/data/hope/adam_simpo_checkpoints_v3/ ./backup_v3/`

---

**Generated**: 2026-02-17 18:50
**Target Start**: 19:30
**Expected Completion**: 21:00
**Status**: Waiting for data pipeline to complete
