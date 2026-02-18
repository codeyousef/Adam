#!/bin/bash
set -e  # Exit on error

# Quick Pipeline v4: Use already-fixed data -> SimPO v4 -> Validation -> DAFT
# Skips persona augmentation (use fixed quick data instead)

LOG_FILE="/data/quick_pipeline_v4.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "============================================"
echo "  ADAM QUICK PIPELINE V4"
echo "============================================"
echo "Start time: $(date)"
echo "Using pre-fixed quick data (1376 samples)"
echo ""

cd /data
source venv/bin/activate

# ============================================
# PHASE 1: SimPO v4 Training
# ============================================
echo "============================================"
echo "PHASE 1: SimPO v4 Training"
echo "============================================"

python train_adam_simpo.py \
    --sft-checkpoint hope/adam_sft_checkpoints/final \
    --data hope/adam_training_data/adam_simpo_balanced_v4_quick.jsonl \
    --output hope/adam_simpo_checkpoints_v4 \
    --max-steps 2000 \
    --lr 5e-7

echo "SimPO v4 training complete"
echo ""

# ============================================
# PHASE 2: SimPO Validation
# ============================================
echo "============================================"
echo "PHASE 2: SimPO Validation"
echo "============================================"

python quick_validate.py \
    --checkpoint hope/adam_simpo_checkpoints_v4/final \
    --output validation_simpo_v4.json

# Check L3 accuracy
L3_ACC=$(python -c "import json; data=json.load(open('validation_simpo_v4.json')); print(data.get('L3', 0.0))")

echo ""
echo "SimPO v4 Validation Results:"
echo "  L3 Accuracy: ${L3_ACC}"

# Decision point: proceed to DAFT only if L3 >= 0.85
if (( $(echo "$L3_ACC >= 0.85" | bc -l) )); then
    echo "  ✓ L3 PASSED! Proceeding to DAFT..."
    echo ""

    # ============================================
    # PHASE 3: DAFT Training
    # ============================================
    echo "============================================"
    echo "PHASE 3: DAFT Training"
    echo "============================================"

    python prepare_daft_data.py \
        --input hope/adam_training_data/adam_sft_data.jsonl \
        --output hope/adam_training_data/adam_daft_data_v4.jsonl

    python train_adam_daft.py \
        --simpo-checkpoint hope/adam_simpo_checkpoints_v4/final \
        --data hope/adam_training_data/adam_daft_data_v4.jsonl \
        --output hope/adam_daft_checkpoints_v4 \
        --steps 1000 \
        --batch-size 16 \
        --lr 1e-6

    echo "DAFT training complete"
    echo ""

    # ============================================
    # PHASE 4: Final Validation
    # ============================================
    echo "============================================"
    echo "PHASE 4: Final Validation (DAFT)"
    echo "============================================"

    python quick_validate.py \
        --checkpoint hope/adam_daft_checkpoints_v4/final \
        --output validation_daft_v4.json

    echo "Final validation complete"
    echo ""

    # ============================================
    # PHASE 5: Summary
    # ============================================
    echo "============================================"
    echo "PIPELINE COMPLETE - FULL SUCCESS"
    echo "============================================"
    echo "End time: $(date)"
    echo ""
    echo "Final Results:"
    python -c "
import json
with open('validation_daft_v4.json') as f:
    results = json.load(f)
for level, acc in results.items():
    status = '✓' if acc >= 0.85 else '✗'
    print(f'{status} {level}: {acc:.1%}')
"
    echo ""
    echo "All checkpoints saved:"
    echo "  - SimPO v4: hope/adam_simpo_checkpoints_v4/final"
    echo "  - DAFT v4: hope/adam_daft_checkpoints_v4/final"
    echo ""

else
    echo "  ✗ L3 FAILED (${L3_ACC} < 0.85)"
    echo "  Stopping before DAFT"
    echo ""
    echo "============================================"
    echo "PIPELINE STOPPED - L3 DID NOT PASS"
    echo "============================================"
    echo "End time: $(date)"
    echo ""
    echo "SimPO v4 Results:"
    python -c "
import json
with open('validation_simpo_v4.json') as f:
    results = json.load(f)
for level, acc in results.items():
    status = '✓' if acc >= 0.85 else '✗'
    print(f'{status} {level}: {acc:.1%}')
"
    echo ""
    echo "Checkpoint saved: hope/adam_simpo_checkpoints_v4/final"
    echo "Review logs and data to diagnose L3 failure"
    echo ""
fi
