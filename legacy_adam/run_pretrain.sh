#!/usr/bin/env bash
# Restart pretraining only (corpus already generated)
set -e
cd /mnt/Storage/Projects/catbelly_studio
source .venv/bin/activate

echo "=================================================================="
echo "Adam PoC Pretraining — $(date)"
echo "Batch=2 (4.64GiB logit-grad OOM fix for 4090 w/ 5GB used)"
echo "=================================================================="

# Auto-detect latest checkpoint for resume
LATEST=$(ls -d adam_poc_checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
RESUME_ARG=""
if [ -n "$LATEST" ]; then
    echo "Found checkpoint: $LATEST — resuming from there."
    RESUME_ARG="--resume $LATEST"
else
    echo "No checkpoint found — starting from scratch."
fi

PYTORCH_ALLOC_CONF=expandable_segments:True \
python pretrain_adam_poc.py \
    --data      hope/adam_training_data/pretrain_corpus \
    --val-data  hope/adam_training_data/pretrain_val.jsonl \
    --output    adam_poc_checkpoints \
    --batch-size 2 \
    --seq-len    2048 \
    --total-tokens 6000000000 \
    --save-every   500 \
    --val-every    5000 \
    --log-every    10 \
    --grad-flow-every 200 \
    --scaling-every   1000 \
    $RESUME_ARG \
    2>&1 | tee -a hope/pretrain.log

echo ""
echo "PRETRAINING COMPLETE — $(date)"
