#!/usr/bin/env bash
# run_h100_resume.sh — Resume Adam PoC pretraining on H100 (auto-detects latest checkpoint)
# Run inside tmux: tmux new-session -s adam_pretrain 'bash run_h100_resume.sh'
set -e
cd /root

echo "=================================================================="
echo "Adam PoC — H100 Resume (auto-detect latest checkpoint)"
echo "Start: $(date)"
echo "=================================================================="

# ── Auto-detect latest checkpoint ─────────────────────────────────────────
LATEST_CKPT=$(ls -d adam_poc_checkpoints/checkpoint-* 2>/dev/null \
    | sort -t- -k2 -n | tail -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in adam_poc_checkpoints/. Aborting."
    exit 1
fi

echo "Latest checkpoint: $LATEST_CKPT"

# ── Stage 1: Regenerate corpus (improved with span-extraction patterns) ────
echo ""
echo "STAGE 1: Corpus generation (6B tokens, 32 workers)"
echo "------------------------------------------------------------------"
python3 data/data_forge_pretrain.py \
    --output-dir adam_training_data/pretrain_corpus \
    --val-output  adam_training_data/pretrain_val.jsonl \
    --total-tokens 6000000000 \
    --val-tokens   10000000 \
    --shard-size   100000000 \
    --workers      32 \
    2>&1 | tee data_gen.log

echo ""
echo "STAGE 1 COMPLETE — $(date)"

# ── Stage 2: Resume pretraining from latest checkpoint ────────────────────
echo ""
echo "STAGE 2: Resuming pretraining from $LATEST_CKPT"
echo "  (LR schedule preserved — same batch_size=4 as original run)"
echo "------------------------------------------------------------------"
python3 pretrain_adam_poc.py \
    --data      adam_training_data/pretrain_corpus \
    --val-data  adam_training_data/pretrain_val.jsonl \
    --output    adam_poc_checkpoints \
    --batch-size 4 \
    --seq-len    2048 \
    --total-tokens 6000000000 \
    --resume    "$LATEST_CKPT" \
    --save-every   500 \
    --keep-checkpoints 5 \
    --val-every    2000 \
    --log-every    10 \
    --grad-flow-every 200 \
    --scaling-every   500 \
    2>&1 | tee pretrain_resume.log

echo ""
echo "=================================================================="
echo "PIPELINE COMPLETE — $(date)"
echo "=================================================================="
