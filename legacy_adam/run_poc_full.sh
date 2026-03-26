#!/usr/bin/env bash
# Full Adam PoC pipeline: corpus generation → pretraining
# Run inside tmux: tmux new-session -s adam_pretrain 'bash run_poc_full.sh'

set -e
cd /mnt/Storage/Projects/catbelly_studio
source .venv/bin/activate

echo "=================================================================="
echo "Adam PoC Full Pipeline — $(date)"
echo "=================================================================="

# ── Stage 1: Generate 6B token corpus ────────────────────────────────
echo ""
echo "STAGE 1: Corpus generation (6B tokens, ~1hr on 32 cores)"
echo "------------------------------------------------------------------"
python data/data_forge_pretrain.py \
    --output-dir hope/adam_training_data/pretrain_corpus \
    --val-output  hope/adam_training_data/pretrain_val.jsonl \
    --total-tokens 6000000000 \
    --val-tokens   10000000 \
    --shard-size   100000000 \
    --workers      8 \
    2>&1 | tee hope/data_gen.log

echo ""
echo "STAGE 1 COMPLETE — $(date)"

# ── Stage 2: Pretrain 494M model ─────────────────────────────────────
echo ""
echo "STAGE 2: Pretraining 494M model (~33hr on RTX 4090)"
echo "------------------------------------------------------------------"
PYTORCH_ALLOC_CONF=expandable_segments:True \
python pretrain_adam_poc.py \
    --data      hope/adam_training_data/pretrain_corpus \
    --val-data  hope/adam_training_data/pretrain_val.jsonl \
    --output    adam_poc_checkpoints \
    --batch-size 4 \
    --seq-len    2048 \
    --total-tokens 6000000000 \
    --save-every   500 \
    --val-every    5000 \
    --log-every    10 \
    --grad-flow-every 200 \
    --scaling-every   1000 \
    2>&1 | tee hope/pretrain.log

echo ""
echo "PIPELINE COMPLETE — $(date)"
