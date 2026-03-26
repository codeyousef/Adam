#!/bin/bash
set -e

cd /data
source /data/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "============================================"
echo "  Resuming SimPO Training on H100"
echo "============================================"
echo "  SFT checkpoint: hope/adam_sft_checkpoints/final"
echo "  Data: hope/adam_training_data/adam_simpo_persona.jsonl"
echo "  Resume from: hope/cloud_backup/simpo_checkpoint-250/checkpoint-250"
echo "  Max steps: 2000, Beta: 2.0"
echo "============================================"

python train_adam_simpo.py \
  --sft-checkpoint hope/adam_sft_checkpoints/final \
  --data hope/adam_training_data/adam_simpo_persona.jsonl \
  --output hope/adam_simpo_checkpoints \
  --max-steps 2000 --beta 2.0 \
  --resume hope/cloud_backup/simpo_checkpoint-250/checkpoint-250

echo ""
echo "SimPO training complete!"
echo "Next: Run DAFT training"
echo ""

# After SimPO completes, start DAFT
echo "============================================"
echo "  Starting DAFT Training"
echo "============================================"

python train_adam_daft.py \
  --data hope/adam_training_data/adam_daft_ready.jsonl \
  --output hope/adam_daft_checkpoints \
  --simpo-checkpoint hope/adam_simpo_checkpoints/final \
  --steps 3000

echo ""
echo "============================================"
echo "  All training complete!"
echo "============================================"
