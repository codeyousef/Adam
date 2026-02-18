#!/bin/bash
# Fresh H200 Setup - No persistent /data
set -e

H200_IP="204.12.171.168"
SSH_KEY="$HOME/.ssh/id_adam"
SSH_CMD="ssh -i $SSH_KEY ubuntu@$H200_IP"

echo "=========================================="
echo "  Fresh H200 Setup - L3 Fix v3"
echo "=========================================="
echo "  Target: $H200_IP"
echo "  User: ubuntu (fresh instance, no /data persistence)"
echo ""

# Step 1: Create directory structure
echo "[1/6] Creating directory structure on H200..."
$SSH_CMD "mkdir -p /data/hope/adam_sft_checkpoints /data/hope/adam_training_data"
echo "  ✓ Directories created"

# Step 2: Upload SFT checkpoint (472M - will take a few minutes)
echo "[2/6] Uploading SFT checkpoint (472M, ~2-3 minutes)..."
rsync -az --progress -e "ssh -i $SSH_KEY" \
    hope/adam_sft_checkpoints/final/ \
    ubuntu@$H200_IP:/data/hope/adam_sft_checkpoints/final/
echo "  ✓ SFT checkpoint uploaded"

# Step 3: Upload training scripts
echo "[3/6] Uploading fixed training scripts..."
rsync -az -e "ssh -i $SSH_KEY" \
    train_adam_simpo.py \
    train_adam_daft.py \
    run_validation_daft.py \
    validation_probes.py \
    daft_model.py \
    ubuntu@$H200_IP:/data/
echo "  ✓ Scripts uploaded"

# Step 4: Upload verified training data
echo "[4/6] Uploading verified training data..."
rsync -az -e "ssh -i $SSH_KEY" \
    hope/adam_training_data/adam_simpo_balanced_v3_quick.jsonl \
    ubuntu@$H200_IP:/data/hope/adam_training_data/
VERIFIED_COUNT=$(wc -l < hope/adam_training_data/adam_simpo_balanced_v3_quick.jsonl)
echo "  ✓ Uploaded $VERIFIED_COUNT verified training examples"

# Step 5: Set up Python environment
echo "[5/6] Setting up Python environment on H200..."
$SSH_CMD bash << 'REMOTE_SETUP'
set -e
cd /data

# Create venv if it doesn't exist
if [ ! -d "/data/venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA 12.8
    pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    
    # Install training dependencies
    pip install transformers==5.1.0 datasets==2.15.0 accelerate==0.27.0 peft==0.8.2
    pip install trl==0.27.2 bitsandbytes==0.43.0 scipy scikit-learn sentencepiece
    
    echo "✓ Python environment ready"
else
    echo "✓ Python environment already exists"
fi
REMOTE_SETUP

echo "  ✓ Python environment ready"

# Step 6: Create training script
echo "[6/6] Creating training script on H200..."
$SSH_CMD "cat > /data/train_simpo_v3.sh" << 'TRAIN_SCRIPT'
#!/bin/bash
set -e

cd /data
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "============================================"
echo "  SimPO v3 - L3 Fix Training"
echo "============================================"
echo "  Date: $(date)"
echo "  SFT checkpoint: hope/adam_sft_checkpoints/final"
echo "  Data: adam_simpo_balanced_v3_quick.jsonl (1384 examples)"
echo "  Output: hope/adam_simpo_checkpoints_v3"
echo ""
echo "  Fixes implemented:"
echo "  - ReplayDataCollator integration ✓"
echo "  - Balanced 1:1 data (4+4 patterns) ✓"
echo "  - gamma_l3=0.3 ✓"
echo "  - l3_replay_ratio=0.15 ✓"
echo "============================================"
echo ""

python train_adam_simpo.py \
    --sft-checkpoint hope/adam_sft_checkpoints/final \
    --data hope/adam_training_data/adam_simpo_balanced_v3_quick.jsonl \
    --output hope/adam_simpo_checkpoints_v3 \
    --max-steps 2000 \
    --beta 2.0

echo ""
echo "============================================"
echo "  SimPO v3 COMPLETE"
echo "============================================"
date
echo ""
echo "Next: Run validation to check L3 accuracy"
echo "  python run_validation_daft.py --checkpoint hope/adam_simpo_checkpoints_v3/final --all"
TRAIN_SCRIPT

$SSH_CMD "chmod +x /data/train_simpo_v3.sh"

echo ""
echo "=========================================="
echo "  SETUP COMPLETE"
echo "=========================================="
echo ""
echo "  All files uploaded to H200"
echo "  Training script ready: /data/train_simpo_v3.sh"
echo ""
echo "To start training:"
echo "  ssh -i ~/.ssh/id_adam ubuntu@$H200_IP"
echo "  tmux new-session -s adam_training"
echo "  bash /data/train_simpo_v3.sh 2>&1 | tee /data/training_v3.log"
echo ""
echo "To monitor remotely:"
echo "  ssh -i ~/.ssh/id_adam ubuntu@$H200_IP 'tail -f /data/training_v3.log'"
echo ""
echo "Expected completion: ~1.7 hours (2000 steps @ 3s/step)"
echo "Cost: ~$5.35 @ $3.14/hour"
echo ""
echo "=========================================="
