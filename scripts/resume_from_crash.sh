#!/bin/bash
# Resume training after spot instance crash
# Usage: ./resume_from_crash.sh <new_instance_ip>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <new_instance_ip>"
    echo "Example: $0 86.38.238.16"
    exit 1
fi

NEW_IP="$1"
SSH_KEY="$HOME/.ssh/id_adam"
SSH_CMD="ssh -i $SSH_KEY root@$NEW_IP"

echo "=========================================="
echo "  H200 Resume from Crash"
echo "=========================================="
echo "  New IP: $NEW_IP"
echo ""

# --- Step 1: Check connectivity ---
echo "[1/6] Checking connectivity..."
if ! $SSH_CMD "echo ok" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to $NEW_IP"
    echo "Make sure:"
    echo "  - Instance is running"
    echo "  - Port 22 is open"
    echo "  - IP is correct"
    exit 1
fi
echo "  Connected successfully."

# --- Step 2: Check /data mount ---
echo "[2/6] Checking /data mount..."
if ! $SSH_CMD "test -d /data/hope"; then
    echo "ERROR: /data drive not mounted or missing hope/"
    echo "Mount the persistent /data volume first."
    exit 1
fi
echo "  /data is mounted."

# --- Step 3: Find latest checkpoint ---
echo "[3/6] Finding latest checkpoint..."
LATEST=$($SSH_CMD "ls -d /data/hope/adam_simpo_checkpoints_v2/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo 'none'")
if [ "$LATEST" = "none" ]; then
    echo "ERROR: No checkpoints found in /data/hope/adam_simpo_checkpoints_v2/"
    exit 1
fi
STEP=$(basename "$LATEST" | grep -oP '\d+')
echo "  Found checkpoint: $LATEST (step $STEP)"

# --- Step 4: Upload updated scripts ---
echo "[4/6] Uploading H200-optimized scripts..."
rsync -az -e "ssh -i $SSH_KEY" \
    train_adam_simpo.py \
    train_adam_daft.py \
    run_validation_daft.py \
    validation_probes.py \
    daft_model.py \
    root@$NEW_IP:/data/

# --- Step 5: Create resume script ---
echo "[5/6] Creating resume script on H200..."
$SSH_CMD "cat > /data/resume_training.sh" << 'REMOTE_SCRIPT'
#!/bin/bash
set -e

cd /data
source /data/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Find latest checkpoint
LATEST=$(ls -d /data/hope/adam_simpo_checkpoints_v2/checkpoint-* 2>/dev/null | sort -V | tail -1)
STEP=$(basename "$LATEST" | grep -oP '\d+')

echo "============================================"
echo "  RESUMING from checkpoint-$STEP"
echo "============================================"
echo "  Data: adam_simpo_balanced_v2.jsonl"
echo "  gamma_l3=0.3, L3 replay=15%"
echo "  H200 141GB: batch=16, grad_accum=4"
echo "  max_steps=2000"
echo "============================================"
echo ""
date
echo ""

python train_adam_simpo.py \
    --sft-checkpoint hope/adam_sft_checkpoints/final \
    --data hope/adam_training_data/adam_simpo_balanced_v2.jsonl \
    --output hope/adam_simpo_checkpoints_v2 \
    --max-steps 2000 \
    --beta 2.0 \
    --resume "$LATEST"

echo ""
echo "============================================"
echo "  SimPO COMPLETE"
echo "============================================"
date
echo ""

echo "============================================"
echo "  Starting DAFT"
echo "============================================"
echo ""

python train_adam_daft.py \
    --data hope/adam_training_data/adam_daft_ready_v2.jsonl \
    --output hope/adam_daft_checkpoints_v2 \
    --simpo-checkpoint hope/adam_simpo_checkpoints_v2/final \
    --steps 3000

echo ""
echo "============================================"
echo "  DAFT COMPLETE"
echo "============================================"
date
echo ""

echo "============================================"
echo "  Running Validation"
echo "============================================"
echo ""

python run_validation_daft.py \
    --checkpoint hope/adam_daft_checkpoints_v2/final \
    --all

echo ""
echo "============================================"
echo "  ALL DONE"
echo "============================================"
date
REMOTE_SCRIPT

$SSH_CMD "chmod +x /data/resume_training.sh"

# --- Step 6: Launch training ---
echo "[6/6] Launching training in tmux..."
$SSH_CMD "tmux kill-session -t adam_training 2>/dev/null || true; tmux new-session -d -s adam_training 'bash /data/resume_training.sh 2>&1 | tee /data/training_resume.log'"

echo ""
echo "=========================================="
echo "  TRAINING RESUMED"
echo "=========================================="
echo ""
echo "  Resumed from: checkpoint-$STEP/2000"
echo "  Monitor: ssh -i ~/.ssh/id_adam root@$NEW_IP 'tmux attach -t adam_training'"
echo "  Log: ssh -i ~/.ssh/id_adam root@$NEW_IP 'tail -f /data/training_resume.log'"
echo ""
echo "  Remaining: ~$((2000 - STEP)) SimPO steps + 3000 DAFT steps"
echo "  ETA: ~3-4 hours on H200"
echo ""
echo "=========================================="
