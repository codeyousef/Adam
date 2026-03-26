#!/bin/bash
set -e

echo "============================================"
echo "  Adam H100 Training Environment Setup"
echo "============================================"

# Work from /data
cd /data

# Install system dependencies
apt-get update && apt-get install -y tmux htop nvtop git

# Create Python virtual environment
python3 -m venv /data/venv
source /data/venv/bin/activate

# Install PyTorch with CUDA 12.4 support
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install training dependencies
pip install -r /data/requirements.txt

# Verify setup
echo ""
echo "============================================"
echo "  Verifying Installation"
echo "============================================"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'Flash Attention: supported')
import transformers
print(f'Transformers: {transformers.__version__}')
import trl
print(f'TRL: {trl.__version__}')
import peft
print(f'PEFT: {peft.__version__}')
import bitsandbytes
print(f'BitsAndBytes: {bitsandbytes.__version__}')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo "To activate: source /data/venv/bin/activate"
echo "To start training: bash /data/start_training.sh"
