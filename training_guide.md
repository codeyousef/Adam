# Adam Mamba Training Guide

Complete guide to set up and train a Mamba 2.7B model on NVIDIA Blackwell (B200) GPUs.

---

## Prerequisites

- Ubuntu 22.04 (or compatible Linux)
- NVIDIA Blackwell GPU (B200, sm_100/sm_120)
- CUDA 12.8+ installed system-wide
- Python 3.10
- `uv` package manager

---

## Step 1: Install System Dependencies

```bash
# Install Python dev headers (required for compiling CUDA extensions)
apt-get update && apt-get install -y python3.10-dev
```

---

## Step 2: Create Virtual Environment

```bash
cd /data
uv venv adam_env --python 3.10
source adam_env/bin/activate
```

---

## Step 3: Install PyTorch Nightly (Blackwell Support)

Standard PyTorch releases do not support Blackwell GPUs (sm_100). You **must** use the nightly build with CUDA 12.8:

```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Verify Blackwell Support

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

Expected output should include `sm_100` and/or `sm_120`:
```
['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

---

## Step 4: Install Build Dependencies

```bash
uv pip install ninja wheel setuptools
```

---

## Step 5: Build Mamba & Causal-Conv1d from Source

Pre-built wheels are not compatible with PyTorch nightly. Compile from source:

```bash
uv pip install --no-binary causal-conv1d,mamba-ssm --no-build-isolation causal-conv1d>=1.4.0 mamba-ssm
```

> ⚠️ **Note**: This compilation takes 10-30 minutes depending on your system.

---

## Step 6: Install Remaining Dependencies

```bash
uv pip install datasets spacy einops transformers huggingface_hub
```

---

## Step 7: Verify Installation

```bash
python << 'EOF'
import torch
print('PyTorch:', torch.__version__)
print('CUDA archs:', torch.cuda.get_arch_list())
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
import causal_conv1d
print('causal_conv1d OK')
import mamba_ssm
print('mamba_ssm OK')
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
print('MambaLMHeadModel imported successfully')
print()
print('🟢 SYSTEM SAFE')
EOF
```

---

## Step 8: Generate Training Data

Run the data forge script to create the training dataset:

```bash
python data_forge.py
```

This creates `adam_skeleton_data.jsonl` with ~5M samples from:
- Wikipedia (general knowledge)
- Cosmopedia (structured explanations)
- Orca Math (mathematical reasoning)
- The Stack (code from multiple languages)

---

## Step 9: Configure Training Script

Edit `train_adam_phoenix.py` and set your HuggingFace credentials:

```python
HF_REPO_ID = "YOUR_USERNAME/adam-mamba-2.7b-logic"  # Create this repo on HF first
HF_TOKEN = "hf_..."                                   # Your write token
```

---

## Step 10: Start Training

### Option A: Direct Run

```bash
source adam_env/bin/activate
python train_adam_phoenix.py
```

### Option B: Persistent Training with Watchdog (Recommended)

Use the eternal watchdog script for automatic restart on failures:

```bash
# Start in a tmux session
tmux new -s adam
./eternal_adam.sh
```

The watchdog will:
- Automatically restart training on crashes
- Monitor GPU temperature (pause if > 80°C)
- Upload checkpoints to HuggingFace hourly
- Log all metrics to CSV

---

## Quick Reference: Full Setup Commands

```bash
# One-shot setup (copy-paste friendly)
apt-get update && apt-get install -y python3.10-dev

cd /data
uv venv adam_env --python 3.10
source adam_env/bin/activate

uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install ninja wheel setuptools
uv pip install --no-binary causal-conv1d,mamba-ssm --no-build-isolation causal-conv1d>=1.4.0 mamba-ssm
uv pip install datasets spacy einops transformers huggingface_hub

# Verify
python -c "import torch; print('sm_100' in torch.cuda.get_arch_list())"
python -c "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel; print('OK')"
```

---

## Troubleshooting

### Error: `NVIDIA B200 with CUDA capability sm_100 is not compatible`

**Cause**: Using stable PyTorch instead of nightly.

**Fix**: Install from nightly cu128 index:
```bash
uv pip uninstall torch torchvision torchaudio
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Error: `Python.h: No such file or directory`

**Cause**: Missing Python development headers.

**Fix**:
```bash
apt-get install -y python3.10-dev
```

### Error: `undefined symbol` when importing mamba_ssm

**Cause**: Pre-built wheels incompatible with PyTorch version (ABI mismatch).

**Fix**: Rebuild from source:
```bash
uv pip uninstall causal-conv1d mamba-ssm
uv pip install --no-binary causal-conv1d,mamba-ssm --no-build-isolation causal-conv1d>=1.4.0 mamba-ssm
```

### Error: `Invalid pattern: '**' can only be an entire path component`

**Cause**: Old `datasets` library version with fsspec incompatibility.

**Fix**:
```bash
uv pip install --upgrade datasets fsspec
```

### Training crashes with OOM

**Fix**: Reduce batch size in `train_adam_phoenix.py`:
```python
BATCH_SIZE = 4  # or lower
GRAD_ACCUM = 8  # increase to compensate
```

---

## File Overview

| File | Purpose |
|------|---------|
| `data_forge.py` | Generates training data from HF datasets |
| `train_adam_phoenix.py` | Main training script with checkpointing |
| `eternal_adam.sh` | Watchdog for persistent training |
| `adam_skeleton_data.jsonl` | Generated training data |
| `adam_checkpoints/` | Local checkpoint storage |
| `adam_research_metrics.csv` | Training telemetry |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | B200 (192GB) | B200 |
| RAM | 64GB | 128GB |
| Storage | 500GB SSD | 1TB NVMe |
| CUDA | 12.8 | 12.8+ |

---

## Version Reference (Tested Configuration)

```
torch==2.11.0.dev20260109+cu128
mamba-ssm==2.2.6.post3
causal-conv1d==1.5.3.post1
datasets==4.4.2
transformers==4.57.3
```
