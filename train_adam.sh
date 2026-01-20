#!/bin/bash
#===============================================================================
# ADAM TRAINING PIPELINE
# Bulletproof training for Parametric Ignorance
#
# Usage:
#   ./train_adam.sh              # Run full pipeline (auto-setup environment)
#   ./train_adam.sh --resume     # Resume from last checkpoint
#   ./train_adam.sh --phase 2    # Start from specific phase
#   ./train_adam.sh --data-only  # Only generate data
#   ./train_adam.sh --setup-only # Only setup environment
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/adam_logs"
STATE_FILE="${SCRIPT_DIR}/.adam_training_state"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Environment configuration
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_VERSION="3.11"

# Training directories
DATA_DIR="${SCRIPT_DIR}/adam_training_data"
SFT_DIR="${SCRIPT_DIR}/adam_sft_checkpoints"
SIMPO_DIR="${SCRIPT_DIR}/adam_simpo_checkpoints"
FINAL_DIR="${SCRIPT_DIR}/adam_final"

# Default hyperparameters (can be overridden via environment)
SFT_MAX_STEPS=${SFT_MAX_STEPS:-10000}
SIMPO_MAX_STEPS=${SIMPO_MAX_STEPS:-5000}
SFT_LR=${SFT_LR:-2e-5}
SIMPO_LR=${SIMPO_LR:-1e-6}
SIMPO_BETA=${SIMPO_BETA:-2.0}
SIMPO_GAMMA=${SIMPO_GAMMA:-1.0}

# HuggingFace Hub (optional - set to push final model)
# Export these or set in .env file
export HF_TOKEN="${HF_TOKEN:-}"
export HF_REPO_ID="${HF_REPO_ID:-}"

#===============================================================================
# UTILITY FUNCTIONS
#===============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

save_state() {
    echo "$1" > "${STATE_FILE}"
    log_info "Saved state: $1"
}

load_state() {
    if [[ -f "${STATE_FILE}" ]]; then
        cat "${STATE_FILE}"
    else
        echo "none"
    fi
}

find_latest_checkpoint() {
    local dir="$1"
    if [[ -d "${dir}" ]]; then
        local latest=$(ls -d "${dir}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        if [[ -n "${latest}" ]]; then
            echo "${latest}"
        fi
    fi
}

#===============================================================================
# ENVIRONMENT SETUP
#===============================================================================

install_uv() {
    log_step "Installing uv package manager..."

    if command -v uv &> /dev/null; then
        log_info "uv already installed: $(uv --version)"
        return 0
    fi

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        log_success "uv installed successfully: $(uv --version)"
    else
        log_error "Failed to install uv"
        exit 1
    fi
}

setup_venv() {
    log_step "Setting up Python virtual environment..."

    if [[ -d "${VENV_DIR}" ]] && [[ -f "${VENV_DIR}/bin/activate" ]]; then
        log_info "Virtual environment already exists at ${VENV_DIR}"
        return 0
    fi

    log_info "Creating virtual environment with Python ${PYTHON_VERSION}..."
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"

    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        log_success "Virtual environment created"
    else
        log_error "Failed to create virtual environment"
        exit 1
    fi
}

activate_venv() {
    log_step "Activating virtual environment..."

    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        log_error "Virtual environment not found. Run setup first."
        exit 1
    fi

    source "${VENV_DIR}/bin/activate"
    log_info "Activated: $(which python)"
}

install_requirements() {
    log_step "Installing Python dependencies..."

    # Check if already installed by looking for key packages
    if python -c "import torch; import transformers; import peft; import trl" 2>/dev/null; then
        log_info "Core dependencies already installed"

        # Quick version check
        python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
        python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
        return 0
    fi

    log_info "Installing PyTorch with CUDA support (>=2.6 for CVE-2025-32434 fix)..."
    uv pip install 'torch>=2.6' 'torchvision>=0.21' 'torchaudio>=2.6' --index-url https://download.pytorch.org/whl/cu124

    log_info "Installing training dependencies..."
    uv pip install \
        transformers \
        datasets \
        accelerate \
        peft \
        trl \
        bitsandbytes \
        scipy \
        sentencepiece \
        protobuf

    log_info "Installing evaluation dependencies..."
    uv pip install \
        scikit-learn \
        pandas \
        matplotlib \
        seaborn

    # Verify installation
    log_info "Verifying installation..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')

import transformers
print(f'Transformers: {transformers.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

import trl
print(f'TRL: {trl.__version__}')

import bitsandbytes
print(f'BitsAndBytes: {bitsandbytes.__version__}')
"

    log_success "All dependencies installed"
}

check_gpu() {
    log_step "Checking GPU availability..."

    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found. Training will be slow without GPU."
        return 0
    fi

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    log_success "GPU detected: ${GPU_NAME} (${GPU_MEM})"

    # Check CUDA in Python
    python -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  [{i}] {props.name}: {props.total_memory / 1024**3:.1f} GB')
else:
    print('  WARNING: CUDA not available in PyTorch!')
"
}

full_setup() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              ENVIRONMENT SETUP                            ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""

    install_uv
    setup_venv
    activate_venv
    install_requirements
    check_gpu

    log_success "Environment setup complete!"
    echo ""
}

ensure_environment() {
    # Quick check if environment is ready
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        source "${VENV_DIR}/bin/activate"

        if python -c "import torch; import transformers; import peft; import trl" 2>/dev/null; then
            log_info "Environment ready"
            return 0
        fi
    fi

    # Need to setup
    log_warning "Environment not ready, running setup..."
    full_setup
}

#===============================================================================
# TRAINING PHASES
#===============================================================================

phase_0_data() {
    log_info "=========================================="
    log_info "PHASE 0: DATA GENERATION"
    log_info "=========================================="

    if [[ -f "${DATA_DIR}/adam_sft_data.jsonl" ]] && [[ -f "${DATA_DIR}/adam_preference_data.jsonl" ]]; then
        log_info "Training data already exists at ${DATA_DIR}"
        read -p "Regenerate data? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping data generation"
            return 0
        fi
    fi

    log_info "Generating training data..."

    python "${SCRIPT_DIR}/data_forge_adam.py" \
        --target 50000 \
        --output-dir "${DATA_DIR}" \
        2>&1 | tee "${LOG_DIR}/data_generation_${TIMESTAMP}.log"

    if [[ -f "${DATA_DIR}/adam_sft_data.jsonl" ]]; then
        local sft_count=$(wc -l < "${DATA_DIR}/adam_sft_data.jsonl")
        local pref_count=$(wc -l < "${DATA_DIR}/adam_preference_data.jsonl")
        log_success "Data generation complete: ${sft_count} SFT samples, ${pref_count} preference pairs"
        save_state "data_complete"
    else
        log_error "Data generation failed"
        exit 1
    fi
}

phase_1_sft() {
    log_info "=========================================="
    log_info "PHASE 1: COUNTERFACTUAL SFT"
    log_info "=========================================="
    log_info "Learning Rate: ${SFT_LR}"
    log_info "Max Steps: ${SFT_MAX_STEPS}"
    log_info "Output: ${SFT_DIR}"

    # Check if data exists
    if [[ ! -f "${DATA_DIR}/adam_sft_data.jsonl" ]]; then
        log_error "SFT data not found. Run data generation first."
        exit 1
    fi

    # Check for existing checkpoint to resume
    local resume_arg=""
    local latest_ckpt=$(find_latest_checkpoint "${SFT_DIR}")
    if [[ -n "${latest_ckpt}" ]]; then
        log_info "Found existing checkpoint: ${latest_ckpt}"
        read -p "Resume from checkpoint? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            resume_arg="--resume ${latest_ckpt}"
            log_info "Resuming from ${latest_ckpt}"
        fi
    fi

    save_state "sft_running"

    python "${SCRIPT_DIR}/train_adam_sft.py" \
        --data "${DATA_DIR}/adam_sft_data.jsonl" \
        --output "${SFT_DIR}" \
        --max-steps ${SFT_MAX_STEPS} \
        --lr ${SFT_LR} \
        ${resume_arg} \
        2>&1 | tee "${LOG_DIR}/sft_training_${TIMESTAMP}.log"

    # Check if training completed successfully
    if [[ -d "${SFT_DIR}/final" ]]; then
        log_success "Phase 1 SFT complete"
        save_state "sft_complete"

        # Report validation results
        if [[ -f "${SFT_DIR}/validation/cpi_history.json" ]]; then
            log_info "Validation history saved to ${SFT_DIR}/validation/"
        fi
    else
        log_error "Phase 1 SFT failed or was interrupted"
        save_state "sft_failed"
        exit 1
    fi
}

phase_2_simpo() {
    log_info "=========================================="
    log_info "PHASE 2: SimPO PREFERENCE OPTIMIZATION"
    log_info "=========================================="
    log_info "Learning Rate: ${SIMPO_LR}"
    log_info "Beta: ${SIMPO_BETA}, Gamma: ${SIMPO_GAMMA}"
    log_info "Max Steps: ${SIMPO_MAX_STEPS}"
    log_info "Output: ${SIMPO_DIR}"

    # Check if SFT checkpoint exists
    local sft_checkpoint="${SFT_DIR}/final"
    if [[ ! -d "${sft_checkpoint}" ]]; then
        log_error "SFT checkpoint not found at ${sft_checkpoint}"
        log_error "Run Phase 1 first."
        exit 1
    fi

    # Check if preference data exists
    if [[ ! -f "${DATA_DIR}/adam_preference_data.jsonl" ]]; then
        log_error "Preference data not found. Run data generation first."
        exit 1
    fi

    # Check for existing checkpoint to resume
    local resume_arg=""
    local latest_ckpt=$(find_latest_checkpoint "${SIMPO_DIR}")
    if [[ -n "${latest_ckpt}" ]]; then
        log_info "Found existing SimPO checkpoint: ${latest_ckpt}"
        read -p "Resume from checkpoint? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            resume_arg="--resume ${latest_ckpt}"
            log_info "Resuming from ${latest_ckpt}"
        fi
    fi

    save_state "simpo_running"

    python "${SCRIPT_DIR}/train_adam_simpo.py" \
        --sft-checkpoint "${sft_checkpoint}" \
        --data "${DATA_DIR}/adam_preference_data.jsonl" \
        --output "${SIMPO_DIR}" \
        --max-steps ${SIMPO_MAX_STEPS} \
        --lr ${SIMPO_LR} \
        --beta ${SIMPO_BETA} \
        --gamma ${SIMPO_GAMMA} \
        ${resume_arg} \
        2>&1 | tee "${LOG_DIR}/simpo_training_${TIMESTAMP}.log"

    # Check if training completed successfully
    if [[ -d "${SIMPO_DIR}/final" ]]; then
        log_success "Phase 2 SimPO complete"
        save_state "simpo_complete"
    else
        log_error "Phase 2 SimPO failed or was interrupted"
        save_state "simpo_failed"
        exit 1
    fi
}

phase_3_finalize() {
    log_info "=========================================="
    log_info "PHASE 3: FINALIZE MODEL"
    log_info "=========================================="

    # Determine best model
    local best_model=""

    if [[ -d "${SIMPO_DIR}/final" ]]; then
        best_model="${SIMPO_DIR}/final"
    elif [[ -d "${SFT_DIR}/final" ]]; then
        log_warning "SimPO not complete, using SFT model"
        best_model="${SFT_DIR}/final"
    else
        log_error "No trained model found"
        exit 1
    fi

    log_info "Best model: ${best_model}"

    # Copy to final directory
    mkdir -p "${FINAL_DIR}"
    cp -r "${best_model}"/* "${FINAL_DIR}/"

    # Run final validation
    log_info "Running final validation..."
    python -c "
from validation_probes import run_validation, LEVEL1_PROBES, LEVEL2_PROBES, LEVEL3_PROBES, LEVEL4_PROBES
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import json

print('Loading final model...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-Coder-3B-Instruct',
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, '${FINAL_DIR}')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-3B-Instruct', trust_remote_code=True)

print('Running final validation...')
report = run_validation(model, tokenizer, step=0, levels=[1,2,3,4])

print()
print('='*50)
print('FINAL MODEL VALIDATION RESULTS')
print('='*50)
print(f'Level 1 (Basic Override):      {report.level1_accuracy:.1%}')
print(f'Level 2 (Numerical Override):  {report.level2_accuracy:.1%}')
print(f'Level 3 (Underdetermined):     {report.level3_accuracy:.1%}')
print(f'Level 4 (Constraint):          {report.level4_accuracy:.1%}')
print(f'Counterfactual Accuracy (A_CF): {report.counterfactual_accuracy:.1%}')
print('='*50)

# Save final report
with open('${FINAL_DIR}/final_validation.json', 'w') as f:
    json.dump(report.to_dict(), f, indent=2)
"

    log_success "Final model saved to ${FINAL_DIR}"
    save_state "complete"

    # Print summary
    echo ""
    echo "=========================================="
    echo "TRAINING COMPLETE"
    echo "=========================================="
    echo "Final model: ${FINAL_DIR}"
    echo "Validation:  ${FINAL_DIR}/final_validation.json"
    echo ""
    echo "Paper metrics:"
    echo "  SFT:   ${SFT_DIR}/paper_metrics/"
    echo "  SimPO: ${SIMPO_DIR}/paper_metrics/"
    echo ""
    echo "To use the model:"
    echo "  from peft import PeftModel"
    echo "  model = PeftModel.from_pretrained(base_model, '${FINAL_DIR}')"
    echo "=========================================="
}

#===============================================================================
# MAIN
#===============================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --resume       Resume from last saved state"
    echo "  --phase N      Start from phase N (0=data, 1=sft, 2=simpo, 3=finalize)"
    echo "  --data-only    Only generate training data"
    echo "  --setup-only   Only setup environment (install dependencies)"
    echo "  --status       Show current training status"
    echo "  --clean        Remove all training artifacts and start fresh"
    echo "  --clean-all    Remove everything including virtual environment"
    echo "  --help         Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  SFT_MAX_STEPS    Max SFT steps (default: 10000)"
    echo "  SIMPO_MAX_STEPS  Max SimPO steps (default: 5000)"
    echo "  SFT_LR           SFT learning rate (default: 2e-5)"
    echo "  SIMPO_LR         SimPO learning rate (default: 1e-6)"
    echo "  SIMPO_BETA       SimPO beta (default: 2.0)"
    echo "  SIMPO_GAMMA      SimPO gamma/margin (default: 1.0)"
    echo ""
    echo "  HF_TOKEN         HuggingFace token (for pushing final model)"
    echo "  HF_REPO_ID       HuggingFace repo ID (e.g. username/adam-model)"
}

main() {
    # Parse arguments
    local start_phase=0
    local resume=false
    local data_only=false
    local setup_only=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --resume)
                resume=true
                shift
                ;;
            --phase)
                start_phase="$2"
                shift 2
                ;;
            --data-only)
                data_only=true
                shift
                ;;
            --setup-only)
                setup_only=true
                shift
                ;;
            --status)
                local state=$(load_state)
                echo "Current state: ${state}"
                if [[ -d "${VENV_DIR}" ]]; then
                    echo "Environment: ${VENV_DIR} (exists)"
                else
                    echo "Environment: not setup"
                fi
                exit 0
                ;;
            --clean)
                log_warning "This will delete all training artifacts!"
                read -p "Are you sure? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf "${DATA_DIR}" "${SFT_DIR}" "${SIMPO_DIR}" "${FINAL_DIR}" "${STATE_FILE}" "${LOG_DIR}"
                    log_info "Cleaned all training artifacts"
                fi
                exit 0
                ;;
            --clean-all)
                log_warning "This will delete EVERYTHING including the virtual environment!"
                read -p "Are you sure? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf "${DATA_DIR}" "${SFT_DIR}" "${SIMPO_DIR}" "${FINAL_DIR}" "${STATE_FILE}" "${LOG_DIR}" "${VENV_DIR}"
                    log_info "Cleaned everything"
                fi
                exit 0
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Create log directory
    mkdir -p "${LOG_DIR}"

    # Header
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                  ADAM TRAINING PIPELINE                   ║"
    echo "║         Bulletproof Parametric Ignorance Training         ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""

    # Setup-only mode
    if [[ "${setup_only}" == true ]]; then
        full_setup
        exit 0
    fi

    # Ensure environment is ready
    ensure_environment

    # Handle resume
    if [[ "${resume}" == true ]]; then
        local state=$(load_state)
        log_info "Resuming from state: ${state}"

        case ${state} in
            "data_complete")
                start_phase=1
                ;;
            "sft_running"|"sft_failed")
                start_phase=1
                ;;
            "sft_complete")
                start_phase=2
                ;;
            "simpo_running"|"simpo_failed")
                start_phase=2
                ;;
            "simpo_complete")
                start_phase=3
                ;;
            "complete")
                log_info "Training already complete!"
                exit 0
                ;;
            *)
                log_info "No previous state found, starting from beginning"
                start_phase=0
                ;;
        esac
    fi

    # Data-only mode
    if [[ "${data_only}" == true ]]; then
        phase_0_data
        exit 0
    fi

    # Check GPU
    check_gpu

    # Run phases
    if [[ ${start_phase} -le 0 ]]; then
        phase_0_data
    fi

    if [[ ${start_phase} -le 1 ]]; then
        phase_1_sft
    fi

    if [[ ${start_phase} -le 2 ]]; then
        phase_2_simpo
    fi

    if [[ ${start_phase} -le 3 ]]; then
        phase_3_finalize
    fi

    log_success "All phases complete!"
}

# Run main
main "$@"
