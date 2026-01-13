#!/bin/bash

# =============================================================================
# ETERNAL ADAM V2 - Curriculum Learning Pipeline
# Based on docs/private/Data Curation.md
# =============================================================================

# --- CONFIG ---
LOG_FILE="adam_curriculum_training.log"
HF_TOKEN="hf"
SAFE_TEMP=80
POWER_CAP=600  # Conservative for B200 while you sleep
MAX_RETRIES=100
DATA_DIR="adam_curriculum_data"

# Training mode: "fresh" starts new training, "resume" continues from checkpoint
# Set to "fresh" for first run with new curriculum data
TRAIN_MODE="${1:-fresh}"  # Pass "resume" as argument to continue

# --- 0. PRE-FLIGHT CHECK ---
source adam_env/bin/activate
echo "Adam Curriculum Training Initiated" | tee $LOG_FILE
echo "Following: docs/private/Data Curation.md" | tee -a $LOG_FILE
echo "============================================" | tee -a $LOG_FILE

# --- 1. ARCHIVE OLD DATA (for paper comparison) ---
OLD_DATA_FILE="adam_skeleton_data.jsonl"
if [ -f "$OLD_DATA_FILE" ]; then
    echo "Archiving old data file for paper comparison..." | tee -a $LOG_FILE
    mkdir -p baseline_data
    mv "$OLD_DATA_FILE" "baseline_data/adam_skeleton_data_v1.jsonl"
    echo "  Moved to baseline_data/adam_skeleton_data_v1.jsonl" | tee -a $LOG_FILE
fi

# --- 2. CHECK/GENERATE CURRICULUM DATA ---
if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/phase1_axiomatic.jsonl" ] && [ -f "$DATA_DIR/phase2_algorithmic.jsonl" ] && [ -f "$DATA_DIR/phase3_crystallization.jsonl" ]; then
    echo "Curriculum data exists, skipping forge." | tee -a $LOG_FILE
    echo "  Phase 1: $(wc -l < $DATA_DIR/phase1_axiomatic.jsonl) samples" | tee -a $LOG_FILE
    echo "  Phase 2: $(wc -l < $DATA_DIR/phase2_algorithmic.jsonl) samples" | tee -a $LOG_FILE
    echo "  Phase 3: $(wc -l < $DATA_DIR/phase3_crystallization.jsonl) samples" | tee -a $LOG_FILE
else
    echo "Starting Data Forge V4 (Pure Logic Distillation)..." | tee -a $LOG_FILE
    python data_forge.py 2>&1 | tee -a $LOG_FILE

    if [ $? -ne 0 ]; then
        echo "Forge crashed! Aborting." | tee -a $LOG_FILE
        exit 1
    fi

    # Verify all phases were created
    if [ ! -f "$DATA_DIR/phase1_axiomatic.jsonl" ] || [ ! -f "$DATA_DIR/phase2_algorithmic.jsonl" ] || [ ! -f "$DATA_DIR/phase3_crystallization.jsonl" ]; then
        echo "Forge incomplete! Missing phase files." | tee -a $LOG_FILE
        exit 1
    fi

    echo "Forge Complete! Curriculum data ready." | tee -a $LOG_FILE
fi

# --- 3. HARDWARE SAFETY ---
echo "Enforcing Power Cap: ${POWER_CAP}W..." | tee -a $LOG_FILE
nvidia-smi -i 0 -pl $POWER_CAP 2>/dev/null || echo "Power cap not supported on this GPU" | tee -a $LOG_FILE

# --- 4. THE WATCHDOG (TRAINING LOOP) ---
RETRY_COUNT=0

while true; do
    # Thermal Check
    CURRENT_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0")
    if [ "$CURRENT_TEMP" -ge "$SAFE_TEMP" ]; then
        echo "GPU too hot (${CURRENT_TEMP}C). Sleeping 60s..." | tee -a $LOG_FILE
        sleep 60
        continue
    fi

    echo "" | tee -a $LOG_FILE
    echo "============================================" | tee -a $LOG_FILE
    echo "Starting Curriculum Training (Attempt $RETRY_COUNT, Mode: $TRAIN_MODE)..." | tee -a $LOG_FILE
    echo "============================================" | tee -a $LOG_FILE

    # Run Training with appropriate flag
    if [ "$TRAIN_MODE" = "fresh" ] && [ $RETRY_COUNT -eq 0 ]; then
        # First attempt with fresh mode - archives old checkpoints as baseline
        python -u train_adam_phoenix.py --fresh 2>&1 | tee -a $LOG_FILE
        # After first attempt, switch to resume mode for retries
        TRAIN_MODE="resume"
    else
        python -u train_adam_phoenix.py --resume 2>&1 | tee -a $LOG_FILE
    fi

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "" | tee -a $LOG_FILE
        echo "============================================" | tee -a $LOG_FILE
        echo "CURRICULUM TRAINING COMPLETE!" | tee -a $LOG_FILE
        echo "============================================" | tee -a $LOG_FILE
        break
    elif [ $EXIT_CODE -eq 2 ]; then
        echo "" | tee -a $LOG_FILE
        echo "DIVERGENCE FAILURE: Loss spiked. Data or LR problem." | tee -a $LOG_FILE
        echo "This is a fatal error - not retrying." | tee -a $LOG_FILE
        echo "Check adam_research_metrics.csv for diagnostics." | tee -a $LOG_FILE
        break
    elif [ $EXIT_CODE -eq 130 ]; then
        echo "" | tee -a $LOG_FILE
        echo "User interrupted (Ctrl+C). Stopping." | tee -a $LOG_FILE
        break
    else
        echo "" | tee -a $LOG_FILE
        echo "Crashed (Exit $EXIT_CODE). Restarting in 10s..." | tee -a $LOG_FILE
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
            echo "Too many crashes. Giving up." | tee -a $LOG_FILE
            break
        fi
        sleep 10
    fi
done

echo "" | tee -a $LOG_FILE
echo "Training session ended at $(date)" | tee -a $LOG_FILE
