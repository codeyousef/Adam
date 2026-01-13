#!/bin/bash

# --- CONFIG ---
LOG_FILE="adam_sleep_mode.log"
HF_TOKEN="hf"
SAFE_TEMP=80
POWER_CAP=600  # Conservative for B200 while you sleep
MAX_RETRIES=100

# --- 0. PRE-FLIGHT CHECK ---
source adam_env/bin/activate
echo "🐕 Adam Sleep Mode Initiated (Resume Mode)." | tee $LOG_FILE

# --- 1. SKIP DELETION - PRESERVE EXISTING DATA ---
echo "📂 Preserving existing checkpoints and data..." | tee -a $LOG_FILE

# --- 2. THE FORGE (DATA GENERATION) - SKIP IF DATA EXISTS ---
if [ -f "adam_skeleton_data.jsonl" ]; then
    echo "✅ Data file exists, skipping forge." | tee -a $LOG_FILE
else
    echo "⚒️  Starting Data Forge V3 (Pure Logic Distillation)..." | tee -a $LOG_FILE
    python data_forge.py

    if [ $? -ne 0 ]; then
        echo "❌ Forge crashed! Aborting." | tee -a $LOG_FILE
        exit 1
    fi
    echo "✅ Forge Complete! Data is ready." | tee -a $LOG_FILE
fi

# --- 3. HARDWARE SAFETY ---
echo "🧊 Enforcing Power Cap: ${POWER_CAP}W..." | tee -a $LOG_FILE
nvidia-smi -i 0 -pl $POWER_CAP

# --- 4. THE WATCHDOG (TRAINING LOOP) ---
RETRY_COUNT=0

while true; do
    # Thermal Check
    CURRENT_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    if [ "$CURRENT_TEMP" -ge "$SAFE_TEMP" ]; then
        echo "🔥 GPU too hot ($CURRENT_TEMP°C). Sleeping 60s..." | tee -a $LOG_FILE
        sleep 60
        continue
    fi

    echo "🚀 Starting Adam Phoenix Training (Attempt $RETRY_COUNT)..." | tee -a $LOG_FILE
    
    # Run Training
    python -u train_adam_phoenix.py
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "🎉 Training Finished Successfully! Go wake up." | tee -a $LOG_FILE
        break
    elif [ $EXIT_CODE -eq 2 ]; then
        echo "❌ DIVERGENCE FAILURE: Loss spiked. Data or LR problem." | tee -a $LOG_FILE
        echo "   This is a fatal error - not retrying." | tee -a $LOG_FILE
        break
    elif [ $EXIT_CODE -eq 3 ]; then
        echo "❌ STAGNATION FAILURE: Loss plateaued with no improvement." | tee -a $LOG_FILE
        echo "   This is a fatal error - not retrying." | tee -a $LOG_FILE
        break
    elif [ $EXIT_CODE -eq 130 ]; then
        echo "🛑 User interrupted (Ctrl+C). Stopping." | tee -a $LOG_FILE
        break
    else
        echo "⚠️  Crashed (Exit $EXIT_CODE). Restarting in 10s..." | tee -a $LOG_FILE
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
            echo "❌ Too many crashes. Giving up." | tee -a $LOG_FILE
            break
        fi
        sleep 10
    fi
done