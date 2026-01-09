#!/bin/bash

# --- CONFIG ---
LOG_FILE="adam_watchdog.log"
MAX_RETRIES=50
RETRY_COUNT=0
SAFE_TEMP=80        # Won't start if hotter than this (°C)
POWER_CAP=600       # Watts (Standard B200 is ~1000W)

# --- ACTIVATION ---
# Ensure we are using the specific environment with the correct PyTorch Nightly
source /root/adam_env/bin/activate

echo "🐕 Adam Watchdog (Thermal Edition) started." | tee -a $LOG_FILE

# 1. APPLY POWER LIMIT (The Critical Fix)
echo "🧊 Enforcing Power Cap: ${POWER_CAP}W..." | tee -a $LOG_FILE
nvidia-smi -i 0 -pl $POWER_CAP

while true; do
    # 2. THERMAL INTERLOCK
    # Check temp before asking the GPU to work hard again
    CURRENT_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    
    if [ "$CURRENT_TEMP" -ge "$SAFE_TEMP" ]; then
        echo "🔥 GPU is too hot ($CURRENT_TEMP°C). Cooling down for 60s..." | tee -a $LOG_FILE
        sleep 60
        continue  # Skip to top of loop to check temp again
    fi

    # 3. RUN TRAINING
    echo "🚀 Starting Adam (Attempt $RETRY_COUNT) [Temp: $CURRENT_TEMP°C]..." | tee -a $LOG_FILE
    
    # Run Python (using the full path to be safe)
    /root/adam_env/bin/python -u train_adam.py
    
    # 4. CAPTURE EXIT CODE
    EXIT_CODE=$?
    
    # 5. DECISION LOGIC
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training Finished Successfully!" | tee -a $LOG_FILE
        break
    else
        echo "⚠️  Adam crashed with exit code $EXIT_CODE." | tee -a $LOG_FILE
        
        # Handle Ctrl+C (SIGINT)
        if [ $EXIT_CODE -eq 130 ]; then
            echo "🛑 Stopped by user."
            break
        fi

        RETRY_COUNT=$((RETRY_COUNT+1))
        
        if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
            echo "❌ Too many crashes ($RETRY_COUNT). Giving up." | tee -a $LOG_FILE
            break
        fi
        
        echo "💤 Resting for 10 seconds before restart..."
        sleep 10
    fi
done