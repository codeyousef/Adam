#!/bin/bash

# --- CONFIG ---
LOG_FILE="adam_watchdog.log"
MAX_RETRIES=100
RETRY_COUNT=0
SAFE_TEMP=80
POWER_CAP=600

# --- SETUP ---
# Ensure we use the correct environment if you have one, 
# otherwise we use the system python we just installed.
# source adam_env/bin/activate  <-- Commented out since this is a fresh instance

echo "🐕 Adam Phoenix Watchdog started." | tee -a $LOG_FILE

# 1. APPLY POWER LIMIT (Critical for B200 stability)
echo "🧊 Enforcing Power Cap: ${POWER_CAP}W..." | tee -a $LOG_FILE
nvidia-smi -i 0 -pl $POWER_CAP

while true; do
    # 2. THERMAL INTERLOCK
    CURRENT_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    
    if [ "$CURRENT_TEMP" -ge "$SAFE_TEMP" ]; then
        echo "🔥 GPU is too hot ($CURRENT_TEMP°C). Cooling down for 60s..." | tee -a $LOG_FILE
        sleep 60
        continue
    fi

    # 3. RUN THE PHOENIX SCRIPT
    # Note: We changed the filename here!
    echo "🚀 Starting Adam Phoenix (Attempt $RETRY_COUNT) [Temp: $CURRENT_TEMP°C]..." | tee -a $LOG_FILE
    
    # Run unbuffered (-u) so logs appear instantly
    python -u train_adam_phoenix.py
    
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