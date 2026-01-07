#!/bin/bash

# --- CONFIG ---
LOG_FILE="adam_watchdog.log"
MAX_RETRIES=50
RETRY_COUNT=0

echo "🐕 Adam Watchdog started. Writing logs to $LOG_FILE"

while true; do
    # 1. Run the training script
    echo "🚀 Starting Adam (Attempt $RETRY_COUNT)..." | tee -a $LOG_FILE
    
    # Using unbuffered output (-u) so you see logs immediately in tmux
    python -u train_adam.py
    
    # 2. Capture Exit Code
    EXIT_CODE=$?
    
    # 3. Decision Logic
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training Finished Successfully!" | tee -a $LOG_FILE
        break
    else
        echo "⚠️  Adam crashed with exit code $EXIT_CODE." | tee -a $LOG_FILE
        
        # Check for user interrupt (Ctrl+C usually sends 130) to allow manual stop
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