#!/usr/bin/fish

# --- 1. Environment Setup (Critical for Systemd) ---
# Systemd doesn't see your shell variables, so we define them here.
set -x CUDA_HOME /opt/cuda-12.8
set -x LD_LIBRARY_PATH $CUDA_HOME/lib64 $LD_LIBRARY_PATH
set -x PATH $CUDA_HOME/bin $PATH

set STUDIO_DIR "/mnt/Storage/Projects/catbelly_studio"
set LOGFILE "$STUDIO_DIR/adam_training_log.txt"

function log_msg
    echo ">>> ["(date)"] Catbelly Studio: $argv" >> $LOGFILE
end

# --- 2. Automatic Duration Calculation ---
# Day 1=Mon ... 5=Fri, 6=Sat, 7=Sun
set DAY (date +%u)

# Schedule Logic:
# If Friday (5) or Saturday (6): Run for 21 hours (2am -> 11pm)
# Else (Sun-Thu): Run for 15.5 hours (12:30am -> 4pm)

if contains $DAY 5 6
    set DURATION "21h"
    log_msg "Detected WEEKEND schedule (Fri/Sat). Duration set to $DURATION."
else
    set DURATION "15.5h"
    log_msg "Detected WEEKDAY schedule (Sun-Thu). Duration set to $DURATION."
end

# --- 3. Distraction Free Mode ---
set APPS_TO_KILL firefox chrome chromium discord steam spotify code obs vlc kcalc
for app in $APPS_TO_KILL
    if pgrep -x $app > /dev/null
        pkill -x $app
    end
end

# --- 4. Hardware Safety Limits ---
# Force CPU performance (CachyOS optimizations)
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

# Force 4090 Power Limit to 300W
# This protects your hardware during unattended 21-hour training sessions.
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300

# --- 5. Launch Training ---
cd $STUDIO_DIR
source .venv/bin/activate.fish

# 'timeout -s SIGINT' is the key here. It sends a gentle CTRL+C signal 
# when the duration expires, allowing train_adam.py to save a checkpoint safely.
timeout -s SIGINT $DURATION python train_adam.py >> $LOGFILE 2>&1

log_msg "Cycle Complete (Duration Reached)."