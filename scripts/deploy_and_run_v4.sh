#!/bin/bash
# Deploy v4 pipeline to H200 and start it

set -e

H200="ubuntu@204.12.171.168"
SSH_KEY="$HOME/.ssh/id_adam"

echo "============================================"
echo "  Deploying ADAM v4 Pipeline to H200"
echo "============================================"
echo ""

# Upload new files
echo "Uploading pipeline files..."
scp -i "$SSH_KEY" \
    fix_and_regenerate_data.py \
    full_pipeline_v4.sh \
    quick_validate_v2.py \
    "${H200}:/data/"

# Make scripts executable
ssh -i "$SSH_KEY" "$H200" 'chmod +x /data/full_pipeline_v4.sh /data/fix_and_regenerate_data.py /data/quick_validate_v2.py'

# Update quick_validate.py symlink
ssh -i "$SSH_KEY" "$H200" 'cd /data && ln -sf quick_validate_v2.py quick_validate.py'

echo "Files uploaded"
echo ""

# Start pipeline in tmux
echo "Starting pipeline in tmux session 'adam_v4'..."
ssh -i "$SSH_KEY" "$H200" << 'REMOTE'
cd /data

# Kill old session if exists
tmux kill-session -t adam_v4 2>/dev/null || true

# Start new session
tmux new-session -d -s adam_v4 "bash /data/full_pipeline_v4.sh"

echo "Pipeline started in tmux session 'adam_v4'"
echo ""
echo "Monitor with:"
echo "  ssh -i ~/.ssh/id_adam ubuntu@204.12.171.168"
echo "  tmux attach -t adam_v4"
echo ""
echo "Or check logs:"
echo "  tail -f /data/pipeline_v4.log"
REMOTE

echo ""
echo "============================================"
echo "Deployment complete!"
echo "============================================"
echo ""
echo "Pipeline stages:"
echo "  1. Fix quick data (2 mins)"
echo "  2. Persona augmentation - 10 personas (~5 hours)"
echo "  3. Fix full data (5 mins)"
echo "  4. NLI verification (~30 mins)"
echo "  5. SimPO v4 training (~3 hours)"
echo "  6. Validation (5 mins)"
echo "  7. DAFT training if L3 >= 85% (~1.5 hours)"
echo "  8. Final validation (5 mins)"
echo ""
echo "Total ETA: ~10 hours (if all passes)"
echo ""
echo "Check status:"
echo "  ssh -i ~/.ssh/id_adam ubuntu@204.12.171.168 'tail -100 /data/pipeline_v4.log'"
