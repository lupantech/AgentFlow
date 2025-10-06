#!/bin/bash

# ===========================================================================
# Script: serve_vllm.sh
# Description:
#   Launch model using vLLM in a tmux window
#   - Uses GPU 0
#   - tensor-parallel-size=1
#   - Port 8000
# ===========================================================================

MODEL="AgentFlow/agentflow-planner-7b"
GPU="0"
PORT=8000
TMUX_SESSION="vllm_agentflow"
TP=1

VENV_ACTIVATE="source .venv/bin/activate"

echo "Launching model: $MODEL"
echo "  Port: $PORT"
echo "  GPU: $GPU"
echo "  Tensor Parallel Size: $TP"

# Create tmux session and run vLLM
tmux new-session -d -s "$TMUX_SESSION"

CMD_START="
    $VENV_ACTIVATE;
    export CUDA_VISIBLE_DEVICES=$GPU;
    echo '--- Starting $MODEL on port $PORT with TP=$TP ---';
    echo 'CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES';
    echo 'Current virtual env: \$(python -c \"import sys; print(sys.prefix)\")';
    vllm serve \"$MODEL\" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP
"

tmux send-keys -t "${TMUX_SESSION}:0" "$CMD_START" C-m

echo ""
echo "✅ Model launched in tmux session: '$TMUX_SESSION'"
echo "💡 View logs:   tmux attach-session -t $TMUX_SESSION"
echo "💡 Detach:      Ctrl+B, then D"
echo "💡 Kill session: tmux kill-session -t $TMUX_SESSION"
