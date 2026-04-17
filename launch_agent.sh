#!/bin/bash
# launch_agent.sh — Kick off the autonomous research agent.
#
# Run with:
#   chmod +x launch_agent.sh
#   nohup ./launch_agent.sh &
#   disown
#   echo "PID: $!"
#   tail -f /root/research/research_agent_output.log
#
# Prerequisites:
#   1. npm install -g @anthropic-ai/claude-code
#   2. claude login  (or export ANTHROPIC_API_KEY)
#   3. instructions.md in /root/research/
#   4. hf auth login  (for checkpoint upload to HuggingFace)

# ─── Force unbuffered output everywhere ──────────────────────────
export PYTHONUNBUFFERED=1
export FORCE_COLOR=0
exec > >(tee -a "/root/research/research_agent_output.log") 2>&1

# ─── Config ───────────────────────────────────────────────────────
WORKDIR="/root/research"
AGENT_INSTRUCTIONS="/root/research/instructions.md"
LOG_FILE="/root/research/research_agent_output.log"
PID_FILE="/root/research/agent.pid"
MAX_TURNS=200
MODEL="claude-opus-4-6"

# ─── Setup workspace ─────────────────────────────────────────────
mkdir -p "$WORKDIR"

# ─── Logging helper ──────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ─── Launch ───────────────────────────────────────────────────────
log "=== Starting autonomous research agent ==="
log "  Workspace  : $WORKDIR"
log "  Log file   : $LOG_FILE"
log "  Max turns  : $MAX_TURNS"
log "  Model      : $MODEL"
log "  PID        : $$"
log "  Data dir   : /vast/adi/discrete_wm/"
log "  HF repo    : adipanda/discrete-wm"
log "============================================"

# Save PID for easy killing later: kill $(cat /root/research/agent.pid)
echo $$ > "$PID_FILE"

cd "$WORKDIR"

# ─── Run the agent ────────────────────────────────────────────────
# --output-format stream-json: emits every event (tool calls, text, results)
#   as newline-delimited JSON so nothing is lost
# --verbose: extra debug info
# The `stdbuf -oL` forces line-buffered output so `tail -f` updates live
IS_SANDBOX=1 stdbuf -oL \
  claude \
    --dangerously-skip-permissions \
    --model "$MODEL" \
    --max-turns "$MAX_TURNS" \
    --output-format stream-json \
    --verbose \
    -p "$(cat "$AGENT_INSTRUCTIONS")"

EXIT_CODE=$?

log ""
log "=== Agent finished ==="
log "  Exit code : $EXIT_CODE"
log "  Time      : $(date)"
log "  Check $WORKDIR/research_log.md for results"
