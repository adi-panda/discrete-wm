#!/bin/bash
# launch_research.sh — Kick off the autonomous research agent.
#
# === Recommended: run inside tmux ===
#   chmod +x launch_research.sh
#   tmux new -s research -d './launch_research.sh'
#
#   # Reattach from any terminal to watch live:
#   tmux attach -t research
#   # Ctrl+B then D to detach (keeps running). Ctrl+B then [ to scroll.
#
#   # Kill it cleanly:
#   tmux kill-session -t research
#   # Or from inside the session: Ctrl+C
#
# === Alternative: nohup (fire-and-forget, no interactive reattach) ===
#   nohup ./launch_research.sh &
#   disown
#   echo "PID: $!"
#   tail -f /root/research/research_agent_output.log
#
#   # Kill:
#   kill $(cat /root/research/agent.pid)
#
# Output:
#   /root/research/research_agent_output.log  ← clean, human-readable
#     (Claude's narration + tool calls + 1-line results)
#
# The terminal (tmux attach) and the log file both see this clean view in real time.
#
# Prerequisites:
#   1. npm install -g @anthropic-ai/claude-code
#   2. claude login  (or export ANTHROPIC_API_KEY)
#   3. instructions.md in /root/research/
#   4. hf auth login  (for checkpoint upload to HuggingFace)

# ─── Force unbuffered output everywhere ──────────────────────────
export PYTHONUNBUFFERED=1
export FORCE_COLOR=0

# ─── Config ───────────────────────────────────────────────────────
WORKDIR="/root/research"
AGENT_INSTRUCTIONS="/root/research/instructions.md"
LOG_FILE="/root/research/research_agent_output.log"   # filtered, human-readable
PID_FILE="/root/research/agent.pid"
MAX_TURNS=200
MODEL="claude-opus-4-6"

# ─── Setup workspace ─────────────────────────────────────────────
mkdir -p "$WORKDIR"

# ─── Logging helper (writes to terminal + log file) ──────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

# ─── Launch banner ────────────────────────────────────────────────
log "=== Starting autonomous research agent ==="
log "  Workspace  : $WORKDIR"
log "  Log file   : $LOG_FILE"
log "  Max turns  : $MAX_TURNS"
log "  Model      : $MODEL"
log "  PID        : $$"
log "  Data dir   : /vast/adi/discrete_wm/"
log "  HF repo    : adipanda/discrete-wm"
log "============================================"

echo $$ > "$PID_FILE"
cd "$WORKDIR"

# ─── jq filter for clean output ──────────────────────────────────
# Keeps only:
#   - Claude's narration text (no prefix)
#   - Tool calls as "→ ToolName: <short description>"
#   - Tool results as "← <first line, truncated to 160 chars>"
#   - Final DONE banner with subtype + turn count
# Everything else (system init, thinking, partial chunks, metadata) is dropped.
FILTER='
  if .type == "assistant" then
    (.message.content // [])[]?
    | if .type == "text" then
        .text
      elif .type == "tool_use" then
        "→ \(.name): \(.input.description // .input.command // .input.file_path // .input.pattern // .input.prompt // "")"
          | if length > 200 then .[0:200] + "…" else . end
      else empty end
  elif .type == "user" then
    (.message.content // [])[]?
    | if .type == "tool_result" then
        "← " + (
          (.content // "")
          | if type == "array" then (map(select(.type=="text") | .text) | join(" ")) else tostring end
          | split("\n")[0]
          | if length > 160 then .[0:160] + "…" else . end
        )
      else empty end
  elif .type == "result" then
    "\n═══ DONE (subtype=\(.subtype // "?"), turns=\(.num_turns // "?"), cost=$\(.total_cost_usd // 0)) ═══"
  else empty end
'

# ─── Run the agent ────────────────────────────────────────────────
# Pipeline:
#   claude --output-format stream-json  →  emits NDJSON events
#     | jq (FILTER)                      →  extract only interesting lines
#     | tee -a $LOG_FILE                  →  clean view to terminal + log file
#
# --verbose is required by the CLI when combining --print with stream-json.
# stdbuf -oL forces line buffering end-to-end so `tmux attach` updates live.

IS_SANDBOX=1 stdbuf -oL \
  claude \
    --dangerously-skip-permissions \
    --model "$MODEL" \
    --max-turns "$MAX_TURNS" \
    --output-format stream-json \
    --verbose \
    -p "$(cat "$AGENT_INSTRUCTIONS")" \
  2>&1 \
  | stdbuf -oL jq --unbuffered -r "$FILTER" \
  | stdbuf -oL tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

log ""
log "=== Agent finished ==="
log "  Exit code : $EXIT_CODE"
log "  Time      : $(date)"
log "  Log file  : $LOG_FILE"
log "  Research log: $WORKDIR/research_log.md"
