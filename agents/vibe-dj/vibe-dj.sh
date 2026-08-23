#!/bin/zsh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${VIBE_DJ_MODEL:-opencode/big-pickle}"
OPENCODE_BIN="${OPENCODE_BIN:-/Users/xnch/.opencode/bin/opencode}"
STATE_FILE="$DIR/state.txt"
LOG="/tmp/vibe-dj.log"

# Quiet hours 00:00-08:00 local — DJ sleeps, leave playback untouched.
hour=$(date +%H)
if (( hour >= 0 && hour < 8 )); then
  echo "[$(date -Iseconds)] quiet hours, exiting" >> "$LOG"
  exit 0
fi

# Read deterministic state (defaults when absent).
bucket=""; lastSwitchAt=0; lastSkipAt=0
if [[ -f "$STATE_FILE" ]]; then
  b=$(grep -m1 '^bucket=' "$STATE_FILE" 2>/dev/null | cut -d= -f2-); [[ -n "$b" ]] && bucket="$b"
  s=$(grep -m1 '^lastSwitchAt=' "$STATE_FILE" 2>/dev/null | cut -d= -f2-); [[ "$s" =~ ^[0-9]+$ ]] && lastSwitchAt="$s"
  k=$(grep -m1 '^lastSkipAt=' "$STATE_FILE" 2>/dev/null | cut -d= -f2-); [[ "$k" =~ ^[0-9]+$ ]] && lastSkipAt="$k"
fi

now=$(date +%s)
secs_since_switch=$(( now - lastSwitchAt ))
secs_since_skip=$(( now - lastSkipAt ))
switch_allowed=no; (( secs_since_switch >= 1800 )) && switch_allowed=yes
skip_allowed=no;   (( secs_since_skip >= 900 )) && skip_allowed=yes

prompt="$(cat "$DIR/dj-prompt.md")"
prompt+=$'\n\n## Current state (injected — trust these values, do not re-read them)\n'
prompt+="current_bucket=$bucket\n"
prompt+="lastSwitchAt=$lastSwitchAt\n"
prompt+="lastSkipAt=$lastSkipAt\n"
prompt+="switch_allowed=$switch_allowed\n"
prompt+="skip_allowed=$skip_allowed\n"
prompt+="state_file=$STATE_FILE\n"

echo "[$(date -Iseconds)] tick start (model=$MODEL, bucket=$bucket, switch_allowed=$switch_allowed)" >> "$LOG"
"$OPENCODE_BIN" run --pure --auto --dir "$DIR" --model "$MODEL" "$prompt" >> "$LOG" 2>&1
echo "[$(date -Iseconds)] tick end" >> "$LOG"
