#!/usr/bin/env bash
# keepalive.sh -- foreground "tick" loop to defeat SageMaker Studio
# JupyterLab's auto-shutdown extension.
#
# The Studio idle extension treats a terminal as idle when no data has
# flowed through its PTY for N minutes.  A backgrounded ``nohup`` job is
# invisible to it because its stdout is redirected to a log file and
# does not pass through the terminal.  Running this script in the
# FOREGROUND of the same terminal pushes a "tick" line through the PTY
# every ``INTERVAL`` seconds, which (server-side) updates
# ``last_activity`` on the terminal model and resets the idle timer.
#
# IMPORTANT: this only works while the browser tab is open.  If you
# disconnect, the websocket dies and the ticks no longer reach the
# server -- the extension will then shut the app down.  For
# fire-and-forget overnight runs use a Jupyter notebook driver or
# disable the extension instead.
#
# Usage:
#   bash scripts/keepalive.sh             # tick every 60s, runs forever
#   bash scripts/keepalive.sh 30          # tick every 30s
#   bash scripts/keepalive.sh 60 4h       # tick every 60s, stop after 4h
#
# Combined with a nohup'd sweep:
#   nohup python -u scripts/run_seed_sweep.py ... > sweep.log 2>&1 &
#   disown
#   bash scripts/keepalive.sh 60          # leave this running in foreground

set -u
INTERVAL="${1:-60}"
DURATION="${2:-}"   # optional: e.g. 4h, 30m, 7200 (seconds); empty = forever

if [[ -n "$DURATION" ]]; then
    # Normalise human suffixes (h/m/s) to seconds for the deadline math.
    case "$DURATION" in
        *h) deadline=$(( $(date +%s) + ${DURATION%h} * 3600 )) ;;
        *m) deadline=$(( $(date +%s) + ${DURATION%m} * 60 )) ;;
        *s) deadline=$(( $(date +%s) + ${DURATION%s} )) ;;
        *)  deadline=$(( $(date +%s) + DURATION )) ;;
    esac
    echo "[keepalive] tick every ${INTERVAL}s until $(date -d "@$deadline" 2>/dev/null || date -r "$deadline" 2>/dev/null)"
else
    deadline=""
    echo "[keepalive] tick every ${INTERVAL}s (Ctrl-C to stop)"
fi
echo "[keepalive] NOTE: this only works while the browser tab stays connected."

i=0
while true; do
    i=$((i + 1))
    printf '[keepalive %05d %s] still here -- tail your sweep log with: tail -F <your-sweep>.log\n' \
        "$i" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ -n "$deadline" ]] && (( $(date +%s) >= deadline )); then
        echo "[keepalive] deadline reached, exiting."
        break
    fi
    sleep "$INTERVAL"
done
