#!/usr/bin/env bash
# Runs queued leader-loop proposals one after another once the current sweep has finished. leader_loop.py itself
# refuses to start when a leader has already emerged or the budget is spent, so the queue stops on its own.
cd "$(dirname "$0")/.."
while pgrep -f "[s]warmlab.cli sweep" >/dev/null; do sleep 20; done
while read -r cfg cap why; do
  [ -z "$cfg" ] && continue
  OPENROUTER_API_KEY=$(tr -d '[:space:]' < ~/anthropic-key/openrouter.txt) CUDA_VISIBLE_DEVICES=0 \
    ~/miniconda3/envs/guardbench/bin/python scripts/leader_loop.py "$cfg" --cap "$cap" --why "$why" || break
done < "${1:-runs/leader_queue.txt}"
