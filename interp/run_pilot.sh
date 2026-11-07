#!/usr/bin/env bash
# Q2 interpretability pilot: 2 local models x 3 frames x 20 seeds on GPU0, steering on the agent round-2 frame.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/guardbench/bin/python}
export CUDA_VISIBLE_DEVICES=0
for m in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
  tag=$(basename "$m")
  $PY interp/q2_probe.py --model "$m" --frame chat   --stage round2 --seeds 20 --bs 4 --out runs/interp_q2/${tag}_chat
  $PY interp/q2_probe.py --model "$m" --frame agent  --stage round1 --seeds 20 --bs 4 --out runs/interp_q2/${tag}_agent_r1
  $PY interp/q2_probe.py --model "$m" --frame agent  --stage round2 --seeds 20 --bs 4 --out runs/interp_q2/${tag}_agent_r2 --steer --alphas 0.25,0.5,1
done
