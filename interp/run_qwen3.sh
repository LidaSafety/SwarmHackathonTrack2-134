#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/guardbench/bin/python}
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
m=Qwen/Qwen3-8B; tag=Qwen3-8B
$PY interp/q2_probe.py --model $m --frame chat  --stage round2 --seeds 20 --bs 4 --out runs/interp_q2/${tag}_chat
$PY interp/q2_probe.py --model $m --frame agent --stage round1 --seeds 20 --bs 4 --out runs/interp_q2/${tag}_agent_r1
$PY interp/q2_probe.py --model $m --frame agent --stage round2 --seeds 20 --bs 4 --out runs/interp_q2/${tag}_agent_r2 --steer --alphas 0.25,0.5,1
