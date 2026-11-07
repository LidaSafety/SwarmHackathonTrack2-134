#!/usr/bin/env bash
# Q2 full agent loop on local Qwen3-8B (GPU0): 20-seed baseline, then 10 seeds steered at +0.5x refusal vector.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/guardbench/bin/python}
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY interp/q2_loop.py --seeds 20 --out runs/interp_q2_loop/Qwen3-8B
$PY interp/q2_loop.py --seeds 10 --steer-alpha 0.5 --out runs/interp_q2_loop/Qwen3-8B_steer0.5
