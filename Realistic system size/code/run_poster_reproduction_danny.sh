#!/usr/bin/env bash
set -euo pipefail

root="/home/samsmart/Research/RPM"
python="/home/samsmart/miniforge3/envs/rpm/bin/python"
output="$root/outputs/poster_reproduction_multisample"
stage="${1:-smoke}"
workers="${POSTER_WORKERS:-32}"

export PYTHONPATH="$root/Realistic system size/code"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$output/logs"
arguments=(
  --stage "$stage"
  --workers "$workers"
  --max-wall-hours 23.5
  --output-dir "$output"
  --resume
)

exec "$python" -m poster_reproduction.runner "${arguments[@]}"
