#!/usr/bin/env bash
set -euo pipefail

root="/home/samsmart/Research/RPM"
python="/home/samsmart/miniforge3/envs/rpm/bin/python"
output="$root/outputs/poster_reproduction_multisample"
stage="${1:-smoke}"
figures="${POSTER_FIGURES:-2,3,4,5}"
kinds="${POSTER_KINDS:-quantum,overhauser,entanglement,n10}"
workers="${POSTER_WORKERS:-32}"
overhauser_samples="${POSTER_OVERHAUSER_SAMPLES:-262144}"
overhauser_replicates="${POSTER_OVERHAUSER_REPLICATES:-8}"
atomwise_samples="${POSTER_ATOMWISE_SAMPLES:-4096}"

export PYTHONPATH="$root/Realistic system size/code"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$output/logs"
arguments=(
  --stage "$stage"
  --workers "$workers"
  --overhauser-samples "$overhauser_samples"
  --overhauser-replicates "$overhauser_replicates"
  --atomwise-samples "$atomwise_samples"
  --max-wall-hours 23.5
  --output-dir "$output"
  --figures "$figures"
  --kinds "$kinds"
  --resume
)

exec "$python" -m poster_reproduction.runner "${arguments[@]}"
