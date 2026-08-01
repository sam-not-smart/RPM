#!/usr/bin/env bash
set -euo pipefail

root="/home/samsmart/Research/RPM"
python="/home/samsmart/miniforge3/envs/rpm/bin/python"
runner="$root/Realistic system size/code/rpm_quantum_needle_jd_sweeps.py"
output="$root/outputs/quantum_needle_efimova_hore_24h"

mkdir -p "$output"
if [[ ! -f "$output/runtime_benchmark.json" ]]; then
  cp "$root/outputs/quantum_needle_jd_sweeps/runtime_benchmark.json" "$output/"
fi
if [[ ! -f "$output/runtime_benchmark.csv" && -f "$root/outputs/quantum_needle_jd_sweeps/runtime_benchmark.csv" ]]; then
  cp "$root/outputs/quantum_needle_jd_sweeps/runtime_benchmark.csv" "$output/"
fi

arguments=(
  --stage efimova-hore-24h
  --workers 32
  --max-wall-hours 23.5
  --output-dir "$output"
)
if [[ -f "$output/manifest.json" ]]; then
  arguments+=(--resume)
fi

exec "$python" "$runner" "${arguments[@]}"
