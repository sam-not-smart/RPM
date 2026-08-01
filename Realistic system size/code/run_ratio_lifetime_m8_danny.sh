#!/usr/bin/env bash
set -euo pipefail

root=/home/samsmart/Research/RPM
output="$root/outputs/quantum_needle_jd_lifetime_m8"
runner="$root/Realistic system size/code/rpm_quantum_needle_jd_sweeps.py"
python=/home/samsmart/miniforge3/envs/rpm/bin/python

mkdir -p "$output"
if [[ ! -f "$output/runtime_benchmark.json" ]]; then
    cp "$root/outputs/quantum_needle_jd_sweeps/runtime_benchmark.json" "$output/"
    cp "$root/outputs/quantum_needle_jd_sweeps/runtime_benchmark.csv" "$output/"
fi

while true; do
    resume=()
    if [[ -f "$output/manifest.json" ]]; then
        resume=(--resume)
    fi
    "$python" "$runner" \
        --stage ratio-lifetime-focus \
        "${resume[@]}" \
        --ratios 5 8 -4 \
        --samples 8 \
        --lifetimes 5 10 \
        --workers 32 \
        --max-wall-hours 20 \
        --output-dir "$output"
    if grep -q '"status": "ratio-lifetime-focus_complete"' "$output/manifest.json"; then
        break
    fi
done
