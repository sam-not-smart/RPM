"""Focused paper-quality refinement of the 50 uT D/J=-3 quantum needle."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from rpm_quantum_needle_jd_sweeps import (
    QUADRUPOLE_SCALE,
    SweepConfig,
    _write_csv,
    _write_json,
    checkpoint_path,
    efimova_hore_ratio_condition,
    load_cell,
    run_tasks,
)


SCHEMA_VERSION = 1
REFINED_ANGLES_DEG = tuple(
    [85.0, 86.0]
    + list(np.round(np.arange(87.0, 93.0 + 0.125, 0.25), 10))
    + [94.0, 95.0]
)


def _manifest_payload(config: SweepConfig, samples: int) -> dict[str, object]:
    condition = efimova_hore_ratio_condition(-3.0)
    stable = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "full14_D_over_J_minus3_quantum_needle_refinement",
        "angles_deg": list(REFINED_ANGLES_DEG),
        "samples": int(samples),
        "condition": condition.stable_dict(),
        "condition_hash": condition.configuration_hash,
        "sweep_configuration": config.stable_dict(),
        "sweep_configuration_hash": config.configuration_hash,
        "quadrupole_scale": QUADRUPOLE_SCALE,
    }
    digest = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"configuration_hash": digest, **stable}


def ensure_manifest(
    output_dir: Path, config: SweepConfig, samples: int, resume: bool
) -> dict[str, object]:
    path = output_dir / "manifest.json"
    expected = _manifest_payload(config, samples)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("configuration_hash") != expected["configuration_hash"]:
            raise RuntimeError("Existing output has an incompatible configuration.")
        if not resume:
            raise RuntimeError("Output exists; pass --resume to continue it.")
        return manifest
    manifest = {
        **expected,
        "status": "created",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    _write_json(path, manifest)
    return manifest


def _halfwidth(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(
        scipy.stats.t.ppf(0.975, values.size - 1)
        * np.std(values, ddof=1)
        / math.sqrt(values.size)
    )


def write_report(
    output_dir: Path, config: SweepConfig, samples: int
) -> dict[str, object]:
    condition = efimova_hore_ratio_condition(-3.0)
    cells: list[dict[str, object]] = []
    for theta in REFINED_ANGLES_DEG:
        for sample in range(samples):
            path = checkpoint_path(output_dir, condition, theta, sample)
            if not path.exists():
                continue
            cell = load_cell(output_dir, condition, theta, sample, config)
            cells.append(
                {
                    **condition.stable_dict(),
                    "theta_deg": theta,
                    "sample_id": sample,
                    "yield": float(cell["yield"]),
                    "yield_quadrature_error": float(cell["yield_quadrature_error"]),
                    "yield_propagation_bound": float(cell["yield_propagation_bound"]),
                    "yield_tail_bound": float(cell["yield_tail_bound"]),
                    "accepted_steps": int(cell["accepted_steps"]),
                    "rejected_steps": int(cell["rejected_steps"]),
                    "matvecs": int(cell["matvecs"]),
                    "total_seconds": float(cell["total_seconds"]),
                    "peak_rss_mib": float(cell["peak_rss_mib"]),
                }
            )
    if cells:
        _write_csv(output_dir / "needle_refinement_cells.csv", cells)

    total = len(REFINED_ANGLES_DEG) * samples
    complete = len(cells) == total
    summary: dict[str, object] = {
        "complete": complete,
        "completed_trajectories": len(cells),
        "total_trajectories": total,
        "angles": len(REFINED_ANGLES_DEG),
        "samples": samples,
        "field_microT": config.field_microT,
        "lifetime_us": config.lifetime_us,
        "ratio_D_over_J": -3.0,
        "quadrupole_scale": QUADRUPOLE_SCALE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not complete:
        _write_json(output_dir / "needle_refinement_summary.json", summary)
        return summary

    by_key = {(float(row["theta_deg"]), int(row["sample_id"])): row for row in cells}
    curve_rows: list[dict[str, object]] = []
    for theta in REFINED_ANGLES_DEG:
        selected = [by_key[(theta, sample)] for sample in range(samples)]
        values = np.asarray([row["yield"] for row in selected], dtype=float)
        statistical = _halfwidth(values)
        systematic = max(
            float(row["yield_quadrature_error"])
            + float(row["yield_propagation_bound"])
            + float(row["yield_tail_bound"])
            for row in selected
        )
        curve_rows.append(
            {
                "theta_deg": theta,
                "samples": samples,
                "mean_yield": float(np.mean(values)),
                "nuclear_95_halfwidth": statistical,
                "systematic_halfwidth": systematic,
                "total_95_halfwidth": statistical + systematic,
            }
        )
    _write_csv(output_dir / "needle_refinement_curve.csv", curve_rows)

    sample_depths = []
    for sample in range(samples):
        background = 0.5 * (
            float(by_key[(85.0, sample)]["yield"])
            + float(by_key[(95.0, sample)]["yield"])
        )
        sample_depths.append(background - float(by_key[(90.0, sample)]["yield"]))
    sample_depths_array = np.asarray(sample_depths)
    depth = float(np.mean(sample_depths_array))
    depth_statistical = _halfwidth(sample_depths_array)

    means = np.asarray([float(row["mean_yield"]) for row in curve_rows])
    angles = np.asarray(REFINED_ANGLES_DEG)
    minimum_index = int(np.argmin(means))
    summary.update(
        {
            "needle_depth": depth,
            "needle_depth_nuclear_95_halfwidth": depth_statistical,
            "minimum_yield": float(means[minimum_index]),
            "minimum_angle_deg": float(angles[minimum_index]),
            "mean_trajectory_seconds": float(
                np.mean([float(row["total_seconds"]) for row in cells])
            ),
            "maximum_peak_rss_mib": float(
                np.max([float(row["peak_rss_mib"]) for row in cells])
            ),
        }
    )
    _write_json(output_dir / "needle_refinement_summary.json", summary)
    _plot(output_dir, curve_rows, summary)
    return summary


def _plot(
    output_dir: Path,
    curve_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    theta = np.asarray([float(row["theta_deg"]) for row in curve_rows])
    mean = np.asarray([float(row["mean_yield"]) for row in curve_rows])
    halfwidth = np.asarray([float(row["total_95_halfwidth"]) for row in curve_rows])
    background = np.interp(theta, [85.0, 95.0], [mean[0], mean[-1]])

    fig, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.fill_between(
        theta,
        mean - halfwidth,
        mean + halfwidth,
        color="#2878B5",
        alpha=0.18,
        linewidth=0,
        label="Conservative total 95% interval",
    )
    axis.plot(
        theta,
        mean,
        color="#2878B5",
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        label=f"Full-state M={summary['samples']} mean",
    )
    axis.plot(theta, background, color="0.4", linestyle="--", linewidth=1.5, label="Linear 85-95 degree background")
    axis.axvline(90.0, color="0.55", linestyle=":", linewidth=1.2)
    axis.set(
        xlabel=r"Magnetic-field polar angle $\theta$ (degrees)",
        ylabel=r"Singlet yield $\Phi_S$",
        title="Full-state D/J=-3 quantum-needle refinement at 50 microtesla",
        xlim=(84.8, 95.2),
    )
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    axis.text(
        0.5,
        1.01,
        "14 nuclei; lifetime=5 microseconds; Q=0; 0.25-degree central grid",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        color="0.35",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "needle_refinement.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "report", "auto"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-wall-hours", type=float, default=23.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "quantum_needle_dj_m3_refined_m8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.workers < 1:
        raise ValueError("Samples must be at least two and workers must be positive.")
    if args.max_wall_hours <= 0.0:
        raise ValueError("--max-wall-hours must be positive.")
    config = replace(SweepConfig(), seed=args.seed, lifetime_us=5.0)
    condition = efimova_hore_ratio_condition(-3.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_manifest(args.output_dir, config, args.samples, args.resume)

    complete = all(
        checkpoint_path(args.output_dir, condition, theta, sample).exists()
        for theta in REFINED_ANGLES_DEG
        for sample in range(args.samples)
    )
    if args.stage in ("run", "auto") and not complete:
        complete = run_tasks(
            args.output_dir,
            (condition,),
            REFINED_ANGLES_DEG,
            args.samples,
            config,
            args.workers,
            time.time() + args.max_wall_hours * 3600.0,
            estimated_wave_seconds=16000.0,
        )

    summary = write_report(args.output_dir, config, args.samples)
    manifest["status"] = "complete" if summary["complete"] else "checkpointed"
    manifest["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["summary"] = summary
    _write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.stage == "report" and not summary["complete"]:
        raise SystemExit("Run is incomplete; report contains partial inventory only.")


if __name__ == "__main__":
    main()
