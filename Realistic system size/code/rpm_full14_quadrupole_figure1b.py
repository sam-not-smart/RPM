"""Fig. 1B-style paired Q-off/Q-on extension of the completed low-field run.

Only the angle inventory changes. Reuse validated trajectories from the
11-angle run and retain the full coupled 14-nucleus propagation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import rpm_full14_quadrupole_experiment as base

import matplotlib.pyplot as plt
import numpy as np


GRID_SETTINGS = {
    # Full-range step, refinement start, refinement stop, refinement step.
    "dense": (5, 87, 93, .25),
    "reduced": (10, 85, 95, 1.),
}
DEFAULT_OUTPUT = Path("outputs/quadrupole tests/full14_quadrupole_figure1b_tau5")
REDUCED_OUTPUT = DEFAULT_OUTPUT.with_name("full14_quadrupole_figure1b_tau5_29angles")
TAGS = ("off", "on")


def angular_grids(grid):
    coarse_step, start, stop, step = GRID_SETTINGS[grid]
    coarse = tuple(float(a) for a in range(0, 181, coarse_step))
    refined = tuple(start + i * step for i in range(round((stop - start) / step) + 1))
    return coarse, refined


COARSE_ANGLES, REFINED_ANGLES = angular_grids("dense")
ANGLES = tuple(sorted(set(COARSE_ANGLES + REFINED_ANGLES + base.ANGLES)))


def configuration(grid="dense"):
    coarse, refined = angular_grids(grid)
    return replace(base.Config(), angles_deg=tuple(sorted(set(coarse + refined + base.ANGLES))))


def ordered_tasks(config, grid="dense"):
    """Fill the full-range grid first, then the fine needle grid."""
    coarse, _ = angular_grids(grid)
    return sorted(base.tasks(config), key=lambda task: (
        task.theta_deg not in coarse, abs(task.theta_deg - 90),
        task.theta_deg, task.sample_id, task.scale,
    ))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reuse_completed(source, output, config, source_config=None):
    """Import compatible complete cells, leaving source checkpoints untouched."""
    source, output = source.resolve(), output.resolve()
    source_config = base.Config() if source_config is None else source_config
    if source == output:
        raise ValueError("source and extension outputs must be different")
    if replace(config, angles_deg=source_config.angles_deg).digest != source_config.digest:
        raise ValueError("only the angular inventory may change when reusing cells")
    manifest = read_json(source / "manifest.json")
    if manifest["configuration_hash"] != source_config.digest:
        raise ValueError("source manifest is incompatible with the previous run")
    status = read_json(source / "status.json")
    source_tasks = base.tasks(source_config)
    if status["status"] != "complete" or status["completed"] != len(source_tasks):
        raise ValueError("source run is not complete")
    module_dir = Path(base.__file__).parent
    for name, path in {
        "rpm_statevector_engine.py": module_dir / "rpm_statevector_engine.py",
        "rpm_quantum_needle_model.py": module_dir / "rpm_quantum_needle_model.py",
        "core.py": module_dir / "poster_reproduction" / "core.py",
    }.items():
        if sha256(path) != manifest["source_sha256"][name]:
            raise ValueError(f"source physics code does not match: {name}")
    target_tasks = set(base.tasks(config))
    if not set(source_tasks).issubset(target_tasks):
        raise ValueError("new inventory must include all original cells")
    if not all(base.cell_path(source, task).is_file() for task in source_tasks):
        raise ValueError("source inventory has missing checkpoint files")

    nuclei, _ = base.system(config)
    dimensions = base.dimensions_for(nuclei)
    initial_hashes = {
        sample: hashlib.sha256(base.random_phase_initial_state(
            dimensions, config.seed, sample).tobytes()).hexdigest()
        for sample in range(config.samples)
    }
    time_grid = config.sweep.time_grid_us
    digest, source_digest = config.digest, source_config.digest
    imported, already_present = 0, 0
    for task in source_tasks:
        original = base.cell_path(source, task)
        destination = base.cell_path(output, task)
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata_json"]))
            if metadata["configuration_hash"] != digest or metadata["task"] != asdict(task):
                raise ValueError(f"incompatible existing cell: {task.key}")
            already_present += 1
            continue
        with np.load(original, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"]))
            if (metadata["configuration_hash"] != source_digest
                    or metadata["task"] != asdict(task)
                    or metadata["initial_state_sha256"] != initial_hashes[task.sample_id]
                    or not np.array_equal(saved["times_us"], time_grid)):
                raise ValueError(f"incompatible source cell: {task.key}")
            values = {key: saved[key] for key in saved.files}
        metadata["configuration_hash"] = digest
        metadata["reused_from"] = {
            "path": str(original), "configuration_hash": source_digest,
            "file_sha256": sha256(original),
        }
        values["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
        base.atomic_npz(destination, values)
        imported += 1
    provenance = {
        "source_directory": str(source), "source_configuration_hash": source_digest,
        "configuration_hash": digest, "original_cells": len(source_tasks),
        "newly_imported": imported, "already_present": already_present,
        "note": "Only the inventory hash and provenance metadata change in imported copies.",
    }
    base.atomic_json(output / "reuse_provenance.json", provenance)
    return provenance


def curve_data(output, config):
    """Use only angles with all M samples in both conditions, even in previews."""
    angles, values, errors = [], [], []
    for angle in sorted(config.angles_deg):
        paths = [[base.cell_path(output, base.Task(q, angle, s))
                  for s in range(config.samples)] for q in base.SCALES]
        if not all(path.exists() for group in paths for path in group):
            continue
        y = np.empty((len(base.SCALES), config.samples))
        e = np.empty_like(y)
        for q, group in enumerate(paths):
            for s, path in enumerate(group):
                with np.load(path, allow_pickle=False) as data:
                    y[q, s] = float(data["yield"])
                    e[q, s] = sum(float(data[key]) for key in (
                        "yield_propagation_estimate", "quadrature_error", "yield_tail_bound"))
        angles.append(angle)
        values.append(y)
        errors.append(e)
    shape = (len(angles), len(base.SCALES), config.samples)
    return (np.array(angles), np.asarray(values).reshape(shape).transpose(1, 0, 2),
            np.asarray(errors).reshape(shape).transpose(1, 0, 2))


def intervals(values, errors, draws):
    lo, hi = np.quantile(values[..., draws].mean(axis=-1), [.025, .975], axis=-1)
    estimate = errors.mean(axis=-1)
    return lo - estimate, hi + estimate


def plot_curves(output, config, zoom_window=(87., 93.)):
    angles, values, errors = curve_data(output, config)
    if not len(angles) or 90. not in angles:
        return
    draws = np.random.default_rng(1701).integers(
        0, config.samples, (config.bootstrap_replicates, config.samples))
    lo, hi = intervals(values, errors, draws)
    at_90 = int(np.flatnonzero(angles == 90.)[0])
    zoom_start, zoom_stop = zoom_window
    zoom = (angles >= zoom_start) & (angles <= zoom_stop)
    delta = 1e3 * (values[:, zoom] - values[:, at_90, None, :])
    delta_errors = 1e3 * (errors[:, zoom] + errors[:, at_90, None, :])
    delta_errors[:, angles[zoom] == 90.] = 0.  # exact self-contrast
    zoom_lo, zoom_hi = intervals(delta, delta_errors, draws)
    complete = len(angles) == len(config.angles_deg)
    mode = "-" if complete else "none"
    suffix = "" if complete else "_preview"
    for q, (tag, color) in enumerate(zip(TAGS, ("#226ba4", "#cb621d"))):
        fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(9.6, 4.8),
            gridspec_kw={"width_ratios": (2.5, 1.2)}, layout="constrained")
        condition = "Quadrupoles off" if q == 0 else "All three nuclear quadrupoles on"
        fig.suptitle(f"{condition} | 14 nuclei, 50 μT, τ = 5 μs, D/J = −3")
        ax.plot(angles, values[q].mean(axis=-1), linestyle=mode,
            marker="o", ms=3., lw=1.45, color=color)
        if complete:
            ax.fill_between(angles, lo[q], hi[q], color=color, alpha=.2, linewidth=0)
        else:
            ax.errorbar(angles, values[q].mean(axis=-1),
                yerr=(values[q].mean(axis=-1)-lo[q], hi[q]-values[q].mean(axis=-1)),
                fmt="none", ecolor=color, alpha=.5, capsize=2)
        ax.set(xlim=(-2, 182), xticks=np.arange(0, 181, 30),
            xlabel="Field polar angle θ (degrees)", ylabel="Singlet yield Φₛ")
        ax_zoom.plot(angles[zoom], delta[q].mean(axis=-1), linestyle=mode,
            marker="o", ms=3., lw=1.45, color=color)
        if complete:
            ax_zoom.fill_between(angles[zoom], zoom_lo[q], zoom_hi[q],
                color=color, alpha=.2, linewidth=0)
        else:
            ax_zoom.errorbar(angles[zoom], delta[q].mean(axis=-1),
                yerr=(delta[q].mean(axis=-1)-zoom_lo[q], zoom_hi[q]-delta[q].mean(axis=-1)),
                fmt="none", ecolor=color, alpha=.5, capsize=2)
        ax_zoom.set(xlim=(zoom_start-.1, zoom_stop+.1), xticks=[zoom_start, 90, zoom_stop], title="Needle zoom",
            xlabel="Field angle (degrees)", ylabel="Yield relative to 90° (×10⁻³)")
        for panel, low, high in ((ax, lo, hi), (ax_zoom, zoom_lo, zoom_hi)):
            pad = max(1e-7, .07 * (high.max() - low.min()))
            panel.set_ylim(low.min()-pad, high.max()+pad)
            panel.axvline(90., color="#666666", lw=.75, ls=":", zorder=0)
            panel.grid(alpha=.2)
        label = (f"M={config.samples} paired samples; pointwise exploratory 95% intervals plus numerical estimates."
                 if complete else f"PREVIEW — {len(angles)}/{len(config.angles_deg)} angles, M={config.samples} paired. "
                 "Bars: exploratory 95% intervals + numerical estimates.")
        fig.supxlabel(label, fontsize=9)
        fig.savefig(output / f"figure1b_quadrupoles_{tag}{suffix}.png", dpi=250, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), default="run")
    parser.add_argument("--grid", choices=tuple(GRID_SETTINGS), default="dense",
        help="Dense: 61 angles. Reduced: 29 angles and 144 new trajectories after reuse.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare" and args.reuse_from is None:
        parser.error("--stage prepare requires --reuse-from")
    default_output = REDUCED_OUTPUT if args.grid == "reduced" else DEFAULT_OUTPUT
    output = (args.output_dir or default_output).resolve()
    config = configuration(args.grid)
    coarse_step, zoom_start, zoom_stop, refined_step = GRID_SETTINGS[args.grid]
    zoom_window = (zoom_start, zoom_stop)
    if args.reuse_from is not None and output == args.reuse_from.resolve():
        parser.error("source and extension outputs must be different")
    base.ensure_manifest(output, config)
    lock = base.acquire_run_lock(output)
    try:
        if args.reuse_from is not None:
            print(json.dumps(reuse_completed(args.reuse_from, output, config)), flush=True)
        done, planned = base.count_completed(output, config), len(base.tasks(config))
        base.atomic_json(output / "figure1b_design.json", {
            "reference": "Hiscock et al. (2016), Fig. 1B; single lifetime, extended Hamiltonian",
            "field_microT": 50, "field_plane": "molecular zx", "lifetime_us": 5.,
            "configuration_hash": config.digest, "samples": config.samples, "grid": args.grid,
            "angles_deg": list(config.angles_deg), "planned": planned,
            "original_cells": len(base.tasks(base.Config())), "new_trajectories": planned-len(base.tasks(base.Config())),
            "full_range_step_deg": coarse_step, "refined_window_deg": list(zoom_window), "refined_step_deg": refined_step,
        })
        if args.stage == "prepare":
            plot_curves(output, config, zoom_window)
            base.atomic_json(output / "status.json", {
                "status": "prepared", "grid": args.grid, "completed": done, "planned": planned,
                "new_trajectories_started": 0,
            })
            return
        if args.stage == "run":
            result = base.run_tasks(ordered_tasks(config, args.grid), config, output, args.workers,
                math.inf, 0., args.resume, "figure1b", "spare-capacity")
        base.report(output, config)
        plot_curves(output, config, zoom_window)
        if args.stage == "run":
            base.atomic_json(output / "status.json", {
                "status": result["status"], "completed": result["completed"], "planned": planned,
                "workers": args.workers, "wall_time_limit_enabled": False, "grid": args.grid,
            })
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
