"""Analyze checkpointed poster reruns and create the requested data products."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize

from rpm_quantum_needle_model import FAD_NUCLEI, TRP_NUCLEI, add_quadrupole_argument

from .core import (
    BOOTSTRAP_REPLICATES,
    ENTANGLEMENT_GRID_VERSION,
    ENTANGLEMENT_NUCLEAR_COUNTS,
    MASTER_SEED,
    N10_THETA_DEG,
    OVERHAUSER_FIELDS_MT,
    OVERHAUSER_STANDARD_ERROR_TARGET,
    POSTER_FIELDS_MT,
    POSTER_THETA_DEG,
    PRODUCTION_MODEL,
    angular_metrics,
    atomwise_overhauser_sample_mhz,
    bootstrap_curve_metric,
    entanglement_time_grids_us,
    mean_se_ci,
    mixed_two_electron_log_negativity_batch,
    overhauser_yield,
    scrambled_sobol_overhauser_samples_mhz,
)
from .runner import DEFAULT_OUTPUT, CellTask, cell_path


PROTOCOLS = ("baseline", "n10_transverse_deleted", "one_pulse", "periodic_0.02", "periodic_0.05")
PROTOCOL_LABELS = {
    "baseline": "baseline",
    "n10_transverse_deleted": "N10 transverse deleted",
    "one_pulse": "one N10 z pi pulse",
    "periodic_0.02": "periodic N10 z pi, 0.02 us",
    "periodic_0.05": "periodic N10 z pi, 0.05 us",
}


def _slug(value: float, digits: int = 3) -> str:
    return f"{float(value):0{digits + 4}.{digits}f}".replace(".", "p")


def quantum_task(field: float, theta: float, sample_id: int, samples: int, seed: int) -> CellTask:
    key = f"B{_slug(field)}_th{_slug(theta)}_s{sample_id:03d}"
    return CellTask("quantum", key, field, theta, sample_id, master_seed=seed, samples=samples)


def overhauser_task(
    field: float,
    theta: float,
    count: int,
    samples: int,
    seed: int,
    replicates: int = 8,
    atomwise_samples: int = 4096,
) -> CellTask:
    key = (
        f"B{_slug(field)}_th{_slug(theta)}_n{count:07d}"
        f"_r{replicates:02d}_a{atomwise_samples:05d}"
    )
    return CellTask(
        "overhauser",
        key,
        field,
        theta,
        overhauser_samples=count,
        overhauser_replicates=replicates,
        atomwise_samples=atomwise_samples,
        master_seed=seed,
        samples=samples,
    )


def entanglement_task(count: int, sample_id: int, samples: int, seed: int) -> CellTask:
    key = f"N{count:02d}_s{sample_id:03d}_{ENTANGLEMENT_GRID_VERSION}"
    return CellTask("entanglement", key, 0.05, 90.0, sample_id, count, master_seed=seed, samples=samples)


def n10_task(theta: float, protocol: str, sample_id: int, samples: int, seed: int) -> CellTask:
    key = f"{protocol}_th{_slug(theta)}_s{sample_id:03d}"
    return CellTask("n10", key, 0.05, theta, sample_id, protocol=protocol, master_seed=seed, samples=samples)


def reported_quadrupole_scale(output: Path) -> float:
    """Use the run manifest rather than the current default in report labels."""
    path = output / "model_manifest.json"
    if not path.exists():
        return 0.0  # Pre-manifest legacy reports only.
    return float(json.loads(path.read_text(encoding="utf-8"))["quadrupole_scale"])


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_best_overhauser_cell(
    output: Path,
    field: float,
    theta: float,
    minimum_samples: int,
    replicates: int,
    atomwise_samples: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Load the highest-precision compatible checkpoint for one OH cell.

    The production sweep establishes a common minimum sample count.  A small
    number of cells can then be refined independently when their replicate
    standard error misses the precision gate, without recomputing the other
    hundreds of completed cells.
    """

    pattern = (
        f"B{_slug(field)}_th{_slug(theta)}_n*_r{replicates:02d}"
        f"_a{atomwise_samples:05d}.npz"
    )
    candidates: list[tuple[int, Path]] = []
    for path in (output / "checkpoints" / "overhauser").glob(pattern):
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"]))
            task = metadata["task"]
            count = int(task["overhauser_samples"])
            if (
                count >= minimum_samples
                and int(task["overhauser_replicates"]) == replicates
                and int(task["atomwise_samples"]) == atomwise_samples
            ):
                candidates.append((count, path))
        except (KeyError, ValueError, OSError, json.JSONDecodeError, TypeError):
            continue
    if not candidates:
        expected = cell_path(
            output,
            overhauser_task(
                field,
                theta,
                minimum_samples,
                PRODUCTION_MODEL.samples,
                PRODUCTION_MODEL.master_seed,
                replicates,
                atomwise_samples,
            ),
        )
        raise FileNotFoundError(expected)
    count, path = max(candidates, key=lambda item: item[0])
    return _load(path), count


def analyze_runtime(output: Path) -> dict[str, float | int]:
    """Build a checkpoint-derived runtime ledger across all completed cells."""

    rows: list[dict[str, object]] = []
    cell_elapsed: list[float] = []
    starts: list[float] = []
    finishes: list[float] = []
    statevector_cells = 0
    semiclassical_realizations = 0
    checkpoint_root = output / "checkpoints"
    for path in sorted(checkpoint_root.glob("*/*.npz")):
        try:
            with np.load(path, allow_pickle=False) as cell:
                metadata = json.loads(str(cell["metadata_json"]))
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
        task = metadata.get("task", {})
        kind = str(task.get("kind", path.parent.name))
        elapsed = float(metadata.get("elapsed_seconds", math.nan))
        finish = float(path.stat().st_mtime)
        if np.isfinite(elapsed) and elapsed >= 0.0:
            cell_elapsed.append(elapsed)
            starts.append(finish - elapsed)
            finishes.append(finish)
        if kind in {"quantum", "entanglement", "n10"}:
            statevector_cells += 1
        elif kind == "overhauser":
            semiclassical_realizations += int(task.get("overhauser_samples", 0))
            semiclassical_realizations += int(task.get("atomwise_samples", 0))
        rows.append(
            {
                "record_type": "checkpoint_cell",
                "kind": kind,
                "key": task.get("key", path.stem),
                "protocol": task.get("protocol", ""),
                "field_mT": task.get("field_mT", ""),
                "theta_deg": task.get("theta_deg", ""),
                "sample_id": task.get("sample_id", ""),
                "elapsed_seconds": elapsed,
                "finished_unix_seconds": finish,
                "relative_path": str(path.relative_to(output)),
            }
        )

    invocation_elapsed: list[float] = []
    for path in sorted(output.glob("run_manifest_*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        elapsed = float(manifest.get("elapsed_seconds_this_invocation", math.nan))
        if np.isfinite(elapsed) and elapsed >= 0.0:
            invocation_elapsed.append(elapsed)
        rows.append(
            {
                "record_type": "runner_invocation",
                "kind": "-".join(manifest.get("kinds", [])),
                "key": path.stem,
                "elapsed_seconds": elapsed,
                "workers": manifest.get("workers", ""),
                "submitted_cells": manifest.get("submitted_this_invocation", ""),
                "completed_cells": manifest.get("completed_cells", ""),
                "status": manifest.get("status", ""),
                "relative_path": str(path.relative_to(output)),
            }
        )
    history_elapsed: list[float] = []
    for path in sorted((output / "run_history").glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        elapsed = float(manifest.get("elapsed_seconds_this_invocation", math.nan))
        if np.isfinite(elapsed) and elapsed >= 0.0:
            history_elapsed.append(elapsed)
        rows.append(
            {
                "record_type": "runner_invocation_history",
                "kind": "-".join(manifest.get("kinds", [])),
                "key": path.stem,
                "elapsed_seconds": elapsed,
                "workers": manifest.get("workers", ""),
                "submitted_cells": manifest.get("submitted_this_invocation", ""),
                "completed_cells": manifest.get("completed_cells", ""),
                "status": manifest.get("status", ""),
                "relative_path": str(path.relative_to(output)),
            }
        )
    _write_csv(output / "runtime_audit.csv", rows)
    return {
        "completed_checkpoint_cells": sum(
            row["record_type"] == "checkpoint_cell" for row in rows
        ),
        "completed_statevector_trajectories": statevector_cells,
        "completed_semiclassical_realizations": semiclassical_realizations,
        "aggregate_cell_compute_hours": float(np.sum(cell_elapsed) / 3600.0),
        "surviving_manifest_invocation_hours": float(
            np.sum(invocation_elapsed) / 3600.0
        ),
        "preserved_run_history_hours": float(np.sum(history_elapsed) / 3600.0),
        "first_to_last_checkpoint_span_hours": (
            float((max(finishes) - min(starts)) / 3600.0)
            if starts and finishes
            else 0.0
        ),
        "median_cell_seconds": (
            float(np.median(cell_elapsed)) if cell_elapsed else math.nan
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _bootstrap_indices(samples: int, seed: int, replicates: int = BOOTSTRAP_REPLICATES) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, samples, size=(replicates, samples))


def _bootstrap_electron_entanglement(
    density_samples: np.ndarray,
    sample_indices: np.ndarray,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Bootstrap log negativity after averaging density matrices first."""

    density = np.asarray(density_samples, dtype=complex)
    result = np.empty((sample_indices.shape[0], density.shape[1]), dtype=float)
    for start in range(0, sample_indices.shape[0], chunk_size):
        stop = min(sample_indices.shape[0], start + chunk_size)
        ids = sample_indices[start:stop]
        weights = np.zeros((ids.shape[0], density.shape[0]), dtype=float)
        for row in range(ids.shape[0]):
            np.add.at(weights[row], ids[row], 1.0)
        averaged = np.einsum("bm,mtij->btij", weights, density, optimize=True) / ids.shape[1]
        result[start:stop] = mixed_two_electron_log_negativity_batch(averaged)
    return result


def load_quantum_curves(output: Path, fields, theta, samples: int, seed: int) -> dict[float, np.ndarray]:
    result = {}
    for field in fields:
        values = np.empty((samples, len(theta)), dtype=float)
        for angle_index, angle in enumerate(theta):
            for sample_id in range(samples):
                task = quantum_task(float(field), float(angle), sample_id, samples, seed)
                values[sample_id, angle_index] = float(_load(cell_path(output, task))["yield"])
        result[float(field)] = values
    return result


def load_quantum_curves_adaptive(
    output: Path,
    fields,
    theta,
    minimum_samples: int,
    seed: int,
) -> dict[float, np.ndarray]:
    """Load every complete paired sample available for each field curve.

    A deadline run may reinforce selected Figure 3 fields with more nuclear
    samples than the all-field Figure 2 sweep.  Sample counts are chosen per
    field, never per angle, so every reconstructed angular curve remains
    paired and has a rectangular sample-by-angle array.
    """

    result: dict[float, np.ndarray] = {}
    angles = np.asarray(theta, dtype=float)
    for field_value in fields:
        field = float(field_value)
        count = 0
        while all(
            cell_path(output, quantum_task(field, float(angle), count, count + 1, seed)).exists()
            for angle in angles
        ):
            count += 1
        if count < minimum_samples:
            raise FileNotFoundError(
                f"Field {field:g} mT has {count} complete paired quantum samples; "
                f"at least {minimum_samples} are required."
            )
        values = np.empty((count, angles.size), dtype=float)
        for angle_index, angle in enumerate(angles):
            for sample_id in range(count):
                task = quantum_task(field, float(angle), sample_id, count, seed)
                values[sample_id, angle_index] = float(
                    _load(cell_path(output, task))["yield"]
                )
        result[field] = values
    return result


def analyze_figure2(output: Path, samples: int, seed: int) -> tuple[dict[float, np.ndarray], list[dict[str, object]]]:
    theta = np.asarray(POSTER_THETA_DEG)
    q = reported_quadrupole_scale(output)
    curves = load_quantum_curves(output, POSTER_FIELDS_MT, theta, samples, seed)
    rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    colors = plt.cm.viridis(np.linspace(0.02, 0.98, len(POSTER_FIELDS_MT)))
    fig, axis = plt.subplots(figsize=(8.7, 5.6))
    for color, field in zip(colors, POSTER_FIELDS_MT):
        values = curves[field]
        mean, se, low, high = mean_se_ci(values)
        for index, angle in enumerate(theta):
            rows.append(
                {
                    "field_mT": field,
                    "theta_deg": angle,
                    "samples": samples,
                    "mean_yield": mean[index],
                    "standard_error": se[index],
                    "ci95_low": low[index],
                    "ci95_high": high[index],
                    "quadrupole_scale": q,
                }
            )
        axis.plot(theta, mean, color=color, marker="o", markersize=2.2, linewidth=1.2, label=f"{field:g} mT")
        axis.fill_between(theta, low, high, color=color, alpha=0.08)
        derivative = np.gradient(mean, theta, edge_order=2)
        centers = (theta[int(np.argmin(mean))], theta[int(np.argmax(mean))], theta[int(np.argmax(np.abs(derivative)))])
        refined_theta = np.array(
            sorted(
                set(theta.tolist())
                | {
                    float(value)
                    for center in centers
                    for value in np.arange(max(0.0, center - 5.0), min(180.0, center + 5.0) + 0.5, 1.0)
                }
            )
        )
        refined_values = np.empty((samples, refined_theta.size), dtype=float)
        refinement_complete = True
        for angle_index, angle in enumerate(refined_theta):
            coarse_match = np.where(np.isclose(theta, angle))[0]
            if coarse_match.size:
                refined_values[:, angle_index] = values[:, coarse_match[0]]
                continue
            for sample_id in range(samples):
                path = cell_path(output, quantum_task(field, float(angle), sample_id, samples, seed))
                if not path.exists():
                    refinement_complete = False
                    break
                refined_values[sample_id, angle_index] = float(_load(path)["yield"])
            if not refinement_complete:
                break
        metric_theta = refined_theta if refinement_complete else theta
        metric_values = refined_values if refinement_complete else values
        point = angular_metrics(metric_theta, np.mean(metric_values, axis=0))
        boot_ids = _bootstrap_indices(samples, seed ^ int(round(field * 1000)) ^ 0xF2)
        boot_metrics = [angular_metrics(metric_theta, np.mean(metric_values[ids], axis=0)) for ids in boot_ids]
        metric_row: dict[str, object] = {
            "field_mT": field,
            "samples": samples,
            "one_degree_refinement_complete": refinement_complete,
            **point,
        }
        for name in ("contrast", "relative_contrast", "maximum_abs_slope_per_deg"):
            distribution = np.array([item[name] for item in boot_metrics])
            metric_row[f"{name}_ci95_low"], metric_row[f"{name}_ci95_high"] = np.quantile(distribution, [0.025, 0.975])
        metrics_rows.append(metric_row)
    axis.set(xlabel="magnetic-field polar angle (deg)", ylabel="singlet yield",
             title=f"Full-state field dependence (Q scale={q:g})")
    axis.legend(ncol=2, fontsize=7, frameon=False)
    _save_figure(fig, output / "figures" / "figure2_field_dependence")

    fields = np.asarray(POSTER_FIELDS_MT)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.0), sharex=True)
    for axis, key, ylabel in zip(
        axes,
        ("contrast", "relative_contrast", "maximum_abs_slope_per_deg"),
        ("absolute contrast", "relative contrast", "maximum |slope| per degree"),
    ):
        values = np.array([row[key] for row in metrics_rows])
        low = np.array([row[f"{key}_ci95_low"] for row in metrics_rows])
        high = np.array([row[f"{key}_ci95_high"] for row in metrics_rows])
        axis.errorbar(fields, values, yerr=np.vstack([values - low, high - values]), marker="o", capsize=2)
        axis.set_xscale("log")
        axis.set_ylabel(ylabel)
    axes[-1].set_xlabel("field strength (mT)")
    fig.suptitle(f"Directional-response metrics (Q scale={q:g})")
    _save_figure(fig, output / "figures" / "figure2b_contrast_vs_field")

    matrix = np.array([np.mean(curves[field], axis=0) for field in POSTER_FIELDS_MT])
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    mesh = axis.pcolormesh(theta, fields, matrix - matrix.mean(axis=1, keepdims=True), shading="nearest", cmap="coolwarm")
    axis.set_yscale("log")
    axis.set(xlabel="magnetic-field polar angle (deg)", ylabel="field strength (mT)",
             title=f"Mean-subtracted directional singlet yield (Q scale={q:g})")
    fig.colorbar(mesh, ax=axis, label="yield minus angular mean")
    _save_figure(fig, output / "figures" / "figure2c_field_angle_heatmap")
    _write_csv(output / "field_dependence.csv", rows)
    _write_csv(output / "field_metrics.csv", metrics_rows)
    return curves, metrics_rows


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def analyze_figure3(
    output: Path,
    quantum_curves: dict[float, np.ndarray],
    samples: int,
    overhauser_samples: int,
    seed: int,
    overhauser_replicates: int = 8,
    atomwise_samples: int = 4096,
) -> list[dict[str, object]]:
    theta = np.asarray(POSTER_THETA_DEG)
    curve_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    dynamics_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True)
    error_field: list[float] = []
    error_rmse: list[float] = []
    contrast_ratio: list[float] = []
    for field in POSTER_FIELDS_MT:
        quantum = quantum_curves[field]
        quantum_samples = int(quantum.shape[0])
        q_mean, q_se, q_low, q_high = mean_se_ci(quantum)
        oh = np.empty((overhauser_replicates, len(theta)))
        atomwise = np.empty((atomwise_samples, len(theta)))
        oh_counts = np.empty(len(theta), dtype=np.int64)
        for index, angle in enumerate(theta):
            cell, actual_count = _load_best_overhauser_cell(
                output,
                field,
                float(angle),
                overhauser_samples,
                overhauser_replicates,
                atomwise_samples,
            )
            oh[:, index] = cell["replicate_means"]
            atomwise[:, index] = cell["atomwise_sample_yields"]
            oh_counts[index] = actual_count
        o_mean, o_se, o_low, o_high = mean_se_ci(oh)
        a_mean, a_se, a_low, a_high = mean_se_ci(atomwise)
        for index, angle in enumerate(theta):
            for model_name, mean, se, low, high, count in (
                ("full_quantum", q_mean, q_se, q_low, q_high, quantum_samples),
                (
                    "gaussian_static_overhauser",
                    o_mean,
                    o_se,
                    o_low,
                    o_high,
                    int(oh_counts[index]),
                ),
                ("atomwise_static_spin", a_mean, a_se, a_low, a_high, atomwise_samples),
            ):
                curve_rows.append({"field_mT": field, "theta_deg": angle, "model": model_name, "samples": count, "mean_yield": mean[index], "standard_error": se[index], "ci95_low": low[index], "ci95_high": high[index]})
        difference = q_mean - o_mean
        q_metric = angular_metrics(theta, q_mean)
        o_metric = angular_metrics(theta, o_mean)
        row = {
            "field_mT": field,
            "rmse_theta": float(np.sqrt(np.mean(difference**2))),
            "max_abs_error": float(np.max(np.abs(difference))),
            "mean_yield_bias_OH_minus_Q": float(np.mean(o_mean - q_mean)),
            "pearson_shape_correlation": _correlation(q_mean, o_mean),
            "quantum_contrast": q_metric["contrast"],
            "overhauser_contrast": o_metric["contrast"],
            "contrast_ratio_OH_over_Q": o_metric["contrast"] / q_metric["contrast"] if q_metric["contrast"] > 0 else math.nan,
            "minimum_orientation_difference_deg": o_metric["theta_minimum_deg"] - q_metric["theta_minimum_deg"],
            "maximum_orientation_difference_deg": o_metric["theta_maximum_deg"] - q_metric["theta_maximum_deg"],
            "quantum_samples": quantum_samples,
            "overhauser_samples": int(np.min(oh_counts)),
            "minimum_overhauser_samples": int(np.min(oh_counts)),
            "maximum_overhauser_samples": int(np.max(oh_counts)),
            "overhauser_replicates": overhauser_replicates,
            "points_per_replicate": int(np.min(oh_counts))
            // overhauser_replicates,
            "minimum_points_per_replicate": int(np.min(oh_counts))
            // overhauser_replicates,
            "maximum_points_per_replicate": int(np.max(oh_counts))
            // overhauser_replicates,
        }
        rng = np.random.default_rng(seed ^ int(round(field * 1000.0)) ^ 0xF311)
        bootstrap_values = {key: [] for key in (
            "rmse_theta",
            "max_abs_error",
            "mean_yield_bias_OH_minus_Q",
            "quantum_contrast",
            "overhauser_contrast",
            "contrast_ratio_OH_over_Q",
        )}
        for _ in range(BOOTSTRAP_REPLICATES):
            q_ids = rng.integers(0, quantum_samples, size=quantum_samples)
            o_ids = rng.integers(0, overhauser_replicates, size=overhauser_replicates)
            q_curve = np.mean(quantum[q_ids], axis=0)
            o_curve = np.mean(oh[o_ids], axis=0)
            q_boot_metric = angular_metrics(theta, q_curve)
            o_boot_metric = angular_metrics(theta, o_curve)
            delta = q_curve - o_curve
            bootstrap_values["rmse_theta"].append(float(np.sqrt(np.mean(delta**2))))
            bootstrap_values["max_abs_error"].append(float(np.max(np.abs(delta))))
            bootstrap_values["mean_yield_bias_OH_minus_Q"].append(float(np.mean(o_curve - q_curve)))
            bootstrap_values["quantum_contrast"].append(q_boot_metric["contrast"])
            bootstrap_values["overhauser_contrast"].append(o_boot_metric["contrast"])
            bootstrap_values["contrast_ratio_OH_over_Q"].append(
                o_boot_metric["contrast"] / q_boot_metric["contrast"]
                if q_boot_metric["contrast"] > 0.0
                else math.nan
            )
        for name, distribution in bootstrap_values.items():
            row[f"{name}_ci95_low"], row[f"{name}_ci95_high"] = np.nanquantile(distribution, [0.025, 0.975])
        metric_rows.append(row)
        error_field.append(field)
        error_rmse.append(row["rmse_theta"])
        contrast_ratio.append(row["contrast_ratio_OH_over_Q"])
        if field in OVERHAUSER_FIELDS_MT:
            panel = OVERHAUSER_FIELDS_MT.index(field)
            axis = axes[panel]
            axis.plot(theta, q_mean, color="black", label="full quantum")
            axis.fill_between(theta, q_low, q_high, color="black", alpha=0.15)
            axis.plot(theta, o_mean, color="#d95f02", label="Gaussian static")
            axis.fill_between(theta, o_low, o_high, color="#d95f02", alpha=0.15)
            axis.plot(theta, a_mean, color="#1b9e77", linestyle="--", label="atomwise static")
            axis.set_title(f"{field:g} mT")
            axis.set_xlabel("angle (deg)")
            if panel == 0:
                axis.set_ylabel("singlet yield")
        representative = (
            sorted({0.0, 90.0, 180.0, q_metric["theta_minimum_deg"], q_metric["theta_maximum_deg"]})
            if field in OVERHAUSER_FIELDS_MT
            else []
        )
        for angle in representative:
            angle_index = int(np.where(np.isclose(theta, angle))[0][0])
            q_pop = []
            for sample_id in range(quantum_samples):
                cell = _load(
                    cell_path(
                        output,
                        quantum_task(
                            field,
                            angle,
                            sample_id,
                            quantum_samples,
                            seed,
                        ),
                    )
                )
                q_pop.append(cell["singlet_population"])
            q_pop_mean_full = np.mean(q_pop, axis=0)
            cell, _ = _load_best_overhauser_cell(
                output,
                field,
                angle,
                overhauser_samples,
                overhauser_replicates,
                atomwise_samples,
            )
            if cell["singlet_population_samples"].size:
                times = cell["times_us"]
                oh_pop_mean = np.mean(cell["singlet_population_samples"], axis=0)
            else:
                times = np.linspace(0.0, PRODUCTION_MODEL.sweep_config.maximum_time_us, 1001)
                replicate_dynamics = np.zeros((overhauser_replicates, times.size))
                dynamics_points = 256
                for replicate_id in range(overhauser_replicates):
                    fad_fields, trp_fields = scrambled_sobol_overhauser_samples_mhz(
                        seed, replicate_id, dynamics_points
                    )
                    for sample_id in range(dynamics_points):
                        _, pop = overhauser_yield(
                            angle,
                            field,
                            (fad_fields[sample_id], trp_fields[sample_id]),
                            return_dynamics=True,
                            dynamics_times_us=times,
                        )
                        assert pop is not None
                        replicate_dynamics[replicate_id] += pop / dynamics_points
                oh_pop_mean = np.mean(replicate_dynamics, axis=0)
            q_times = PRODUCTION_MODEL.sweep_config.time_grid_us
            q_pop_mean = np.interp(times, q_times, q_pop_mean_full)
            rmse_t = float(np.sqrt(np.mean((q_pop_mean - oh_pop_mean) ** 2)))
            iae = float(scipy.integrate.simpson(np.abs(q_pop_mean - oh_pop_mean), x=times))
            for time_value, q_value, o_value in zip(times, q_pop_mean, oh_pop_mean):
                dynamics_rows.append({"field_mT": field, "theta_deg": angle, "time_us": time_value, "quantum_mean_Ps": q_value, "overhauser_mean_Ps": o_value, "time_domain_rmse": rmse_t, "integrated_absolute_error": iae})
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Full quantum (Q scale={reported_quadrupole_scale(output):g}) "
        "vs covariance-matched static Overhauser"
    )
    _save_figure(fig, output / "figures" / "figure3_overhauser_comparison")
    fig, left = plt.subplots(figsize=(7.2, 4.5))
    right = left.twinx()
    left.plot(error_field, error_rmse, marker="o", color="#7570b3", label="angular RMSE")
    right.plot(error_field, contrast_ratio, marker="s", color="#e7298a", label="contrast ratio")
    left.set_xscale("log")
    left.set(xlabel="field strength (mT)", ylabel="angular RMSE")
    right.set_ylabel("Overhauser / quantum contrast")
    left.set_title("Static-Overhauser approximation error")
    _save_figure(fig, output / "figures" / "figure3b_overhauser_error_vs_field")
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 8.4), sharex=True, sharey=True)
    for row_axis, field in zip(axes, OVERHAUSER_FIELDS_MT):
        selected = [row for row in dynamics_rows if row["field_mT"] == field and row["theta_deg"] in (0.0, 90.0, 180.0)]
        for axis, angle in zip(row_axis, (0.0, 90.0, 180.0)):
            rows = [row for row in selected if row["theta_deg"] == angle]
            axis.plot([row["time_us"] for row in rows], [row["quantum_mean_Ps"] for row in rows], color="black")
            axis.plot([row["time_us"] for row in rows], [row["overhauser_mean_Ps"] for row in rows], color="#d95f02")
            axis.set_title(f"{field:g} mT, {angle:g} deg")
    fig.supxlabel("time (us)")
    fig.supylabel("singlet population")
    _save_figure(fig, output / "figures" / "figure3c_overhauser_dynamics")
    _write_csv(output / "overhauser_curves.csv", curve_rows)
    _write_csv(output / "overhauser_metrics.csv", metric_rows)
    _write_csv(output / "overhauser_dynamics.csv", dynamics_rows)
    return metric_rows


def _fit_models(x: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    rss = float(np.sum((y - predicted) ** 2))
    rows.append({"model": "linear", "intercept": coefficients[0], "slope": coefficients[1], "rss": rss, "aic": len(x) * math.log(max(rss / len(x), 1e-30)) + 4})
    def power(value, a, b, c):
        return a + b * value**c
    try:
        parameters, _ = scipy.optimize.curve_fit(power, x, y, p0=(0.0, 0.2, 0.8), maxfev=20000)
        residual = y - power(x, *parameters)
        rss = float(np.sum(residual**2))
        rows.append({"model": "power", "a": parameters[0], "b": parameters[1], "exponent": parameters[2], "rss": rss, "aic": len(x) * math.log(max(rss / len(x), 1e-30)) + 6})
    except (RuntimeError, ValueError):
        rows.append({"model": "power", "a": math.nan, "b": math.nan, "exponent": math.nan, "rss": math.inf, "aic": math.inf})
    return rows


def analyze_figure4(output: Path, samples: int, seed: int) -> list[dict[str, object]]:
    scaling_rows: list[dict[str, object]] = []
    electron_rows: list[dict[str, object]] = []
    electron_average_rows: list[dict[str, object]] = []
    initial_rows: list[dict[str, object]] = []
    quadrature_rows: list[dict[str, object]] = []
    expected_times, coarse_indices = entanglement_time_grids_us()
    radical_means = []
    en_means = []
    physical_maxima = []
    time_by_count = {}
    electron_curve_by_count = {}
    electron_curve_interval_by_count = {}
    radical_samples_by_count = {}
    en_samples_by_count = {}
    electron_bootstrap_ids = _bootstrap_indices(samples, seed ^ 0xE1EC)
    for count in ENTANGLEMENT_NUCLEAR_COUNTS:
        radical_sample = []
        en_sample = []
        radical_quadrature_difference = []
        en_quadrature_difference = []
        rho_samples = []
        times = None
        for sample_id in range(samples):
            cell = _load(cell_path(output, entanglement_task(count, sample_id, samples, seed)))
            times = cell["times_us"]
            if not np.allclose(times, expected_times, rtol=0.0, atol=1e-13):
                raise RuntimeError("Figure 4 checkpoint uses an incompatible time grid.")
            radical_sample.append(scipy.integrate.simpson(cell["radical_log_negativity"], x=times) / (times[-1] - times[0]))
            en_sample.append(scipy.integrate.simpson(cell["electron_nuclear_log_negativity"], x=times) / (times[-1] - times[0]))
            radical_coarse = scipy.integrate.simpson(
                cell["radical_log_negativity"][coarse_indices], x=times[coarse_indices]
            ) / (times[-1] - times[0])
            en_coarse = scipy.integrate.simpson(
                cell["electron_nuclear_log_negativity"][coarse_indices], x=times[coarse_indices]
            ) / (times[-1] - times[0])
            radical_quadrature_difference.append(abs(radical_sample[-1] - radical_coarse))
            en_quadrature_difference.append(abs(en_sample[-1] - en_coarse))
            rho_samples.append(cell["electron_density_real"] + 1.0j * cell["electron_density_imag"])
            initial_rows.append({"nuclear_count": count, "sample_id": sample_id, "radical_initial": float(cell["initial_radical_log_negativity"]), "electron_nuclear_initial": float(cell["initial_electron_nuclear_log_negativity"]), "radical_generated_time_average": radical_sample[-1] - float(cell["initial_radical_log_negativity"]), "electron_nuclear_generated_time_average": en_sample[-1]})
        assert times is not None
        radical_sample = np.asarray(radical_sample)
        en_sample = np.asarray(en_sample)
        radical_samples_by_count[count] = radical_sample
        en_samples_by_count[count] = en_sample
        rho_samples = np.asarray(rho_samples)
        rho_mean = np.mean(rho_samples, axis=0)
        electron_curve = mixed_two_electron_log_negativity_batch(rho_mean)
        electron_average = float(scipy.integrate.simpson(electron_curve, x=times) / (times[-1] - times[0]))
        electron_bootstrap_curves = _bootstrap_electron_entanglement(
            rho_samples, electron_bootstrap_ids
        )
        electron_curve_se = np.std(electron_bootstrap_curves, axis=0, ddof=1)
        electron_curve_low, electron_curve_high = np.quantile(
            electron_bootstrap_curves, [0.025, 0.975], axis=0
        )
        electron_bootstrap_average = (
            scipy.integrate.simpson(electron_bootstrap_curves, x=times, axis=1)
            / (times[-1] - times[0])
        )
        electron_average_se = float(np.std(electron_bootstrap_average, ddof=1))
        electron_average_low, electron_average_high = np.quantile(
            electron_bootstrap_average, [0.025, 0.975]
        )
        electron_coarse_average = float(
            scipy.integrate.simpson(electron_curve[coarse_indices], x=times[coarse_indices])
            / (times[-1] - times[0])
        )
        for cut, differences in (
            ("radical_partition", radical_quadrature_difference),
            ("electron_nuclear_trajectory_proxy", en_quadrature_difference),
        ):
            quadrature_rows.append(
                {
                    "nuclear_count": count,
                    "cut": cut,
                    "fine_time_points": times.size,
                    "coarse_time_points": coarse_indices.size,
                    "mean_absolute_fine_minus_coarse": float(np.mean(differences)),
                    "maximum_absolute_fine_minus_coarse": float(np.max(differences)),
                }
            )
        quadrature_rows.append(
            {
                "nuclear_count": count,
                "cut": "ensemble_averaged_two_electron",
                "fine_time_points": times.size,
                "coarse_time_points": coarse_indices.size,
                "mean_absolute_fine_minus_coarse": abs(electron_average - electron_coarse_average),
                "maximum_absolute_fine_minus_coarse": abs(electron_average - electron_coarse_average),
            }
        )
        time_by_count[count] = times
        electron_curve_by_count[count] = electron_curve
        electron_curve_interval_by_count[count] = (electron_curve_low, electron_curve_high)
        electron_average_rows.append(
            {
                "nuclear_count": count,
                "samples": samples,
                "time_average": electron_average,
                "bootstrap_standard_error": electron_average_se,
                "bootstrap_ci95_low": float(electron_average_low),
                "bootstrap_ci95_high": float(electron_average_high),
            }
        )
        per = count // 2
        dim_fad = 2 * math.prod(n.dimension for n in FAD_NUCLEI[:per])
        dim_trp = 2 * math.prod(n.dimension for n in TRP_NUCLEI[:per])
        physical_maximum = math.log2(min(dim_fad, dim_trp))
        physical_maxima.append(physical_maximum)
        radical_mean, radical_se, radical_low, radical_high = mean_se_ci(radical_sample[:, None])
        en_mean, en_se, en_low, en_high = mean_se_ci(en_sample[:, None])
        radical_means.append(float(radical_mean[0]))
        en_means.append(float(en_mean[0]))
        for cut, values, mean, se, low, high in (
            ("radical_partition", radical_sample, radical_mean, radical_se, radical_low, radical_high),
            ("electron_nuclear_trajectory_proxy", en_sample, en_mean, en_se, en_low, en_high),
        ):
            scaling_rows.append({"nuclear_count": count, "cut": cut, "samples": samples, "mean_time_averaged_log_negativity": mean[0], "standard_error": se[0], "ci95_low": low[0], "ci95_high": high[0], "physical_maximum": physical_maximum if cut == "radical_partition" else 2.0, "label": "Mean pure-trajectory log negativity for a product-state decomposition of the nuclear mixture"})
        for index, (t, value) in enumerate(zip(times, electron_curve)):
            electron_rows.append(
                {
                    "nuclear_count": count,
                    "time_us": t,
                    "ensemble_averaged_two_electron_log_negativity": value,
                    "bootstrap_standard_error": electron_curve_se[index],
                    "bootstrap_ci95_low": electron_curve_low[index],
                    "bootstrap_ci95_high": electron_curve_high[index],
                    "time_average": electron_average,
                    "samples": samples,
                }
            )
    fit_rows = _fit_models(np.asarray(ENTANGLEMENT_NUCLEAR_COUNTS, dtype=float), np.asarray(radical_means))
    def saturation(value, a, n0):
        return 2.0 - a * np.exp(-value / n0)
    try:
        parameters, _ = scipy.optimize.curve_fit(saturation, ENTANGLEMENT_NUCLEAR_COUNTS, en_means, p0=(2.0, 5.0), bounds=([0.0, 0.01], [10.0, 100.0]))
        fit_rows.append({"model": "bounded_saturation_electron_nuclear", "a": parameters[0], "N0": parameters[1]})
    except (RuntimeError, ValueError):
        fit_rows.append({"model": "bounded_saturation_electron_nuclear", "a": math.nan, "N0": math.nan})
    selected_radical = min((row for row in fit_rows if row["model"] in ("linear", "power")), key=lambda row: row["aic"])
    selected_radical["selected_by"] = "minimum_AIC"
    counts = np.asarray(ENTANGLEMENT_NUCLEAR_COUNTS, dtype=float)
    extrapolated = np.arange(16.0, 27.0, 2.0)
    extrapolated_physical_maxima = physical_maxima[-1] + (extrapolated - counts[-1]) / 2.0
    bootstrap_ids = _bootstrap_indices(samples, seed ^ 0xF417)
    radical_predictions = []
    en_predictions = []
    for ids in bootstrap_ids:
        radical_boot = np.array([np.mean(radical_samples_by_count[count][ids]) for count in ENTANGLEMENT_NUCLEAR_COUNTS])
        if selected_radical["model"] == "linear":
            coefficients, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(counts), counts]), radical_boot, rcond=None)
            radical_predictions.append(coefficients[0] + coefficients[1] * extrapolated)
        else:
            def power(value, a, b, c):
                return a + b * value**c
            try:
                parameters, _ = scipy.optimize.curve_fit(power, counts, radical_boot, p0=(0.0, 0.2, 0.8), maxfev=5000)
                radical_predictions.append(power(extrapolated, *parameters))
            except (RuntimeError, ValueError):
                pass
        en_boot = np.array([np.mean(en_samples_by_count[count][ids]) for count in ENTANGLEMENT_NUCLEAR_COUNTS])
        try:
            parameters, _ = scipy.optimize.curve_fit(saturation, counts, en_boot, p0=(2.0, 5.0), bounds=([0.0, 0.01], [10.0, 100.0]), maxfev=5000)
            en_predictions.append(saturation(extrapolated, *parameters))
        except (RuntimeError, ValueError):
            pass
    radical_prediction_low, radical_prediction_high = np.quantile(radical_predictions, [0.025, 0.975], axis=0)
    en_prediction_low, en_prediction_high = np.quantile(en_predictions, [0.025, 0.975], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    radical_summary = [row for row in scaling_rows if row["cut"] == "radical_partition"]
    en_summary = [row for row in scaling_rows if row["cut"] == "electron_nuclear_trajectory_proxy"]
    radical_low = np.array([row["ci95_low"] for row in radical_summary])
    radical_high = np.array([row["ci95_high"] for row in radical_summary])
    en_low = np.array([row["ci95_low"] for row in en_summary])
    en_high = np.array([row["ci95_high"] for row in en_summary])
    axes[0].errorbar(
        counts,
        radical_means,
        yerr=np.vstack([np.asarray(radical_means) - radical_low, radical_high - np.asarray(radical_means)]),
        marker="o",
        capsize=3,
        label="explicit mean and 95% CI",
    )
    axes[0].plot(
        np.concatenate([counts, extrapolated]),
        np.concatenate([physical_maxima, extrapolated_physical_maxima]),
        linestyle="--",
        label="physical maximum (spin-1/2 continuation after N=14)",
    )
    if selected_radical["model"] == "linear":
        prediction = selected_radical["intercept"] + selected_radical["slope"] * extrapolated
    else:
        prediction = selected_radical["a"] + selected_radical["b"] * extrapolated ** selected_radical["exponent"]
    axes[0].plot(extrapolated, prediction, marker="x", linestyle=":", label="phenomenological prediction")
    axes[0].fill_between(extrapolated, radical_prediction_low, radical_prediction_high, alpha=0.18, label="bootstrap 95% prediction band")
    axes[0].set(xlabel="nuclear count", ylabel="time-averaged log negativity", title="radical partition")
    axes[1].errorbar(
        counts,
        en_means,
        yerr=np.vstack([np.asarray(en_means) - en_low, en_high - np.asarray(en_means)]),
        marker="o",
        capsize=3,
        label="trajectory proxy and 95% CI",
    )
    bounded = next(row for row in fit_rows if row["model"] == "bounded_saturation_electron_nuclear")
    if np.isfinite(bounded["a"]):
        bounded_prediction = saturation(extrapolated, bounded["a"], bounded["N0"])
        axes[1].plot(extrapolated, bounded_prediction, marker="x", linestyle=":", label="bounded phenomenological prediction")
        axes[1].fill_between(extrapolated, en_prediction_low, en_prediction_high, alpha=0.18, label="bootstrap 95% prediction band")
    axes[1].axhline(2.0, linestyle="--", color="gray", label="physical maximum")
    axes[1].set(xlabel="nuclear count", title="electron--nuclear trajectory proxy")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Entanglement scaling (Q scale={reported_quadrupole_scale(output):g})")
    _save_figure(fig, output / "figures" / "figure4_entanglement_scaling")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for count in ENTANGLEMENT_NUCLEAR_COUNTS:
        low, high = electron_curve_interval_by_count[count]
        axes[0].plot(time_by_count[count], electron_curve_by_count[count], label=f"N={count}")
        axes[0].fill_between(time_by_count[count], low, high, alpha=0.08)
    axes[0].set(
        xlabel="time (us)",
        ylabel="two-electron log negativity",
        title="ensemble-averaged dynamics",
    )
    axes[0].set_xlim(0.0, 0.1)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    electron_averages = np.array([row["time_average"] for row in electron_average_rows])
    electron_average_low = np.array([row["bootstrap_ci95_low"] for row in electron_average_rows])
    electron_average_high = np.array([row["bootstrap_ci95_high"] for row in electron_average_rows])
    axes[1].errorbar(
        counts,
        electron_averages,
        yerr=np.vstack(
            [
                np.maximum(0.0, electron_averages - electron_average_low),
                np.maximum(0.0, electron_average_high - electron_averages),
            ]
        ),
        marker="o",
        capsize=3,
    )
    axes[1].set(
        xlabel="nuclear count",
        ylabel="0--2 us time-averaged log negativity",
        title="physical electron-entanglement average",
    )
    fig.suptitle("Ensemble-averaged physical two-electron entanglement")
    _save_figure(fig, output / "figures" / "figure4b_two_electron_entanglement")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), sharex="col")
    for axis, key, expected, title in (
        (axes[0, 0], "radical_initial", 1.0, "radical cut: initial"),
        (axes[0, 1], "electron_nuclear_initial", 0.0, "electron--nuclear cut: initial"),
    ):
        for count in ENTANGLEMENT_NUCLEAR_COUNTS:
            values = [row[key] for row in initial_rows if row["nuclear_count"] == count]
            axis.scatter([count] * len(values), values, s=10)
        axis.axhline(expected, color="black", linestyle="--")
        axis.set(xlabel="nuclear count", ylabel="initial log negativity", title=title)
    for axis, key, title in (
        (axes[1, 0], "radical_generated_time_average", "radical cut: generated change"),
        (axes[1, 1], "electron_nuclear_generated_time_average", "electron--nuclear cut: generated proxy"),
    ):
        for count in ENTANGLEMENT_NUCLEAR_COUNTS:
            values = np.asarray([row[key] for row in initial_rows if row["nuclear_count"] == count])
            mean, se, low, high = mean_se_ci(values[:, None])
            axis.errorbar(
                count,
                mean[0],
                yerr=[[mean[0] - low[0]], [high[0] - mean[0]]],
                marker="o",
                capsize=3,
            )
        axis.axhline(0.0, color="black", linestyle="--")
        axis.set(xlabel="nuclear count", ylabel="time-averaged change", title=title)
    _save_figure(fig, output / "figures" / "figure4c_initial_state_check")
    for index, count in enumerate(extrapolated):
        fit_rows.append({
            "model": f"{selected_radical['model']}_radical_prediction",
            "nuclear_count": count,
            "prediction": prediction[index],
            "prediction_ci95_low": radical_prediction_low[index],
            "prediction_ci95_high": radical_prediction_high[index],
            "label": "phenomenological prediction",
        })
        if np.isfinite(bounded["a"]):
            fit_rows.append({
                "model": "bounded_saturation_electron_nuclear_prediction",
                "nuclear_count": count,
                "prediction": bounded_prediction[index],
                "prediction_ci95_low": en_prediction_low[index],
                "prediction_ci95_high": en_prediction_high[index],
                "label": "phenomenological prediction",
            })
        fit_rows.append(
            {
                "model": "radical_partition_physical_maximum",
                "nuclear_count": count,
                "prediction": extrapolated_physical_maxima[index],
                "label": "spin-1/2 continuation after N=14; bound, not an entanglement prediction",
            }
        )
    _write_csv(output / "entanglement_scaling.csv", scaling_rows)
    _write_csv(output / "entanglement_fit_parameters.csv", fit_rows)
    _write_csv(output / "two_electron_entanglement.csv", electron_rows)
    _write_csv(output / "two_electron_entanglement_time_average.csv", electron_average_rows)
    _write_csv(output / "entanglement_initial_state_check.csv", initial_rows)
    scaling_se = {
        (int(row["nuclear_count"]), str(row["cut"])): float(row["standard_error"])
        for row in scaling_rows
    }
    electron_se = {
        int(row["nuclear_count"]): float(row["bootstrap_standard_error"])
        for row in electron_average_rows
    }
    for row in quadrature_rows:
        count = int(row["nuclear_count"])
        cut = str(row["cut"])
        uncertainty = (
            electron_se[count]
            if cut == "ensemble_averaged_two_electron"
            else scaling_se[(count, cut)]
        )
        difference = float(row["mean_absolute_fine_minus_coarse"])
        ratio = difference / uncertainty if uncertainty > 0.0 else math.inf
        row.update(
            {
                "uncertainty_standard_error": uncertainty,
                "mean_difference_to_standard_error_ratio": ratio,
                "gate_maximum_ratio": 0.25,
                "passed_quadrature_gate": ratio <= 0.25,
            }
        )
    _write_csv(output / "entanglement_quadrature_check.csv", quadrature_rows)
    return scaling_rows


def _n10_metrics(theta: np.ndarray, curve: np.ndarray) -> dict[str, float]:
    metrics = angular_metrics(theta, curve)
    derivative = np.gradient(curve, theta, edge_order=2)
    curvature = np.gradient(derivative, theta, edge_order=2)
    center = int(np.argmin(np.abs(theta - 90.0)))
    metrics["curvature_at_90_per_deg2"] = float(curvature[center])
    return metrics


def analyze_figure5(output: Path, samples: int, seed: int) -> list[dict[str, object]]:
    theta = np.asarray(N10_THETA_DEG)
    curves = {}
    rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    dynamics_rows: list[dict[str, object]] = []
    common_boot_ids = _bootstrap_indices(samples, seed ^ 0xF510)
    bootstrap_metrics_by_protocol = {}
    for protocol in PROTOCOLS:
        values = np.empty((samples, len(theta)))
        standard = np.empty_like(values)
        for angle_index, angle in enumerate(theta):
            for sample_id in range(samples):
                cell = _load(cell_path(output, n10_task(float(angle), protocol, sample_id, samples, seed)))
                values[sample_id, angle_index] = cell["conditional_normalized_yield"]
                standard[sample_id, angle_index] = cell["standard_truncated_yield"]
        curves[protocol] = values
        mean, se, low, high = mean_se_ci(values)
        standard_mean = np.mean(standard, axis=0)
        for index, angle in enumerate(theta):
            rows.append({"protocol": protocol, "theta_deg": angle, "samples": samples, "conditional_mean_yield": mean[index], "standard_mean_yield": standard_mean[index], "standard_error": se[index], "ci95_low": low[index], "ci95_high": high[index]})
        point = _n10_metrics(theta, mean)
        distributions = [_n10_metrics(theta, np.mean(values[ids], axis=0)) for ids in common_boot_ids]
        bootstrap_metrics_by_protocol[protocol] = distributions
        row = {"protocol": protocol, "samples": samples, **point}
        for key in ("contrast", "maximum_abs_slope_per_deg", "curvature_at_90_per_deg2"):
            row[f"{key}_ci95_low"], row[f"{key}_ci95_high"] = np.quantile([item[key] for item in distributions], [0.025, 0.975])
        metrics_rows.append(row)
    baseline = next(row for row in metrics_rows if row["protocol"] == "baseline")
    deleted_mean = np.mean(curves["n10_transverse_deleted"], axis=0)
    for row in metrics_rows:
        protocol = row["protocol"]
        row["absolute_slope_reduction_from_baseline"] = baseline["maximum_abs_slope_per_deg"] - row["maximum_abs_slope_per_deg"]
        row["percentage_slope_reduction_from_baseline"] = 100.0 * row["absolute_slope_reduction_from_baseline"] / baseline["maximum_abs_slope_per_deg"]
        row["L2_distance_from_transverse_deleted"] = float(np.sqrt(np.mean((np.mean(curves[protocol], axis=0) - deleted_mean) ** 2)))
        slope_reductions = []
        percentage_reductions = []
        l2_distances = []
        for boot_index, ids in enumerate(common_boot_ids):
            baseline_slope = bootstrap_metrics_by_protocol["baseline"][boot_index]["maximum_abs_slope_per_deg"]
            protocol_slope = bootstrap_metrics_by_protocol[protocol][boot_index]["maximum_abs_slope_per_deg"]
            reduction = baseline_slope - protocol_slope
            slope_reductions.append(reduction)
            percentage_reductions.append(100.0 * reduction / baseline_slope)
            protocol_curve = np.mean(curves[protocol][ids], axis=0)
            deleted_curve = np.mean(curves["n10_transverse_deleted"][ids], axis=0)
            l2_distances.append(float(np.sqrt(np.mean((protocol_curve - deleted_curve) ** 2))))
        for name, distribution in (
            ("absolute_slope_reduction_from_baseline", slope_reductions),
            ("percentage_slope_reduction_from_baseline", percentage_reductions),
            ("L2_distance_from_transverse_deleted", l2_distances),
        ):
            row[f"{name}_ci95_low"], row[f"{name}_ci95_high"] = np.quantile(distribution, [0.025, 0.975])
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    for protocol in PROTOCOLS:
        mean, _, low, high = mean_se_ci(curves[protocol])
        axis.plot(theta, mean, label=PROTOCOL_LABELS[protocol])
        axis.fill_between(theta, low, high, alpha=0.12)
    axis.set(xlabel="magnetic-field polar angle (deg)", ylabel="conditionally normalized truncated singlet yield", title="FAD-N10 control near 90 degrees")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(fig, output / "figures" / "figure5_n10_control")
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    for protocol in PROTOCOLS:
        mean = np.mean(curves[protocol], axis=0)
        axis.plot(theta, np.gradient(mean, theta, edge_order=2), label=PROTOCOL_LABELS[protocol])
    axis.set(xlabel="angle (deg)", ylabel="d yield / d angle", title="N10-control angular derivatives")
    axis.legend(frameon=False, fontsize=8)
    _save_figure(fig, output / "figures" / "figure5b_n10_slope")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for axis, angle in zip(axes, (89.0, 90.0, 91.0)):
        for protocol in PROTOCOLS:
            populations = []
            times = None
            for sample_id in range(samples):
                cell = _load(cell_path(output, n10_task(angle, protocol, sample_id, samples, seed)))
                times = cell["times_us"]
                populations.append(cell["singlet_population"])
            mean = np.mean(populations, axis=0)
            axis.plot(times, mean, label=PROTOCOL_LABELS[protocol])
            for time_value, population in zip(times, mean):
                dynamics_rows.append({"protocol": protocol, "theta_deg": angle, "time_us": time_value, "mean_singlet_population": population, "samples": samples})
        axis.set_title(f"{angle:g} deg")
        axis.set_xlabel("time (us)")
        axis.set_xlim(0.0, 10.0)
    axes[0].set_ylabel("singlet population")
    axes[0].legend(frameon=False, fontsize=7)
    _save_figure(fig, output / "figures" / "figure5c_n10_dynamics")
    _write_csv(output / "n10_control_curves.csv", rows)
    _write_csv(output / "n10_control_metrics.csv", metrics_rows)
    _write_csv(output / "n10_control_dynamics.csv", dynamics_rows)
    return metrics_rows


def sample_convergence(output: Path, fields: dict[float, np.ndarray]) -> None:
    rows = []
    for field, values in fields.items():
        for count in (4, 8, 16):
            if count > values.shape[0]:
                continue
            _, se, low, high = mean_se_ci(values[:count])
            rows.append({"field_mT": field, "samples": count, "maximum_standard_error": float(np.nanmax(se)), "maximum_ci95_half_width": float(np.nanmax((high - low) / 2.0))})
    _write_csv(output / "sample_convergence.csv", rows)


def analyze_overhauser_sampling(
    output: Path,
    samples: int,
    overhauser_samples: int,
    seed: int,
    overhauser_replicates: int = 8,
    atomwise_samples: int = 4096,
) -> list[dict[str, object]]:
    """Validate the completed semiclassical ensemble without quantum curves."""

    rows: list[dict[str, object]] = []
    theta = np.asarray(POSTER_THETA_DEG)
    field_max_se = []
    field_max_atomwise_se = []
    for field in POSTER_FIELDS_MT:
        maximum_se = 0.0
        maximum_atomwise_se = 0.0
        for angle in theta:
            cell, actual_overhauser_samples = _load_best_overhauser_cell(
                output,
                field,
                float(angle),
                overhauser_samples,
                overhauser_replicates,
                atomwise_samples,
            )
            gaussian = np.asarray(cell["replicate_means"], dtype=float)
            atomwise = np.asarray(cell["atomwise_sample_yields"], dtype=float)
            gaussian_mean, gaussian_se, gaussian_low, gaussian_high = mean_se_ci(gaussian[:, None])
            atom_mean, atom_se, atom_low, atom_high = mean_se_ci(atomwise[:, None])
            maximum_se = max(maximum_se, float(gaussian_se[0]))
            maximum_atomwise_se = max(maximum_atomwise_se, float(atom_se[0]))
            rows.append(
                {
                    "field_mT": field,
                    "theta_deg": angle,
                    "overhauser_samples": actual_overhauser_samples,
                    "overhauser_replicates": overhauser_replicates,
                    "points_per_replicate": actual_overhauser_samples
                    // overhauser_replicates,
                    "gaussian_mean_yield": gaussian_mean[0],
                    "gaussian_standard_error": gaussian_se[0],
                    "gaussian_ci95_low": gaussian_low[0],
                    "gaussian_ci95_high": gaussian_high[0],
                    "gaussian_standard_error_target": OVERHAUSER_STANDARD_ERROR_TARGET,
                    "gaussian_precision_gate_passed": gaussian_se[0]
                    <= OVERHAUSER_STANDARD_ERROR_TARGET,
                    "atomwise_mean_yield": atom_mean[0],
                    "atomwise_standard_error": atom_se[0],
                    "atomwise_ci95_low": atom_low[0],
                    "atomwise_ci95_high": atom_high[0],
                }
            )
        field_max_se.append(maximum_se)
        field_max_atomwise_se.append(maximum_atomwise_se)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(POSTER_FIELDS_MT, field_max_se, marker="o", label="Gaussian static")
    axis.plot(POSTER_FIELDS_MT, field_max_atomwise_se, marker="s", label="atomwise static")
    axis.axhline(
        OVERHAUSER_STANDARD_ERROR_TARGET,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="conservative M=16 quantum-noise target",
    )
    axis.set_xscale("log")
    axis.set(
        xlabel="field strength (mT)",
        ylabel="maximum pointwise Monte Carlo standard error",
        title=f"Semiclassical ensemble precision (at least {overhauser_samples:,} Gaussian samples)",
    )
    axis.legend(frameon=False)
    _save_figure(fig, output / "figures" / "overhauser_sampling_precision")
    _write_csv(output / "overhauser_sampling_diagnostics.csv", rows)
    return rows


def _conclusion_from_metrics(field_metrics, overhauser_metrics, n10_metrics) -> str:
    geomagnetic = next(row for row in field_metrics if np.isclose(row["field_mT"], 0.05))
    two_mt = next(row for row in field_metrics if np.isclose(row["field_mT"], 2.0))
    survived_field = two_mt["relative_contrast"] > geomagnetic["relative_contrast"]
    oh_50 = next(row for row in overhauser_metrics if np.isclose(row["field_mT"], 0.05))
    oh_20 = next(row for row in overhauser_metrics if np.isclose(row["field_mT"], 20.0))
    overhauser_improves = oh_20["rmse_theta"] < oh_50["rmse_theta"]
    baseline = next(row for row in n10_metrics if row["protocol"] == "baseline")
    p005 = next(row for row in n10_metrics if row["protocol"] == "periodic_0.05")
    pulse_flattens = p005["maximum_abs_slope_per_deg"] < baseline["maximum_abs_slope_per_deg"]
    return (
        f"- Field-transition observation survived: **{survived_field}**.\n"
        f"- Static-Overhauser error decreased from 0.05 to 20 mT: **{overhauser_improves}**.\n"
        f"- The 0.05-us periodic N10 sequence reduced the local maximum slope: **{pulse_flattens}**.\n"
    )


def write_summary(
    output: Path,
    samples: int,
    field_metrics,
    overhauser_metrics,
    entanglement_rows,
    n10_metrics,
    runtime: dict[str, float | int],
) -> None:
    run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    text = f"""# Poster-style rerun with the current production model

This is not an exact reproduction of the APS poster.  It uses the current
nonfactorized 14-nucleus state-vector Hamiltonian, Efimova--Hore D/J=-3 electron
coupling at r=2 nm, random-phase trace sampling, adaptive Krylov propagation,
and Simpson-corrected yield integration.

## Validation and sampling

- Nuclear samples per quantum curve: **{samples}**.
- Master seed: **{run['seed']}** with paired sample IDs across conditions.
- Quantum quadrupole scale: **{reported_quadrupole_scale(output):g}**.
  Static Overhauser calculations have no explicit nuclear quadrupole term.
- Completed checkpoint cells: **{runtime['completed_checkpoint_cells']}**.
- Completed full-state trajectories: **{runtime['completed_statevector_trajectories']}**.
- Completed semiclassical realizations: **{runtime['completed_semiclassical_realizations']:,}**.
- Aggregate checkpointed cell compute: **{runtime['aggregate_cell_compute_hours']:.2f} worker-h**.
- Invocation time retained by the current manifests: **{runtime['surviving_manifest_invocation_hours']:.2f} h**;
  resumed invocations overwrite their earlier manifest record, so the
  checkpoint-derived worker-hour total above is authoritative.
- Preserved non-overwriting run-history time: **{runtime['preserved_run_history_hours']:.2f} h**
  for invocations launched after run-history recording was enabled.
- First-to-last checkpoint span, including unattended gaps: **{runtime['first_to_last_checkpoint_span_hours']:.2f} h**.
- Uncertainty is across nuclear sample IDs; nonlinear metrics bootstrap complete
  paired sample curves rather than treating angles as replicates.

## Poster-conclusion checks

{_conclusion_from_metrics(field_metrics, overhauser_metrics, n10_metrics)}

## Interpretation limits

The Figure 4 pure-trajectory values are decomposition-dependent entanglement-
generation proxies from a computational-basis product-state decomposition of
the nuclear mixture.  They are not the log negativity of the mixed nuclear
ensemble.  The separately reported two-electron curve averages reduced density
matrices before evaluating log negativity and is a genuine mixed-state measure.

Any change from the old poster combines two deliberate changes: reliable
multi-sample trace estimation and the added D/J electron interaction.  This run
does not isolate those effects by parameter tuning; the D=J=0 legacy one-sample
check is interpretation-only and the current-model curves remain primary.

All requested CSV files and 300-dpi PNG/vector-PDF figures are in this directory.
"""
    (output / "summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=PRODUCTION_MODEL.samples)
    parser.add_argument("--overhauser-samples", type=int, default=262144)
    parser.add_argument("--overhauser-replicates", type=int, default=8)
    parser.add_argument("--atomwise-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    add_quadrupole_argument(parser, default=None)
    parser.add_argument(
        "--components",
        default="2,3,4,5",
        help="Comma-separated completed components: 2,3,4,5,overhauser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.quadrupole_scale is not None and
            args.quadrupole_scale != reported_quadrupole_scale(args.output_dir)):
        raise ValueError("Requested quadrupole scale does not match the archived model.")
    components = tuple(item.strip() for item in args.components.split(",") if item.strip())
    if not components or any(item not in {"2", "3", "4", "5", "overhauser"} for item in components):
        raise ValueError("--components may contain only 2,3,4,5,overhauser.")
    quantum = None
    field_metrics = overhauser_metrics = entanglement_rows = n10_metrics = None
    runtime = analyze_runtime(args.output_dir)
    if "overhauser" in components:
        analyze_overhauser_sampling(
            args.output_dir,
            args.samples,
            args.overhauser_samples,
            args.seed,
            args.overhauser_replicates,
            args.atomwise_samples,
        )
    if "2" in components:
        quantum, field_metrics = analyze_figure2(args.output_dir, args.samples, args.seed)
        sample_convergence(args.output_dir, quantum)
    if "3" in components:
        quantum = load_quantum_curves_adaptive(
            args.output_dir,
            POSTER_FIELDS_MT,
            np.asarray(POSTER_THETA_DEG),
            args.samples,
            args.seed,
        )
        overhauser_metrics = analyze_figure3(
            args.output_dir,
            quantum,
            args.samples,
            args.overhauser_samples,
            args.seed,
            args.overhauser_replicates,
            args.atomwise_samples,
        )
    if "4" in components:
        entanglement_rows = analyze_figure4(args.output_dir, args.samples, args.seed)
    if "5" in components:
        n10_metrics = analyze_figure5(args.output_dir, args.samples, args.seed)
    if all(value is not None for value in (field_metrics, overhauser_metrics, entanglement_rows, n10_metrics)):
        write_summary(
            args.output_dir,
            args.samples,
            field_metrics,
            overhauser_metrics,
            entanglement_rows,
            n10_metrics,
            runtime,
        )


if __name__ == "__main__":
    main()
