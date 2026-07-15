"""Budgeted validation of full-pair random-phase/Krylov RPM dynamics.

The exact radical-factorized calculation is used only as an oracle.  Every
sampled trajectory evolves the complete 221,184-component radical-pair state.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    # Multiprocessing provides the trajectory-level parallelism.  Force each
    # numerical library to remain single-threaded so worker counts are real and
    # reproducible even when the login shell exports a larger default.
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.linalg

import rpm_quantum_needle_reproduction as exact
from rpm_quantum_needle_model import (
    GAMMA_E_MHZ_PER_MT,
    NUCLEAR_DIMENSION,
    STATE_DIMENSION,
    SYSTEM_DIMENSIONS,
    TWOPI,
    field_vector_mT,
    spin_operators,
)
from rpm_statevector_engine import (
    AdaptiveLanczosPropagator,
    LocalTerm,
    MatrixFreeHamiltonian,
    build_quantum_needle_hamiltonian,
    quadrupole_operator,
    singlet_initial_state,
    singlet_population,
    two_spin_coupling_operator,
)


STAGE1_ANGLES = (0.0, 60.0, 80.0, 90.0, 110.0)
PREFERRED_SWEEP = tuple(float(value) for value in range(0, 181, 15))
FALLBACK_SWEEP = tuple(float(value) for value in range(0, 181, 30))
NESTED_COUNTS = (1, 2, 4, 8, 16)
_EXACT_REFERENCE_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


@dataclass(frozen=True)
class ValidationConfig:
    seed: int = 20260714
    field_microT: float = 50.0
    lifetime_us: float = 5.0
    survival_tail_bound: float = 1e-4
    output_step_us: float = 0.01
    detailed_trace_us: float = 5.0
    sweep_angles_deg: tuple[float, ...] = PREFERRED_SWEEP
    stage1_max_samples: int = 16
    sweep_initial_samples: int = 4
    sweep_max_samples: int = 8
    trace_rms_target: float = 1e-3
    yield_error_target: float = 1e-3
    angular_correlation_target: float = 0.8
    anisotropy_relative_error_target: float = 0.30
    state_error_budget: float = 2.4e-5
    lanczos_minimum_dimension: int = 24
    lanczos_maximum_dimension: int = 64
    lanczos_dimension_increment: int = 8
    lanczos_embedded_offset: int = 4
    lanczos_safety_factor: float = 2.0
    lanczos_initial_step_us: float = 0.05
    lanczos_maximum_step_us: float = 0.25

    @property
    def maximum_time_us(self) -> float:
        return -self.lifetime_us * math.log(self.survival_tail_bound)

    @property
    def time_grid_us(self) -> np.ndarray:
        count = int(math.floor(self.maximum_time_us / self.output_step_us))
        values = np.arange(count + 1, dtype=float) * self.output_step_us
        if values[-1] < self.maximum_time_us - 1e-12:
            values = np.append(values, self.maximum_time_us)
        return values

    def stable_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["sweep_angles_deg"] = list(self.sweep_angles_deg)
        result["stage1_angles_deg"] = list(STAGE1_ANGLES)
        result["state_dimension"] = STATE_DIMENSION
        result["nuclear_dimension"] = NUCLEAR_DIMENSION
        result["J_MHz"] = 0.0
        result["dipolar_tensor_MHz"] = [[0.0] * 3 for _ in range(3)]
        return result

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TrajectoryTask:
    theta_deg: float
    sample_id: int
    config: ValidationConfig


def _seed(base: int, *keys: int) -> int:
    sequence = np.random.SeedSequence([int(base), *(int(key) for key in keys)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _peak_rss_mib() -> float | None:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            value /= 1024.0
        return value / 1024.0
    except (ImportError, AttributeError):
        return None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _propagator(hamiltonian: MatrixFreeHamiltonian, config: ValidationConfig):
    return AdaptiveLanczosPropagator(
        hamiltonian,
        minimum_dimension=config.lanczos_minimum_dimension,
        maximum_dimension=config.lanczos_maximum_dimension,
        dimension_increment=config.lanczos_dimension_increment,
        embedded_offset=config.lanczos_embedded_offset,
        safety_factor=config.lanczos_safety_factor,
        state_error_budget=config.state_error_budget,
        initial_step_us=config.lanczos_initial_step_us,
        maximum_step_us=config.lanczos_maximum_step_us,
    )


def integrate_singlet_yield(
    times_us: np.ndarray,
    singlet_population_values: np.ndarray,
    lifetime_us: float,
    *,
    method: str = "simpson",
) -> float:
    """Integrate an already-propagated singlet trajectory without changing it."""

    times = np.asarray(times_us, dtype=float)
    populations = np.asarray(singlet_population_values, dtype=float)
    if times.shape != populations.shape or times.ndim != 1:
        raise ValueError("Yield times and singlet populations must be matching 1-D arrays.")
    integrand = np.exp(-times / lifetime_us) * populations / lifetime_us
    if method == "simpson":
        return float(scipy.integrate.simpson(integrand, x=times))
    if method == "trapezoid":
        return float(np.trapezoid(integrand, times))
    raise ValueError(f"Unknown yield-integration method: {method}")


def propagate_full_state(
    theta_deg: float,
    sample_id: int,
    times_us: np.ndarray,
    config: ValidationConfig,
) -> dict[str, object]:
    total_started = time.perf_counter()
    build_started = time.perf_counter()
    hamiltonian = build_quantum_needle_hamiltonian(
        theta_deg,
        config.field_microT,
        exchange_J_MHz=0.0,
        dipolar_tensor_MHz=np.zeros((3, 3)),
    )
    construction_seconds = time.perf_counter() - build_started
    initial_state = singlet_initial_state(
        SYSTEM_DIMENSIONS,
        np.random.default_rng(_seed(config.seed, 0xA11CE, sample_id)),
    )
    propagation_started = time.perf_counter()
    result = _propagator(hamiltonian, config).propagate(
        initial_state,
        times_us,
        lambda state: singlet_population(state, SYSTEM_DIMENSIONS),
    )
    propagation_seconds = time.perf_counter() - propagation_started
    weights = np.exp(-times_us / config.lifetime_us) / config.lifetime_us
    yield_value = integrate_singlet_yield(
        times_us,
        result.observable_values,
        config.lifetime_us,
        method="simpson",
    )
    yield_propagation_bound = float(
        np.trapezoid(weights * result.observable_error_bounds, times_us)
    )
    return {
        "theta_deg": float(theta_deg),
        "sample_id": int(sample_id),
        "times_us": times_us,
        "singlet_population": result.observable_values,
        "population_bounds": result.observable_error_bounds,
        "yield": yield_value,
        "yield_integration_method": "composite_simpson",
        "yield_propagation_bound": yield_propagation_bound,
        "yield_tail_bound": config.survival_tail_bound,
        "construction_seconds": construction_seconds,
        "propagation_seconds": propagation_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "peak_rss_mib": _peak_rss_mib(),
        "accepted_steps": result.stats.accepted_steps,
        "rejected_steps": result.stats.rejected_steps,
        "matvecs": result.stats.matvecs,
        "maximum_krylov_dimension": result.stats.maximum_krylov_dimension,
        "minimum_step_us": result.stats.minimum_step_us,
        "maximum_step_us": result.stats.maximum_step_us,
        "accumulated_state_error": result.stats.accumulated_state_error,
        "maximum_norm_error": result.stats.maximum_norm_error,
    }


def _angle_token(theta_deg: float) -> str:
    return f"{theta_deg:07.3f}".replace(".", "p")


def checkpoint_path(output_dir: Path, theta_deg: float, sample_id: int) -> Path:
    return output_dir / "checkpoints" / f"theta_{_angle_token(theta_deg)}_sample_{sample_id:03d}.npz"


def save_trajectory(
    output_dir: Path,
    result: dict[str, object],
    config: ValidationConfig,
    *,
    retain_trace: bool,
) -> None:
    path = checkpoint_path(output_dir, float(result["theta_deg"]), int(result["sample_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        stored = dict(result)
        if not retain_trace:
            for key in ("times_us", "singlet_population", "population_bounds"):
                stored.pop(key, None)
        arrays = {
            "configuration_hash": np.array(config.configuration_hash),
            **{key: np.asarray(value) for key, value in stored.items()},
        }
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_trajectory(
    output_dir: Path, theta_deg: float, sample_id: int, config: ValidationConfig
) -> dict[str, object]:
    with np.load(checkpoint_path(output_dir, theta_deg, sample_id), allow_pickle=False) as data:
        if str(data["configuration_hash"]) != config.configuration_hash:
            raise RuntimeError("Checkpoint configuration hash mismatch.")
        result: dict[str, object] = {}
        for key in data.files:
            if key == "configuration_hash":
                continue
            value = data[key]
            result[key] = value.item() if value.ndim == 0 else value.copy()
        return result


def _worker(task: TrajectoryTask) -> dict[str, object]:
    return propagate_full_state(
        task.theta_deg, task.sample_id, task.config.time_grid_us, task.config
    )


def run_tasks(
    output_dir: Path,
    angles: tuple[float, ...],
    sample_count: int,
    config: ValidationConfig,
    workers: int,
    deadline: float,
) -> bool:
    tasks = [
        TrajectoryTask(theta, sample_id, config)
        for theta in angles
        for sample_id in range(sample_count)
        if not checkpoint_path(output_dir, theta, sample_id).exists()
    ]
    if not tasks:
        return True
    print(f"running {len(tasks)} full-space trajectories with {workers} workers", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(tasks), workers):
            if time.time() >= deadline:
                return False
            batch = tasks[start : start + workers]
            futures = {executor.submit(_worker, task): task for task in batch}
            for future in as_completed(futures):
                result = future.result()
                save_trajectory(
                    output_dir,
                    result,
                    config,
                    retain_trace=float(result["theta_deg"]) in STAGE1_ANGLES,
                )
                completed += 1
                if completed == 1 or completed == len(tasks) or completed % max(8, workers) == 0:
                    print(
                        f"[{completed}/{len(tasks)}] theta={result['theta_deg']:g}, "
                        f"sample={result['sample_id']}, {result['total_seconds']:.1f}s, "
                        f"matvecs={result['matvecs']}",
                        flush=True,
                    )
    return True


def exact_reference(
    theta_deg: float, times_us: np.ndarray, config: ValidationConfig
) -> dict[str, object]:
    cache_key = (
        float(theta_deg),
        config.field_microT,
        config.lifetime_us,
        len(times_us),
        float(times_us[0]) if len(times_us) else None,
        float(times_us[-1]) if len(times_us) else None,
    )
    cached = _EXACT_REFERENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    fad = exact.diagonalize_radical(exact._FAD_MODEL, theta_deg, config.field_microT)
    trp = exact.diagonalize_radical(exact._TRP_MODEL, theta_deg, config.field_microT)
    # Match the reference runner's established 16 bins per recombination
    # linewidth.  A fixed 0.00125 rad/us grid was ten times finer than needed
    # for the 5-us validation lifetime and made the oracle unnecessarily slow.
    delta_omega = (1.0 / config.lifetime_us) / 16.0
    omega, coefficients, initial = exact.convolved_correlation_spectrum(
        fad, trp, delta_omega
    )
    populations = np.empty(len(times_us), dtype=float)
    for start in range(0, len(times_us), 32):
        stop = min(start + 32, len(times_us))
        phases = np.exp(1.0j * np.outer(times_us[start:stop], omega))
        populations[start:stop] = (0.25 + phases @ coefficients).real
    yield_value = exact.yields_from_spectrum(
        omega, coefficients, (config.lifetime_us,)
    )[config.lifetime_us]
    result = {
        "theta_deg": theta_deg,
        "initial_singlet": initial,
        "singlet_population": populations,
        "yield": yield_value,
        "delta_omega_rad_per_us": delta_omega,
    }
    _EXACT_REFERENCE_CACHE[cache_key] = result
    return result


def reference_grid_assessment(config: ValidationConfig) -> dict[str, float | bool]:
    """Compare Simpson and trapezoidal bias on the unchanged output grid."""
    reference = exact_reference(90.0, config.time_grid_us, config)
    old_grid_yield = integrate_singlet_yield(
        config.time_grid_us,
        np.asarray(reference["singlet_population"]),
        config.lifetime_us,
        method="trapezoid",
    )
    grid_yield = integrate_singlet_yield(
        config.time_grid_us,
        np.asarray(reference["singlet_population"]),
        config.lifetime_us,
        method="simpson",
    )
    old_bias = abs(old_grid_yield - float(reference["yield"]))
    bias = abs(grid_yield - float(reference["yield"]))
    return {
        "theta_deg": 90.0,
        "infinite_time_reference_yield": float(reference["yield"]),
        "trapezoid_finite_grid_yield": old_grid_yield,
        "trapezoid_absolute_bias_including_tail": old_bias,
        "finite_grid_reference_yield": grid_yield,
        "absolute_bias_including_tail": bias,
        "absolute_bias_reduction": old_bias - bias,
        "relative_bias_reduction": (old_bias - bias) / old_bias,
        "integration_method": "composite_simpson",
        "survival_tail_bound": config.survival_tail_bound,
        "output_step_us": config.output_step_us,
        "passed": bool(bias <= config.yield_error_target / 4.0),
    }


def reintegrate_saved_trajectories(
    output_dir: Path, config: ValidationConfig
) -> dict[str, object]:
    """Reintegrate retained trajectories; never propagates or edits checkpoints."""

    cell_rows: list[dict[str, object]] = []
    missing_trace_cells: list[str] = []
    for theta in STAGE1_ANGLES:
        exact_yield = float(exact_reference(theta, np.array([0.0]), config)["yield"])
        for sample_id in range(config.stage1_max_samples):
            path = checkpoint_path(output_dir, theta, sample_id)
            if not path.exists():
                missing_trace_cells.append(f"theta={theta:g},sample={sample_id}")
                continue
            cell = load_trajectory(output_dir, theta, sample_id, config)
            if "times_us" not in cell or "singlet_population" not in cell:
                missing_trace_cells.append(f"theta={theta:g},sample={sample_id}")
                continue
            old_yield = float(cell["yield"])
            new_yield = integrate_singlet_yield(
                np.asarray(cell["times_us"]),
                np.asarray(cell["singlet_population"]),
                config.lifetime_us,
                method="simpson",
            )
            cell_rows.append(
                {
                    "theta_deg": theta,
                    "sample_id": sample_id,
                    "exact_yield": exact_yield,
                    "trapezoid_yield": old_yield,
                    "simpson_yield": new_yield,
                    "trapezoid_signed_error": old_yield - exact_yield,
                    "simpson_signed_error": new_yield - exact_yield,
                }
            )
    if not cell_rows:
        raise RuntimeError("No retained P_S(t) trajectories were available to reintegrate.")
    _write_csv(output_dir / "yield_reintegration_cells.csv", cell_rows)

    aggregate_rows: list[dict[str, object]] = []
    for count in NESTED_COUNTS:
        for theta in STAGE1_ANGLES:
            selected = [
                row
                for row in cell_rows
                if row["theta_deg"] == theta and int(row["sample_id"]) < count
            ]
            if len(selected) != count:
                continue
            exact_yield = float(selected[0]["exact_yield"])
            old_mean = float(np.mean([row["trapezoid_yield"] for row in selected]))
            new_mean = float(np.mean([row["simpson_yield"] for row in selected]))
            old_bias = abs(old_mean - exact_yield)
            new_bias = abs(new_mean - exact_yield)
            aggregate_rows.append(
                {
                    "M": count,
                    "theta_deg": theta,
                    "exact_yield": exact_yield,
                    "trapezoid_mean_yield": old_mean,
                    "simpson_mean_yield": new_mean,
                    "trapezoid_absolute_bias": old_bias,
                    "simpson_absolute_bias": new_bias,
                    "absolute_bias_reduction": old_bias - new_bias,
                    "relative_bias_reduction": (old_bias - new_bias) / old_bias
                    if old_bias
                    else math.nan,
                }
            )
    _write_csv(output_dir / "yield_reintegration_summary.csv", aggregate_rows)
    final_rows = [row for row in aggregate_rows if row["M"] == max(NESTED_COUNTS)]
    grid = reference_grid_assessment(config)
    summary = {
        "propagation_rerun": False,
        "checkpoints_modified": False,
        "retained_trajectory_cells_reintegrated": len(cell_rows),
        "missing_representative_trace_cells": missing_trace_cells,
        "M16_mean_trapezoid_absolute_bias": float(
            np.mean([row["trapezoid_absolute_bias"] for row in final_rows])
        ),
        "M16_mean_simpson_absolute_bias": float(
            np.mean([row["simpson_absolute_bias"] for row in final_rows])
        ),
        "M16_max_trapezoid_absolute_bias": float(
            max(row["trapezoid_absolute_bias"] for row in final_rows)
        ),
        "M16_max_simpson_absolute_bias": float(
            max(row["simpson_absolute_bias"] for row in final_rows)
        ),
        "exact_reference_grid": grid,
        "note": (
            "Sweep-only checkpoints without retained P_S(t) were not reintegrated; "
            "no state-vector trajectory was rerun."
        ),
    }
    old_mean = float(summary["M16_mean_trapezoid_absolute_bias"])
    new_mean = float(summary["M16_mean_simpson_absolute_bias"])
    summary["M16_mean_absolute_bias_reduction"] = old_mean - new_mean
    summary["M16_mean_relative_bias_reduction"] = (old_mean - new_mean) / old_mean
    _write_json(output_dir / "yield_reintegration_report.json", summary)
    return summary


def stage1_metrics(
    output_dir: Path, config: ValidationConfig, available_count: int
) -> tuple[list[dict[str, object]], int | None]:
    times = config.time_grid_us
    detailed = times <= config.detailed_trace_us + 1e-12
    references = {theta: exact_reference(theta, times[detailed], config) for theta in STAGE1_ANGLES}
    rows: list[dict[str, object]] = []
    recommended: int | None = None
    for count in (value for value in NESTED_COUNTS if value <= available_count):
        count_passed = True
        for theta in STAGE1_ANGLES:
            cells = [load_trajectory(output_dir, theta, sample, config) for sample in range(count)]
            traces = np.stack([cell["singlet_population"] for cell in cells])
            yields = np.array([cell["yield"] for cell in cells], dtype=float)
            mean_trace = traces.mean(axis=0)[detailed]
            exact_trace = np.asarray(references[theta]["singlet_population"])
            difference = mean_trace - exact_trace
            rms = float(np.sqrt(np.mean(difference * difference)))
            maximum = float(np.max(np.abs(difference)))
            yield_mean = float(yields.mean())
            yield_error = abs(yield_mean - float(references[theta]["yield"]))
            yield_sem = float(yields.std(ddof=1) / math.sqrt(count)) if count > 1 else math.nan
            propagation_bound = float(
                max(float(cell["yield_propagation_bound"]) for cell in cells)
            )
            passed = bool(
                rms <= config.trace_rms_target
                and yield_error <= config.yield_error_target
                and propagation_bound <= config.yield_error_target / 10.0
            )
            count_passed &= passed
            rows.append(
                {
                    "M": count,
                    "theta_deg": theta,
                    "trace_rms_error": rms,
                    "trace_max_error": maximum,
                    "direct_yield": yield_mean,
                    "exact_yield": references[theta]["yield"],
                    "yield_absolute_error": yield_error,
                    "yield_sample_sem": yield_sem,
                    "maximum_yield_propagation_bound": propagation_bound,
                    "p_s_0_error": abs(float(traces[:, 0].mean()) - 1.0),
                    "mean_runtime_seconds": float(np.mean([cell["total_seconds"] for cell in cells])),
                    "passed": passed,
                }
            )
        if count_passed and recommended is None:
            recommended = count
    return rows, recommended


def plot_stage1(output_dir: Path, config: ValidationConfig, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for theta in STAGE1_ANGLES:
        selected = [row for row in rows if row["theta_deg"] == theta]
        axes[0].plot([row["M"] for row in selected], [row["trace_rms_error"] for row in selected], marker="o", label=f"{theta:g}°")
        axes[1].plot([row["M"] for row in selected], [row["yield_absolute_error"] for row in selected], marker="o")
    for axis, target, title in (
        (axes[0], config.trace_rms_target, "RMS trace error"),
        (axes[1], config.yield_error_target, "Singlet-yield error"),
    ):
        axis.axhline(target, color="black", linestyle="--", linewidth=1)
        axis.set(xscale="log", yscale="log", xlabel="nested random-phase samples M", title=title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_nested_convergence.png", dpi=180)
    plt.close(fig)

    times = config.time_grid_us
    detailed = times <= config.detailed_trace_us + 1e-12
    fig, axes = plt.subplots(len(STAGE1_ANGLES), 1, figsize=(8.5, 12), sharex=True)
    for axis, theta in zip(axes, STAGE1_ANGLES):
        reference = exact_reference(theta, times[detailed], config)
        traces = np.stack(
            [load_trajectory(output_dir, theta, sample, config)["singlet_population"] for sample in range(config.stage1_max_samples)]
        )
        mean = traces.mean(axis=0)[detailed]
        sem = traces.std(axis=0, ddof=1)[detailed] / math.sqrt(config.stage1_max_samples)
        axis.plot(times[detailed], reference["singlet_population"], color="black", label="exact oracle")
        axis.plot(times[detailed], mean, color="tab:blue", label="full-pair RPS/Krylov")
        axis.fill_between(times[detailed], mean - 1.96 * sem, mean + 1.96 * sem, color="tab:blue", alpha=0.2)
        axis.set_ylabel(f"{theta:g}°")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("time (μs)")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_exact_vs_direct_traces.png", dpi=180)
    plt.close(fig)


def angular_metrics(
    output_dir: Path, config: ValidationConfig, sample_count: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    exact_values = []
    direct_values = []
    direct_sems = []
    per_angle_samples = []
    for theta in config.sweep_angles_deg:
        reference = exact_reference(theta, np.array([0.0]), config)
        cells = [load_trajectory(output_dir, theta, sample, config) for sample in range(sample_count)]
        samples = np.array([cell["yield"] for cell in cells], dtype=float)
        exact_values.append(float(reference["yield"]))
        direct_values.append(float(samples.mean()))
        direct_sems.append(float(samples.std(ddof=1) / math.sqrt(sample_count)))
        per_angle_samples.append(samples)
    exact_array = np.asarray(exact_values)
    direct_array = np.asarray(direct_values)
    exact_min = int(np.argmin(exact_array))
    exact_max = int(np.argmax(exact_array))
    direct_min = int(np.argmin(direct_array))
    direct_max = int(np.argmax(direct_array))
    exact_delta = float(exact_array[exact_max] - exact_array[exact_min])
    direct_delta = float(direct_array[direct_max] - direct_array[direct_min])
    paired = per_angle_samples[exact_max] - per_angle_samples[exact_min]
    direct_at_exact_extrema = float(direct_array[exact_max] - direct_array[exact_min])
    anisotropy_sign_preserved = bool(
        np.sign(direct_at_exact_extrema) == np.sign(exact_delta)
    )
    correlation = float(np.corrcoef(exact_array, direct_array)[0, 1])
    rmse = float(np.sqrt(np.mean((direct_array - exact_array) ** 2)))
    delta_relative_error = abs(direct_delta - exact_delta) / max(abs(exact_delta), np.finfo(float).tiny)
    grid_step = min(np.diff(config.sweep_angles_deg))
    passed = bool(
        correlation >= config.angular_correlation_target
        and rmse <= config.yield_error_target
        and anisotropy_sign_preserved
        and delta_relative_error <= config.anisotropy_relative_error_target
        and abs(config.sweep_angles_deg[direct_min] - config.sweep_angles_deg[exact_min]) <= grid_step
        and abs(config.sweep_angles_deg[direct_max] - config.sweep_angles_deg[exact_max]) <= grid_step
    )
    rows = [
        {
            "theta_deg": theta,
            "exact_yield": exact_values[index],
            "direct_yield": direct_values[index],
            "direct_sem": direct_sems[index],
            "absolute_error": abs(direct_values[index] - exact_values[index]),
            "M": sample_count,
        }
        for index, theta in enumerate(config.sweep_angles_deg)
    ]
    summary = {
        "M": sample_count,
        "curve_rmse": rmse,
        "curve_correlation": correlation,
        "exact_minimum_angle_deg": config.sweep_angles_deg[exact_min],
        "direct_minimum_angle_deg": config.sweep_angles_deg[direct_min],
        "exact_maximum_angle_deg": config.sweep_angles_deg[exact_max],
        "direct_maximum_angle_deg": config.sweep_angles_deg[direct_max],
        "exact_delta_phi_s": exact_delta,
        "direct_delta_phi_s": direct_delta,
        "direct_delta_at_exact_extrema": direct_at_exact_extrema,
        "delta_absolute_error": abs(direct_delta - exact_delta),
        "delta_relative_error": delta_relative_error,
        "paired_delta_sem": float(paired.std(ddof=1) / math.sqrt(sample_count)),
        "anisotropy_sign_preserved": anisotropy_sign_preserved,
        "passed": passed,
    }
    return rows, summary


def plot_angular(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    theta = np.array([row["theta_deg"] for row in rows])
    exact_curve = np.array([row["exact_yield"] for row in rows])
    direct_curve = np.array([row["direct_yield"] for row in rows])
    sem = np.array([row["direct_sem"] for row in rows])
    fig, axis = plt.subplots(figsize=(8.5, 5))
    axis.plot(theta, exact_curve, color="black", marker="o", label="exact oracle")
    axis.errorbar(theta, direct_curve, yerr=1.96 * sem, color="tab:blue", marker="o", capsize=3, label="full-pair RPS/Krylov")
    axis.set(xlabel="field angle (degrees)", ylabel="singlet yield", title=f"Coarse magnetosensitivity, M={summary['M']}")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "stage2_exact_vs_direct_angular_yield.png", dpi=180)
    plt.close(fig)


def reduced_validation(output_dir: Path, seed: int = 9917) -> dict[str, object]:
    dimensions = (2, 2, 3, 2)
    sx, sy, sz = spin_operators(0.5)
    field = field_vector_mT(67.0, 50.0)
    zeeman = TWOPI * GAMMA_E_MHZ_PER_MT * sum(value * op for value, op in zip(field, (sx, sy, sz)))
    terms = [LocalTerm((0,), zeeman, "eA Zeeman"), LocalTerm((1,), zeeman, "eB Zeeman")]
    terms.append(LocalTerm((0, 2), two_spin_coupling_operator(0.5, 1.0, TWOPI * np.array([[0.15, 0.02, 0.01], [0.02, -0.08, 0.03], [0.01, 0.03, 1.2]])), "A hyperfine"))
    terms.append(LocalTerm((1, 3), two_spin_coupling_operator(0.5, 0.5, TWOPI * np.array([[0.2, -0.03, 0.01], [-0.03, 0.1, 0.02], [0.01, 0.02, 0.8]])), "B hyperfine"))
    electron_tensor = TWOPI * (np.array([[0.12, 0.03, -0.02], [0.03, -0.07, 0.04], [-0.02, 0.04, -0.05]]) - 2.0 * 0.25 * np.eye(3))
    terms.append(LocalTerm((0, 1), two_spin_coupling_operator(0.5, 0.5, electron_tensor), "nonfactorizable J+D"))
    q_tensor = TWOPI * np.diag([-0.025, -0.025, 0.05])
    terms.append(LocalTerm((2,), quadrupole_operator(1.0, q_tensor), "spin-1 quadrupole"))
    hamiltonian = MatrixFreeHamiltonian(dimensions, terms)
    dense = hamiltonian.explicit_matrix()
    times = np.linspace(0.0, 2.0, 41)
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    nuclear_dimension = 6
    projector = np.kron(np.outer(singlet, singlet.conj()), np.eye(nuclear_dimension))
    rho0 = np.kron(np.outer(singlet, singlet.conj()), np.eye(nuclear_dimension) / nuclear_dimension)
    exact_trace = []
    exhaustive = []
    unitaries = []
    for time_us in times:
        unitary = scipy.linalg.expm(-1.0j * dense * time_us)
        unitaries.append(unitary)
        exact_trace.append(float(np.trace(projector @ unitary @ rho0 @ unitary.conj().T).real))
        basis_values = []
        for basis_index in range(nuclear_dimension):
            nuclear = np.eye(nuclear_dimension)[:, basis_index]
            state = np.kron(singlet, nuclear)
            evolved = unitary @ state
            basis_values.append(float(np.vdot(evolved, projector @ evolved).real))
        exhaustive.append(float(np.mean(basis_values)))
    exact_trace = np.asarray(exact_trace)
    exhaustive = np.asarray(exhaustive)
    rng = np.random.default_rng(seed)
    sample_traces = []
    maximum_krylov_error = 0.0
    for _ in range(256):
        nuclear = np.exp(1.0j * rng.uniform(0.0, TWOPI, nuclear_dimension)) / math.sqrt(nuclear_dimension)
        initial = np.kron(singlet, nuclear)
        dense_trace = np.array([float(np.vdot(unitary @ initial, projector @ (unitary @ initial)).real) for unitary in unitaries])
        sample_traces.append(dense_trace)
    for sample_index in range(8):
        rng_sample = np.random.default_rng(_seed(seed, sample_index))
        nuclear = np.exp(1.0j * rng_sample.uniform(0.0, TWOPI, nuclear_dimension)) / math.sqrt(nuclear_dimension)
        initial = np.kron(singlet, nuclear)
        result = AdaptiveLanczosPropagator(
            hamiltonian,
            minimum_dimension=8,
            maximum_dimension=24,
            dimension_increment=4,
            embedded_offset=2,
            state_error_budget=1e-10,
            initial_step_us=0.25,
            maximum_step_us=0.25,
        ).propagate(initial, times, lambda state: singlet_population(state, dimensions))
        dense_trace = np.array([float(np.vdot(unitary @ initial, projector @ (unitary @ initial)).real) for unitary in unitaries])
        maximum_krylov_error = max(maximum_krylov_error, float(np.max(np.abs(result.observable_values - dense_trace))))
    sample_traces = np.asarray(sample_traces)
    convergence_rows = []
    for count in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        mean = sample_traces[:count].mean(axis=0)
        convergence_rows.append({"M": count, "rms_error": float(np.sqrt(np.mean((mean - exact_trace) ** 2))), "max_error": float(np.max(np.abs(mean - exact_trace)))})
    _write_csv(output_dir / "reduced_model_convergence.csv", convergence_rows)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(times, exact_trace, color="black", label="exact density matrix")
    axis.plot(times, exhaustive, linestyle="--", label="exhaustive basis average")
    axis.plot(times, sample_traces[:16].mean(axis=0), label="random phase M=16")
    axis.set(xlabel="time (μs)", ylabel="singlet population", title="Reduced nonfactorizable J+D+Q validation")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "reduced_model_exact_comparison.png", dpi=180)
    plt.close(fig)
    summary = {
        "dimensions": dimensions,
        "J_MHz": 0.25,
        "dipolar_tensor_MHz": (electron_tensor / TWOPI + 0.5 * np.eye(3)).tolist(),
        "quadrupole_tensor_MHz": (q_tensor / TWOPI).tolist(),
        "exact_vs_exhaustive_max_error": float(np.max(np.abs(exact_trace - exhaustive))),
        "krylov_vs_dense_max_error": maximum_krylov_error,
        "random_phase_M16_rms_error": convergence_rows[4]["rms_error"],
        "random_phase_M256_rms_error": convergence_rows[-1]["rms_error"],
    }
    summary["passed"] = bool(summary["exact_vs_exhaustive_max_error"] <= 1e-12 and maximum_krylov_error <= 1e-8 and summary["random_phase_M256_rms_error"] <= 1e-2)
    _write_json(output_dir / "reduced_model_summary.json", summary)
    return summary


def _benchmark_worker(payload: tuple[ValidationConfig, float]) -> dict[str, object]:
    config, full_horizon = payload
    short_time = 0.1
    scaled = replace(config, state_error_budget=config.state_error_budget * short_time / full_horizon)
    benchmark_times = np.arange(
        0.0, short_time + 0.5 * config.output_step_us, config.output_step_us
    )
    return propagate_full_state(90.0, os.getpid(), benchmark_times, scaled)


def benchmark_concurrency(output_dir: Path, config: ValidationConfig) -> tuple[int, float, list[dict[str, object]]]:
    candidates = [value for value in (1, 2, 4, 8, 16, 32, 64) if value <= (os.cpu_count() or 1)]
    rows = []
    for workers in candidates:
        started = time.perf_counter()
        if workers == 1:
            results = [_benchmark_worker((config, config.maximum_time_us))]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_benchmark_worker, [(config, config.maximum_time_us)] * workers))
        elapsed = time.perf_counter() - started
        throughput = workers * 0.1 / elapsed
        rows.append({
            "workers": workers,
            "elapsed_seconds": elapsed,
            "simulated_us_per_wall_second": throughput,
            "mean_construction_seconds": float(np.mean([item["construction_seconds"] for item in results])),
            "mean_propagation_seconds": float(np.mean([item["propagation_seconds"] for item in results])),
            "maximum_peak_rss_mib": max((item["peak_rss_mib"] or 0.0) for item in results),
        })
        print(f"concurrency benchmark {workers}: {throughput:.6g} simulated us/s", flush=True)
    best = max(float(row["simulated_us_per_wall_second"]) for row in rows)
    selected = int(next(row["workers"] for row in rows if float(row["simulated_us_per_wall_second"]) >= 0.95 * best))
    selected_throughput = float(next(row["simulated_us_per_wall_second"] for row in rows if row["workers"] == selected))
    _write_csv(output_dir / "runtime_benchmark.csv", rows)
    return selected, selected_throughput, rows


def _unique_task_count(config: ValidationConfig, sweep_samples: int) -> int:
    return len(
        {
            (theta, sample)
            for theta in STAGE1_ANGLES
            for sample in range(config.stage1_max_samples)
        }
        | {
            (theta, sample)
            for theta in config.sweep_angles_deg
            for sample in range(sweep_samples)
        }
    )


def choose_workload(config: ValidationConfig, throughput: float, target_hours: float = 12.0) -> tuple[ValidationConfig, dict[str, object]]:
    candidates = [config, replace(config, sweep_angles_deg=FALLBACK_SWEEP), replace(config, lifetime_us=2.0, sweep_angles_deg=FALLBACK_SWEEP)]
    selected = candidates[-1]
    for candidate in candidates:
        projected = _unique_task_count(candidate, candidate.sweep_initial_samples) * candidate.maximum_time_us / throughput / 3600.0
        if projected <= target_hours:
            selected = candidate
            break
    projected = _unique_task_count(selected, selected.sweep_initial_samples) * selected.maximum_time_us / throughput / 3600.0
    if projected > 24.0:
        raise RuntimeError(f"Central validation projects to {projected:.1f} h; refusing to exceed 24 h.")
    return selected, {"projected_initial_hours": projected, "task_count": _unique_task_count(selected, selected.sweep_initial_samples), "target_hours": target_hours}


def ensure_manifest(output_dir: Path, config: ValidationConfig, resume: bool, workers: int, throughput: float, projection: dict[str, object]) -> dict[str, object]:
    path = output_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["configuration_hash"] != config.configuration_hash:
            raise RuntimeError("Existing validation output uses a different configuration.")
        if not resume:
            raise RuntimeError("Output exists; pass --resume.")
        return manifest
    manifest = {
        "status": "created",
        "configuration_hash": config.configuration_hash,
        "configuration": config.stable_dict(),
        "selected_workers": workers,
        "measured_simulated_us_per_wall_second": throughput,
        "runtime_projection": projection,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_started_epoch": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    _write_json(path, manifest)
    return manifest


def final_report(output_dir: Path, config: ValidationConfig, manifest: dict[str, object], reduced: dict[str, object]) -> dict[str, object]:
    stage1_rows, recommended = stage1_metrics(output_dir, config, config.stage1_max_samples)
    _write_csv(output_dir / "stage1_metrics.csv", stage1_rows)
    plot_stage1(output_dir, config, stage1_rows)
    sample_count = int(manifest.get("stage2_M", config.sweep_initial_samples))
    angular_rows, angular_summary = angular_metrics(output_dir, config, sample_count)
    _write_csv(output_dir / "stage2_angular_curve.csv", angular_rows)
    _write_json(output_dir / "stage2_angular_summary.json", angular_summary)
    plot_angular(output_dir, angular_rows, angular_summary)
    grid_assessment = reference_grid_assessment(config)
    _write_json(output_dir / "reference_time_grid_assessment.json", grid_assessment)
    cells = [load_trajectory(output_dir, theta, sample, config) for theta in set(STAGE1_ANGLES) | set(config.sweep_angles_deg) for sample in range(config.stage1_max_samples if theta in STAGE1_ANGLES else sample_count) if checkpoint_path(output_dir, theta, sample).exists()]
    runtime = {
        "projected_total_wall_hours": float(manifest["runtime_projection"]["projected_initial_hours"]),
        "actual_elapsed_wall_hours": (time.time() - float(manifest["run_started_epoch"])) / 3600.0,
        "actual_total_trajectory_seconds": float(sum(float(cell["total_seconds"]) for cell in cells)),
        "mean_trajectory_seconds": float(np.mean([cell["total_seconds"] for cell in cells])),
        "maximum_peak_rss_mib": max(float(cell["peak_rss_mib"] or 0.0) for cell in cells),
        "completed_trajectories": len(cells),
        "projected_one_full_horizon_trajectory_hours_at_selected_throughput": config.maximum_time_us / float(manifest["measured_simulated_us_per_wall_second"]) / 3600.0,
    }
    _write_csv(output_dir / "runtime_summary.csv", [runtime])
    stage1_passed = recommended is not None
    result = {
        "stage1_passed": stage1_passed,
        "stage2_passed": bool(angular_summary["passed"]),
        "reduced_nonfactorizable_passed": bool(reduced["passed"]),
        "reference_time_grid_passed": bool(grid_assessment["passed"]),
        "recommended_full_model_random_phase_samples": recommended,
        "runtime": runtime,
        "angular_summary": angular_summary,
        "reference_time_grid": grid_assessment,
    }
    result["validation_passed"] = bool(
        result["stage1_passed"]
        and result["stage2_passed"]
        and result["reduced_nonfactorizable_passed"]
        and result["reference_time_grid_passed"]
    )
    _write_json(output_dir / "validation_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "auto",
            "benchmark",
            "reduced",
            "stage1",
            "stage2",
            "report",
            "reintegrate",
        ),
        default="auto",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--walltime-hours", type=float, default=23.5)
    parser.add_argument("--target-hours", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "quantum_needle_fullspace_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.walltime_hours <= 24.0:
        raise ValueError("--walltime-hours must be in (0, 24].")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = ValidationConfig()
    benchmark_path = output_dir / "benchmark_selection.json"
    if benchmark_path.exists() and args.resume:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        workers = int(benchmark["selected_workers"])
        throughput = float(benchmark["throughput"])
    else:
        workers, throughput, benchmark_rows = benchmark_concurrency(output_dir, base_config)
        if args.workers > 0:
            workers = args.workers
            matching = [
                row for row in benchmark_rows if int(row["workers"]) == workers
            ]
            if not matching:
                raise ValueError(
                    "--workers must be one of the benchmarked powers of two."
                )
            throughput = float(matching[0]["simulated_us_per_wall_second"])
        _write_json(
            benchmark_path,
            {
                "selected_workers": workers,
                "throughput": throughput,
                "rows": benchmark_rows,
            },
        )
    if args.stage == "benchmark":
        return
    config, projection = choose_workload(base_config, throughput, args.target_hours)
    manifest = ensure_manifest(output_dir, config, args.resume, workers, throughput, projection)
    if args.stage == "reintegrate":
        print(
            json.dumps(
                reintegrate_saved_trajectories(output_dir, config),
                indent=2,
                sort_keys=True,
            )
        )
        return
    deadline = time.time() + args.walltime_hours * 3600.0
    reduced_path = output_dir / "reduced_model_summary.json"
    reduced = json.loads(reduced_path.read_text()) if reduced_path.exists() else reduced_validation(output_dir)
    if args.stage == "reduced":
        return
    if args.stage in ("auto", "stage1"):
        for count in (4, 8, 12, 16):
            if not run_tasks(output_dir, STAGE1_ANGLES, count, config, workers, deadline):
                manifest["status"] = "walltime_limit_reached"
                _write_json(output_dir / "manifest.json", manifest)
                return
            rows, recommendation = stage1_metrics(output_dir, config, count)
            _write_csv(output_dir / f"stage1_metrics_M{count}.csv", rows)
            print(f"stage1 M={count}, recommended={recommendation}", flush=True)
        manifest["status"] = "stage1_complete"
        _write_json(output_dir / "manifest.json", manifest)
    if args.stage == "stage1":
        return
    if args.stage in ("auto", "stage2"):
        stage2_M = config.sweep_initial_samples
        if not run_tasks(output_dir, config.sweep_angles_deg, stage2_M, config, workers, deadline):
            manifest["status"] = "walltime_limit_reached"
            _write_json(output_dir / "manifest.json", manifest)
            return
        _, summary = angular_metrics(output_dir, config, stage2_M)
        if not summary["passed"] and config.sweep_max_samples > stage2_M:
            extra_hours = (_unique_task_count(config, config.sweep_max_samples) - _unique_task_count(config, stage2_M)) * config.maximum_time_us / throughput / 3600.0
            if time.time() + extra_hours * 3600.0 < deadline:
                stage2_M = config.sweep_max_samples
                if not run_tasks(
                    output_dir,
                    config.sweep_angles_deg,
                    stage2_M,
                    config,
                    workers,
                    deadline,
                ):
                    manifest["status"] = "walltime_limit_reached"
                    _write_json(output_dir / "manifest.json", manifest)
                    return
        manifest.update({"status": "stage2_complete", "stage2_M": stage2_M})
        _write_json(output_dir / "manifest.json", manifest)
    if args.stage == "stage2":
        return
    result = final_report(output_dir, config, manifest, reduced)
    manifest.update({"status": "complete", "result": result})
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
