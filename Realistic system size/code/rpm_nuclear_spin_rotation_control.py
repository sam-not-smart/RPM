"""Adaptive FAD-N5 nuclear-spin rotation control experiment.

This experiment asks whether ideal periodic pi rotations of the dominant FAD
nitrogen suppress the low-field angular singlet-yield feature.  It uses the
current complete 14-nucleus, nonfactorized Danny Hamiltonian and a 10-us
recombination lifetime.  A paired M=1 five-angle interval screen is followed
by an M=4 85--95 degree confirmation of the best sequence when a prespecified
damping gate is met.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from poster_reproduction.core import (
    MASTER_SEED,
    PRODUCTION_MODEL,
    ProductionModel,
    build_hamiltonian,
    dimensions_for,
    local_unitary_apply,
    make_propagator,
    random_phase_initial_state,
    spin_pi_pulse,
)
from rpm_quantum_needle_jd_sweeps import SweepConfig, integrate_yield
from rpm_quantum_needle_model import ALL_NUCLEI, FAD_NUCLEI, NUCLEAR_RADICALS, Nucleus, spin_operators
from rpm_statevector_engine import SingletPopulationObservable, singlet_population


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("outputs") / "nuclear_spin_rotation_control_tau10"
SCREEN_ANGLES_DEG = (88.0, 89.0, 90.0, 91.0, 92.0)
CONFIRM_ANGLES_DEG = tuple(float(value) for value in range(85, 96))
PULSE_INTERVALS_US = (0.02, 0.03, 0.05, 0.075, 0.10, 0.20)
PRIMARY_TARGET = "FAD:N5"
POSTER_TARGET = "FAD:N10"


@dataclass(frozen=True)
class ControlConfig:
    master_seed: int = MASTER_SEED
    field_mT: float = 0.05
    lifetime_us: float = 10.0
    survival_tail_bound: float = 1.0e-4
    integration_step_us: float = 0.00125
    coarse_integration_step_us: float = 0.0025
    state_error_budget: float = 2.4e-5
    screen_angles_deg: tuple[float, ...] = SCREEN_ANGLES_DEG
    confirm_angles_deg: tuple[float, ...] = CONFIRM_ANGLES_DEG
    pulse_intervals_us: tuple[float, ...] = PULSE_INTERVALS_US
    screen_samples: int = 1
    confirm_samples: int = 4
    primary_target: str = PRIMARY_TARGET
    poster_target: str = POSTER_TARGET
    damping_gate_fraction: float = 0.10
    bootstrap_replicates: int = 2000
    smoke: bool = False

    @property
    def model(self) -> ProductionModel:
        return replace(
            PRODUCTION_MODEL,
            master_seed=self.master_seed,
            samples=self.confirm_samples,
            lifetime_us=self.lifetime_us,
        )

    @property
    def sweep(self) -> SweepConfig:
        return replace(
            self.model.sweep_config,
            seed=self.master_seed,
            field_microT=self.field_mT * 1000.0,
            lifetime_us=self.lifetime_us,
            survival_tail_bound=self.survival_tail_bound,
            integration_step_us=self.integration_step_us,
            coarse_integration_step_us=self.coarse_integration_step_us,
            state_error_budget=self.state_error_budget,
        )

    @property
    def maximum_time_us(self) -> float:
        return self.sweep.maximum_time_us

    @property
    def time_grid_us(self) -> np.ndarray:
        return self.sweep.time_grid_us

    def stable_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "schema_version": SCHEMA_VERSION,
                "maximum_time_us": self.maximum_time_us,
                "model": self.model.stable_dict(),
                "pulse": {
                    "rotation_angle_rad": math.pi,
                    "axis": "target hyperfine tensor dominant principal axis",
                    "timing": "instantaneous pulses at t=n*interval through the full reaction horizon",
                    "interpretation": "ideal mechanistic control; no finite pulse width or drive Hamiltonian",
                },
            }
        )
        return result

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def smoke_config(seed: int = MASTER_SEED) -> ControlConfig:
    return ControlConfig(
        master_seed=seed,
        lifetime_us=0.05,
        survival_tail_bound=math.exp(-1.0),
        integration_step_us=0.0025,
        coarse_integration_step_us=0.005,
        state_error_budget=1.0e-6,
        screen_angles_deg=(89.0, 90.0, 91.0),
        confirm_angles_deg=(89.0, 90.0, 91.0),
        pulse_intervals_us=(0.01, 0.02),
        screen_samples=1,
        confirm_samples=2,
        damping_gate_fraction=0.0,
        bootstrap_replicates=100,
        smoke=True,
    )


@dataclass(frozen=True)
class ControlTask:
    target: str
    protocol: str
    theta_deg: float
    sample_id: int
    interval_us: float = 0.0
    role: str = "screen"

    @property
    def key(self) -> str:
        target = self.target.replace(":", "_")
        interval = f"_dt{self.interval_us:.6f}" if self.protocol == "periodic" else ""
        return (
            f"{target}_{self.protocol}{interval}_th{self.theta_deg:07.3f}_s{self.sample_id:03d}"
            .replace(".", "p")
        )

    @property
    def label(self) -> str:
        if self.protocol == "baseline":
            return "baseline"
        if self.protocol == "transverse_deleted":
            return f"{self.target} transverse deleted"
        return f"{self.target} periodic pi, {self.interval_us:g} us"


def target_components(
    target: str,
    nuclei: Sequence[Nucleus],
    radicals: Sequence[str],
) -> tuple[int, Nucleus, str]:
    keys = [f"{radical}:{nucleus.name}" for nucleus, radical in zip(nuclei, radicals)]
    if target not in keys:
        raise ValueError(f"Target {target!r} is not present in this system: {keys}")
    local_index = keys.index(target)
    return local_index + 2, nuclei[local_index], radicals[local_index]


def dominant_principal_axis(tensor: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(tensor, dtype=float) + np.asarray(tensor, dtype=float).T)
    values, vectors = np.linalg.eigh(symmetric)
    axis = vectors[:, int(np.argmax(np.abs(values)))].copy()
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis *= -1.0
    return axis


def transverse_deleted_tensor(tensor: np.ndarray) -> np.ndarray:
    """Delete molecular-frame Ix/Iy columns while retaining the Iz column."""

    deleted = np.asarray(tensor, dtype=float).copy()
    deleted[:, :2] = 0.0
    return deleted


def pulse_schedule(interval_us: float, final_time_us: float) -> np.ndarray:
    interval = float(interval_us)
    if interval <= 0.0:
        raise ValueError("Pulse interval must be positive.")
    return np.arange(interval, float(final_time_us), interval)


def _system(config: ControlConfig) -> tuple[tuple[Nucleus, ...], tuple[str, ...]]:
    if config.smoke:
        return tuple(FAD_NUCLEI[:2]), ("FAD", "FAD")
    return tuple(ALL_NUCLEI), tuple(NUCLEAR_RADICALS)


def _coarse_indices(config: ControlConfig, times: np.ndarray) -> np.ndarray:
    ratio = int(round(config.coarse_integration_step_us / config.integration_step_us))
    if ratio < 1 or not math.isclose(
        ratio * config.integration_step_us,
        config.coarse_integration_step_us,
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        raise ValueError("Coarse integration step must be an integer multiple of the fine step.")
    indices = np.arange(0, times.size, ratio, dtype=int)
    if indices[-1] != times.size - 1:
        indices = np.append(indices, times.size - 1)
    return indices


def propagate_control_task(task: ControlTask, config: ControlConfig) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    nuclei, radicals = _system(config)
    dimensions = dimensions_for(nuclei)
    target_axis, nucleus, radical = target_components(task.target, nuclei, radicals)
    overrides = None
    if task.protocol == "transverse_deleted":
        overrides = {task.target: transverse_deleted_tensor(nucleus.tensor_mT)}
    hamiltonian = build_hamiltonian(
        task.theta_deg,
        config.field_mT,
        model=config.model,
        nuclei=nuclei,
        radicals=radicals,
        hyperfine_overrides_mT=overrides,
    )
    state = random_phase_initial_state(dimensions, config.master_seed, task.sample_id)
    times = config.time_grid_us
    pulse_times = (
        pulse_schedule(task.interval_us, float(times[-1]))
        if task.protocol == "periodic"
        else np.array([], dtype=float)
    )
    pulse = spin_pi_pulse(nucleus.spin, dominant_principal_axis(nucleus.tensor_mT))
    observable = SingletPopulationObservable(dimensions)
    populations = np.empty(times.size, dtype=float)
    bounds = np.empty(times.size, dtype=float)
    populations[0] = singlet_population(state, dimensions)
    bounds[0] = 0.0
    cursor = 1
    current = 0.0
    accepted = rejected = matvecs = 0
    maximum_bound = 0.0
    accumulated_state_error = 0.0
    for boundary in np.append(pulse_times, times[-1]):
        boundary = float(boundary)
        stop = int(np.searchsorted(times, boundary + 1e-12, side="right"))
        requested = times[cursor:stop] - current
        duration = boundary - current
        if requested.size == 0 or requested[-1] < duration - 1e-12:
            local_times = np.append(requested, duration)
        else:
            local_times = requested
        if duration > 0.0:
            propagator = make_propagator(hamiltonian, config.model)
            propagator.state_error_budget = config.state_error_budget * duration / times[-1]
            result = propagator.propagate(state, local_times, observable)
            count = requested.size
            populations[cursor:stop] = result.observable_values[:count]
            global_point_errors = accumulated_state_error + result.state_error_estimates[:count]
            bounds[cursor:stop] = 2.0 * global_point_errors + global_point_errors**2
            state = result.final_state
            accepted += result.stats.accepted_steps
            rejected += result.stats.rejected_steps
            matvecs += result.stats.matvecs
            accumulated_state_error += result.stats.accumulated_state_error
            if count:
                maximum_bound = max(maximum_bound, float(np.max(bounds[cursor:stop])))
        cursor = stop
        current = boundary
        if boundary < times[-1] - 1e-12:
            state = local_unitary_apply(state, dimensions, target_axis, pulse)
    fine_yield = integrate_yield(times, populations, config.lifetime_us)
    coarse_ids = _coarse_indices(config, times)
    coarse_yield = integrate_yield(times[coarse_ids], populations[coarse_ids], config.lifetime_us)
    yield_population_error_bound = integrate_yield(times, bounds, config.lifetime_us)
    retain = math.isclose(task.theta_deg, 90.0, abs_tol=1e-12)
    metadata = {
        "task": asdict(task),
        "label": task.label,
        "target_axis": target_axis,
        "target_spin": nucleus.spin,
        "target_radical": radical,
        "pulse_count": int(pulse_times.size),
        "configuration_hash": config.configuration_hash,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {
        "yield": np.array(fine_yield),
        "coarse_yield": np.array(coarse_yield),
        "quadrature_error": np.array(abs(fine_yield - coarse_yield)),
        "yield_population_error_bound": np.array(yield_population_error_bound),
        "accumulated_state_error": np.array(accumulated_state_error),
        "initial_singlet": np.array(populations[0]),
        "maximum_population_error_bound": np.array(maximum_bound),
        "times_us": times if retain else np.array([], dtype=float),
        "singlet_population": populations if retain else np.array([], dtype=float),
        "population_error_bound": bounds if retain else np.array([], dtype=float),
        "accepted_steps": np.array(accepted),
        "rejected_steps": np.array(rejected),
        "matvecs": np.array(matvecs),
        "pulse_count": np.array(pulse_times.size),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def cell_path(output_dir: Path, task: ControlTask) -> Path:
    return output_dir / "checkpoints" / "cells" / f"{task.key}.npz"


def atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ensure_manifest(output_dir: Path, config: ControlConfig) -> None:
    path = output_dir / "manifest.json"
    payload = {
        "experiment": "dominant-nitrogen periodic spin-rotation control",
        "configuration_hash": config.configuration_hash,
        "configuration": config.stable_dict(),
        "git_commit": _git_commit(),
        "created_host": socket.gethostname(),
        "python": platform.python_version(),
        "system": {
            "primary_target": PRIMARY_TARGET,
            "poster_reference_target": POSTER_TARGET,
            "quadrupole_scale": config.model.quadrupole_scale,
            "state_dimension": math.prod(dimensions_for(ALL_NUCLEI)),
        },
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("configuration_hash") != config.configuration_hash:
            raise ValueError(f"Incompatible resume manifest: {path}")
        return
    atomic_json(path, payload)


def screen_protocols(config: ControlConfig) -> tuple[tuple[str, float], ...]:
    return (
        ("baseline", 0.0),
        ("transverse_deleted", 0.0),
        *(("periodic", interval) for interval in config.pulse_intervals_us),
    )


def screen_tasks(config: ControlConfig) -> list[ControlTask]:
    tasks = [
        ControlTask(config.primary_target, protocol, theta, 0, interval, "screen")
        for theta in config.screen_angles_deg
        for protocol, interval in screen_protocols(config)
    ]
    if not config.smoke:
        tasks.extend(
            [
                ControlTask(config.poster_target, "transverse_deleted", 90.0, 0, role="poster_anchor"),
                ControlTask(config.poster_target, "periodic", 90.0, 0, 0.02, "poster_anchor"),
                ControlTask(config.poster_target, "periodic", 90.0, 0, 0.05, "poster_anchor"),
            ]
        )
    return tasks


def confirmation_tasks(config: ControlConfig, best_interval_us: float) -> list[ControlTask]:
    protocols = (("baseline", 0.0), ("transverse_deleted", 0.0), ("periodic", best_interval_us))
    return [
        ControlTask(config.primary_target, protocol, theta, sample_id, interval, "confirm")
        for theta in config.confirm_angles_deg
        for protocol, interval in protocols
        for sample_id in range(config.confirm_samples)
    ]


def _curve(output_dir: Path, tasks: Sequence[ControlTask]) -> np.ndarray:
    return np.array([float(np.load(cell_path(output_dir, task))["yield"]) for task in tasks])


def local_metrics(theta: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(theta, dtype=float)
    y = np.asarray(values, dtype=float)
    derivative = np.gradient(y, x, edge_order=2 if x.size >= 3 else 1)
    center = int(np.argmin(np.abs(x - 90.0)))
    edge_reference = 0.5 * (y[0] + y[-1])
    return {
        "local_range": float(np.max(y) - np.min(y)),
        "maximum_abs_slope_per_deg": float(np.max(np.abs(derivative))),
        "theta_minimum_deg": float(x[int(np.argmin(y))]),
        "theta_maximum_deg": float(x[int(np.argmax(y))]),
        "yield_at_90": float(y[center]),
        "signed_center_feature": float(y[center] - edge_reference),
        "absolute_center_feature": float(abs(y[center] - edge_reference)),
    }


def select_best_interval(output_dir: Path, config: ControlConfig) -> dict[str, object]:
    theta = np.asarray(config.screen_angles_deg)

    def tasks_for(protocol: str, interval: float = 0.0) -> list[ControlTask]:
        return [ControlTask(config.primary_target, protocol, float(angle), 0, interval) for angle in theta]

    baseline_values = _curve(output_dir, tasks_for("baseline"))
    deleted_values = _curve(output_dir, tasks_for("transverse_deleted"))
    baseline = local_metrics(theta, baseline_values)
    deleted = local_metrics(theta, deleted_values)
    rows: list[dict[str, float]] = []
    for interval in config.pulse_intervals_us:
        values = _curve(output_dir, tasks_for("periodic", interval))
        metrics = local_metrics(theta, values)
        range_ratio = metrics["local_range"] / max(baseline["local_range"], 1e-30)
        slope_ratio = metrics["maximum_abs_slope_per_deg"] / max(
            baseline["maximum_abs_slope_per_deg"], 1e-30
        )
        rows.append(
            {
                "interval_us": float(interval),
                **metrics,
                "range_ratio_to_baseline": float(range_ratio),
                "slope_ratio_to_baseline": float(slope_ratio),
                "damping_score": float(0.5 * (range_ratio + slope_ratio)),
                "rmse_to_transverse_deleted": float(np.sqrt(np.mean((values - deleted_values) ** 2))),
            }
        )
    best = min(rows, key=lambda row: (row["damping_score"], row["rmse_to_transverse_deleted"]))
    threshold = 1.0 - config.damping_gate_fraction
    gate_passed = bool(
        min(best["range_ratio_to_baseline"], best["slope_ratio_to_baseline"]) <= threshold
        and max(best["range_ratio_to_baseline"], best["slope_ratio_to_baseline"]) <= 1.10
    )
    return {
        "configuration_hash": config.configuration_hash,
        "best_interval_us": best["interval_us"],
        "gate_passed": gate_passed,
        "gate_definition": (
            f"at least {100*config.damping_gate_fraction:.1f}% reduction in local range or maximum slope, "
            "with the other metric no more than 10% worse"
        ),
        "baseline": baseline,
        "transverse_deleted": deleted,
        "interval_metrics": rows,
    }


def load_or_select(output_dir: Path, config: ControlConfig) -> dict[str, object]:
    path = output_dir / "selection.json"
    selection = select_best_interval(output_dir, config)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("configuration_hash") != config.configuration_hash:
            raise ValueError("Selection file is incompatible with this configuration.")
    atomic_json(path, selection)
    return selection


def _run_one(task: ControlTask, config: ControlConfig) -> tuple[ControlTask, dict[str, np.ndarray]]:
    return task, propagate_control_task(task, config)


def run_tasks(
    tasks: Sequence[ControlTask],
    output_dir: Path,
    config: ControlConfig,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    start_guard_seconds: float,
) -> dict[str, int]:
    pending: list[ControlTask] = []
    for task in tasks:
        path = cell_path(output_dir, task)
        if path.exists():
            if resume:
                continue
            raise FileExistsError(f"Checkpoint exists; pass --resume: {path}")
        pending.append(task)
    completed = skipped = len(tasks) - len(pending)
    if not pending:
        return {"planned": len(tasks), "completed": completed, "skipped": skipped, "remaining": 0}
    active = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            while pending and len(active) < workers and time.monotonic() + start_guard_seconds < deadline:
                task = pending.pop(0)
                active[pool.submit(_run_one, task, config)] = task
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                returned_task, values = future.result()
                if returned_task != task:
                    raise RuntimeError("Worker returned a mismatched task.")
                atomic_npz(cell_path(output_dir, task), values)
                completed += 1
                metadata = json.loads(str(values["metadata_json"]))
                print(
                    f"completed {completed}/{len(tasks)} {task.key} "
                    f"yield={float(values['yield']):.9f} elapsed={metadata['elapsed_seconds']/3600:.2f} h",
                    flush=True,
                )
    return {
        "planned": len(tasks),
        "completed": completed,
        "skipped": skipped,
        "remaining": len(pending),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_raw_cell_results(output_dir: Path, tasks: Sequence[ControlTask]) -> None:
    """Export every completed unique cell without discarding checkpoint detail."""

    rows: list[dict[str, object]] = []
    unique = {task.key: task for task in tasks}
    for task in sorted(unique.values(), key=lambda item: item.key):
        path = cell_path(output_dir, task)
        if not path.exists():
            continue
        with np.load(path) as cell:
            metadata = json.loads(str(cell["metadata_json"]))
            rows.append(
                {
                    **asdict(task),
                    "key": task.key,
                    "label": task.label,
                    "yield": float(cell["yield"]),
                    "coarse_yield": float(cell["coarse_yield"]),
                    "quadrature_error": float(cell["quadrature_error"]),
                    "yield_population_error_bound": float(cell["yield_population_error_bound"]),
                    "maximum_population_error_bound": float(cell["maximum_population_error_bound"]),
                    "accumulated_state_error": float(cell["accumulated_state_error"]),
                    "accepted_steps": int(cell["accepted_steps"]),
                    "rejected_steps": int(cell["rejected_steps"]),
                    "matvecs": int(cell["matvecs"]),
                    "pulse_count": int(cell["pulse_count"]),
                    "elapsed_seconds": float(metadata["elapsed_seconds"]),
                    "host": metadata["host"],
                    "pid": metadata["pid"],
                }
            )
    _write_csv(output_dir / "raw_cell_yields.csv", rows)


def _point_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0)
    if values.shape[0] < 2:
        nan = np.full_like(mean, np.nan)
        return mean, nan, nan, nan
    se = scipy.stats.sem(values, axis=0)
    half = scipy.stats.t.ppf(0.975, values.shape[0] - 1) * se
    return mean, se, mean - half, mean + half


def generate_report(output_dir: Path, config: ControlConfig, selection: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    screen_rows = list(selection["interval_metrics"])
    _write_csv(output_dir / "screen_interval_metrics.csv", screen_rows)

    theta_screen = np.asarray(config.screen_angles_deg)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for protocol, interval in screen_protocols(config):
        tasks = [ControlTask(config.primary_target, protocol, float(angle), 0, interval) for angle in theta_screen]
        if not all(cell_path(output_dir, task).exists() for task in tasks):
            continue
        axis.plot(theta_screen, _curve(output_dir, tasks), marker="o", label=tasks[0].label)
    axis.set(xlabel="field orientation (degrees)", ylabel="singlet yield", title="N5 pulse-interval screen at 50 uT; tau=10 us")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "n5_interval_screen_curves.png", dpi=220)
    plt.close(fig)

    if screen_rows:
        fig, axis = plt.subplots(figsize=(6.8, 4.5))
        intervals = [row["interval_us"] for row in screen_rows]
        axis.plot(intervals, [1.0 - row["range_ratio_to_baseline"] for row in screen_rows], marker="o", label="local-range suppression")
        axis.plot(intervals, [1.0 - row["slope_ratio_to_baseline"] for row in screen_rows], marker="s", label="maximum-slope suppression")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(xlabel="pulse interval (us)", ylabel="fractional suppression", title="N5 control suppression versus pulse interval")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / "n5_suppression_vs_interval.png", dpi=220)
        plt.close(fig)

    if not config.smoke:
        anchor_specs = (
            (
                ControlTask(config.primary_target, "baseline", 90.0, 0, role="poster_anchor"),
                "baseline",
            ),
            (
                ControlTask(config.poster_target, "transverse_deleted", 90.0, 0, role="poster_anchor"),
                "N10 transverse deleted",
            ),
            (
                ControlTask(config.poster_target, "periodic", 90.0, 0, 0.02, "poster_anchor"),
                "N10 periodic pi, 0.02 us",
            ),
            (
                ControlTask(config.poster_target, "periodic", 90.0, 0, 0.05, "poster_anchor"),
                "N10 periodic pi, 0.05 us",
            ),
        )
        if all(cell_path(output_dir, task).exists() for task, _label in anchor_specs):
            anchor_rows: list[dict[str, object]] = []
            for task, label in anchor_specs:
                with np.load(cell_path(output_dir, task)) as cell:
                    anchor_rows.append(
                        {
                            "protocol": label,
                            "target": task.target,
                            "interval_us": task.interval_us,
                            "theta_deg": task.theta_deg,
                            "sample_id": task.sample_id,
                            "yield": float(cell["yield"]),
                            "quadrature_error": float(cell["quadrature_error"]),
                            "yield_population_error_bound": float(cell["yield_population_error_bound"]),
                        }
                    )
            _write_csv(output_dir / "n10_poster_anchor_90deg.csv", anchor_rows)
            fig, axis = plt.subplots(figsize=(7.2, 4.6))
            positions = np.arange(len(anchor_rows))
            axis.bar(
                positions,
                [float(row["yield"]) for row in anchor_rows],
                color=("#4c78a8", "#f58518", "#54a24b", "#e45756"),
            )
            axis.set_xticks(positions, [str(row["protocol"]) for row in anchor_rows], rotation=18, ha="right")
            axis.set(
                ylabel="singlet yield",
                title="N10 poster-anchor controls at 90 degrees (M=1)",
            )
            axis.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures / "n10_poster_anchor_90deg.png", dpi=220)
            plt.close(fig)

    best = float(selection["best_interval_us"])
    confirm = confirmation_tasks(config, best)
    complete = all(cell_path(output_dir, task).exists() for task in confirm)
    if not complete:
        text = (
            "# Nuclear spin rotation control\n\n"
            f"Screen complete. Best interval: **{best:g} us**. "
            f"Confirmation gate passed: **{selection['gate_passed']}**.\n\n"
            "The M=4 confirmation is incomplete; resume the runner to generate final confidence intervals.\n"
        )
        (output_dir / "report.md").write_text(text, encoding="utf-8")
        return

    theta = np.asarray(config.confirm_angles_deg)
    protocol_specs = (
        ("baseline", 0.0, "baseline"),
        ("transverse_deleted", 0.0, "N5 transverse deleted"),
        ("periodic", best, f"N5 periodic pi, {best:g} us"),
    )
    curve_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(config.master_seed + 991)
    bootstrap_ids = rng.integers(0, config.confirm_samples, size=(config.bootstrap_replicates, config.confirm_samples))
    for protocol, interval, label in protocol_specs:
        values = np.empty((config.confirm_samples, theta.size), dtype=float)
        for sample in range(config.confirm_samples):
            tasks = [ControlTask(config.primary_target, protocol, float(angle), sample, interval, "confirm") for angle in theta]
            values[sample] = _curve(output_dir, tasks)
        curves[label] = values
        mean, se, low, high = _point_interval(values)
        for index, angle in enumerate(theta):
            curve_rows.append(
                {
                    "protocol": label,
                    "theta_deg": angle,
                    "samples": config.confirm_samples,
                    "mean_yield": mean[index],
                    "standard_error": se[index],
                    "ci95_low": low[index],
                    "ci95_high": high[index],
                }
            )
        point = local_metrics(theta, mean)
        boot = [local_metrics(theta, np.mean(values[ids], axis=0)) for ids in bootstrap_ids]
        row: dict[str, object] = {"protocol": label, "samples": config.confirm_samples, **point}
        for metric in ("local_range", "maximum_abs_slope_per_deg", "absolute_center_feature"):
            distribution = np.array([item[metric] for item in boot])
            row[f"{metric}_ci95_low"] = float(np.quantile(distribution, 0.025))
            row[f"{metric}_ci95_high"] = float(np.quantile(distribution, 0.975))
        metric_rows.append(row)
    _write_csv(output_dir / "confirmation_curves.csv", curve_rows)
    _write_csv(output_dir / "confirmation_metrics.csv", metric_rows)

    fig, axis = plt.subplots(figsize=(7.4, 5.0))
    for label, values in curves.items():
        mean, _se, low, high = _point_interval(values)
        line = axis.plot(theta, mean, marker="o", label=label)[0]
        axis.fill_between(theta, low, high, color=line.get_color(), alpha=0.18)
    axis.set(xlabel="field orientation (degrees)", ylabel="singlet yield", title="Dominant-nitrogen spin-rotation control; 50 uT, tau=10 us")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "n5_control_angular_curves_with_95ci.png", dpi=220)
    plt.close(fig)

    dynamics_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True, sharey=True)
    for axis, (protocol, interval, label) in zip(axes, protocol_specs):
        populations = []
        times = None
        for sample in range(config.confirm_samples):
            task = ControlTask(config.primary_target, protocol, 90.0, sample, interval, "confirm")
            with np.load(cell_path(output_dir, task)) as cell:
                times = np.asarray(cell["times_us"])
                populations.append(np.asarray(cell["singlet_population"]))
        population_array = np.asarray(populations)
        mean = np.mean(population_array, axis=0)
        axis.plot(times, mean, linewidth=1.0, label=label)
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        for time_value, population in zip(times, mean):
            dynamics_rows.append({"protocol": label, "time_us": time_value, "mean_singlet_population": population, "samples": config.confirm_samples})
    axes[-1].set_xlabel("time (us)")
    fig.supylabel("singlet population")
    fig.suptitle("N5-controlled dynamics at 90 degrees")
    fig.tight_layout()
    fig.savefig(figures / "n5_control_dynamics_90deg.png", dpi=220)
    plt.close(fig)
    _write_csv(output_dir / "confirmation_dynamics_90deg.csv", dynamics_rows)

    baseline_metric = next(row for row in metric_rows if row["protocol"] == "baseline")
    best_metric = next(row for row in metric_rows if row["protocol"].startswith("N5 periodic"))
    range_suppression = 1.0 - float(best_metric["local_range"]) / max(float(baseline_metric["local_range"]), 1e-30)
    slope_suppression = 1.0 - float(best_metric["maximum_abs_slope_per_deg"]) / max(float(baseline_metric["maximum_abs_slope_per_deg"]), 1e-30)
    text = f"""# Nuclear spin rotation control

Complete 14-nucleus nonfactorized state-vector calculation at 50 uT with a
10-us recombination lifetime and Q=0.

- Best screened N5 pulse interval: **{best:g} us**
- Screen gate passed: **{selection['gate_passed']}**
- Confirmed local-range suppression: **{100*range_suppression:.2f}%**
- Confirmed maximum-slope suppression: **{100*slope_suppression:.2f}%**
- Nuclear trace samples: **M={config.confirm_samples}**, paired across protocols and angles
- Pulse model: ideal instantaneous pi rotations throughout the {config.maximum_time_us:.3f}-us reaction horizon

The transverse-deleted curve is a mechanistic benchmark, not a separately
physical Hamiltonian. Pulse intervals this short are also an ideal-control
probe rather than an experimental hardware prescription.
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def write_inventory(output_dir: Path, tasks: Sequence[ControlTask]) -> None:
    rows = [{**asdict(task), "key": task.key, "label": task.label, "checkpoint": str(cell_path(output_dir, task))} for task in tasks]
    _write_csv(output_dir / "task_inventory.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "screen", "confirm", "auto", "report"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--max-wall-hours", type=float, default=24.0)
    parser.add_argument("--start-guard-hours", type=float, default=7.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-confirm", action="store_true", help="Run confirmation even when the screen gate fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.max_wall_hours <= 0.0 or args.start_guard_hours < 0.0:
        raise ValueError("Wall-time values must be nonnegative, with a positive maximum.")
    config = smoke_config(args.seed) if args.stage == "smoke" else ControlConfig(master_seed=args.seed)
    ensure_manifest(args.output_dir, config)
    deadline = time.monotonic() + 3600.0 * args.max_wall_hours
    guard = 0.0 if config.smoke else 3600.0 * args.start_guard_hours
    screen = screen_tasks(config)
    write_inventory(args.output_dir, screen)
    if args.stage in ("smoke", "screen", "auto"):
        snapshot = run_tasks(
            screen,
            args.output_dir,
            config,
            workers=min(args.workers, len(screen)),
            resume=args.resume,
            deadline=deadline,
            start_guard_seconds=guard,
        )
        atomic_json(args.output_dir / "screen_status.json", snapshot)
    write_raw_cell_results(args.output_dir, screen)
    screen_complete = all(cell_path(args.output_dir, task).exists() for task in screen)
    if not screen_complete:
        print("Screen is incomplete; resume before selection or confirmation.", flush=True)
        return
    selection = load_or_select(args.output_dir, config)
    generate_report(args.output_dir, config, selection)
    if args.stage in ("screen", "report"):
        return
    if not selection["gate_passed"] and not args.force_confirm and not config.smoke:
        print("No interval passed the damping gate; confirmation was not started.", flush=True)
        return
    confirm = confirmation_tasks(config, float(selection["best_interval_us"]))
    write_inventory(args.output_dir, list({task.key: task for task in screen + confirm}.values()))
    if args.stage in ("smoke", "confirm", "auto"):
        snapshot = run_tasks(
            confirm,
            args.output_dir,
            config,
            workers=min(args.workers, len(confirm)),
            resume=True if args.stage == "auto" else args.resume,
            deadline=deadline,
            start_guard_seconds=guard,
        )
        atomic_json(args.output_dir / "confirmation_status.json", snapshot)
    write_raw_cell_results(args.output_dir, screen + confirm)
    generate_report(args.output_dir, config, selection)


if __name__ == "__main__":
    main()
