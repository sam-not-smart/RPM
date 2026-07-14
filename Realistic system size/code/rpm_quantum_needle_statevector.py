"""Precision-gated matrix-free state-vector reproduction of the quantum needle.

The production workflow is intentionally resumable.  Each (angle, nuclear
sample) calculation is checkpointed independently, and a manifest prevents
incompatible configurations from being mixed.
"""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from rpm_quantum_needle_model import (
    DEFAULT_FIELD_MICROT,
    DEFAULT_LIFETIMES_US,
    DIAGNOSTIC_ANGLES_DEG,
    NUCLEAR_DIMENSION,
    PILOT_ANGLES_DEG,
    SPIN_ORDER,
    STATE_DIMENSION,
    SYSTEM_DIMENSIONS,
    diagnostic_time_grid_us,
    theta_grid,
)
from rpm_statevector_engine import (
    AdaptiveLanczosPropagator,
    build_quantum_needle_hamiltonian,
    singlet_initial_state,
    singlet_population,
)


@dataclass(frozen=True)
class RunConfig:
    field_microT: float = DEFAULT_FIELD_MICROT
    lifetimes_us: tuple[float, ...] = DEFAULT_LIFETIMES_US
    seed: int = 20260713
    sobol_scrambles: int = 8
    bootstrap_replicates: int = 2000
    stochastic_halfwidth_target: float = 2e-4
    propagation_bound_target: float = 5e-5
    pilot_start_m: int = 8
    pilot_start_q: int = 32
    pilot_max_m: int = 256
    pilot_max_q: int = 256
    lanczos_minimum_dimension: int = 24
    lanczos_maximum_dimension: int = 64
    lanczos_dimension_increment: int = 8
    lanczos_embedded_offset: int = 4
    lanczos_safety_factor: float = 2.0
    lanczos_state_error_budget: float = 2.4e-5
    lanczos_initial_step_us: float = 0.05
    lanczos_maximum_step_us: float = 0.25

    def stable_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["lifetimes_us"] = list(self.lifetimes_us)
        result["spin_order"] = list(SPIN_ORDER)
        result["dimensions"] = list(SYSTEM_DIMENSIONS)
        return result

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CellTask:
    theta_deg: float
    sample_id: int
    q_points: int
    include_trace: bool
    config: RunConfig


@dataclass
class CellResult:
    theta_deg: float
    sample_id: int
    q_points: int
    yields: np.ndarray
    propagation_bounds: np.ndarray
    trace_times_us: np.ndarray
    trace_population: np.ndarray
    trace_bounds: np.ndarray
    accepted_steps: int
    rejected_steps: int
    matvecs: int
    maximum_krylov_dimension: int
    minimum_step_us: float
    maximum_step_us: float
    accumulated_state_error: float
    maximum_norm_error: float
    elapsed_seconds: float


@dataclass(frozen=True)
class Uncertainty:
    mean: float
    nuclear_halfwidth: float
    reaction_time_halfwidth: float
    joint_halfwidth: float
    nuclear_sample_std: float


def _seed(base: int, *keys: int) -> int:
    sequence = np.random.SeedSequence([int(base), *(int(key) for key in keys)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def reaction_time_schedule(config: RunConfig, q_points: int) -> np.ndarray:
    if q_points < 1 or q_points & (q_points - 1):
        raise ValueError("The Sobol point count Q must be a positive power of two.")
    exponent = int(math.log2(q_points))
    schedule = np.empty(
        (len(config.lifetimes_us), config.sobol_scrambles, q_points), dtype=float
    )
    for scramble in range(config.sobol_scrambles):
        engine = qmc.Sobol(
            d=1,
            scramble=True,
            seed=_seed(config.seed, 0x50B01, scramble),
        )
        uniform = engine.random_base2(exponent)[:, 0]
        uniform = np.minimum(uniform, np.nextafter(1.0, 0.0))
        for lifetime_index, lifetime in enumerate(config.lifetimes_us):
            schedule[lifetime_index, scramble, :] = -float(lifetime) * np.log1p(-uniform)
    return schedule


def run_cell(task: CellTask) -> CellResult:
    started = time.perf_counter()
    config = task.config
    schedule = reaction_time_schedule(config, task.q_points)
    reaction_times = schedule.reshape(-1)
    trace_times = diagnostic_time_grid_us() if task.include_trace else np.empty(0)
    observation_times = np.unique(np.concatenate(([0.0], reaction_times, trace_times)))

    hamiltonian = build_quantum_needle_hamiltonian(task.theta_deg, config.field_microT)
    rng = np.random.default_rng(_seed(config.seed, 0xA11CE, task.sample_id))
    initial_state = singlet_initial_state(SYSTEM_DIMENSIONS, rng)
    propagator = AdaptiveLanczosPropagator(
        hamiltonian,
        minimum_dimension=config.lanczos_minimum_dimension,
        maximum_dimension=config.lanczos_maximum_dimension,
        dimension_increment=config.lanczos_dimension_increment,
        embedded_offset=config.lanczos_embedded_offset,
        safety_factor=config.lanczos_safety_factor,
        state_error_budget=config.lanczos_state_error_budget,
        initial_step_us=config.lanczos_initial_step_us,
        maximum_step_us=config.lanczos_maximum_step_us,
    )
    propagated = propagator.propagate(
        initial_state,
        observation_times,
        lambda state: singlet_population(state, SYSTEM_DIMENSIONS),
    )
    reaction_indices = np.searchsorted(observation_times, reaction_times)
    populations = propagated.observable_values[reaction_indices].reshape(schedule.shape)
    bounds = propagated.observable_error_bounds[reaction_indices].reshape(schedule.shape)
    yields = populations.mean(axis=2)
    propagation_bounds = bounds.mean(axis=2)
    if task.include_trace:
        trace_indices = np.searchsorted(observation_times, trace_times)
        trace_population = propagated.observable_values[trace_indices]
        trace_bounds = propagated.observable_error_bounds[trace_indices]
    else:
        trace_population = np.empty(0)
        trace_bounds = np.empty(0)
    stats = propagated.stats
    return CellResult(
        theta_deg=task.theta_deg,
        sample_id=task.sample_id,
        q_points=task.q_points,
        yields=yields,
        propagation_bounds=propagation_bounds,
        trace_times_us=trace_times,
        trace_population=trace_population,
        trace_bounds=trace_bounds,
        accepted_steps=stats.accepted_steps,
        rejected_steps=stats.rejected_steps,
        matvecs=stats.matvecs,
        maximum_krylov_dimension=stats.maximum_krylov_dimension,
        minimum_step_us=stats.minimum_step_us,
        maximum_step_us=stats.maximum_step_us,
        accumulated_state_error=stats.accumulated_state_error,
        maximum_norm_error=stats.maximum_norm_error,
        elapsed_seconds=time.perf_counter() - started,
    )


def _angle_token(theta_deg: float) -> str:
    return f"{theta_deg:07.3f}".replace("-", "m").replace(".", "p")


def checkpoint_path(output_dir: Path, theta_deg: float, sample_id: int, q_points: int) -> Path:
    return (
        output_dir
        / "checkpoints"
        / f"theta_{_angle_token(theta_deg)}_sample_{sample_id:04d}_q_{q_points:04d}.npz"
    )


def save_cell(result: CellResult, output_dir: Path, configuration_hash: str) -> Path:
    path = checkpoint_path(output_dir, result.theta_deg, result.sample_id, result.q_points)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            configuration_hash=np.array(configuration_hash),
            theta_deg=np.array(result.theta_deg),
            sample_id=np.array(result.sample_id),
            q_points=np.array(result.q_points),
            yields=result.yields,
            propagation_bounds=result.propagation_bounds,
            trace_times_us=result.trace_times_us,
            trace_population=result.trace_population,
            trace_bounds=result.trace_bounds,
            accepted_steps=np.array(result.accepted_steps),
            rejected_steps=np.array(result.rejected_steps),
            matvecs=np.array(result.matvecs),
            maximum_krylov_dimension=np.array(result.maximum_krylov_dimension),
            minimum_step_us=np.array(result.minimum_step_us),
            maximum_step_us=np.array(result.maximum_step_us),
            accumulated_state_error=np.array(result.accumulated_state_error),
            maximum_norm_error=np.array(result.maximum_norm_error),
            elapsed_seconds=np.array(result.elapsed_seconds),
        )
    os.replace(temporary, path)
    return path


def load_cell(
    output_dir: Path,
    theta_deg: float,
    sample_id: int,
    q_points: int,
    configuration_hash: str,
) -> CellResult:
    path = checkpoint_path(output_dir, theta_deg, sample_id, q_points)
    with np.load(path, allow_pickle=False) as data:
        stored_hash = str(data["configuration_hash"].item())
        if stored_hash != configuration_hash:
            raise RuntimeError(f"Checkpoint configuration mismatch: {path}")
        return CellResult(
            theta_deg=float(data["theta_deg"]),
            sample_id=int(data["sample_id"]),
            q_points=int(data["q_points"]),
            yields=data["yields"].copy(),
            propagation_bounds=data["propagation_bounds"].copy(),
            trace_times_us=data["trace_times_us"].copy(),
            trace_population=data["trace_population"].copy(),
            trace_bounds=data["trace_bounds"].copy(),
            accepted_steps=int(data["accepted_steps"]),
            rejected_steps=int(data["rejected_steps"]),
            matvecs=int(data["matvecs"]),
            maximum_krylov_dimension=int(data["maximum_krylov_dimension"]),
            minimum_step_us=float(data["minimum_step_us"]),
            maximum_step_us=float(data["maximum_step_us"]),
            accumulated_state_error=float(data["accumulated_state_error"]),
            maximum_norm_error=float(data["maximum_norm_error"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
        )


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def ensure_manifest(
    output_dir: Path,
    config: RunConfig,
    *,
    resume: bool,
    allow_existing: bool = False,
) -> dict[str, object]:
    path = output_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("configuration_hash") != config.configuration_hash:
            raise RuntimeError("Existing output manifest uses an incompatible configuration.")
        if not resume and not allow_existing:
            raise RuntimeError("Output already contains a run; pass --resume to continue it.")
        return manifest
    manifest: dict[str, object] = {
        "schema_version": 1,
        "configuration_hash": config.configuration_hash,
        "configuration": config.stable_dict(),
        "status": "created",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": git_commit(),
        "state_dimension": STATE_DIMENSION,
        "nuclear_dimension": NUCLEAR_DIMENSION,
    }
    write_json_atomic(path, manifest)
    return manifest


def run_tasks(
    angles_deg: Iterable[float],
    sample_count: int,
    q_points: int,
    config: RunConfig,
    output_dir: Path,
    workers: int,
) -> None:
    angles_deg = tuple(float(theta) for theta in angles_deg)
    tasks: list[CellTask] = []
    for theta in angles_deg:
        for sample_id in range(sample_count):
            path = checkpoint_path(output_dir, float(theta), sample_id, q_points)
            if not path.exists():
                tasks.append(
                    CellTask(
                        theta_deg=float(theta),
                        sample_id=sample_id,
                        q_points=q_points,
                        include_trace=any(
                            np.isclose(theta, diagnostic) for diagnostic in DIAGNOSTIC_ANGLES_DEG
                        ),
                        config=config,
                    )
                )
    if not tasks:
        print(f"All {len(angles_deg) * sample_count} requested cells are checkpointed.")
        return
    print(f"Running {len(tasks)} cells with {workers} worker(s), M={sample_count}, Q={q_points}")
    completed = 0
    if workers == 1:
        for task in tasks:
            result = run_cell(task)
            save_cell(result, output_dir, config.configuration_hash)
            completed += 1
            print(
                f"[{completed}/{len(tasks)}] theta={task.theta_deg:g}, sample={task.sample_id}, "
                f"{result.elapsed_seconds:.1f}s, matvecs={result.matvecs}",
                flush=True,
            )
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_cell, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            save_cell(result, output_dir, config.configuration_hash)
            completed += 1
            print(
                f"[{completed}/{len(tasks)}] theta={task.theta_deg:g}, sample={task.sample_id}, "
                f"{result.elapsed_seconds:.1f}s, matvecs={result.matvecs}",
                flush=True,
            )


def load_cells(
    output_dir: Path,
    theta_deg: float,
    sample_count: int,
    q_points: int,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray, list[CellResult]]:
    cells = [
        load_cell(output_dir, theta_deg, sample_id, q_points, config.configuration_hash)
        for sample_id in range(sample_count)
    ]
    yields = np.stack([cell.yields for cell in cells], axis=0)
    bounds = np.stack([cell.propagation_bounds for cell in cells], axis=0)
    return yields, bounds, cells


def _halfwidth(samples: np.ndarray, center: float) -> float:
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(max(center - lower, upper - center))


def uncertainty_components(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> Uncertainty:
    """Two-level bootstrap for an M nuclear-state by S scramble table."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("Uncertainty analysis needs at least two rows and columns.")
    nuclear_count, scramble_count = values.shape
    center = float(values.mean())
    rng = np.random.default_rng(seed)
    nuclear_indices = rng.integers(0, nuclear_count, size=(replicates, nuclear_count))
    scramble_indices = rng.integers(0, scramble_count, size=(replicates, scramble_count))
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    nuclear_samples = row_means[nuclear_indices].mean(axis=1)
    time_samples = column_means[scramble_indices].mean(axis=1)
    joint_samples = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        joint_samples[replicate] = values[
            nuclear_indices[replicate][:, None], scramble_indices[replicate][None, :]
        ].mean()
    return Uncertainty(
        mean=center,
        nuclear_halfwidth=_halfwidth(nuclear_samples, center),
        reaction_time_halfwidth=_halfwidth(time_samples, center),
        joint_halfwidth=_halfwidth(joint_samples, center),
        nuclear_sample_std=float(row_means.std(ddof=1)),
    )


def pilot_metrics(
    output_dir: Path,
    sample_count: int,
    q_points: int,
    config: RunConfig,
) -> dict[str, float]:
    angle_values: dict[float, np.ndarray] = {}
    maximum_nuclear = 0.0
    maximum_time = 0.0
    maximum_joint = 0.0
    maximum_propagation = 0.0
    for angle_index, theta in enumerate(PILOT_ANGLES_DEG):
        yields, bounds, _ = load_cells(output_dir, theta, sample_count, q_points, config)
        angle_values[theta] = yields
        maximum_propagation = max(maximum_propagation, float(bounds.mean(axis=(0, 1)).max()))
        for lifetime_index in range(len(config.lifetimes_us)):
            uncertainty = uncertainty_components(
                yields[:, :, lifetime_index],
                config.bootstrap_replicates,
                _seed(config.seed, 0xB007, angle_index, lifetime_index, sample_count, q_points),
            )
            maximum_nuclear = max(maximum_nuclear, uncertainty.nuclear_halfwidth)
            maximum_time = max(maximum_time, uncertainty.reaction_time_halfwidth)
            maximum_joint = max(maximum_joint, uncertainty.joint_halfwidth)
    for contrast_index, (left, right) in enumerate(((90.0, 80.0), (90.0, 100.0), (0.0, 180.0))):
        difference = angle_values[left] - angle_values[right]
        for lifetime_index in range(len(config.lifetimes_us)):
            uncertainty = uncertainty_components(
                difference[:, :, lifetime_index],
                config.bootstrap_replicates,
                _seed(config.seed, 0xC017, contrast_index, lifetime_index, sample_count, q_points),
            )
            maximum_joint = max(maximum_joint, uncertainty.joint_halfwidth)
    return {
        "maximum_nuclear_halfwidth": maximum_nuclear,
        "maximum_reaction_time_halfwidth": maximum_time,
        "maximum_joint_halfwidth": maximum_joint,
        "maximum_propagation_bound": maximum_propagation,
    }


def run_pilot(
    output_dir: Path,
    config: RunConfig,
    workers: int,
    manifest: dict[str, object],
) -> tuple[bool, int, int]:
    sample_count = int(manifest.get("selected_M", config.pilot_start_m))
    q_points = int(manifest.get("selected_Q", config.pilot_start_q))
    history = list(manifest.get("pilot_history", []))
    while True:
        run_tasks(PILOT_ANGLES_DEG, sample_count, q_points, config, output_dir, workers)
        metrics = pilot_metrics(output_dir, sample_count, q_points, config)
        record = {"M": sample_count, "Q": q_points, **metrics}
        history.append(record)
        print("Pilot precision:", json.dumps(record, sort_keys=True))
        stochastic_ok = metrics["maximum_joint_halfwidth"] <= config.stochastic_halfwidth_target
        propagation_ok = metrics["maximum_propagation_bound"] <= config.propagation_bound_target
        manifest.update(
            {
                "status": "pilot_passed" if stochastic_ok and propagation_ok else "pilot_running",
                "selected_M": sample_count,
                "selected_Q": q_points,
                "pilot_history": history,
            }
        )
        write_json_atomic(output_dir / "manifest.json", manifest)
        if stochastic_ok and propagation_ok:
            return True, sample_count, q_points
        can_grow_m = sample_count < config.pilot_max_m
        can_grow_q = q_points < config.pilot_max_q
        if not can_grow_m and not can_grow_q:
            manifest["status"] = "uncertified"
            manifest["uncertified_reason"] = "Pilot precision targets were not met at M/Q caps."
            write_json_atomic(output_dir / "manifest.json", manifest)
            return False, sample_count, q_points
        nuclear_dominates = (
            metrics["maximum_nuclear_halfwidth"] >= metrics["maximum_reaction_time_halfwidth"]
        )
        if (nuclear_dominates and can_grow_m) or not can_grow_q:
            sample_count = min(config.pilot_max_m, 2 * sample_count)
        else:
            q_points = min(config.pilot_max_q, 2 * q_points)


def write_rows(path: Path, rows: Iterable[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_reference(path: Path | None) -> dict[tuple[float, float], float]:
    if path is None or not path.exists():
        return {}
    result: dict[tuple[float, float], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            theta = float(row["theta_deg"])
            for lifetime in DEFAULT_LIFETIMES_US:
                result[(theta, lifetime)] = float(row[f"phi_s_tau_{lifetime:g}_us"])
    return result


def create_report(
    output_dir: Path,
    angles_deg: np.ndarray,
    sample_count: int,
    q_points: int,
    config: RunConfig,
    reference_csv: Path | None,
) -> dict[str, object]:
    reference = read_reference(reference_csv)
    summary_rows: list[dict[str, object]] = []
    all_cells: dict[float, tuple[np.ndarray, np.ndarray, list[CellResult]]] = {}
    for angle_index, theta in enumerate(angles_deg):
        yields, bounds, cells = load_cells(
            output_dir, float(theta), sample_count, q_points, config
        )
        all_cells[float(theta)] = yields, bounds, cells
        for lifetime_index, lifetime in enumerate(config.lifetimes_us):
            uncertainty = uncertainty_components(
                yields[:, :, lifetime_index],
                config.bootstrap_replicates,
                _seed(config.seed, 0x5A11, angle_index, lifetime_index),
            )
            propagation_bound = float(bounds[:, :, lifetime_index].mean())
            exact = reference.get((float(theta), float(lifetime)))
            total = uncertainty.joint_halfwidth + propagation_bound
            summary_rows.append(
                {
                    "theta_deg": float(theta),
                    "lifetime_us": float(lifetime),
                    "phi_s": uncertainty.mean,
                    "nuclear_sample_std": uncertainty.nuclear_sample_std,
                    "nuclear_halfwidth_95": uncertainty.nuclear_halfwidth,
                    "reaction_time_halfwidth_95": uncertainty.reaction_time_halfwidth,
                    "joint_stochastic_halfwidth_95": uncertainty.joint_halfwidth,
                    "propagation_bound": propagation_bound,
                    "total_halfwidth_95": total,
                    "reference_phi_s": "" if exact is None else exact,
                    "reference_difference": "" if exact is None else uncertainty.mean - exact,
                    "reference_within_interval": "" if exact is None else abs(uncertainty.mean - exact) <= total,
                    "M": sample_count,
                    "S": config.sobol_scrambles,
                    "Q": q_points,
                }
            )
    summary_columns = list(summary_rows[0])
    write_rows(output_dir / "quantum_needle_statevector_summary.csv", summary_rows, summary_columns)

    cell_columns = [
        "theta_deg", "sample_id", "scramble_id", "lifetime_us", "phi_s",
        "propagation_bound", "Q", "matvecs", "accepted_steps", "rejected_steps",
        "maximum_krylov_dimension", "maximum_norm_error", "elapsed_seconds",
    ]
    cell_path = output_dir / "quantum_needle_statevector_cells.csv"
    temporary = cell_path.with_suffix(cell_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_columns)
        writer.writeheader()
        for theta in angles_deg:
            yields, bounds, cells = all_cells[float(theta)]
            for sample_id, cell in enumerate(cells):
                for scramble in range(config.sobol_scrambles):
                    for lifetime_index, lifetime in enumerate(config.lifetimes_us):
                        writer.writerow(
                            {
                                "theta_deg": float(theta),
                                "sample_id": sample_id,
                                "scramble_id": scramble,
                                "lifetime_us": lifetime,
                                "phi_s": yields[sample_id, scramble, lifetime_index],
                                "propagation_bound": bounds[sample_id, scramble, lifetime_index],
                                "Q": q_points,
                                "matvecs": cell.matvecs,
                                "accepted_steps": cell.accepted_steps,
                                "rejected_steps": cell.rejected_steps,
                                "maximum_krylov_dimension": cell.maximum_krylov_dimension,
                                "maximum_norm_error": cell.maximum_norm_error,
                                "elapsed_seconds": cell.elapsed_seconds,
                            }
                        )
    os.replace(temporary, cell_path)
    save_diagnostic_traces(output_dir, all_cells, sample_count, config)
    save_curve_summary(output_dir, all_cells, angles_deg, sample_count, config)
    plot_report(output_dir, summary_rows, config)
    certification = certify_report(summary_rows, config)
    write_json_atomic(output_dir / "certification.json", certification)
    write_json_atomic(
        output_dir / "quantum_needle_statevector_metadata.json",
        {
            "paper": "Hiscock et al., PNAS 113, 4634-4639 (2016)",
            "doi": "10.1073/pnas.1600341113",
            "configuration_hash": config.configuration_hash,
            "configuration": config.stable_dict(),
            "M": sample_count,
            "S": config.sobol_scrambles,
            "Q": q_points,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "git_commit": git_commit(),
            "certification": certification,
        },
    )
    return certification


def save_diagnostic_traces(
    output_dir: Path,
    all_cells: dict[float, tuple[np.ndarray, np.ndarray, list[CellResult]]],
    sample_count: int,
    config: RunConfig,
) -> None:
    rows: list[dict[str, object]] = []
    for angle_index, theta in enumerate(DIAGNOSTIC_ANGLES_DEG):
        if theta not in all_cells:
            continue
        cells = all_cells[theta][2]
        traces = np.stack([cell.trace_population for cell in cells])
        bounds = np.stack([cell.trace_bounds for cell in cells])
        times = cells[0].trace_times_us
        rng = np.random.default_rng(_seed(config.seed, 0x7ACE, angle_index))
        indices = rng.integers(0, sample_count, size=(config.bootstrap_replicates, sample_count))
        bootstrap = traces[indices].mean(axis=1)
        mean = traces.mean(axis=0)
        lower, upper = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        halfwidth = np.maximum(mean - lower, upper - mean)
        for time_index, time_us in enumerate(times):
            rows.append(
                {
                    "theta_deg": theta,
                    "time_us": time_us,
                    "mean_p_s": mean[time_index],
                    "nuclear_halfwidth_95": halfwidth[time_index],
                    "propagation_bound": bounds[:, time_index].mean(),
                    "M": sample_count,
                }
            )
    if rows:
        write_rows(output_dir / "diagnostic_traces.csv", rows, list(rows[0]))


def save_curve_summary(
    output_dir: Path,
    all_cells: dict[float, tuple[np.ndarray, np.ndarray, list[CellResult]]],
    angles_deg: np.ndarray,
    sample_count: int,
    config: RunConfig,
) -> None:
    """Bootstrap extrema after resampling each complete paired angular curve."""

    rows: list[dict[str, object]] = []
    angle_array = np.asarray(angles_deg, dtype=float)
    for lifetime_index, lifetime in enumerate(config.lifetimes_us):
        curves = np.stack(
            [all_cells[float(theta)][0][:, :, lifetime_index] for theta in angle_array], axis=0
        )
        mean_curve = curves.mean(axis=(1, 2))
        rng = np.random.default_rng(_seed(config.seed, 0xC0A7E, lifetime_index))
        anisotropies = np.empty(config.bootstrap_replicates)
        minimum_angles = np.empty(config.bootstrap_replicates)
        maximum_angles = np.empty(config.bootstrap_replicates)
        for replicate in range(config.bootstrap_replicates):
            nuclear_indices = rng.integers(0, sample_count, size=sample_count)
            scramble_indices = rng.integers(
                0, config.sobol_scrambles, size=config.sobol_scrambles
            )
            curve = curves[:, nuclear_indices][:, :, scramble_indices].mean(axis=(1, 2))
            minimum = int(np.argmin(curve))
            maximum = int(np.argmax(curve))
            anisotropies[replicate] = curve[maximum] - curve[minimum]
            minimum_angles[replicate] = angle_array[minimum]
            maximum_angles[replicate] = angle_array[maximum]
        minimum = int(np.argmin(mean_curve))
        maximum = int(np.argmax(mean_curve))
        anisotropy = float(mean_curve[maximum] - mean_curve[minimum])
        anisotropy_low, anisotropy_high = np.percentile(anisotropies, [2.5, 97.5])
        min_angle_low, min_angle_high = np.percentile(minimum_angles, [2.5, 97.5])
        max_angle_low, max_angle_high = np.percentile(maximum_angles, [2.5, 97.5])
        rows.append(
            {
                "lifetime_us": lifetime,
                "minimum_phi_s": mean_curve[minimum],
                "minimum_angle_deg": angle_array[minimum],
                "minimum_angle_ci_low_deg": min_angle_low,
                "minimum_angle_ci_high_deg": min_angle_high,
                "maximum_phi_s": mean_curve[maximum],
                "maximum_angle_deg": angle_array[maximum],
                "maximum_angle_ci_low_deg": max_angle_low,
                "maximum_angle_ci_high_deg": max_angle_high,
                "absolute_anisotropy": anisotropy,
                "anisotropy_ci_low": anisotropy_low,
                "anisotropy_ci_high": anisotropy_high,
            }
        )
    write_rows(output_dir / "curve_summary.csv", rows, list(rows[0]))


def plot_report(output_dir: Path, rows: list[dict[str, object]], config: RunConfig) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 5.6))
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(config.lifetimes_us)))
    for lifetime, color in zip(config.lifetimes_us, colors):
        selected = [row for row in rows if float(row["lifetime_us"]) == lifetime]
        theta = np.array([row["theta_deg"] for row in selected], dtype=float)
        mean = np.array([row["phi_s"] for row in selected], dtype=float)
        error = np.array([row["total_halfwidth_95"] for row in selected], dtype=float)
        axis.plot(theta, mean, color=color, linewidth=1.25, label=f"{lifetime:g} us")
        axis.fill_between(theta, mean - error, mean + error, color=color, alpha=0.13)
    axis.set(xlabel="field angle theta (degrees)", ylabel="singlet yield Phi_S", xlim=(0, 180))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Matrix-free state-vector quantum-needle reproduction")
    fig.tight_layout()
    fig.savefig(output_dir / "quantum_needle_statevector_fig1b.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharex=True)
    labels = (
        ("nuclear_halfwidth_95", "nuclear sampling"),
        ("reaction_time_halfwidth_95", "reaction-time sampling"),
        ("propagation_bound", "Lanczos propagation"),
    )
    for axis, (column, title) in zip(axes, labels):
        for lifetime, color in zip(config.lifetimes_us, colors):
            selected = [row for row in rows if float(row["lifetime_us"]) == lifetime]
            axis.plot(
                [row["theta_deg"] for row in selected],
                [row[column] for row in selected],
                color=color,
                linewidth=1.0,
            )
        axis.set_title(title)
        axis.set_xlabel("theta (degrees)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("95% half-width or bound")
    fig.tight_layout()
    fig.savefig(output_dir / "uncertainty_components.png", dpi=200)
    plt.close(fig)


def certify_report(rows: list[dict[str, object]], config: RunConfig) -> dict[str, object]:
    stochastic_ok = all(
        float(row["joint_stochastic_halfwidth_95"]) <= config.stochastic_halfwidth_target
        for row in rows
    )
    propagation_ok = all(
        float(row["propagation_bound"]) <= config.propagation_bound_target for row in rows
    )
    comparable = [row for row in rows if row["reference_phi_s"] != ""]
    result: dict[str, object] = {
        "stochastic_precision_passed": stochastic_ok,
        "propagation_precision_passed": propagation_ok,
        "reference_points": len(comparable),
    }
    if comparable:
        differences = np.array([float(row["reference_difference"]) for row in comparable])
        halfwidths = np.array([float(row["total_halfwidth_95"]) for row in comparable])
        coverage = float(np.mean(np.abs(differences) <= halfwidths))
        standardized = np.abs(differences) / np.maximum(halfwidths, np.finfo(float).tiny)
        rms = float(np.sqrt(np.mean(differences * differences)))
        statevector_100 = [row for row in comparable if float(row["lifetime_us"]) == 100.0]
        needle_row = min(statevector_100, key=lambda row: float(row["phi_s"]))
        reference_row = min(statevector_100, key=lambda row: float(row["reference_phi_s"]))
        needle_error = abs(float(needle_row["theta_deg"]) - float(reference_row["theta_deg"]))
        result.update(
            {
                "pointwise_interval_coverage": coverage,
                "maximum_standardized_difference": float(standardized.max()),
                "curve_rmse": rms,
                "needle_angle_error_deg": needle_error,
                "reference_acceptance_passed": bool(
                    coverage >= 0.90
                    and standardized.max() <= 3.0
                    and rms <= 2e-4
                    and needle_error <= 0.25
                ),
            }
        )
    else:
        result["reference_acceptance_passed"] = False
        result["reference_note"] = "No exact-reference CSV was supplied."
    result["certified"] = bool(
        stochastic_ok and propagation_ok and result["reference_acceptance_passed"]
    )
    return result


def _benchmark_task(_: int) -> float:
    started = time.perf_counter()
    hamiltonian = build_quantum_needle_hamiltonian(90.0)
    rng = np.random.default_rng(12345)
    vector = singlet_initial_state(SYSTEM_DIMENSIONS, rng)
    hamiltonian.matvec(vector)
    hamiltonian.matvec(vector)
    return time.perf_counter() - started


def benchmark_workers(output_dir: Path) -> int:
    candidates = [value for value in (1, 2, 4, 8, 16, 32) if value <= (os.cpu_count() or 1)]
    measurements: list[dict[str, float]] = []
    for workers in candidates:
        started = time.perf_counter()
        if workers == 1:
            _benchmark_task(0)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_benchmark_task, range(workers)))
        elapsed = time.perf_counter() - started
        throughput = workers / elapsed
        measurements.append({"workers": workers, "elapsed_seconds": elapsed, "throughput": throughput})
        print(f"worker benchmark: {workers} -> {throughput:.3f} tasks/s")
    best = max(item["throughput"] for item in measurements)
    selected = int(next(item["workers"] for item in measurements if item["throughput"] >= 0.95 * best))
    write_json_atomic(
        output_dir / "worker_benchmark.json",
        {"measurements": measurements, "selected_workers": selected},
    )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("auto", "pilot", "full", "report"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "quantum_needle_statevector")
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--workers", type=int, default=0, help="Zero benchmarks 1-32 workers and selects automatically.")
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("outputs") / "quantum_needle_2016" / "quantum_needle_fig1b.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(seed=args.seed)
    output_dir = args.output_dir
    manifest = ensure_manifest(
        output_dir,
        config,
        resume=args.resume,
        allow_existing=args.stage == "report",
    )
    workers = args.workers
    if workers < 0:
        raise ValueError("--workers cannot be negative.")
    if workers == 0 and args.stage != "report":
        workers = benchmark_workers(output_dir)
    workers = max(1, workers)

    certified = manifest.get("status") in {
        "pilot_passed", "full_running", "full_complete", "full_complete_uncertified", "certified"
    }
    sample_count = int(manifest.get("selected_M", config.pilot_start_m))
    q_points = int(manifest.get("selected_Q", config.pilot_start_q))
    if args.stage in {"auto", "pilot"} and not certified:
        certified, sample_count, q_points = run_pilot(
            output_dir, config, workers, manifest
        )
        if not certified:
            print("Pilot stopped uncertified; the full sweep was not launched.")
            return
    if args.stage == "pilot":
        print(f"Pilot certified M={sample_count}, Q={q_points}.")
        return
    if args.stage in {"auto", "full"}:
        if not certified:
            raise RuntimeError("A certified pilot is required before the full sweep.")
        angles = theta_grid()
        manifest.update({"status": "full_running", "selected_M": sample_count, "selected_Q": q_points})
        write_json_atomic(output_dir / "manifest.json", manifest)
        run_tasks(angles, sample_count, q_points, config, output_dir, workers)
        manifest["status"] = "full_complete"
        write_json_atomic(output_dir / "manifest.json", manifest)
    else:
        angles = theta_grid()
        if manifest.get("status") not in {"full_complete", "full_complete_uncertified", "certified"}:
            raise RuntimeError("The full checkpoint set is not marked complete.")
    certification = create_report(
        output_dir,
        angles,
        sample_count,
        q_points,
        config,
        args.reference_csv,
    )
    manifest["status"] = "certified" if certification["certified"] else "full_complete_uncertified"
    manifest["certification"] = certification
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(json.dumps(certification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
