"""Full-state dipolar/exchange quantum-needle sweeps with optional quadrupoles.

Every trajectory evolves the complete 14-nucleus radical-pair state with the
matrix-free Hamiltonian. Use ``--quadrupole-scale 0`` for the historical sweeps.
"""

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
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.stats

from rpm_quantum_needle_model import (
    GAMMA_E_MHZ_PER_MT,
    NUCLEAR_DIMENSION,
    SPIN_ORDER,
    STATE_DIMENSION,
    SYSTEM_DIMENSIONS,
    add_quadrupole_argument,
    nominal_quadrupole_metadata,
)
from rpm_statevector_engine import (
    AdaptiveLanczosPropagator,
    SingletPopulationObservable,
    build_quantum_needle_hamiltonian,
    singlet_initial_state,
    singlet_population,
)


SCHEMA_VERSION = 1
QUADRUPOLE_SCALE = 0.0
SCREEN_ANGLES_DEG = tuple(float(value) for value in range(0, 181, 30))
REFINED_ANGLES_DEG = tuple(float(value) for value in range(0, 181, 15))
DIPOLE_PAPER_MAGNITUDES_UT = (
    0.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    300.0,
    347.875,
    400.0,
    500.0,
    600.0,
)
LOW_EXCHANGE_UT = (-10.0, -5.0, 0.0, 5.0, 10.0)
NEGATIVE_RATIOS = (-8.0, -6.0, -5.0, -4.5, -4.25, -4.0, -3.75, -3.5, -3.0, -2.5, -2.0)
POSITIVE_RATIOS = (4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 12.0, 16.0)
R2_D_PAPER_UT = -2783.0 / (2.0**3)
EFIMOVA_HORE_NEGATIVE_RATIOS = (-4.0, -3.5, -3.25, -3.0, -2.75, -2.5, -2.0)
EFIMOVA_HORE_POSITIVE_RATIOS = (4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0)
EFIMOVA_HORE_AXIS = np.array([1.0, 0.0, 0.0])
PUBLISHED_DIPOLE_TENSOR_MHZ = np.array(
    [
        [1.09, -12.12, 4.91],
        [-12.12, -7.80, 7.02],
        [4.91, 7.02, 6.71],
    ],
    dtype=float,
)


def paper_to_code_dipolar_uT(d_paper_uT: float) -> float:
    """Convert Efimova-Hore D to this code's unique axial eigenvalue."""

    return 4.0 * float(d_paper_uT) / 3.0


def equivalent_field_uT_to_MHz(value_uT: float) -> float:
    return float(value_uT) * GAMMA_E_MHZ_PER_MT / 1000.0


def unique_principal_value(tensor: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(tensor, dtype=float))
    return float(eigenvalues[np.argmax(np.abs(eigenvalues))])


def scaled_published_dipolar_tensor_MHz(d_parallel_MHz: float) -> np.ndarray:
    """Scale the full published tensor while preserving axes and rhombicity."""

    target = float(d_parallel_MHz)
    if target == 0.0:
        return np.zeros((3, 3), dtype=float)
    source_parallel = unique_principal_value(PUBLISHED_DIPOLE_TENSOR_MHZ)
    return PUBLISHED_DIPOLE_TENSOR_MHZ * (target / source_parallel)


def _number_token(value: float) -> str:
    return f"{float(value):+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class CouplingCondition:
    condition_id: str
    family: str
    role: str
    d_paper_uT: float
    d_parallel_uT: float
    exchange_J_uT: float
    ratio_D_over_J: float | None

    @property
    def d_parallel_MHz(self) -> float:
        return equivalent_field_uT_to_MHz(self.d_parallel_uT)

    @property
    def exchange_J_MHz(self) -> float:
        return equivalent_field_uT_to_MHz(self.exchange_J_uT)

    @property
    def dipolar_tensor_MHz(self) -> np.ndarray:
        return scaled_published_dipolar_tensor_MHz(self.d_parallel_MHz)

    def stable_dict(self, quadrupole_scale: float = QUADRUPOLE_SCALE) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "d_parallel_MHz": self.d_parallel_MHz,
                "exchange_J_MHz": self.exchange_J_MHz,
                "dipolar_tensor_MHz": self.dipolar_tensor_MHz.tolist(),
                "dipolar_principal_values_MHz": np.linalg.eigvalsh(
                    self.dipolar_tensor_MHz
                ).tolist(),
                "quadrupole_scale": quadrupole_scale,
            }
        )
        return result

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class EfimovaHoreCondition:
    """Electron couplings expressed literally in the paper convention."""

    condition_id: str
    family: str
    role: str
    d_paper_uT: float
    exchange_J_uT: float
    ratio_D_over_J: float | None
    dipolar_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @property
    def d_paper_MHz(self) -> float:
        return equivalent_field_uT_to_MHz(self.d_paper_uT)

    @property
    def exchange_J_MHz(self) -> float:
        return equivalent_field_uT_to_MHz(self.exchange_J_uT)

    def stable_dict(self, quadrupole_scale: float = QUADRUPOLE_SCALE) -> dict[str, object]:
        return {
            **asdict(self),
            "convention": "Efimova-Hore Eqs. 9-10",
            "d_paper_MHz": self.d_paper_MHz,
            "exchange_J_MHz": self.exchange_J_MHz,
            "quadrupole_scale": quadrupole_scale,
        }

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def efimova_hore_uncoupled_control() -> EfimovaHoreCondition:
    return EfimovaHoreCondition(
        condition_id="eh_D0_J0_control",
        family="efimova_hore",
        role="uncoupled_control",
        d_paper_uT=0.0,
        exchange_J_uT=0.0,
        ratio_D_over_J=None,
    )


def efimova_hore_dipole_only_control() -> EfimovaHoreCondition:
    return EfimovaHoreCondition(
        condition_id="eh_Dpaper_m347p875uT_J0_control",
        family="efimova_hore",
        role="dipole_only_control",
        d_paper_uT=R2_D_PAPER_UT,
        exchange_J_uT=0.0,
        ratio_D_over_J=None,
    )


def efimova_hore_ratio_condition(ratio: float) -> EfimovaHoreCondition:
    parsed = float(ratio)
    if parsed == 0.0:
        raise ValueError("Efimova-Hore D/J ratio cannot be zero.")
    return EfimovaHoreCondition(
        condition_id=f"eh_R_{_number_token(parsed)}",
        family="efimova_hore",
        role="paper_cancellation_scan",
        d_paper_uT=R2_D_PAPER_UT,
        exchange_J_uT=R2_D_PAPER_UT / parsed,
        ratio_D_over_J=parsed,
    )


def efimova_hore_screen_conditions() -> tuple[EfimovaHoreCondition, ...]:
    ratios = tuple(
        efimova_hore_ratio_condition(value)
        for value in EFIMOVA_HORE_NEGATIVE_RATIOS
        + EFIMOVA_HORE_POSITIVE_RATIOS
    )
    return (
        efimova_hore_uncoupled_control(),
        efimova_hore_dipole_only_control(),
        *ratios,
    )


def dipole_condition(magnitude_uT: float) -> CouplingCondition:
    d_paper = -abs(float(magnitude_uT))
    d_parallel = paper_to_code_dipolar_uT(d_paper)
    return CouplingCondition(
        condition_id=f"dipole_Dpaper_{_number_token(d_paper)}uT",
        family="dipole",
        role="dipole_control" if magnitude_uT == 0.0 else "dipole_scan",
        d_paper_uT=d_paper,
        d_parallel_uT=d_parallel,
        exchange_J_uT=0.0,
        ratio_D_over_J=None,
    )


def low_exchange_condition(exchange_J_uT: float) -> CouplingCondition:
    d_parallel = paper_to_code_dipolar_uT(R2_D_PAPER_UT)
    return CouplingCondition(
        condition_id=f"exchange_J_{_number_token(exchange_J_uT)}uT",
        family="ratio",
        role="fixed_D_low_exchange_control",
        d_paper_uT=R2_D_PAPER_UT,
        d_parallel_uT=d_parallel,
        exchange_J_uT=float(exchange_J_uT),
        ratio_D_over_J=None if exchange_J_uT == 0.0 else d_parallel / exchange_J_uT,
    )


def ratio_condition(ratio: float) -> CouplingCondition:
    d_parallel = paper_to_code_dipolar_uT(R2_D_PAPER_UT)
    exchange = d_parallel / float(ratio)
    return CouplingCondition(
        condition_id=f"ratio_R_{_number_token(ratio)}",
        family="ratio",
        role="mechanistic_cancellation_scan",
        d_paper_uT=R2_D_PAPER_UT,
        d_parallel_uT=d_parallel,
        exchange_J_uT=exchange,
        ratio_D_over_J=float(ratio),
    )


def dipole_screen_conditions() -> tuple[CouplingCondition, ...]:
    return tuple(dipole_condition(value) for value in DIPOLE_PAPER_MAGNITUDES_UT)


def ratio_screen_conditions() -> tuple[CouplingCondition, ...]:
    controls = tuple(low_exchange_condition(value) for value in LOW_EXCHANGE_UT)
    ratios = tuple(ratio_condition(value) for value in NEGATIVE_RATIOS + POSITIVE_RATIOS)
    return controls + ratios


def all_conditions() -> tuple[CouplingCondition, ...]:
    return dipole_screen_conditions() + ratio_screen_conditions()


@dataclass(frozen=True)
class SweepConfig:
    seed: int = 20260715
    field_microT: float = 50.0
    lifetime_us: float = 2.0
    survival_tail_bound: float = 1e-4
    integration_step_us: float = 0.00125
    coarse_integration_step_us: float = 0.0025
    state_error_budget: float = 2.4e-5
    lanczos_minimum_dimension: int = 24
    lanczos_maximum_dimension: int = 64
    lanczos_dimension_increment: int = 8
    lanczos_embedded_offset: int = 4
    lanczos_safety_factor: float = 2.0
    lanczos_initial_step_us: float = 0.05
    lanczos_maximum_step_us: float = 0.25
    quadrupole_scale: float = QUADRUPOLE_SCALE

    @property
    def maximum_time_us(self) -> float:
        return -self.lifetime_us * math.log(self.survival_tail_bound)

    @property
    def time_grid_us(self) -> np.ndarray:
        count = int(math.floor(self.maximum_time_us / self.integration_step_us))
        values = np.arange(count + 1, dtype=float) * self.integration_step_us
        if values[-1] < self.maximum_time_us - 1e-13:
            values = np.append(values, self.maximum_time_us)
        return values

    def stable_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "schema_version": SCHEMA_VERSION,
                "quadrupole_scale": self.quadrupole_scale,
                **nominal_quadrupole_metadata(self.quadrupole_scale),
                "state_dimension": STATE_DIMENSION,
                "nuclear_dimension": NUCLEAR_DIMENSION,
                "spin_order": list(SPIN_ORDER),
                "system_dimensions": list(SYSTEM_DIMENSIONS),
                "screen_angles_deg": list(SCREEN_ANGLES_DEG),
                "refined_angles_deg": list(REFINED_ANGLES_DEG),
                "dipole_paper_magnitudes_uT": list(DIPOLE_PAPER_MAGNITUDES_UT),
                "low_exchange_uT": list(LOW_EXCHANGE_UT),
                "negative_ratios": list(NEGATIVE_RATIOS),
                "positive_ratios": list(POSITIVE_RATIOS),
                "published_dipolar_tensor_MHz": PUBLISHED_DIPOLE_TENSOR_MHZ.tolist(),
            }
        )
        return result

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TrajectoryTask:
    condition: CouplingCondition | EfimovaHoreCondition
    theta_deg: float
    sample_id: int
    config: SweepConfig
    yield_lifetimes_us: tuple[float, ...] = ()


def _seed(base: int, sample_id: int) -> int:
    sequence = np.random.SeedSequence([int(base), 0xA11CE, int(sample_id)])
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


def integrate_yield(times_us: np.ndarray, populations: np.ndarray, lifetime_us: float) -> float:
    values = np.exp(-times_us / lifetime_us) * populations / lifetime_us
    return float(scipy.integrate.simpson(values, x=times_us))


def coarse_grid_indices(config: SweepConfig) -> np.ndarray:
    ratio = config.coarse_integration_step_us / config.integration_step_us
    stride = int(round(ratio))
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The coarse integration step must be an integer fine-grid multiple.")
    indices = np.arange(0, config.time_grid_us.size, stride, dtype=int)
    if indices[-1] != config.time_grid_us.size - 1:
        indices = np.append(indices, config.time_grid_us.size - 1)
    return indices


def _propagator(condition: CouplingCondition, theta_deg: float, config: SweepConfig):
    if isinstance(condition, EfimovaHoreCondition):
        hamiltonian = build_quantum_needle_hamiltonian(
            theta_deg,
            config.field_microT,
            exchange_J_MHz=condition.exchange_J_MHz,
            efimova_hore_D_MHz=condition.d_paper_MHz,
            efimova_hore_axis=np.asarray(condition.dipolar_axis),
            quadrupole_scale=config.quadrupole_scale,
        )
    else:
        hamiltonian = build_quantum_needle_hamiltonian(
            theta_deg,
            config.field_microT,
            exchange_J_MHz=condition.exchange_J_MHz,
            dipolar_tensor_MHz=condition.dipolar_tensor_MHz,
            quadrupole_scale=config.quadrupole_scale,
        )
    propagator = AdaptiveLanczosPropagator(
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
    return hamiltonian, propagator


def run_trajectory(task: TrajectoryTask, *, times_us: np.ndarray | None = None) -> dict[str, object]:
    started = time.perf_counter()
    build_started = time.perf_counter()
    hamiltonian, propagator = _propagator(task.condition, task.theta_deg, task.config)
    construction_seconds = time.perf_counter() - build_started
    initial = singlet_initial_state(
        SYSTEM_DIMENSIONS,
        np.random.default_rng(_seed(task.config.seed, task.sample_id)),
    )
    initial_singlet = singlet_population(initial, SYSTEM_DIMENSIONS)
    grid = task.config.time_grid_us if times_us is None else np.asarray(times_us, dtype=float)
    propagation_started = time.perf_counter()
    result = propagator.propagate(
        initial,
        grid,
        SingletPopulationObservable(SYSTEM_DIMENSIONS),
    )
    propagation_seconds = time.perf_counter() - propagation_started
    lifetimes = task.yield_lifetimes_us or (task.config.lifetime_us,)
    if any(lifetime <= 0.0 for lifetime in lifetimes):
        raise ValueError("Yield lifetimes must be positive.")
    coarse = coarse_grid_indices(task.config) if times_us is None else None
    fine_yields: list[float] = []
    coarse_yields: list[float] = []
    propagation_bounds: list[float] = []
    tail_bounds: list[float] = []
    for lifetime in lifetimes:
        fine = integrate_yield(grid, result.observable_values, lifetime)
        coarse_value = (
            integrate_yield(grid[coarse], result.observable_values[coarse], lifetime)
            if coarse is not None
            else fine
        )
        weights = np.exp(-grid / lifetime) / lifetime
        fine_yields.append(fine)
        coarse_yields.append(coarse_value)
        propagation_bounds.append(
            float(np.trapezoid(weights * result.observable_error_bounds, grid))
        )
        tail_bounds.append(float(np.exp(-grid[-1] / lifetime)))
    primary_index = int(np.argmin(np.abs(np.asarray(lifetimes) - task.config.lifetime_us)))
    fine_yield = fine_yields[primary_index]
    coarse_yield = coarse_yields[primary_index]
    propagation_bound = propagation_bounds[primary_index]
    return {
        "condition_id": task.condition.condition_id,
        "condition_hash": task.condition.configuration_hash,
        "theta_deg": float(task.theta_deg),
        "sample_id": int(task.sample_id),
        "quadrupole_scale": task.config.quadrupole_scale,
        "initial_singlet": initial_singlet,
        "yield": fine_yield,
        "coarse_yield": coarse_yield,
        "yield_quadrature_error": abs(fine_yield - coarse_yield),
        "yield_propagation_bound": propagation_bound,
        "yield_tail_bound": tail_bounds[primary_index],
        "yield_lifetimes_us": np.asarray(lifetimes, dtype=float),
        "yields": np.asarray(fine_yields, dtype=float),
        "coarse_yields": np.asarray(coarse_yields, dtype=float),
        "yield_quadrature_errors": np.abs(
            np.asarray(fine_yields) - np.asarray(coarse_yields)
        ),
        "yield_propagation_bounds": np.asarray(propagation_bounds, dtype=float),
        "yield_tail_bounds": np.asarray(tail_bounds, dtype=float),
        "times_us": grid,
        "singlet_population": result.observable_values,
        "population_bounds": result.observable_error_bounds,
        "construction_seconds": construction_seconds,
        "propagation_seconds": propagation_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_mib": _peak_rss_mib(),
        "accepted_steps": result.stats.accepted_steps,
        "rejected_steps": result.stats.rejected_steps,
        "matvecs": result.stats.matvecs,
        "maximum_krylov_dimension": result.stats.maximum_krylov_dimension,
        "minimum_step_us": result.stats.minimum_step_us,
        "maximum_step_us": result.stats.maximum_step_us,
        "accumulated_state_error": result.stats.accumulated_state_error,
        "maximum_norm_error": result.stats.maximum_norm_error,
        "hamiltonian_matvecs": hamiltonian.matvec_count,
    }


def _angle_token(theta_deg: float) -> str:
    return f"{theta_deg:07.3f}".replace(".", "p")


def checkpoint_path(
    output_dir: Path, condition: CouplingCondition, theta_deg: float, sample_id: int
) -> Path:
    return (
        output_dir
        / "checkpoints"
        / condition.condition_id
        / f"theta_{_angle_token(theta_deg)}_sample_{sample_id:03d}.npz"
    )


def save_cell(output_dir: Path, task: TrajectoryTask, result: dict[str, object]) -> None:
    path = checkpoint_path(output_dir, task.condition, task.theta_deg, task.sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    retained = dict(result)
    if task.theta_deg not in (0.0, 90.0):
        for key in ("times_us", "singlet_population", "population_bounds"):
            retained.pop(key, None)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            configuration_hash=np.array(task.config.configuration_hash),
            **{key: np.asarray(value) for key, value in retained.items()},
        )
    os.replace(temporary, path)


def load_cell(
    output_dir: Path,
    condition: CouplingCondition,
    theta_deg: float,
    sample_id: int,
    config: SweepConfig,
) -> dict[str, object]:
    path = checkpoint_path(output_dir, condition, theta_deg, sample_id)
    with np.load(path, allow_pickle=False) as data:
        if str(data["configuration_hash"]) != config.configuration_hash:
            raise RuntimeError(f"Configuration mismatch in {path}.")
        if str(data["condition_hash"]) != condition.configuration_hash:
            raise RuntimeError(f"Condition mismatch in {path}.")
        result: dict[str, object] = {}
        for key in data.files:
            if key == "configuration_hash":
                continue
            value = data[key]
            result[key] = value.item() if value.ndim == 0 else value.copy()
        return result


def _worker(task: TrajectoryTask) -> tuple[TrajectoryTask, dict[str, object]]:
    return task, run_trajectory(task)


def run_tasks(
    output_dir: Path,
    conditions: Iterable[CouplingCondition],
    angles_deg: Iterable[float],
    sample_count: int,
    config: SweepConfig,
    workers: int,
    deadline: float,
    estimated_wave_seconds: float,
    yield_lifetimes_us: Iterable[float] = (),
) -> bool:
    yield_lifetimes = tuple(float(value) for value in yield_lifetimes_us)
    tasks = [
        TrajectoryTask(condition, float(theta), sample, config, yield_lifetimes)
        for condition in conditions
        for theta in angles_deg
        for sample in range(sample_count)
        if not checkpoint_path(output_dir, condition, float(theta), sample).exists()
    ]
    if not tasks:
        return True
    print(f"running {len(tasks)} trajectories with {workers} workers", flush=True)
    durations: list[float] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while tasks:
            rolling_wave = (
                float(np.quantile(durations, 0.95)) if durations else estimated_wave_seconds
            )
            remaining = deadline - time.time()
            if rolling_wave > 0.0 and remaining < 1.15 * rolling_wave:
                print("stopping before the next wave to respect the wall-time limit", flush=True)
                return False
            wave, tasks = tasks[:workers], tasks[workers:]
            futures = [executor.submit(_worker, task) for task in wave]
            for future in as_completed(futures):
                task, result = future.result()
                save_cell(output_dir, task, result)
                durations.append(float(result["total_seconds"]))
                completed += 1
                print(
                    f"[{completed}] {task.condition.condition_id} theta={task.theta_deg:g} "
                    f"sample={task.sample_id} {result['total_seconds']:.1f}s "
                    f"matvecs={result['matvecs']}",
                    flush=True,
                )
    return True


def _benchmark_worker(payload: tuple[CouplingCondition, int, SweepConfig]) -> float:
    condition, sample_id, config = payload
    task = TrajectoryTask(condition, 90.0, sample_id, config)
    times = np.linspace(0.0, 0.1, 81)
    started = time.perf_counter()
    run_trajectory(task, times_us=times)
    return time.perf_counter() - started


def benchmark_concurrency(output_dir: Path, config: SweepConfig) -> dict[str, object]:
    conditions = (dipole_condition(600.0), ratio_condition(-2.0))
    candidates = [value for value in (1, 2, 4, 8, 16, 32) if value <= (os.cpu_count() or 1)]
    rows: list[dict[str, object]] = []
    for workers in candidates:
        payloads = [
            (conditions[index % 2], index, config) for index in range(2 * workers)
        ]
        started = time.perf_counter()
        if workers == 1:
            durations = [_benchmark_worker(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                durations = list(executor.map(_benchmark_worker, payloads))
        elapsed = time.perf_counter() - started
        throughput = len(payloads) * 0.1 / elapsed
        rows.append(
            {
                "workers": workers,
                "elapsed_seconds": elapsed,
                "simulated_us_per_wall_second": throughput,
                "mean_task_seconds": float(np.mean(durations)),
                "maximum_task_seconds": float(np.max(durations)),
            }
        )
        print(f"benchmark workers={workers}: {throughput:.6g} simulated us/s", flush=True)
    best = max(float(row["simulated_us_per_wall_second"]) for row in rows)
    selected = next(
        int(row["workers"])
        for row in rows
        if float(row["simulated_us_per_wall_second"]) >= 0.95 * best
    )
    result = {
        "conditions": [condition.stable_dict(config.quadrupole_scale) for condition in conditions],
        "rows": rows,
        "selected_workers": selected,
        "selected_throughput": next(
            float(row["simulated_us_per_wall_second"])
            for row in rows
            if int(row["workers"]) == selected
        ),
    }
    _write_json(output_dir / "runtime_benchmark.json", result)
    _write_csv(output_dir / "runtime_benchmark.csv", rows)
    return result


def _t_halfwidth(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(
        scipy.stats.t.ppf(0.975, values.size - 1)
        * np.std(values, ddof=1)
        / math.sqrt(values.size)
    )


def condition_metrics(
    output_dir: Path,
    condition: CouplingCondition | EfimovaHoreCondition,
    angles_deg: Iterable[float],
    sample_count: int,
    config: SweepConfig,
    control: CouplingCondition | EfimovaHoreCondition | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    angles = tuple(float(value) for value in angles_deg)
    curve_rows: list[dict[str, object]] = []
    means: list[float] = []
    sample_curves: list[np.ndarray] = []
    for sample in range(sample_count):
        sample_curves.append(
            np.array(
                [
                    float(load_cell(output_dir, condition, theta, sample, config)["yield"])
                    for theta in angles
                ]
            )
        )
    curves = np.stack(sample_curves)
    for index, theta in enumerate(angles):
        cells = [load_cell(output_dir, condition, theta, sample, config) for sample in range(sample_count)]
        yields = curves[:, index]
        nuclear = _t_halfwidth(yields)
        systematic = max(
            float(cell["yield_quadrature_error"])
            + float(cell["yield_propagation_bound"])
            + float(cell["yield_tail_bound"])
            for cell in cells
        )
        mean = float(np.mean(yields))
        means.append(mean)
        curve_rows.append(
            {
                "condition_id": condition.condition_id,
                "family": condition.family,
                "theta_deg": theta,
                "samples": sample_count,
                "mean_yield": mean,
                "nuclear_95_halfwidth": nuclear,
                "systematic_halfwidth": systematic,
                "total_halfwidth": systematic + (nuclear if np.isfinite(nuclear) else 0.0),
                "quadrupole_scale": config.quadrupole_scale,
            }
        )
    means_array = np.asarray(means)
    minimum_index = int(np.argmin(means_array))
    maximum_index = int(np.argmax(means_array))
    zero_index = angles.index(0.0)
    ninety_index = angles.index(90.0)
    contrasts = curves[:, zero_index] - curves[:, ninety_index]
    contrast_nuclear = _t_halfwidth(contrasts)
    contrast_systematic = (
        float(curve_rows[zero_index]["systematic_halfwidth"])
        + float(curve_rows[ninety_index]["systematic_halfwidth"])
    )
    correlation = 1.0
    if control is not None and control.condition_id != condition.condition_id:
        control_means = []
        for theta in angles:
            control_means.append(
                np.mean(
                    [
                        float(load_cell(output_dir, control, theta, sample, config)["yield"])
                        for sample in range(sample_count)
                    ]
                )
            )
        correlation = float(scipy.stats.spearmanr(means_array, control_means).statistic)
    absolute_range = float(means_array[maximum_index] - means_array[minimum_index])
    total_contrast_halfwidth = contrast_systematic + (
        contrast_nuclear if np.isfinite(contrast_nuclear) else 0.0
    )
    summary = {
        **condition.stable_dict(config.quadrupole_scale),
        "samples": sample_count,
        "angle_count": len(angles),
        "mean_yield": float(np.mean(means_array)),
        "minimum_yield": float(means_array[minimum_index]),
        "theta_minimum_deg": angles[minimum_index],
        "maximum_yield": float(means_array[maximum_index]),
        "theta_maximum_deg": angles[maximum_index],
        "absolute_angular_range": absolute_range,
        "normalized_anisotropy": absolute_range / float(np.mean(means_array)),
        "signed_0_90_contrast": float(np.mean(contrasts)),
        "contrast_nuclear_95_halfwidth": contrast_nuclear,
        "contrast_systematic_halfwidth": contrast_systematic,
        "contrast_total_halfwidth": total_contrast_halfwidth,
        "rank_correlation_with_control": correlation,
        "nearest_extremum_to_90_deg": min(
            abs(angles[minimum_index] - 90.0), abs(angles[maximum_index] - 90.0)
        ),
        "contrast_resolved": bool(
            sample_count >= 2 and abs(float(np.mean(contrasts))) > total_contrast_halfwidth
        ),
    }
    summary["candidate_passed"] = bool(
        summary["contrast_resolved"]
        and summary["nearest_extremum_to_90_deg"] <= 15.0
        and correlation >= 0.8
    )
    return curve_rows, summary


def _efimova_hore_selection_key(summary: dict[str, object]) -> tuple[object, ...]:
    """Rank angular responses without over-rewarding a noisy M=1 range."""

    correlation = float(summary["rank_correlation_with_control"])
    if not np.isfinite(correlation):
        correlation = -1.0
    extremum_distance = float(summary["nearest_extremum_to_90_deg"])
    return (
        bool(summary["contrast_resolved"]),
        extremum_distance <= 15.0,
        -extremum_distance,
        correlation,
        float(summary["normalized_anisotropy"]),
    )


def select_efimova_hore_confirm_conditions(
    output_dir: Path, config: SweepConfig
) -> tuple[EfimovaHoreCondition, ...]:
    """Select both paper crossings and one non-centre candidate per branch."""

    screen = efimova_hore_screen_conditions()
    if not all(_complete(output_dir, item, SCREEN_ANGLES_DEG, 1) for item in screen):
        raise RuntimeError("The Efimova-Hore screen must finish before confirmation.")
    control = efimova_hore_uncoupled_control()
    selected = [
        control,
        efimova_hore_dipole_only_control(),
        efimova_hore_ratio_condition(-3.0),
        efimova_hore_ratio_condition(6.0),
    ]
    for ratios, centre in (
        (EFIMOVA_HORE_NEGATIVE_RATIOS, -3.0),
        (EFIMOVA_HORE_POSITIVE_RATIOS, 6.0),
    ):
        candidates = [
            efimova_hore_ratio_condition(value)
            for value in ratios
            if not math.isclose(value, centre)
        ]
        selected.append(
            max(
                candidates,
                key=lambda condition: _efimova_hore_selection_key(
                    condition_metrics(
                        output_dir,
                        condition,
                        SCREEN_ANGLES_DEG,
                        1,
                        config,
                        control,
                    )[1]
                ),
            )
        )
    return tuple({item.condition_id: item for item in selected}.values())


def select_efimova_hore_final_conditions(
    output_dir: Path,
    confirmed: Iterable[EfimovaHoreCondition],
    config: SweepConfig,
) -> tuple[EfimovaHoreCondition, ...]:
    """Choose one confirmed response from each cancellation branch."""

    confirmed = tuple(confirmed)
    if not all(_complete(output_dir, item, SCREEN_ANGLES_DEG, 4) for item in confirmed):
        raise RuntimeError("Efimova-Hore M=4 confirmation is incomplete.")
    control = efimova_hore_uncoupled_control()
    selected: list[EfimovaHoreCondition] = [
        control,
        efimova_hore_dipole_only_control(),
    ]
    for sign in (-1.0, 1.0):
        branch = [
            item
            for item in confirmed
            if item.ratio_D_over_J is not None
            and math.copysign(1.0, item.ratio_D_over_J) == sign
        ]
        selected.append(
            max(
                branch,
                key=lambda condition: _efimova_hore_selection_key(
                    condition_metrics(
                        output_dir,
                        condition,
                        SCREEN_ANGLES_DEG,
                        4,
                        config,
                        control,
                    )[1]
                ),
            )
        )
    return tuple(selected)


def _csv_safe(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            key: json.dumps(value) if isinstance(value, (list, tuple, dict)) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def create_efimova_hore_report(
    output_dir: Path,
    config: SweepConfig,
    confirmed: Iterable[EfimovaHoreCondition] = (),
    final: Iterable[EfimovaHoreCondition] = (),
) -> dict[str, object]:
    """Write tables and plots for every completed part of the 24-hour run."""

    control = efimova_hore_uncoupled_control()
    screen = efimova_hore_screen_conditions()
    confirmed = tuple(confirmed)
    final = tuple(final)
    screen_rows: list[dict[str, object]] = []
    screen_summaries: list[dict[str, object]] = []
    if all(_complete(output_dir, item, SCREEN_ANGLES_DEG, 1) for item in screen):
        for condition in screen:
            rows, summary = condition_metrics(
                output_dir, condition, SCREEN_ANGLES_DEG, 1, config, control
            )
            screen_rows.extend(rows)
            screen_summaries.append(summary)
        _write_csv(output_dir / "efimova_hore_screen_curves.csv", _csv_safe(screen_rows))
        _write_csv(
            output_dir / "efimova_hore_screen_metrics.csv",
            _csv_safe(screen_summaries),
        )

    confirm_summaries: list[dict[str, object]] = []
    if confirmed and all(
        _complete(output_dir, item, SCREEN_ANGLES_DEG, 4) for item in confirmed
    ):
        for condition in confirmed:
            _, summary = condition_metrics(
                output_dir, condition, SCREEN_ANGLES_DEG, 4, config, control
            )
            confirm_summaries.append(summary)
        _write_csv(
            output_dir / "efimova_hore_confirm_metrics.csv",
            _csv_safe(confirm_summaries),
        )

    final_rows: list[dict[str, object]] = []
    final_summaries: list[dict[str, object]] = []
    if final and all(_complete(output_dir, item, REFINED_ANGLES_DEG, 4) for item in final):
        for condition in final:
            rows, summary = condition_metrics(
                output_dir, condition, REFINED_ANGLES_DEG, 4, config, control
            )
            final_rows.extend(rows)
            final_summaries.append(summary)
        _write_csv(
            output_dir / "efimova_hore_final_curves.csv", _csv_safe(final_rows)
        )
        _write_csv(
            output_dir / "efimova_hore_final_metrics_m4.csv",
            _csv_safe(final_summaries),
        )

    precision_summaries: list[dict[str, object]] = []
    precision_angles = (0.0, 90.0)
    if final and all(_complete(output_dir, item, precision_angles, 8) for item in final):
        for condition in final:
            _, summary = condition_metrics(
                output_dir, condition, precision_angles, 8, config, control
            )
            precision_summaries.append(summary)
        _write_csv(
            output_dir / "efimova_hore_precision_metrics_m8.csv",
            _csv_safe(precision_summaries),
        )

    raw_rows: list[dict[str, object]] = []
    for condition in screen:
        for theta in REFINED_ANGLES_DEG:
            for sample in range(8):
                if not checkpoint_path(output_dir, condition, theta, sample).exists():
                    continue
                cell = load_cell(output_dir, condition, theta, sample, config)
                raw_rows.append(
                    {
                        **condition.stable_dict(config.quadrupole_scale),
                        "theta_deg": theta,
                        "sample_id": sample,
                        **{
                            key: cell[key]
                            for key in (
                                "yield",
                                "coarse_yield",
                                "yield_quadrature_error",
                                "yield_propagation_bound",
                                "yield_tail_bound",
                                "accepted_steps",
                                "rejected_steps",
                                "matvecs",
                                "maximum_krylov_dimension",
                                "accumulated_state_error",
                                "maximum_norm_error",
                                "total_seconds",
                            )
                        },
                    }
                )
    _write_csv(output_dir / "efimova_hore_yield_cells.csv", _csv_safe(raw_rows))

    if screen_summaries:
        ratios = sorted(
            (
                row
                for row in screen_summaries
                if row["ratio_D_over_J"] is not None
            ),
            key=lambda row: float(row["ratio_D_over_J"]),
        )
        fig, axis = plt.subplots(figsize=(8.0, 4.8))
        for sign, label in ((-1.0, "negative branch"), (1.0, "positive branch")):
            branch = [
                row
                for row in ratios
                if math.copysign(1.0, float(row["ratio_D_over_J"])) == sign
            ]
            axis.plot(
                [float(row["ratio_D_over_J"]) for row in branch],
                [float(row["normalized_anisotropy"]) for row in branch],
                marker="o",
                label=label,
            )
        axis.axvline(-3.0, color="black", linestyle="--", linewidth=1.0)
        axis.axvline(6.0, color="black", linestyle="--", linewidth=1.0)
        axis.set(
            xlabel=r"Efimova-Hore $D/J$",
            ylabel="Normalized angular anisotropy",
            title=f"Literal Efimova-Hore cancellation screen (Q scale={config.quadrupole_scale:g})",
        )
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "efimova_hore_screen_anisotropy.png", dpi=180)
        plt.close(fig)

    if final_rows:
        fig, axis = plt.subplots(figsize=(8.5, 5.2))
        for condition in final:
            rows = [
                row for row in final_rows if row["condition_id"] == condition.condition_id
            ]
            rows.sort(key=lambda row: float(row["theta_deg"]))
            angles = np.array([float(row["theta_deg"]) for row in rows])
            means = np.array([float(row["mean_yield"]) for row in rows])
            halfwidths = np.array([float(row["total_halfwidth"]) for row in rows])
            label = condition.role
            if condition.ratio_D_over_J is not None:
                label = f"D/J={condition.ratio_D_over_J:g}"
            axis.plot(angles, means, marker="o", markersize=3, label=label)
            axis.fill_between(angles, means - halfwidths, means + halfwidths, alpha=0.15)
        axis.set(
            xlabel="Field angle (degrees)",
            ylabel="Singlet yield",
            title=f"Efimova-Hore angular sensitivity (Q scale={config.quadrupole_scale:g}, 2 us)",
        )
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "efimova_hore_final_angular_yields.png", dpi=180)
        plt.close(fig)

    control_precision = next(
        (
            row
            for row in precision_summaries
            if row["condition_id"] == control.condition_id
        ),
        None,
    )
    summary = {
        "convention": "Efimova-Hore Eqs. 9-10 (literal operators)",
        "exchange_hamiltonian": "-J*(S^2-1)",
        "dipolar_hamiltonian": "D*((S.n)^2-S^2/3)",
        "dipolar_axis": EFIMOVA_HORE_AXIS.tolist(),
        "d_paper_uT": R2_D_PAPER_UT,
        "cancellation_centres_D_over_J": [-3.0, 6.0],
        "quadrupole_scale": config.quadrupole_scale,
        "screen_complete": bool(screen_summaries),
        "confirmation_complete": bool(confirm_summaries),
        "refinement_complete": bool(final_summaries),
        "precision_complete": bool(precision_summaries),
        "confirmed_conditions": [item.stable_dict(config.quadrupole_scale) for item in confirmed],
        "final_conditions": [item.stable_dict(config.quadrupole_scale) for item in final],
        "uncoupled_control_precision": control_precision,
        "uncoupled_control_angular_sensitivity_resolved": bool(
            control_precision is not None and control_precision["contrast_resolved"]
        ),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output_dir / "efimova_hore_summary.json", summary)
    return summary


def _complete_sample_count(
    output_dir: Path,
    condition: CouplingCondition,
    angles: Iterable[float],
    maximum: int,
) -> int:
    """Return the largest contiguous sample count complete at every angle."""
    count = 0
    for sample in range(maximum):
        if not _complete(output_dir, condition, angles, sample + 1):
            break
        count = sample + 1
    return count


def create_ratio_focus_report(
    output_dir: Path,
    conditions: Iterable[CouplingCondition],
    angles_deg: Iterable[float],
    requested_samples: int,
    config: SweepConfig,
) -> list[dict[str, object]]:
    """Write a self-contained report for a selected high-M ratio confirmation."""
    conditions = tuple(conditions)
    angles = tuple(float(value) for value in angles_deg)
    raw_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    control = low_exchange_condition(0.0)
    control_samples = _complete_sample_count(output_dir, control, angles, requested_samples)

    for condition in conditions:
        completed_samples = _complete_sample_count(
            output_dir, condition, angles, requested_samples
        )
        for theta in angles:
            for sample in range(requested_samples):
                if not checkpoint_path(output_dir, condition, theta, sample).exists():
                    continue
                cell = load_cell(output_dir, condition, theta, sample, config)
                raw_rows.append(
                    {
                        **condition.stable_dict(config.quadrupole_scale),
                        "theta_deg": theta,
                        "sample_id": sample,
                        **{
                            key: cell[key]
                            for key in (
                                "yield",
                                "coarse_yield",
                                "yield_quadrature_error",
                                "yield_propagation_bound",
                                "yield_tail_bound",
                                "accepted_steps",
                                "rejected_steps",
                                "matvecs",
                                "maximum_krylov_dimension",
                                "accumulated_state_error",
                                "maximum_norm_error",
                                "total_seconds",
                                "peak_rss_mib",
                                "quadrupole_scale",
                            )
                        },
                    }
                )
        if completed_samples != requested_samples:
            continue
        rows, summary = condition_metrics(
            output_dir, condition, angles, requested_samples, config
        )
        if control_samples:
            control_means = np.array(
                [
                    np.mean(
                        [
                            float(load_cell(output_dir, control, theta, sample, config)["yield"])
                            for sample in range(control_samples)
                        ]
                    )
                    for theta in angles
                ]
            )
            condition_means = np.array([float(row["mean_yield"]) for row in rows])
            correlation = float(
                scipy.stats.spearmanr(condition_means, control_means).statistic
            )
        else:
            correlation = float("nan")
        summary["rank_correlation_with_control"] = correlation
        summary["comparison_control_samples"] = control_samples
        summary["candidate_passed"] = bool(
            summary["contrast_resolved"]
            and summary["nearest_extremum_to_90_deg"] <= 15.0
            and np.isfinite(correlation)
            and correlation >= 0.8
        )
        curve_rows.extend(rows)
        summaries.append(summary)

    _write_csv(output_dir / "ratio_focus_yield_cells.csv", raw_rows)
    _write_csv(output_dir / "ratio_focus_angular_curves.csv", curve_rows)
    _write_csv(
        output_dir / "ratio_focus_condition_metrics.csv",
        [
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
            for row in summaries
        ],
    )
    _write_json(
        output_dir / "ratio_focus_summary.json",
        {
            "requested_samples": requested_samples,
            "angles_deg": angles,
            "control_samples": control_samples,
            "quadrupole_scale": config.quadrupole_scale,
            "conditions_complete": len(summaries),
            "conditions_requested": len(conditions),
            "metrics": summaries,
        },
    )

    if curve_rows:
        fig, axis = plt.subplots(figsize=(8.0, 4.8))
        for condition in conditions:
            rows = [
                row for row in curve_rows if row["condition_id"] == condition.condition_id
            ]
            if not rows:
                continue
            x = np.array([float(row["theta_deg"]) for row in rows])
            y = np.array([float(row["mean_yield"]) for row in rows])
            error = np.array([float(row["total_halfwidth"]) for row in rows])
            axis.plot(x, y, marker="o", label=rf"$D_\parallel/J={condition.ratio_D_over_J:g}$")
            axis.fill_between(x, y - error, y + error, alpha=0.16)
        axis.set(
            xlabel="Field angle (degrees)",
            ylabel="Singlet yield",
            title=f"Focused exchange confirmation (M={requested_samples}, Q scale={config.quadrupole_scale:g})",
        )
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "ratio_focus_angular_yields.png", dpi=180)
        plt.close(fig)
    return summaries


def create_ratio_lifetime_report(
    output_dir: Path,
    conditions: Iterable[CouplingCondition],
    angles_deg: Iterable[float],
    requested_samples: int,
    lifetimes_us: Iterable[float],
    config: SweepConfig,
) -> list[dict[str, object]]:
    """Report several recombination lifetimes from each shared propagation."""
    conditions = tuple(conditions)
    angles = tuple(float(value) for value in angles_deg)
    lifetimes = tuple(float(value) for value in lifetimes_us)
    raw_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for condition in conditions:
        completed_samples = _complete_sample_count(
            output_dir, condition, angles, requested_samples
        )
        cells_by_angle: dict[float, list[dict[str, object]]] = {}
        for theta in angles:
            cells: list[dict[str, object]] = []
            for sample in range(min(completed_samples, requested_samples)):
                cell = load_cell(output_dir, condition, theta, sample, config)
                stored_lifetimes = np.atleast_1d(cell["yield_lifetimes_us"]).astype(float)
                if stored_lifetimes.shape != (len(lifetimes),) or not np.allclose(
                    stored_lifetimes, lifetimes, atol=0.0, rtol=0.0
                ):
                    raise RuntimeError("Checkpoint lifetime list is incompatible with this report.")
                cells.append(cell)
                for index, lifetime in enumerate(lifetimes):
                    raw_rows.append(
                        {
                            **condition.stable_dict(config.quadrupole_scale),
                            "theta_deg": theta,
                            "sample_id": sample,
                            "lifetime_us": lifetime,
                            "yield": float(np.atleast_1d(cell["yields"])[index]),
                            "coarse_yield": float(
                                np.atleast_1d(cell["coarse_yields"])[index]
                            ),
                            "yield_quadrature_error": float(
                                np.atleast_1d(cell["yield_quadrature_errors"])[index]
                            ),
                            "yield_propagation_bound": float(
                                np.atleast_1d(cell["yield_propagation_bounds"])[index]
                            ),
                            "yield_tail_bound": float(
                                np.atleast_1d(cell["yield_tail_bounds"])[index]
                            ),
                            "total_seconds": float(cell["total_seconds"]),
                            "matvecs": int(cell["matvecs"]),
                            "accepted_steps": int(cell["accepted_steps"]),
                            "rejected_steps": int(cell["rejected_steps"]),
                            "maximum_krylov_dimension": int(
                                cell["maximum_krylov_dimension"]
                            ),
                            "quadrupole_scale": config.quadrupole_scale,
                        }
                    )
            cells_by_angle[theta] = cells
        if completed_samples != requested_samples:
            continue

        for index, lifetime in enumerate(lifetimes):
            curves = np.array(
                [
                    [
                        float(np.atleast_1d(cell["yields"])[index])
                        for theta in angles
                        for cell in cells_by_angle[theta][sample : sample + 1]
                    ]
                    for sample in range(requested_samples)
                ]
            )
            means = np.mean(curves, axis=0)
            systematic: list[float] = []
            for angle_index, theta in enumerate(angles):
                yields = curves[:, angle_index]
                cells = cells_by_angle[theta]
                nuclear = _t_halfwidth(yields)
                bound = max(
                    float(np.atleast_1d(cell["yield_quadrature_errors"])[index])
                    + float(np.atleast_1d(cell["yield_propagation_bounds"])[index])
                    + float(np.atleast_1d(cell["yield_tail_bounds"])[index])
                    for cell in cells
                )
                systematic.append(bound)
                curve_rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "ratio_D_over_J": condition.ratio_D_over_J,
                        "lifetime_us": lifetime,
                        "theta_deg": theta,
                        "samples": requested_samples,
                        "mean_yield": float(means[angle_index]),
                        "nuclear_95_halfwidth": nuclear,
                        "systematic_halfwidth": bound,
                        "total_halfwidth": bound + nuclear,
                        "quadrupole_scale": config.quadrupole_scale,
                    }
                )
            minimum_index = int(np.argmin(means))
            maximum_index = int(np.argmax(means))
            zero_index = angles.index(0.0)
            ninety_index = angles.index(90.0)
            contrasts = curves[:, zero_index] - curves[:, ninety_index]
            contrast_nuclear = _t_halfwidth(contrasts)
            contrast_systematic = systematic[zero_index] + systematic[ninety_index]
            total_contrast = contrast_systematic + contrast_nuclear
            angular_range = float(means[maximum_index] - means[minimum_index])
            summaries.append(
                {
                    **condition.stable_dict(config.quadrupole_scale),
                    "lifetime_us": lifetime,
                    "samples": requested_samples,
                    "angle_count": len(angles),
                    "mean_yield": float(np.mean(means)),
                    "minimum_yield": float(means[minimum_index]),
                    "theta_minimum_deg": angles[minimum_index],
                    "maximum_yield": float(means[maximum_index]),
                    "theta_maximum_deg": angles[maximum_index],
                    "absolute_angular_range": angular_range,
                    "normalized_anisotropy": angular_range / float(np.mean(means)),
                    "signed_0_90_contrast": float(np.mean(contrasts)),
                    "contrast_nuclear_95_halfwidth": contrast_nuclear,
                    "contrast_systematic_halfwidth": contrast_systematic,
                    "contrast_total_halfwidth": total_contrast,
                    "nearest_extremum_to_90_deg": min(
                        abs(angles[minimum_index] - 90.0),
                        abs(angles[maximum_index] - 90.0),
                    ),
                    "contrast_resolved": bool(
                        abs(float(np.mean(contrasts))) > total_contrast
                    ),
                }
            )

    _write_csv(output_dir / "ratio_lifetime_yield_cells.csv", raw_rows)
    _write_csv(output_dir / "ratio_lifetime_angular_curves.csv", curve_rows)
    _write_csv(
        output_dir / "ratio_lifetime_condition_metrics.csv",
        [
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
            for row in summaries
        ],
    )
    _write_json(
        output_dir / "ratio_lifetime_summary.json",
        {
            "requested_samples": requested_samples,
            "angles_deg": angles,
            "lifetimes_us": lifetimes,
            "quadrupole_scale": config.quadrupole_scale,
            "conditions_complete": len(summaries) // max(1, len(lifetimes)),
            "conditions_requested": len(conditions),
            "metrics": summaries,
        },
    )

    if curve_rows:
        fig, axes = plt.subplots(
            1, len(lifetimes), figsize=(6.4 * len(lifetimes), 4.8), squeeze=False
        )
        for lifetime, axis in zip(lifetimes, axes[0]):
            for condition in conditions:
                rows = [
                    row
                    for row in curve_rows
                    if row["condition_id"] == condition.condition_id
                    and float(row["lifetime_us"]) == lifetime
                ]
                if not rows:
                    continue
                x = np.array([float(row["theta_deg"]) for row in rows])
                y = np.array([float(row["mean_yield"]) for row in rows])
                error = np.array([float(row["total_halfwidth"]) for row in rows])
                axis.plot(
                    x,
                    y,
                    marker="o",
                    label=rf"$D_\parallel/J={condition.ratio_D_over_J:g}$",
                )
                axis.fill_between(x, y - error, y + error, alpha=0.16)
            axis.set(
                xlabel="Field angle (degrees)",
                ylabel="Singlet yield",
                title=rf"$\tau={lifetime:g}\,\mu s$ (M={requested_samples}, Q scale={config.quadrupole_scale:g})",
            )
            axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "ratio_lifetime_angular_yields.png", dpi=180)
        plt.close(fig)
    return summaries


def _complete(
    output_dir: Path,
    condition: CouplingCondition,
    angles: Iterable[float],
    samples: int,
) -> bool:
    return all(
        checkpoint_path(output_dir, condition, float(theta), sample).exists()
        for theta in angles
        for sample in range(samples)
    )


def select_dipole_confirm_conditions(output_dir: Path, config: SweepConfig) -> tuple[CouplingCondition, ...]:
    conditions = dipole_screen_conditions()
    if not all(_complete(output_dir, condition, SCREEN_ANGLES_DEG, 1) for condition in conditions):
        raise RuntimeError("The dipole screen must finish before dipole confirmation.")
    scored = [
        (condition_metrics(output_dir, condition, SCREEN_ANGLES_DEG, 1, config)[1]["absolute_angular_range"], condition)
        for condition in conditions
    ]
    selected = [dipole_condition(value) for value in (0.0, 50.0, 347.875, 600.0)]
    selected.extend((min(scored, key=lambda item: item[0])[1], max(scored, key=lambda item: item[0])[1]))
    return tuple({condition.condition_id: condition for condition in selected}.values())


def select_ratio_confirm_conditions(output_dir: Path, config: SweepConfig) -> tuple[CouplingCondition, ...]:
    conditions = ratio_screen_conditions()
    if not all(_complete(output_dir, condition, SCREEN_ANGLES_DEG, 1) for condition in conditions):
        raise RuntimeError("The ratio screen must finish before ratio confirmation.")
    selected = [low_exchange_condition(value) for value in (-10.0, 0.0, 10.0)]
    selected.extend(ratio_condition(value) for value in (-4.25, -4.0, -3.75, 7.5, 8.0, 8.5))
    for ratios in (NEGATIVE_RATIOS, POSITIVE_RATIOS):
        candidates = [ratio_condition(value) for value in ratios]
        best = max(
            candidates,
            key=lambda condition: condition_metrics(
                output_dir, condition, SCREEN_ANGLES_DEG, 1, config
            )[1]["absolute_angular_range"],
        )
        selected.append(best)
    return tuple({condition.condition_id: condition for condition in selected}.values())


def _best_by_range(
    output_dir: Path,
    conditions: Iterable[CouplingCondition],
    config: SweepConfig,
) -> CouplingCondition:
    return max(
        conditions,
        key=lambda condition: condition_metrics(
            output_dir, condition, SCREEN_ANGLES_DEG, 4, config
        )[1]["absolute_angular_range"],
    )


def ensure_manifest(output_dir: Path, config: SweepConfig, resume: bool) -> dict[str, object]:
    path = output_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["configuration_hash"] != config.configuration_hash:
            raise RuntimeError("Existing output uses an incompatible sweep configuration.")
        if not resume:
            raise RuntimeError("Output already exists; pass --resume to continue it.")
        return manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "created",
        "configuration_hash": config.configuration_hash,
        "configuration": config.stable_dict(),
        "conditions": [condition.stable_dict(config.quadrupole_scale) for condition in all_conditions()],
        "quadrupole_scale": config.quadrupole_scale,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": _git_commit(),
        "stages": {},
    }
    _write_json(path, manifest)
    return manifest


def ensure_efimova_hore_stage_definition(
    output_dir: Path, quadrupole_scale: float = QUADRUPOLE_SCALE
) -> dict[str, object]:
    """Pin adaptive selections and the full 24-hour task budget for safe resume."""

    definition = {
        "schema_version": 1,
        "convention": "Efimova-Hore Eqs. 9-10 (literal operators)",
        "exchange_hamiltonian": "-J*(S^2-1)",
        "dipolar_hamiltonian": "D*((S.n)^2-S^2/3)",
        "dipolar_axis": EFIMOVA_HORE_AXIS.tolist(),
        "d_paper_uT": R2_D_PAPER_UT,
        "quadrupole_scale": quadrupole_scale,
        "negative_ratios": list(EFIMOVA_HORE_NEGATIVE_RATIOS),
        "positive_ratios": list(EFIMOVA_HORE_POSITIVE_RATIOS),
        "screen": {"conditions": 16, "angles_deg": list(SCREEN_ANGLES_DEG), "samples": 1},
        "confirm": {"conditions": 6, "angles_deg": list(SCREEN_ANGLES_DEG), "samples": 4},
        "refine": {"conditions": 4, "angles_deg": list(REFINED_ANGLES_DEG), "samples": 4},
        "precision": {"conditions": 4, "angles_deg": [0.0, 90.0], "samples": 8},
        "new_trajectory_budget": 366,
        "selection_policy": (
            "resolved contrast, extremum within 15 degrees of 90, nearest extremum, "
            "Spearman correlation with D=J=0, normalized anisotropy"
        ),
    }
    payload = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    definition["definition_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    path = output_dir / "efimova_hore_stage_definition.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != definition:
            raise RuntimeError("Existing output uses an incompatible Efimova-Hore stage definition.")
    else:
        _write_json(path, definition)
    return definition


def _control_for(condition: CouplingCondition) -> CouplingCondition:
    return dipole_condition(0.0) if condition.family == "dipole" else low_exchange_condition(0.0)


def create_report(output_dir: Path, config: SweepConfig) -> dict[str, object]:
    raw_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for condition in all_conditions():
        completed = [
            (theta, sample)
            for theta in REFINED_ANGLES_DEG
            for sample in range(4)
            if checkpoint_path(output_dir, condition, theta, sample).exists()
        ]
        for theta, sample in completed:
            cell = load_cell(output_dir, condition, theta, sample, config)
            raw_rows.append(
                {
                    **condition.stable_dict(config.quadrupole_scale),
                    "theta_deg": theta,
                    "sample_id": sample,
                    **{
                        key: cell[key]
                        for key in (
                            "yield",
                            "coarse_yield",
                            "yield_quadrature_error",
                            "yield_propagation_bound",
                            "yield_tail_bound",
                            "initial_singlet",
                            "accepted_steps",
                            "rejected_steps",
                            "matvecs",
                            "maximum_krylov_dimension",
                            "accumulated_state_error",
                            "maximum_norm_error",
                            "total_seconds",
                            "peak_rss_mib",
                            "quadrupole_scale",
                        )
                    },
                }
            )
        sample_count = 4 if any(sample == 3 for _, sample in completed) else 1
        available_angles = tuple(
            theta
            for theta in REFINED_ANGLES_DEG
            if _complete(output_dir, condition, (theta,), sample_count)
        )
        if 0.0 not in available_angles or 90.0 not in available_angles:
            continue
        control = _control_for(condition)
        usable_angles = available_angles
        if not _complete(output_dir, control, usable_angles, sample_count):
            usable_angles = tuple(
                theta
                for theta in available_angles
                if _complete(output_dir, control, (theta,), sample_count)
            )
        rows, summary = condition_metrics(
            output_dir, condition, usable_angles, sample_count, config, control
        )
        curve_rows.extend(rows)
        summaries.append(summary)
    _write_csv(output_dir / "yield_cells.csv", raw_rows)
    _write_csv(output_dir / "angular_curves.csv", curve_rows)
    serializable_summaries = [
        {key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()}
        for row in summaries
    ]
    _write_csv(output_dir / "condition_metrics.csv", serializable_summaries)

    physical = [
        row
        for row in summaries
        if row["family"] == "ratio"
        and row["role"] == "fixed_D_low_exchange_control"
        and 0.0 < abs(float(row["exchange_J_uT"])) <= 10.0
        and row["candidate_passed"]
    ]
    mechanistic = [
        row
        for row in summaries
        if row["role"] == "mechanistic_cancellation_scan" and row["candidate_passed"]
    ]
    best_physical = max(physical, key=lambda row: row["normalized_anisotropy"], default=None)
    best_mechanistic = max(mechanistic, key=lambda row: row["normalized_anisotropy"], default=None)
    candidates = {
        "best_physical_nonzero_exchange_baseline": best_physical,
        "best_mechanistic_cancellation_candidate": best_mechanistic,
        "fallback": (
            None
            if best_physical is not None or best_mechanistic is not None
            else f"Retain the J=D=0 control at Q scale={config.quadrupole_scale:g}."
        ),
    }
    _write_json(output_dir / "candidate_summary.json", candidates)
    _write_json(
        output_dir / "metadata.json",
        {
            "configuration": config.stable_dict(),
            "configuration_hash": config.configuration_hash,
            "quadrupole_scale": config.quadrupole_scale,
            "conditions": [condition.stable_dict(config.quadrupole_scale) for condition in all_conditions()],
            "candidate_summary": candidates,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    _plot_reports(output_dir, curve_rows, summaries, config.quadrupole_scale)
    return candidates


def _plot_reports(
    output_dir: Path,
    curve_rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    quadrupole_scale: float = QUADRUPOLE_SCALE,
) -> None:
    if not curve_rows:
        return
    summary_by_id = {str(row["condition_id"]): row for row in summaries}

    def condition_label(condition_id: str) -> str:
        summary = summary_by_id[condition_id]
        if summary["family"] == "dipole":
            return rf"$D_{{paper}}={float(summary['d_paper_uT']):g}\ \mu T$"
        if summary["role"] == "mechanistic_cancellation_scan":
            return rf"$D_\parallel/J={float(summary['ratio_D_over_J']):g}$"
        return rf"$J={float(summary['exchange_J_uT']):g}\ \mu T$"
    for family, filename, title in (
        ("dipole", "dipole_angular_yields.png", f"Dipole-only angular yields (J=0, Q scale={quadrupole_scale:g})"),
        ("ratio", "exchange_angular_yields.png", f"Fixed-D exchange angular yields (Q scale={quadrupole_scale:g})"),
    ):
        fig, axis = plt.subplots(figsize=(9.0, 5.3))
        ids = sorted({str(row["condition_id"]) for row in curve_rows if row["family"] == family})
        for condition_id in ids:
            rows = [row for row in curve_rows if row["condition_id"] == condition_id]
            rows.sort(key=lambda row: float(row["theta_deg"]))
            axis.plot(
                [float(row["theta_deg"]) for row in rows],
                [float(row["mean_yield"]) for row in rows],
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label=condition_label(condition_id),
            )
        axis.set(xlabel="Field angle (degrees)", ylabel="Singlet yield", title=title)
        axis.legend(fontsize=6, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    def heatmap(
        selected: list[dict[str, object]],
        coordinate: str,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        if not selected:
            return
        selected.sort(key=lambda row: float(row[coordinate]))
        angles = np.asarray(SCREEN_ANGLES_DEG)
        matrix = np.full((len(selected), angles.size), np.nan)
        for row_index, summary in enumerate(selected):
            condition_id = str(summary["condition_id"])
            lookup = {
                float(row["theta_deg"]): float(row["mean_yield"])
                for row in curve_rows
                if row["condition_id"] == condition_id
            }
            for angle_index, angle in enumerate(angles):
                matrix[row_index, angle_index] = lookup.get(float(angle), np.nan)
        coordinates = np.array([float(row[coordinate]) for row in selected])
        fig, axis = plt.subplots(figsize=(8.3, 5.3))
        image = axis.pcolormesh(angles, coordinates, matrix, shading="nearest")
        axis.set(xlabel="Field angle (degrees)", ylabel=ylabel, title=title)
        axis.set_xticks(angles)
        fig.colorbar(image, ax=axis, label="Mean singlet yield")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    heatmap(
        [row for row in summaries if row["family"] == "dipole"],
        "d_paper_uT",
        r"$D_{paper}$ ($\mu$T)",
        f"Dipolar strength and angular response (J=0, Q scale={quadrupole_scale:g})",
        "dipole_angle_heatmap.png",
    )
    heatmap(
        [
            row
            for row in summaries
            if row["role"] == "mechanistic_cancellation_scan"
        ],
        "ratio_D_over_J",
        r"$D_{\parallel}/J$",
        f"Cancellation-ratio angular response (Q scale={quadrupole_scale:g})",
        "ratio_angle_heatmap.png",
    )

    dipole = sorted(
        (row for row in summaries if row["family"] == "dipole"),
        key=lambda row: abs(float(row["d_paper_uT"])),
    )
    if dipole:
        fig, axis = plt.subplots(figsize=(7.6, 4.8))
        axis.plot(
            [abs(float(row["d_paper_uT"])) for row in dipole],
            [float(row["normalized_anisotropy"]) for row in dipole],
            marker="o",
        )
        axis.set(
            xlabel=r"$|D_{paper}|$ ($\mu$T)",
            ylabel="Normalized angular anisotropy",
            title=f"Dipolar-strength sweep (J=0, Q scale={quadrupole_scale:g})",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "anisotropy_vs_dipole.png", dpi=180)
        plt.close(fig)

    ratio = sorted(
        (row for row in summaries if row["ratio_D_over_J"] is not None and row["role"] == "mechanistic_cancellation_scan"),
        key=lambda row: float(row["ratio_D_over_J"]),
    )
    if ratio:
        fig, axis = plt.subplots(figsize=(8.0, 4.8))
        for sign, label in ((-1, "negative-ratio branch"), (1, "positive-ratio branch")):
            rows = [row for row in ratio if math.copysign(1.0, float(row["ratio_D_over_J"])) == sign]
            axis.plot(
                [float(row["ratio_D_over_J"]) for row in rows],
                [float(row["normalized_anisotropy"]) for row in rows],
                marker="o",
                label=label,
            )
        axis.axvline(-4.0, color="black", linestyle="--", linewidth=1.0)
        axis.axvline(8.0, color="black", linestyle="--", linewidth=1.0)
        axis.set(
            xlabel=r"$D_{\parallel}/J$",
            ylabel="Normalized angular anisotropy",
            title=f"Exchange/dipole cancellation scan (Q scale={quadrupole_scale:g})",
        )
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "anisotropy_vs_ratio.png", dpi=180)
        plt.close(fig)


def _mark_stage(
    output_dir: Path, manifest: dict[str, object], stage: str, complete: bool
) -> None:
    manifest["status"] = f"{stage}_{'complete' if complete else 'walltime_limited'}"
    manifest.setdefault("stages", {})[stage] = {
        "complete": complete,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output_dir / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "benchmark",
            "dipole-screen",
            "dipole-confirm",
            "ratio-screen",
            "ratio-confirm",
            "ratio-focus",
            "ratio-lifetime-focus",
            "efimova-hore-24h",
            "report",
        ),
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    add_quadrupole_argument(parser)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-wall-hours", type=float, default=20.0)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        help="D_parallel/J values for a focused ratio stage.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Total nuclear random-phase samples for a focused ratio stage.",
    )
    parser.add_argument(
        "--lifetimes",
        nargs="+",
        type=float,
        help="Recombination lifetimes for --stage ratio-lifetime-focus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "quantum_needle_jd_sweeps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maximum_allowed_wall_hours = 24.0 if args.stage == "efimova-hore-24h" else 20.0
    if not 0.0 < args.max_wall_hours <= maximum_allowed_wall_hours:
        raise ValueError(
            f"--max-wall-hours must be in (0, {maximum_allowed_wall_hours:g}]."
        )
    if args.samples < 1:
        raise ValueError("--samples must be positive.")
    if args.stage in ("ratio-focus", "ratio-lifetime-focus") and not args.ratios:
        raise ValueError(f"--stage {args.stage} requires --ratios.")
    if args.stage == "ratio-lifetime-focus":
        if not args.lifetimes or any(value <= 0.0 for value in args.lifetimes):
            raise ValueError("--stage ratio-lifetime-focus requires positive --lifetimes.")
        lifetimes = tuple(dict.fromkeys(float(value) for value in args.lifetimes))
        config = replace(SweepConfig(), seed=args.seed, lifetime_us=max(lifetimes),
                         quadrupole_scale=args.quadrupole_scale)
    else:
        lifetimes = ()
        config = replace(SweepConfig(), seed=args.seed, quadrupole_scale=args.quadrupole_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_manifest(args.output_dir, config, args.resume)
    if args.stage == "efimova-hore-24h":
        definition = ensure_efimova_hore_stage_definition(args.output_dir, config.quadrupole_scale)
        manifest["efimova_hore_stage_definition"] = definition
        manifest["efimova_hore_conditions"] = [
            item.stable_dict(config.quadrupole_scale) for item in efimova_hore_screen_conditions()
        ]
        _write_json(args.output_dir / "manifest.json", manifest)
    if args.stage == "report":
        create_report(args.output_dir, config)
        _mark_stage(args.output_dir, manifest, "report", True)
        return
    benchmark_path = args.output_dir / "runtime_benchmark.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    else:
        benchmark = benchmark_concurrency(args.output_dir, config)
    if args.stage == "benchmark":
        manifest["runtime_benchmark"] = benchmark
        _mark_stage(args.output_dir, manifest, "benchmark", True)
        return
    workers = args.workers or int(benchmark["selected_workers"])
    matching = [row for row in benchmark["rows"] if int(row["workers"]) == workers]
    if not matching:
        raise ValueError("--workers must be one of the benchmarked worker counts.")
    throughput = float(matching[0]["simulated_us_per_wall_second"])
    estimated_wave_seconds = workers * config.maximum_time_us / throughput
    manifest["selected_workers"] = workers
    manifest["estimated_full_horizon_wave_seconds"] = estimated_wave_seconds
    _write_json(args.output_dir / "manifest.json", manifest)
    deadline = time.time() + args.max_wall_hours * 3600.0
    if args.stage == "efimova-hore-24h":
        screen = efimova_hore_screen_conditions()
        complete = run_tasks(
            args.output_dir,
            screen,
            SCREEN_ANGLES_DEG,
            1,
            config,
            workers,
            deadline,
            estimated_wave_seconds,
        )
        confirmed: tuple[EfimovaHoreCondition, ...] = ()
        final: tuple[EfimovaHoreCondition, ...] = ()
        if complete:
            confirmed = select_efimova_hore_confirm_conditions(args.output_dir, config)
            _write_json(
                args.output_dir / "efimova_hore_confirmation_selection.json",
                [item.stable_dict(config.quadrupole_scale) for item in confirmed],
            )
            complete = run_tasks(
                args.output_dir,
                confirmed,
                SCREEN_ANGLES_DEG,
                4,
                config,
                workers,
                deadline,
                estimated_wave_seconds,
            )
        if complete:
            final = select_efimova_hore_final_conditions(
                args.output_dir, confirmed, config
            )
            _write_json(
                args.output_dir / "efimova_hore_final_selection.json",
                [item.stable_dict(config.quadrupole_scale) for item in final],
            )
            complete = run_tasks(
                args.output_dir,
                final,
                REFINED_ANGLES_DEG,
                4,
                config,
                workers,
                deadline,
                estimated_wave_seconds,
            )
        if complete:
            complete = run_tasks(
                args.output_dir,
                final,
                (0.0, 90.0),
                8,
                config,
                workers,
                deadline,
                estimated_wave_seconds,
            )
        create_efimova_hore_report(args.output_dir, config, confirmed, final)
        _mark_stage(args.output_dir, manifest, args.stage, complete)
        return
    if args.stage == "dipole-screen":
        complete = run_tasks(
            args.output_dir,
            dipole_screen_conditions(),
            SCREEN_ANGLES_DEG,
            1,
            config,
            workers,
            deadline,
            estimated_wave_seconds,
        )
    elif args.stage == "ratio-screen":
        complete = run_tasks(
            args.output_dir,
            ratio_screen_conditions(),
            SCREEN_ANGLES_DEG,
            1,
            config,
            workers,
            deadline,
            estimated_wave_seconds,
        )
    elif args.stage == "ratio-focus":
        selected = tuple(
            {condition.condition_id: condition for condition in map(ratio_condition, args.ratios)}.values()
        )
        _write_json(
            args.output_dir / "ratio_focus_selection.json",
            {
                "conditions": [item.stable_dict(config.quadrupole_scale) for item in selected],
                "angles_deg": SCREEN_ANGLES_DEG,
                "samples": args.samples,
                "quadrupole_scale": config.quadrupole_scale,
            },
        )
        complete = run_tasks(
            args.output_dir,
            selected,
            SCREEN_ANGLES_DEG,
            args.samples,
            config,
            workers,
            deadline,
            estimated_wave_seconds,
        )
        create_ratio_focus_report(
            args.output_dir, selected, SCREEN_ANGLES_DEG, args.samples, config
        )
        _mark_stage(args.output_dir, manifest, args.stage, complete)
        return
    elif args.stage == "ratio-lifetime-focus":
        selected = tuple(
            {
                condition.condition_id: condition
                for condition in map(ratio_condition, args.ratios)
            }.values()
        )
        _write_json(
            args.output_dir / "ratio_lifetime_selection.json",
            {
                "conditions": [item.stable_dict(config.quadrupole_scale) for item in selected],
                "angles_deg": SCREEN_ANGLES_DEG,
                "samples": args.samples,
                "lifetimes_us": lifetimes,
                "propagation_horizon_us": config.maximum_time_us,
                "quadrupole_scale": config.quadrupole_scale,
            },
        )
        complete = run_tasks(
            args.output_dir,
            selected,
            SCREEN_ANGLES_DEG,
            args.samples,
            config,
            workers,
            deadline,
            estimated_wave_seconds,
            lifetimes,
        )
        create_ratio_lifetime_report(
            args.output_dir,
            selected,
            SCREEN_ANGLES_DEG,
            args.samples,
            lifetimes,
            config,
        )
        _mark_stage(args.output_dir, manifest, args.stage, complete)
        return
    elif args.stage == "dipole-confirm":
        selected = select_dipole_confirm_conditions(args.output_dir, config)
        _write_json(args.output_dir / "dipole_confirmation_selection.json", [item.stable_dict(config.quadrupole_scale) for item in selected])
        complete = run_tasks(
            args.output_dir, selected, SCREEN_ANGLES_DEG, 4, config, workers, deadline, estimated_wave_seconds
        )
        if complete:
            nonzero = [condition for condition in selected if condition.d_parallel_uT != 0.0]
            best = _best_by_range(args.output_dir, nonzero, config)
            refined = (dipole_condition(0.0), best)
            _write_json(args.output_dir / "dipole_refinement_selection.json", [item.stable_dict(config.quadrupole_scale) for item in refined])
            complete = run_tasks(
                args.output_dir, refined, REFINED_ANGLES_DEG, 4, config, workers, deadline, estimated_wave_seconds
            )
    else:
        selected = select_ratio_confirm_conditions(args.output_dir, config)
        _write_json(args.output_dir / "ratio_confirmation_selection.json", [item.stable_dict(config.quadrupole_scale) for item in selected])
        complete = run_tasks(
            args.output_dir, selected, SCREEN_ANGLES_DEG, 4, config, workers, deadline, estimated_wave_seconds
        )
        if complete:
            negative = [condition for condition in selected if condition.ratio_D_over_J is not None and condition.ratio_D_over_J < 0]
            positive = [condition for condition in selected if condition.ratio_D_over_J is not None and condition.ratio_D_over_J > 0]
            refined = (
                low_exchange_condition(0.0),
                _best_by_range(args.output_dir, negative, config),
                _best_by_range(args.output_dir, positive, config),
            )
            _write_json(args.output_dir / "ratio_refinement_selection.json", [item.stable_dict(config.quadrupole_scale) for item in refined])
            complete = run_tasks(
                args.output_dir, refined, REFINED_ANGLES_DEG, 4, config, workers, deadline, estimated_wave_seconds
            )
    create_report(args.output_dir, config)
    _mark_stage(args.output_dir, manifest, args.stage, complete)


if __name__ == "__main__":
    main()
