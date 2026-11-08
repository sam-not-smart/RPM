"""Full-bath transverse-to-axial regime-shift discovery experiment.

The experiment compares the complete 14-nucleus production Hamiltonian with a
paired intervention in which only the transverse principal values of FAD N5
and N10 are set to zero.  It retains the Efimova-Hore D/J=-3 cancellation
condition, Q=0, a 5-us lifetime, and the production matrix-free Lanczos
propagator.  The M=1 discovery grid is deliberately multiresolution so it can
complete within a three-day Danny allocation.
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
from typing import Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from poster_reproduction.core import (
    MASTER_SEED,
    PRODUCTION_MODEL,
    ProductionModel,
    build_hamiltonian,
    dimensions_for,
    make_propagator,
    random_phase_initial_state,
)
from rpm_quantum_needle_jd_sweeps import SweepConfig, coarse_grid_indices, integrate_yield
from rpm_quantum_needle_model import ALL_NUCLEI, FAD_NUCLEI, NUCLEAR_RADICALS
from rpm_statevector_engine import SingletPopulationObservable, singlet_population


SCHEMA_VERSION = 1
EXPERIMENT_TITLE = (
    "Full-Bath Transverse-to-Axial Regime Shift Under Exchange-Dipolar Cancellation"
)
DEFAULT_OUTPUT = Path("outputs") / "transverse_axial_regime_shift_tau5"
FIELDS_MT = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.80,
    1.00,
    1.50,
    2.00,
    3.00,
)
EQUATORIAL_ANGLES_DEG = (84.0, 86.0, 88.0, 89.0, 90.0, 91.0, 92.0, 94.0, 96.0)
FULL_CURVE_FIELDS_MT = (0.05, 0.60, 3.00)
COARSE_ANGLES_DEG = tuple(float(value) for value in range(0, 181, 10))
CONDITIONS = ("full", "n5_n10_transverse_ablated")
ABLATION_TARGETS = ("FAD:N5", "FAD:N10")


def principal_axial_only_tensor(tensor_mT: np.ndarray) -> np.ndarray:
    """Retain the largest-magnitude principal value and delete the other two."""

    tensor = np.asarray(tensor_mT, dtype=float)
    symmetric = 0.5 * (tensor + tensor.T)
    values, vectors = np.linalg.eigh(symmetric)
    axial = int(np.argmax(np.abs(values)))
    retained = np.zeros(3, dtype=float)
    retained[axial] = values[axial]
    result = vectors @ np.diag(retained) @ vectors.T
    return 0.5 * (result + result.T)


def ablated_hyperfine_tensors_mT() -> dict[str, np.ndarray]:
    by_key = {
        f"{radical}:{nucleus.name}": nucleus
        for nucleus, radical in zip(ALL_NUCLEI, NUCLEAR_RADICALS)
    }
    return {
        key: principal_axial_only_tensor(by_key[key].tensor_mT)
        for key in ABLATION_TARGETS
    }


@dataclass(frozen=True)
class RegimeConfig:
    master_seed: int = MASTER_SEED
    lifetime_us: float = 5.0
    fields_mT: tuple[float, ...] = FIELDS_MT
    equatorial_angles_deg: tuple[float, ...] = EQUATORIAL_ANGLES_DEG
    full_curve_fields_mT: tuple[float, ...] = FULL_CURVE_FIELDS_MT
    coarse_angles_deg: tuple[float, ...] = COARSE_ANGLES_DEG
    samples: int = 1
    survival_tail_bound: float = 1.0e-4
    integration_step_us: float = 0.00125
    coarse_integration_step_us: float = 0.0025
    state_error_budget: float = 2.4e-5
    smoke: bool = False

    @property
    def model(self) -> ProductionModel:
        return replace(
            PRODUCTION_MODEL,
            master_seed=self.master_seed,
            samples=self.samples,
            lifetime_us=self.lifetime_us,
            quadrupole_scale=0.0,
            ratio_D_over_J=-3.0,
        )

    @property
    def sweep(self) -> SweepConfig:
        return replace(
            self.model.sweep_config,
            seed=self.master_seed,
            lifetime_us=self.lifetime_us,
            survival_tail_bound=self.survival_tail_bound,
            integration_step_us=self.integration_step_us,
            coarse_integration_step_us=self.coarse_integration_step_us,
            state_error_budget=self.state_error_budget,
        )

    @property
    def time_grid_us(self) -> np.ndarray:
        return self.sweep.time_grid_us

    def stable_dict(self) -> dict[str, object]:
        originals = {
            f"{radical}:{nucleus.name}": np.asarray(nucleus.tensor_mT).tolist()
            for nucleus, radical in zip(ALL_NUCLEI, NUCLEAR_RADICALS)
            if f"{radical}:{nucleus.name}" in ABLATION_TARGETS
        }
        return {
            **asdict(self),
            "schema_version": SCHEMA_VERSION,
            "experiment_title": EXPERIMENT_TITLE,
            "conditions": list(CONDITIONS),
            "ablation_targets": list(ABLATION_TARGETS),
            "ablation_definition": (
                "diagonalize each symmetric hyperfine tensor in its principal-axis frame; "
                "retain only the largest-magnitude principal value; rotate back"
            ),
            "original_target_tensors_mT": originals,
            "ablated_target_tensors_mT": {
                key: value.tolist() for key, value in ablated_hyperfine_tensors_mT().items()
            },
            "model": self.model.stable_dict(),
            "propagation": self.sweep.stable_dict(),
        }

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def smoke_config(seed: int = MASTER_SEED) -> RegimeConfig:
    return RegimeConfig(
        master_seed=seed,
        lifetime_us=0.02,
        fields_mT=(0.05,),
        equatorial_angles_deg=(89.0, 90.0),
        full_curve_fields_mT=(),
        coarse_angles_deg=(),
        samples=1,
        survival_tail_bound=math.exp(-1.0),
        integration_step_us=0.00125,
        coarse_integration_step_us=0.0025,
        state_error_budget=1.0e-6,
        smoke=True,
    )


@dataclass(frozen=True)
class RegimeTask:
    condition: str
    field_mT: float
    theta_deg: float
    sample_id: int = 0
    role: str = "equatorial"

    @property
    def key(self) -> str:
        field = f"{self.field_mT:08.3f}".replace(".", "p")
        theta = f"{self.theta_deg:07.3f}".replace(".", "p")
        return f"{self.condition}_B{field}_th{theta}_s{self.sample_id:03d}"


def discovery_tasks(config: RegimeConfig) -> list[RegimeTask]:
    roles: dict[tuple[str, float, float, int], set[str]] = {}
    for field in config.fields_mT:
        for angle in config.equatorial_angles_deg:
            for condition in CONDITIONS:
                roles.setdefault((condition, field, angle, 0), set()).add("equatorial")
    for field in config.full_curve_fields_mT:
        for angle in config.coarse_angles_deg:
            for condition in CONDITIONS:
                roles.setdefault((condition, field, angle, 0), set()).add("full_curve")
    tasks = [
        RegimeTask(condition, field, angle, sample, "+".join(sorted(task_roles)))
        for (condition, field, angle, sample), task_roles in roles.items()
    ]
    return sorted(
        tasks,
        key=lambda task: (task.field_mT, task.theta_deg, task.condition, task.sample_id),
    )


def _system(config: RegimeConfig):
    if config.smoke:
        return tuple(FAD_NUCLEI[:2]), ("FAD", "FAD")
    return tuple(ALL_NUCLEI), tuple(NUCLEAR_RADICALS)


def propagate_task(task: RegimeTask, config: RegimeConfig) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    nuclei, radicals = _system(config)
    dimensions = dimensions_for(nuclei)
    overrides = ablated_hyperfine_tensors_mT() if task.condition == CONDITIONS[1] else None
    if config.smoke and overrides is not None:
        overrides = {key: value for key, value in overrides.items() if key in ABLATION_TARGETS}
    hamiltonian = build_hamiltonian(
        task.theta_deg,
        task.field_mT,
        model=config.model,
        nuclei=nuclei,
        radicals=radicals,
        hyperfine_overrides_mT=overrides,
    )
    initial = random_phase_initial_state(dimensions, config.master_seed, task.sample_id)
    initial_singlet = singlet_population(initial, dimensions)
    result = make_propagator(hamiltonian, config.model).propagate(
        initial,
        config.time_grid_us,
        SingletPopulationObservable(dimensions),
    )
    times = config.time_grid_us
    fine_yield = integrate_yield(times, result.observable_values, config.lifetime_us)
    coarse_ids = coarse_grid_indices(config.sweep)
    coarse_yield = integrate_yield(
        times[coarse_ids], result.observable_values[coarse_ids], config.lifetime_us
    )
    weights = np.exp(-times / config.lifetime_us) / config.lifetime_us
    propagation_bound = float(
        np.trapezoid(weights * result.observable_error_bounds, times)
    )
    retain = task.field_mT in config.full_curve_fields_mT and math.isclose(
        task.theta_deg, 90.0, abs_tol=1e-12
    )
    metadata = {
        "task": asdict(task),
        "configuration_hash": config.configuration_hash,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {
        "yield": np.array(fine_yield),
        "coarse_yield": np.array(coarse_yield),
        "quadrature_error": np.array(abs(fine_yield - coarse_yield)),
        "yield_propagation_bound": np.array(propagation_bound),
        "yield_tail_bound": np.array(config.survival_tail_bound),
        "initial_singlet": np.array(initial_singlet),
        "times_us": times if retain else np.array([], dtype=float),
        "singlet_population": result.observable_values if retain else np.array([], dtype=float),
        "population_error_bound": (
            result.observable_error_bounds if retain else np.array([], dtype=float)
        ),
        "accepted_steps": np.array(result.stats.accepted_steps),
        "rejected_steps": np.array(result.stats.rejected_steps),
        "matvecs": np.array(result.stats.matvecs),
        "maximum_krylov_dimension": np.array(result.stats.maximum_krylov_dimension),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }


def cell_path(output_dir: Path, task: RegimeTask) -> Path:
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


def ensure_manifest(output_dir: Path, config: RegimeConfig) -> None:
    path = output_dir / "manifest.json"
    condition = config.model.condition
    payload = {
        "experiment": EXPERIMENT_TITLE,
        "configuration_hash": config.configuration_hash,
        "configuration": config.stable_dict(),
        "git_commit": _git_commit(),
        "created_host": socket.gethostname(),
        "python": platform.python_version(),
        "physics_summary": {
            "nuclei": 14 if not config.smoke else 2,
            "state_dimension": math.prod(dimensions_for(_system(config)[0])),
            "lifetime_us": config.lifetime_us,
            "quadrupole_scale": config.model.quadrupole_scale,
            "ratio_D_over_J": config.model.ratio_D_over_J,
            "d_paper_uT": condition.d_paper_uT,
            "exchange_J_uT": condition.exchange_J_uT,
            "factorized_evolution": False,
            "samples": config.samples,
        },
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("configuration_hash") != config.configuration_hash:
            raise ValueError(f"Incompatible resume manifest: {path}")
        return
    atomic_json(path, payload)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_inventory(output_dir: Path, tasks: Sequence[RegimeTask]) -> None:
    _write_csv(
        output_dir / "task_inventory.csv",
        [
            {
                **asdict(task),
                "key": task.key,
                "checkpoint": str(cell_path(output_dir, task)),
            }
            for task in tasks
        ],
    )


def write_raw_results(output_dir: Path, tasks: Sequence[RegimeTask]) -> None:
    rows: list[dict[str, object]] = []
    for task in tasks:
        path = cell_path(output_dir, task)
        if not path.exists():
            continue
        with np.load(path) as cell:
            metadata = json.loads(str(cell["metadata_json"]))
            rows.append(
                {
                    **asdict(task),
                    "key": task.key,
                    "yield": float(cell["yield"]),
                    "coarse_yield": float(cell["coarse_yield"]),
                    "quadrature_error": float(cell["quadrature_error"]),
                    "yield_propagation_bound": float(cell["yield_propagation_bound"]),
                    "yield_tail_bound": float(cell["yield_tail_bound"]),
                    "initial_singlet": float(cell["initial_singlet"]),
                    "accepted_steps": int(cell["accepted_steps"]),
                    "rejected_steps": int(cell["rejected_steps"]),
                    "matvecs": int(cell["matvecs"]),
                    "maximum_krylov_dimension": int(cell["maximum_krylov_dimension"]),
                    "elapsed_seconds": float(metadata["elapsed_seconds"]),
                    "host": metadata["host"],
                    "pid": metadata["pid"],
                }
            )
    _write_csv(output_dir / "raw_cell_yields.csv", rows)


def _worker(task: RegimeTask, config: RegimeConfig):
    return task, propagate_task(task, config)


def run_tasks(
    tasks: Sequence[RegimeTask],
    output_dir: Path,
    config: RegimeConfig,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    start_guard_seconds: float,
) -> dict[str, object]:
    pending: list[RegimeTask] = []
    for task in tasks:
        path = cell_path(output_dir, task)
        if path.exists():
            if resume:
                continue
            raise FileExistsError(f"Checkpoint exists; pass --resume: {path}")
        pending.append(task)
    completed = len(tasks) - len(pending)
    skipped = completed
    started = time.monotonic()
    active: dict[object, RegimeTask] = {}
    durations: list[float] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            while (
                pending
                and len(active) < workers
                and time.monotonic() + start_guard_seconds < deadline
            ):
                task = pending.pop(0)
                active[pool.submit(_worker, task, config)] = task
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                returned, values = future.result()
                if returned != task:
                    raise RuntimeError("Worker returned a mismatched task.")
                atomic_npz(cell_path(output_dir, task), values)
                completed += 1
                metadata = json.loads(str(values["metadata_json"]))
                durations.append(float(metadata["elapsed_seconds"]))
                status = {
                    "status": "running",
                    "planned": len(tasks),
                    "completed": completed,
                    "skipped": skipped,
                    "remaining_unstarted": len(pending),
                    "active": len(active),
                    "elapsed_wall_seconds": time.monotonic() - started,
                    "median_completed_cell_seconds": float(np.median(durations)),
                    "updated_unix_seconds": time.time(),
                }
                atomic_json(output_dir / "status.json", status)
                print(
                    f"completed {completed}/{len(tasks)} {task.key} "
                    f"yield={float(values['yield']):.9f} "
                    f"elapsed={metadata['elapsed_seconds']/3600.0:.2f} h",
                    flush=True,
                )
    final = {
        "status": "complete" if completed == len(tasks) else "wall_time_checkpoint",
        "planned": len(tasks),
        "completed": completed,
        "skipped": skipped,
        "remaining_unstarted": len(pending),
        "active": 0,
        "elapsed_wall_seconds": time.monotonic() - started,
        "median_completed_cell_seconds": (
            float(np.median(durations)) if durations else None
        ),
        "updated_unix_seconds": time.time(),
    }
    atomic_json(output_dir / "status.json", final)
    return final


def _load_yield(output_dir: Path, task: RegimeTask) -> float:
    with np.load(cell_path(output_dir, task)) as cell:
        return float(cell["yield"])


def equatorial_metrics(theta: np.ndarray, values: np.ndarray) -> dict[str, float]:
    derivative = np.gradient(values, theta, edge_order=2 if theta.size >= 3 else 1)
    center = int(np.argmin(np.abs(theta - 90.0)))
    edge_reference = 0.5 * (values[0] + values[-1])
    return {
        "local_range": float(np.ptp(values)),
        "maximum_abs_slope_per_deg": float(np.max(np.abs(derivative))),
        "theta_maximum_abs_slope_deg": float(theta[int(np.argmax(np.abs(derivative)))]),
        "yield_at_90": float(values[center]),
        "signed_center_feature": float(values[center] - edge_reference),
        "theta_minimum_deg": float(theta[int(np.argmin(values))]),
        "theta_maximum_deg": float(theta[int(np.argmax(values))]),
    }


def generate_report(output_dir: Path, config: RegimeConfig) -> None:
    tasks = discovery_tasks(config)
    missing = [task.key for task in tasks if not cell_path(output_dir, task).exists()]
    if missing:
        raise RuntimeError(f"Discovery run is incomplete ({len(missing)} cells missing).")
    theta = np.asarray(config.equatorial_angles_deg)
    curve_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    by_field_condition: dict[tuple[float, str], np.ndarray] = {}
    for field in config.fields_mT:
        for condition in CONDITIONS:
            values = np.array(
                [
                    _load_yield(output_dir, RegimeTask(condition, field, angle))
                    for angle in theta
                ]
            )
            by_field_condition[(field, condition)] = values
            metrics = equatorial_metrics(theta, values)
            metric_rows.append({"field_mT": field, "condition": condition, **metrics})
            for angle, value in zip(theta, values):
                curve_rows.append(
                    {
                        "field_mT": field,
                        "condition": condition,
                        "theta_deg": angle,
                        "sample_id": 0,
                        "yield": value,
                    }
                )
    metric_lookup = {
        (float(row["field_mT"]), str(row["condition"])): row for row in metric_rows
    }
    contrast_rows = []
    for field in config.fields_mT:
        full = metric_lookup[(field, CONDITIONS[0])]
        ablated = metric_lookup[(field, CONDITIONS[1])]
        gamma_full = float(full["maximum_abs_slope_per_deg"])
        gamma_ablated = float(ablated["maximum_abs_slope_per_deg"])
        contrast_rows.append(
            {
                "field_mT": field,
                "gamma_equatorial_full": gamma_full,
                "gamma_equatorial_ablated": gamma_ablated,
                "f_FAD": 1.0 - gamma_ablated / max(gamma_full, 1e-30),
                "full_signed_center_feature": full["signed_center_feature"],
                "ablated_signed_center_feature": ablated["signed_center_feature"],
                "samples": 1,
                "inferential_status": "discovery_only_no_sampling_interval",
            }
        )
    _write_csv(output_dir / "equatorial_curves.csv", curve_rows)
    _write_csv(output_dir / "equatorial_metrics.csv", metric_rows)
    _write_csv(output_dir / "fad_ablation_fraction.csv", contrast_rows)

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    fields = np.asarray(config.fields_mT)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True)
    axes[0].plot(fields, [row["f_FAD"] for row in contrast_rows], marker="o")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("FAD transverse-ablation fraction")
    axes[1].plot(
        fields,
        [row["gamma_equatorial_full"] for row in contrast_rows],
        marker="o",
        label="full",
    )
    axes[1].plot(
        fields,
        [row["gamma_equatorial_ablated"] for row in contrast_rows],
        marker="s",
        label="N5+N10 transverse-ablated",
    )
    axes[1].set(xlabel="field magnitude B0 (mT)", ylabel="maximum equatorial |dPhiS/dtheta|")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
    fig.suptitle(f"{EXPERIMENT_TITLE}\nM=1 discovery; tau=5 us; Q=0")
    fig.tight_layout()
    fig.savefig(figures / "regime_shift_discovery_metrics.png", dpi=220)
    plt.close(fig)

    if config.full_curve_fields_mT:
        fig, axes = plt.subplots(
            len(config.full_curve_fields_mT), 1, figsize=(7.4, 9.0), sharex=True
        )
        axes = np.atleast_1d(axes)
        for axis, field in zip(axes, config.full_curve_fields_mT):
            for condition in CONDITIONS:
                angles = sorted(
                    {
                        *config.coarse_angles_deg,
                        *config.equatorial_angles_deg,
                    }
                )
                values = [
                    _load_yield(output_dir, RegimeTask(condition, field, angle))
                    for angle in angles
                ]
                axis.plot(
                    angles,
                    values,
                    marker="o",
                    markersize=3,
                    label=condition.replace("_", " "),
                )
            axis.set_ylabel("singlet yield")
            axis.set_title(f"B0={field:g} mT")
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("field orientation theta (degrees)")
        axes[0].legend(frameon=False)
        fig.suptitle("Representative full angular curves (M=1 discovery)")
        fig.tight_layout()
        fig.savefig(figures / "regime_shift_representative_curves.png", dpi=220)
        plt.close(fig)

    (output_dir / "report.md").write_text(
        f"""# {EXPERIMENT_TITLE}

Discovery sweep complete for the complete 14-nucleus nonfactorized model.

- Lifetime: **5 us**
- D/J: **-3** in the Efimova-Hore convention
- Quadrupole scale: **Q=0**
- Nuclear samples: **M=1**, paired across conditions and angles
- Statistical status: **discovery only**; M=1 does not support a sampling confidence interval
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "discovery", "report"), default="discovery")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--max-wall-hours", type=float, default=72.0)
    parser.add_argument("--start-guard-hours", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.max_wall_hours <= 0.0 or args.start_guard_hours < 0.0:
        raise ValueError("Wall-time values must be nonnegative, with a positive maximum.")
    config = smoke_config(args.seed) if args.stage == "smoke" else RegimeConfig(args.seed)
    ensure_manifest(args.output_dir, config)
    tasks = discovery_tasks(config)
    write_inventory(args.output_dir, tasks)
    if args.stage != "report":
        status = run_tasks(
            tasks,
            args.output_dir,
            config,
            workers=min(args.workers, len(tasks)),
            resume=args.resume,
            deadline=time.monotonic() + args.max_wall_hours * 3600.0,
            start_guard_seconds=args.start_guard_hours * 3600.0,
        )
        write_raw_results(args.output_dir, tasks)
        if status["status"] != "complete":
            print("Discovery run checkpointed at the wall-time gate; resume to continue.", flush=True)
            return
    generate_report(args.output_dir, config)


if __name__ == "__main__":
    main()
