"""Checkpointed runner for the multisample APS-poster experiments.

Examples
--------
Run a plumbing smoke test on Danny::

    python -m poster_reproduction.runner --stage smoke --workers 4 --resume

Run one resumable shard of the pilot::

    python -m poster_reproduction.runner --stage pilot --workers 32 \
        --shard-index 0 --shard-count 1 --resume

The runner never reduces the requested samples or angular grid.  It stops
starting cells at the wall-time guard and can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Iterable

import numpy as np

from rpm_quantum_needle_model import ALL_NUCLEI, SYSTEM_DIMENSIONS
from rpm_statevector_engine import SingletPopulationObservable, singlet_population

from .core import (
    BOOTSTRAP_REPLICATES,
    ENTANGLEMENT_NUCLEAR_COUNTS,
    MASTER_SEED,
    N10_PERIODS_US,
    N10_THETA_DEG,
    OVERHAUSER_FIELDS_MT,
    POSTER_FIELDS_MT,
    POSTER_THETA_DEG,
    PRODUCTION_MODEL,
    ProductionModel,
    atomic_json,
    atomwise_overhauser_sample_mhz,
    build_hamiltonian,
    dimensions_for,
    full_product_basis_indices,
    local_unitary_apply,
    make_propagator,
    mixed_two_electron_log_negativity,
    model_manifest,
    n10_principal_axis,
    n10_transverse_deleted_tensor,
    overhauser_sample_mhz,
    overhauser_yield,
    paired_product_indices,
    product_basis_initial_state,
    pure_log_negativity,
    random_phase_initial_state,
    reduced_electron_density,
    sample_seed,
    selected_nuclei,
    selected_radicals,
    spin_pi_pulse,
)


for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"


OUTPUT_SCHEMA = 1
DEFAULT_OUTPUT = Path("outputs") / "poster_reproduction_multisample"
RECOVERED_POSTER_SETTINGS = """# Poster setting recovery

The original poster-era scripts were recovered from local Codex worktrees and
inspected together with `APS Poster Draft.pptx`.  They are evidence for the
experiment definitions only; their obsolete factorized Hamiltonian is not used.

## Recovered without fallback

- Figure 2: molecular `zx` field plane, 0--180 degrees in 10-degree steps,
  twelve field strengths from 0.05 to 400 mT, and the 400 mT boundary label.
- Figure 3: 0.05, 2, and 20 mT; 0--180 degrees in 10-degree steps; representative
  dynamics at 0, 90, and 180 degrees.
- Figure 4: paired strongest-to-weakest growth at N=4,6,...,14; 50 microtesla,
  theta=90 degrees, and a 0--2 microsecond, 101-point unitary time window.
- Figure 5: 50 microtesla, a 10-microsecond displayed control window, one pulse at
  5 microseconds, pulse intervals 0.02 and 0.05 microseconds, and conditionally
  normalized truncated yield `Phi_trunc/(1-exp(-T/tau))`.  The old transverse
  deletion zeroed the molecular-frame N10 hyperfine tensor columns for Ix and Iy.

## Intentional current-model deviations

- All quantum runs use the complete nonfactorized state-vector Hamiltonian.
- Electron coupling is the current Efimova--Hore D/J=-3 condition at r=2 nm.
- The current validated recombination lifetime is 5 microseconds and the yield
  horizon is selected from the 1e-4 survival-tail rule, rather than the old
  poster's short fixed propagation window.
- Periodic pulses occur at `t=n*tau`, as requested; the old code centered them at
  `(n+1/2)*tau`.
- Nuclear yields use paired random-phase trace samples.  Entanglement uses paired
  computational-basis product-state samples to avoid artificial initial bath
  entanglement.
- Quadrupole operators remain implemented, but the frozen validated production
  setting is `quadrupole_scale=0`.

## Production sample count

No 80-sample current-production manifest was found.  The latest validated full
14-nucleus runs used smaller M=4/M=8/M=16 campaigns.  This experiment therefore
uses M=16 as its unreduced production default and requires the pilot convergence
report before scientific interpretation.
"""


@dataclass(frozen=True)
class CellTask:
    kind: str
    key: str
    field_mT: float = 0.05
    theta_deg: float = 90.0
    sample_id: int = 0
    nuclear_count: int = 14
    protocol: str = "baseline"
    overhauser_samples: int = 4096
    master_seed: int = MASTER_SEED
    samples: int = 16


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in values.items()})
    os.replace(temporary, path)


def _metadata(task: CellTask, model: ProductionModel, elapsed: float) -> str:
    return json.dumps(
        {
            "task": asdict(task),
            "model_hash": model.configuration_hash,
            "elapsed_seconds": elapsed,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        },
        sort_keys=True,
    )


def _quantum_cell(task: CellTask, model: ProductionModel) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    overrides = None
    if task.protocol == "n10_transverse_deleted":
        overrides = {"FAD:N10": n10_transverse_deleted_tensor()}
    hamiltonian = build_hamiltonian(
        task.theta_deg, task.field_mT, model=model, hyperfine_overrides_mT=overrides
    )
    initial = random_phase_initial_state(SYSTEM_DIMENSIONS, model.master_seed, task.sample_id)
    result = make_propagator(hamiltonian, model).propagate(
        initial,
        model.sweep_config.time_grid_us,
        SingletPopulationObservable(SYSTEM_DIMENSIONS),
    )
    times = result.times_us
    populations = result.observable_values
    from rpm_quantum_needle_jd_sweeps import coarse_grid_indices, integrate_yield

    fine = integrate_yield(times, populations, model.lifetime_us)
    coarse_ids = coarse_grid_indices(model.sweep_config)
    coarse = integrate_yield(times[coarse_ids], populations[coarse_ids], model.lifetime_us)
    retain_dynamics = task.field_mT in OVERHAUSER_FIELDS_MT
    return {
        "yield": np.array(fine),
        "coarse_yield": np.array(coarse),
        "quadrature_error": np.array(abs(fine - coarse)),
        "initial_singlet": np.array(singlet_population(initial, SYSTEM_DIMENSIONS)),
        "times_us": times if retain_dynamics else np.array([], dtype=float),
        "singlet_population": populations if retain_dynamics else np.array([], dtype=float),
        "population_error_bound": result.observable_error_bounds if retain_dynamics else np.array([], dtype=float),
        "propagation_stats_json": np.array(json.dumps(asdict(result.stats), sort_keys=True)),
        "metadata_json": np.array(_metadata(task, model, time.perf_counter() - started)),
    }


def _overhauser_cell(task: CellTask, model: ProductionModel) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    yields = np.empty(task.overhauser_samples, dtype=float)
    retained = task.theta_deg in (0.0, 90.0, 180.0)
    dynamics_times = np.linspace(0.0, model.sweep_config.maximum_time_us, 1001)
    dynamics = np.empty((task.overhauser_samples, dynamics_times.size), dtype=float) if retained else None
    atomwise_yields = np.empty(task.overhauser_samples, dtype=float)
    for sample_id in range(task.overhauser_samples):
        local = overhauser_sample_mhz(task.master_seed, sample_id)
        yields[sample_id], population = overhauser_yield(
            task.theta_deg,
            task.field_mT,
            local,
            model,
            return_dynamics=retained,
            dynamics_times_us=dynamics_times,
        )
        if retained:
            assert dynamics is not None and population is not None
            dynamics[sample_id] = population
        atomwise = atomwise_overhauser_sample_mhz(task.master_seed, sample_id)
        atomwise_yields[sample_id], _ = overhauser_yield(
            task.theta_deg, task.field_mT, atomwise, model, return_dynamics=False
        )
    return {
        "sample_yields": yields,
        "atomwise_sample_yields": atomwise_yields,
        "times_us": dynamics_times if retained else np.array([], dtype=float),
        "singlet_population_samples": dynamics if retained else np.empty((0, 0)),
        "metadata_json": np.array(_metadata(task, model, time.perf_counter() - started)),
    }


def _entanglement_cell(task: CellTask, model: ProductionModel) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    nuclei = selected_nuclei(task.nuclear_count)
    radicals = selected_radicals(task.nuclear_count)
    dimensions = dimensions_for(nuclei)
    hamiltonian = build_hamiltonian(0.0 + task.theta_deg, task.field_mT, model=model, nuclei=nuclei, radicals=radicals)
    full_indices = full_product_basis_indices(model.master_seed, task.sample_id)
    indices = paired_product_indices(task.nuclear_count, full_indices)
    initial = product_basis_initial_state(dimensions, indices)
    times = np.linspace(0.0, 2.0, 101)
    result = make_propagator(hamiltonian, model).propagate(
        initial, times, lambda state: singlet_population(state, dimensions), store_states=True
    )
    assert result.states is not None
    per_radical = task.nuclear_count // 2
    radical_left = (0,) + tuple(range(2, 2 + per_radical))
    radical = np.array([pure_log_negativity(state, dimensions, radical_left) for state in result.states])
    electron_nuclear = np.array([pure_log_negativity(state, dimensions, (0, 1)) for state in result.states])
    electron_rho = np.array([reduced_electron_density(state, dimensions) for state in result.states])
    return {
        "times_us": times,
        "radical_log_negativity": radical,
        "electron_nuclear_log_negativity": electron_nuclear,
        "electron_density_real": electron_rho.real,
        "electron_density_imag": electron_rho.imag,
        "initial_radical_log_negativity": np.array(radical[0]),
        "initial_electron_nuclear_log_negativity": np.array(electron_nuclear[0]),
        "propagation_stats_json": np.array(json.dumps(asdict(result.stats), sort_keys=True)),
        "metadata_json": np.array(_metadata(task, model, time.perf_counter() - started)),
    }


def _pulse_schedule(protocol: str, final_time_us: float) -> np.ndarray:
    if protocol == "one_pulse":
        # Recovered poster setting: the single pulse was fixed at 5 us.
        return np.array([5.0]) if final_time_us > 5.0 else np.array([0.5 * final_time_us])
    if protocol.startswith("periodic_"):
        period = float(protocol.split("_", 1)[1])
        return np.arange(period, final_time_us, period)
    return np.array([], dtype=float)


def _pulse_propagate(
    task: CellTask,
    model: ProductionModel,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    overrides = {"FAD:N10": n10_transverse_deleted_tensor()} if task.protocol == "n10_transverse_deleted" else None
    hamiltonian = build_hamiltonian(task.theta_deg, task.field_mT, model=model, hyperfine_overrides_mT=overrides)
    dimensions = SYSTEM_DIMENSIONS
    state = random_phase_initial_state(dimensions, model.master_seed, task.sample_id)
    times = model.sweep_config.time_grid_us
    pulse_times = _pulse_schedule(task.protocol, float(times[-1]))
    pulse = spin_pi_pulse(1.0, n10_principal_axis())
    populations = np.empty(times.size, dtype=float)
    populations[0] = singlet_population(state, dimensions)
    cursor = 1
    current = 0.0
    accepted = rejected = matvecs = 0
    for boundary in np.append(pulse_times, times[-1]):
        boundary = float(boundary)
        stop = int(np.searchsorted(times, boundary + 1e-12, side="right"))
        requested = times[cursor:stop] - current
        duration = boundary - current
        if requested.size == 0 or requested[-1] < duration - 1e-12:
            local_times = np.append(requested, duration)
            endpoint_only = True
        else:
            local_times = requested
            endpoint_only = False
        if duration > 0.0:
            segment_model = replace(
                model,
            )
            # Allocate the global propagation-error budget in proportion to the segment.
            segment_config = replace(
                model.sweep_config,
                state_error_budget=model.sweep_config.state_error_budget * duration / times[-1],
            )
            propagator = make_propagator(hamiltonian, replace(segment_model))
            propagator.state_error_budget = segment_config.state_error_budget
            result = propagator.propagate(
                state, local_times, SingletPopulationObservable(dimensions)
            )
            count = requested.size
            populations[cursor:stop] = result.observable_values[:count]
            state = result.final_state
            accepted += result.stats.accepted_steps
            rejected += result.stats.rejected_steps
            matvecs += result.stats.matvecs
        cursor = stop
        current = boundary
        if boundary < times[-1] - 1e-12:
            state = local_unitary_apply(state, dimensions, 3, pulse)  # axes: eA,eB,N5,N10,...
    return times, populations, {"accepted_steps": accepted, "rejected_steps": rejected, "matvecs": matvecs}


def _n10_cell(task: CellTask, model: ProductionModel) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    times, populations, stats = _pulse_propagate(task, model)
    from rpm_quantum_needle_jd_sweeps import integrate_yield

    truncated = integrate_yield(times, populations, model.lifetime_us)
    recombined_weight = 1.0 - math.exp(-times[-1] / model.lifetime_us)
    retain = task.theta_deg in (89.0, 90.0, 91.0)
    return {
        "standard_truncated_yield": np.array(truncated),
        "conditional_normalized_yield": np.array(truncated / recombined_weight),
        "recombined_weight": np.array(recombined_weight),
        "times_us": times if retain else np.array([], dtype=float),
        "singlet_population": populations if retain else np.array([], dtype=float),
        "propagation_stats_json": np.array(json.dumps(stats, sort_keys=True)),
        "metadata_json": np.array(_metadata(task, model, time.perf_counter() - started)),
    }


def run_cell(task: CellTask) -> tuple[str, dict[str, np.ndarray]]:
    model = replace(PRODUCTION_MODEL, master_seed=task.master_seed, samples=task.samples)
    if task.kind == "quantum":
        result = _quantum_cell(task, model)
    elif task.kind == "overhauser":
        result = _overhauser_cell(task, model)
    elif task.kind == "entanglement":
        result = _entanglement_cell(task, model)
    elif task.kind == "n10":
        result = _n10_cell(task, model)
    else:
        raise ValueError(f"Unknown cell kind {task.kind!r}.")
    return task.key, result


def cell_path(output_dir: Path, task: CellTask) -> Path:
    return output_dir / "checkpoints" / task.kind / f"{task.key}.npz"


def _slug(value: float, digits: int = 3) -> str:
    return f"{float(value):0{digits + 4}.{digits}f}".replace(".", "p")


def planned_tasks(
    stage: str,
    samples: int,
    overhauser_samples: int,
    seed: int,
    figures: tuple[int, ...] = (2, 3, 4, 5),
) -> list[CellTask]:
    if stage == "smoke":
        fields = (0.05,)
        angles = (0.0, 90.0, 180.0)
        n10_angles = (89.0, 90.0, 91.0)
        counts = (4,)
    else:
        fields = POSTER_FIELDS_MT
        angles = POSTER_THETA_DEG
        n10_angles = N10_THETA_DEG
        counts = ENTANGLEMENT_NUCLEAR_COUNTS
    tasks: list[CellTask] = []
    if 2 in figures or 3 in figures:
        quantum_fields = fields if 2 in figures else tuple(value for value in OVERHAUSER_FIELDS_MT if value in fields)
        for field in quantum_fields:
            for theta in angles:
                for sample_id in range(samples):
                    key = f"B{_slug(field)}_th{_slug(theta)}_s{sample_id:03d}"
                    tasks.append(CellTask("quantum", key, field, theta, sample_id, master_seed=seed, samples=samples))
    # The three poster panels are a subset, but the error-vs-field result uses
    # the complete Figure 2 field grid.
    if 3 in figures:
        for field in fields:
            for theta in angles:
                key = f"B{_slug(field)}_th{_slug(theta)}_n{overhauser_samples:05d}"
                tasks.append(CellTask("overhauser", key, field, theta, overhauser_samples=overhauser_samples, master_seed=seed, samples=samples))
    if 4 in figures:
        for count in counts:
            for sample_id in range(samples):
                key = f"N{count:02d}_s{sample_id:03d}"
                tasks.append(CellTask("entanglement", key, 0.05, 90.0, sample_id, count, master_seed=seed, samples=samples))
    protocols = ("baseline", "n10_transverse_deleted", "one_pulse", "periodic_0.02", "periodic_0.05")
    if 5 in figures:
        for theta in n10_angles:
            for protocol in protocols:
                for sample_id in range(samples):
                    key = f"{protocol}_th{_slug(theta)}_s{sample_id:03d}"
                    tasks.append(CellTask("n10", key, 0.05, theta, sample_id, protocol=protocol, master_seed=seed, samples=samples))
    return tasks


def planned_refinement_tasks(output_dir: Path, samples: int, seed: int) -> list[CellTask]:
    """Plan 1-degree windows around coarse extrema and steepest slopes."""

    tasks: list[CellTask] = []
    theta = np.asarray(POSTER_THETA_DEG)
    for field in POSTER_FIELDS_MT:
        values = np.empty((samples, theta.size), dtype=float)
        for angle_index, angle in enumerate(theta):
            for sample_id in range(samples):
                task = CellTask(
                    "quantum",
                    f"B{_slug(field)}_th{_slug(angle)}_s{sample_id:03d}",
                    field,
                    float(angle),
                    sample_id,
                    master_seed=seed,
                    samples=samples,
                )
                values[sample_id, angle_index] = float(np.load(cell_path(output_dir, task))["yield"])
        mean = np.mean(values, axis=0)
        derivative = np.gradient(mean, theta, edge_order=2)
        centers = (
            theta[int(np.argmin(mean))],
            theta[int(np.argmax(mean))],
            theta[int(np.argmax(np.abs(derivative)))],
        )
        angles = sorted(
            {
                float(value)
                for center in centers
                for value in np.arange(max(0.0, center - 5.0), min(180.0, center + 5.0) + 0.5, 1.0)
            }
        )
        for angle in angles:
            if angle in POSTER_THETA_DEG:
                continue
            for sample_id in range(samples):
                key = f"B{_slug(field)}_th{_slug(angle)}_s{sample_id:03d}"
                tasks.append(CellTask("quantum", key, field, angle, sample_id, master_seed=seed, samples=samples))
    return tasks


def _write_task_inventory(path: Path, tasks: Iterable[CellTask]) -> None:
    rows = [asdict(task) for task in tasks]
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def ensure_manifests(
    output_dir: Path,
    model: ProductionModel,
    requested_samples: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_manifest.json"
    current = model_manifest(model)
    if model_path.exists():
        previous = json.loads(model_path.read_text(encoding="utf-8"))
        if previous.get("configuration_hash") != current["configuration_hash"]:
            raise RuntimeError("Existing output directory has an incompatible model manifest.")
    else:
        atomic_json(model_path, current)
    recovery = output_dir / "poster_setting_recovery.md"
    if not recovery.exists():
        recovery.write_text(RECOVERED_POSTER_SETTINGS, encoding="utf-8")
    run_path = output_dir / "run_manifest.json"
    manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "model_hash": model.configuration_hash,
        "stage": args.stage,
        "samples": requested_samples,
        "overhauser_samples": args.overhauser_samples,
        "seed": model.master_seed,
        "workers": args.workers,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_wall_hours": args.max_wall_hours,
        "figures": list(args.figures),
        "git_commit_at_launch": _git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "quadrupole_scale": model.quadrupole_scale,
        "status": "running",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if run_path.exists() and args.resume:
        old = json.loads(run_path.read_text(encoding="utf-8"))
        keys = ("model_hash", "samples", "overhauser_samples", "seed", "shard_count", "figures")
        same_stage = old.get("stage") == manifest["stage"]
        if same_stage and any(old.get(key) != manifest.get(key) for key in keys):
            raise RuntimeError("Resume rejected because the run manifest is incompatible.")
    atomic_json(output_dir / f"run_manifest_{args.stage}.json", manifest)
    atomic_json(run_path, manifest)
    return manifest


def execute(args: argparse.Namespace) -> None:
    samples = {
        "smoke": 4,
        "pilot": 16,
        "production": PRODUCTION_MODEL.samples,
        "refine": PRODUCTION_MODEL.samples,
    }[args.stage]
    if args.samples is not None:
        samples = args.samples
    # The model manifest always describes the unreduced production model.  Smoke
    # and pilot sample counts are execution-stage properties, not new physics.
    model = replace(PRODUCTION_MODEL, master_seed=args.seed)
    manifest = ensure_manifests(args.output_dir, model, samples, args)
    all_tasks = (
        planned_refinement_tasks(args.output_dir, samples, args.seed)
        if args.stage == "refine"
        else planned_tasks(
            args.stage,
            samples,
            args.overhauser_samples,
            args.seed,
            tuple(args.figures),
        )
    )
    tasks = [task for index, task in enumerate(all_tasks) if index % args.shard_count == args.shard_index]
    _write_task_inventory(args.output_dir / f"task_inventory_{args.stage}.csv", tasks)
    pending = [task for task in tasks if not cell_path(args.output_dir, task).exists()]
    completed_at_start = len(tasks) - len(pending)
    counts_by_kind = {
        kind: sum(task.kind == kind for task in tasks)
        for kind in ("quantum", "overhauser", "entanglement", "n10")
    }
    inventory_summary = {
        "stage": args.stage,
        "figures": list(args.figures),
        "samples": samples,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "total_cells": len(tasks),
        "completed_cells": completed_at_start,
        "pending_cells": len(pending),
        "counts_by_kind": counts_by_kind,
        "note": "N10 pulse cells are more expensive than pulse-free quantum cells.",
    }
    atomic_json(args.output_dir / f"task_inventory_{args.stage}.json", inventory_summary)
    if args.dry_run:
        manifest.update({"status": "dry_run", **inventory_summary})
        atomic_json(args.output_dir / "run_manifest.json", manifest)
        atomic_json(args.output_dir / f"run_manifest_{args.stage}.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    started = time.time()
    deadline = started + args.max_wall_hours * 3600.0
    durations: list[float] = []
    submitted = completed = 0
    inflight = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while pending or inflight:
            median = float(np.median(durations)) if durations else args.initial_cell_estimate_minutes * 60.0
            while pending and len(inflight) < args.workers and time.time() + 1.25 * median < deadline:
                task = pending.pop(0)
                future = pool.submit(run_cell, task)
                inflight[future] = (task, time.time())
                submitted += 1
            if not inflight:
                break
            done, _ = wait(tuple(inflight), timeout=30.0, return_when=FIRST_COMPLETED)
            for future in done:
                task, launched = inflight.pop(future)
                key, result = future.result()
                if key != task.key:
                    raise RuntimeError("Worker returned a mismatched checkpoint key.")
                _atomic_npz(cell_path(args.output_dir, task), **result)
                durations.append(time.time() - launched)
                completed += 1
                print(f"[{completed_at_start + completed}/{len(tasks)}] {task.kind}:{task.key}", flush=True)
    remaining = len([task for task in tasks if not cell_path(args.output_dir, task).exists()])
    elapsed = time.time() - started
    manifest.update(
        {
            "status": "complete" if remaining == 0 else "wall_time_checkpoint",
            "completed_cells": len(tasks) - remaining,
            "total_cells": len(tasks),
            "remaining_cells": remaining,
            "submitted_this_invocation": submitted,
            "elapsed_seconds_this_invocation": elapsed,
            "median_cell_seconds_this_invocation": float(np.median(durations)) if durations else None,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    if durations:
        manifest["projected_serial_cell_seconds"] = float(np.median(durations))
        manifest["projected_remaining_wall_hours"] = remaining * float(np.median(durations)) / args.workers / 3600.0
    atomic_json(args.output_dir / "run_manifest.json", manifest)
    atomic_json(args.output_dir / f"run_manifest_{args.stage}.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot", "production", "refine"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--overhauser-samples", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-wall-hours", type=float, default=23.5)
    parser.add_argument("--initial-cell-estimate-minutes", type=float, default=240.0)
    parser.add_argument("--dry-run", action="store_true", help="Write manifests and the exact task inventory without launching cells.")
    parser.add_argument(
        "--figures",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=(2, 3, 4, 5),
        help="Comma-separated poster figure numbers to schedule (default: 2,3,4,5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.overhauser_samples < 1:
        raise ValueError("Worker and sample counts must be positive.")
    if args.samples is not None and args.samples < 1:
        raise ValueError("--samples must be positive.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count.")
    if not args.figures or any(value not in (2, 3, 4, 5) for value in args.figures):
        raise ValueError("--figures may contain only 2,3,4,5.")
    if not 0.0 < args.max_wall_hours <= 23.5:
        raise ValueError("--max-wall-hours must be in (0, 23.5].")
    execute(args)


if __name__ == "__main__":
    main()
