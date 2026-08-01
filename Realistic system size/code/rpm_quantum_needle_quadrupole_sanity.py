"""Paired 0/90-degree sanity check for paper-based 14N quadrupole tensors.

The Q=0 trajectories are loaded from the completed validation checkpoints.
Only the two Q-on trajectories are propagated, using the identical random-phase
state, time grid, and Lanczos settings.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rpm_quantum_needle_model import (
    QUADRUPOLE_TENSORS_MT,
    SYSTEM_DIMENSIONS,
)
from rpm_quantum_needle_validation import (
    ValidationConfig,
    _propagator,
    _seed,
    choose_workload,
    integrate_singlet_yield,
    load_trajectory,
)
from rpm_statevector_engine import (
    build_quantum_needle_hamiltonian,
    singlet_initial_state,
    singlet_population,
)


ANGLES_DEG = (0.0, 90.0)


@dataclass(frozen=True)
class SanityTask:
    theta_deg: float
    sample_id: int
    config: ValidationConfig


def _run_q_on(task: SanityTask) -> dict[str, object]:
    started = time.perf_counter()
    hamiltonian = build_quantum_needle_hamiltonian(
        task.theta_deg,
        task.config.field_microT,
        exchange_J_MHz=0.0,
        dipolar_tensor_MHz=np.zeros((3, 3)),
        quadrupole_scale=1.0,
    )
    initial = singlet_initial_state(
        SYSTEM_DIMENSIONS,
        np.random.default_rng(_seed(task.config.seed, 0xA11CE, task.sample_id)),
    )
    result = _propagator(hamiltonian, task.config).propagate(
        initial,
        task.config.time_grid_us,
        lambda state: singlet_population(state, SYSTEM_DIMENSIONS),
    )
    return {
        "theta_deg": task.theta_deg,
        "sample_id": task.sample_id,
        "times_us": task.config.time_grid_us,
        "singlet_population": result.observable_values,
        "yield": integrate_singlet_yield(
            task.config.time_grid_us,
            result.observable_values,
            task.config.lifetime_us,
        ),
        "runtime_seconds": time.perf_counter() - started,
        "accepted_steps": result.stats.accepted_steps,
        "rejected_steps": result.stats.rejected_steps,
        "matvecs": result.stats.matvecs,
        "maximum_krylov_dimension": result.stats.maximum_krylov_dimension,
        "accumulated_state_error": result.stats.accumulated_state_error,
        "maximum_norm_error": result.stats.maximum_norm_error,
    }


def _load_config(validation_output: Path) -> tuple[ValidationConfig, dict[str, object]]:
    manifest = json.loads((validation_output / "manifest.json").read_text(encoding="utf-8"))
    throughput = float(manifest["measured_simulated_us_per_wall_second"])
    target = float(manifest["runtime_projection"]["target_hours"])
    config, _ = choose_workload(ValidationConfig(), throughput, target)
    if config.configuration_hash != manifest["configuration_hash"]:
        raise RuntimeError("The saved validation manifest does not match the reconstructed configuration.")
    return config, manifest


def _save_npz(path: Path, result: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in result.items()})
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run_sanity(
    validation_output: Path,
    output_dir: Path,
    *,
    sample_id: int = 0,
    workers: int = 2,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config, manifest = _load_config(validation_output)
    q_on_results: dict[float, dict[str, object]] = {}
    missing = []
    for theta in ANGLES_DEG:
        path = output_dir / f"q_on_theta_{theta:03.0f}_sample_{sample_id:03d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                q_on_results[theta] = {
                    key: data[key].item() if data[key].ndim == 0 else data[key].copy()
                    for key in data.files
                }
        else:
            missing.append(SanityTask(theta, sample_id, config))
    if missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as executor:
            for result in executor.map(_run_q_on, missing):
                theta = float(result["theta_deg"])
                q_on_results[theta] = result
                _save_npz(
                    output_dir / f"q_on_theta_{theta:03.0f}_sample_{sample_id:03d}.npz",
                    result,
                )

    rows = []
    q_off_yields: dict[float, float] = {}
    q_on_yields: dict[float, float] = {}
    traces: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for theta in ANGLES_DEG:
        q_off = load_trajectory(validation_output, theta, sample_id, config)
        q_on = q_on_results[theta]
        times = np.asarray(q_off["times_us"])
        if not np.array_equal(times, np.asarray(q_on["times_us"])):
            raise RuntimeError("Q-on and Q-off observation grids differ.")
        off_trace = np.asarray(q_off["singlet_population"])
        on_trace = np.asarray(q_on["singlet_population"])
        off_yield = integrate_singlet_yield(times, off_trace, config.lifetime_us)
        on_yield = float(q_on["yield"])
        detailed = times <= config.detailed_trace_us + 1e-12
        difference = on_trace[detailed] - off_trace[detailed]
        q_off_yields[theta] = off_yield
        q_on_yields[theta] = on_yield
        traces[theta] = (times, off_trace, on_trace)
        rows.append(
            {
                "theta_deg": theta,
                "sample_id": sample_id,
                "q_off_yield": off_yield,
                "q_on_yield": on_yield,
                "yield_shift": on_yield - off_yield,
                "trace_rms_change_0_to_5_us": float(np.sqrt(np.mean(difference**2))),
                "trace_max_change_0_to_5_us": float(np.max(np.abs(difference))),
                "q_on_p_s_0_error": abs(float(on_trace[0]) - 1.0),
                "q_on_runtime_seconds": float(q_on["runtime_seconds"]),
                "q_on_matvecs": int(q_on["matvecs"]),
                "q_on_accumulated_state_error": float(q_on["accumulated_state_error"]),
                "q_on_maximum_norm_error": float(q_on["maximum_norm_error"]),
            }
        )

    off_contrast = q_off_yields[0.0] - q_off_yields[90.0]
    on_contrast = q_on_yields[0.0] - q_on_yields[90.0]
    sign_preserved = bool(np.sign(on_contrast) == np.sign(off_contrast))
    contrast_change = on_contrast - off_contrast
    contrast_tolerance = max(0.5 * abs(off_contrast), 5e-4)
    passed = bool(
        sign_preserved
        and abs(contrast_change) <= contrast_tolerance
        and max(abs(float(row["yield_shift"])) for row in rows) <= 5e-3
        and max(float(row["trace_rms_change_0_to_5_us"]) for row in rows) <= 2e-2
        and max(float(row["q_on_p_s_0_error"]) for row in rows) <= 1e-12
    )
    summary = {
        "passed": passed,
        "sample_id": sample_id,
        "propagated_new_trajectories": len(missing),
        "q_off_source_commit": manifest["git_commit"],
        "lifetime_us": config.lifetime_us,
        "maximum_time_us": config.maximum_time_us,
        "q_off_contrast_phi_0_minus_phi_90": off_contrast,
        "q_on_contrast_phi_0_minus_phi_90": on_contrast,
        "contrast_change": contrast_change,
        "contrast_relative_change": contrast_change / off_contrast,
        "contrast_sign_preserved": sign_preserved,
        "contrast_acceptance_tolerance": contrast_tolerance,
        "angles": rows,
        "quadrupole_tensors_mT": {
            label: tensor.tolist() for label, tensor in QUADRUPOLE_TENSORS_MT.items()
        },
        "orientation_note": (
            "FAD:N5 uses the documented provisional in-plane azimuth; the coupling is nonzero."
        ),
    }
    _write_json(output_dir / "quadrupole_sanity_summary.json", summary)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for theta in ANGLES_DEG:
        times, off_trace, on_trace = traces[theta]
        detailed = times <= config.detailed_trace_us + 1e-12
        axes[0].plot(times[detailed], off_trace[detailed], linestyle="--", label=f"Q=0, {theta:g} deg")
        axes[0].plot(times[detailed], on_trace[detailed], label=f"Q on, {theta:g} deg")
    axes[0].set(xlabel="time (us)", ylabel="singlet population", title="Paired full-state dynamics")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].plot(ANGLES_DEG, [q_off_yields[value] for value in ANGLES_DEG], "o--", label="Q=0")
    axes[1].plot(ANGLES_DEG, [q_on_yields[value] for value in ANGLES_DEG], "o-", label="paper-based Q")
    axes[1].set(xlabel="field angle (degrees)", ylabel="singlet yield", title="Two-angle structure check")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "quadrupole_sanity_comparison.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("outputs") / "quantum_needle_fullspace_validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "quantum_needle_quadrupole_sanity",
    )
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.sample_id < 0 or args.workers < 1:
        raise ValueError("sample-id must be nonnegative and workers must be positive.")
    print(
        json.dumps(
            run_sanity(
                args.validation_output,
                args.output_dir,
                sample_id=args.sample_id,
                workers=args.workers,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
