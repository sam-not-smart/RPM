"""Refine Figure 1 of the transverse-versus-axial two-nucleus study.

The physics is deliberately identical to Experiment 1 in
``RPM_transverse_vs_axial_brief_v2``: two electrons, one spin-1 nucleus on
each radical, a 10-us radical-pair lifetime, equal singlet/triplet
recombination rates, and no exchange, electron dipolar coupling, relaxation,
or nuclear Zeeman interaction.  The only intervention removes the transverse
principal values of the N5-like tensor.

Unlike the historical factorized implementation, this runner constructs the
complete 36-dimensional radical-pair Hamiltonian.  It evaluates the equal-rate
singlet yield analytically in the Hamiltonian eigenbasis, so there is no time
integration or propagation error.  A dense angular grid locates candidate
maxima of |dPhi_S/dtheta| and a second local grid refines each candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from rpm_quantum_needle_model import GAMMA_E_MHZ_PER_MT, TWOPI, add_quadrupole_argument, spin_operators
from rpm_statevector_engine import dense_aligned_toy_quadrupole


DEFAULT_OUTPUT = Path("outputs") / "two_nucleus_regime_shift_refined"
LIFETIME_US = 10.0
K_RECOMBINATION_PER_US = 1.0 / LIFETIME_US
N5_TENSOR_MT = np.diag([-0.0989, -0.0989, 1.7569])
TRP_TENSOR_MT = np.diag([0.0, 0.0, 1.0812])
CONDITIONS = ("full", "transverse_ablated")
ORIGINAL_FIELDS_MT = (0.05, 0.20, 0.50, 0.60, 0.70, 1.00, 10.0, 100.0)


def _embed(operator: np.ndarray, position: int, dimensions: tuple[int, ...]) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for index, dimension in enumerate(dimensions):
        factor = operator if index == position else np.eye(dimension, dtype=complex)
        result = np.kron(result, factor)
    return result


DIMENSIONS = (2, 2, 3, 3)  # eA, eB, N5-like, Trp-N1-like
ELECTRON_A = tuple(_embed(op, 0, DIMENSIONS) for op in spin_operators(0.5))
ELECTRON_B = tuple(_embed(op, 1, DIMENSIONS) for op in spin_operators(0.5))
NUCLEUS_A = tuple(_embed(op, 2, DIMENSIONS) for op in spin_operators(1.0))
NUCLEUS_B = tuple(_embed(op, 3, DIMENSIONS) for op in spin_operators(1.0))
TOTAL_ELECTRON = tuple(a + b for a, b in zip(ELECTRON_A, ELECTRON_B))

_singlet_electrons = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
_singlet_projector_electrons = np.outer(_singlet_electrons, _singlet_electrons.conj())
SINGLET_PROJECTOR = np.kron(_singlet_projector_electrons, np.eye(9, dtype=complex))
NUCLEAR_DIMENSION = 9


def _hyperfine_hamiltonian(tensor_a_mT: np.ndarray) -> np.ndarray:
    result = np.zeros((36, 36), dtype=complex)
    for electron_axis in range(3):
        for nuclear_axis in range(3):
            result += (
                TWOPI
                * GAMMA_E_MHZ_PER_MT
                * tensor_a_mT[electron_axis, nuclear_axis]
                * (ELECTRON_A[electron_axis] @ NUCLEUS_A[nuclear_axis])
            )
            result += (
                TWOPI
                * GAMMA_E_MHZ_PER_MT
                * TRP_TENSOR_MT[electron_axis, nuclear_axis]
                * (ELECTRON_B[electron_axis] @ NUCLEUS_B[nuclear_axis])
            )
    return result


HYPERFINE = {
    "full": _hyperfine_hamiltonian(N5_TENSOR_MT),
    "transverse_ablated": _hyperfine_hamiltonian(np.diag([0.0, 0.0, N5_TENSOR_MT[2, 2]])),
}
QUADRUPOLE = dense_aligned_toy_quadrupole(DIMENSIONS, ("FAD:N5", "TrpH:N1"))


def hamiltonian(condition: str, field_mT: float, theta_deg: float,
                quadrupole_scale: float = 0.0) -> np.ndarray:
    theta = math.radians(theta_deg)
    zeeman = field_mT * (
        math.sin(theta) * TOTAL_ELECTRON[0] + math.cos(theta) * TOTAL_ELECTRON[2]
    )
    h = HYPERFINE[condition] + TWOPI * GAMMA_E_MHZ_PER_MT * zeeman
    return h if quadrupole_scale == 0 else h + quadrupole_scale * QUADRUPOLE


def exact_singlet_yield(condition: str, field_mT: float, theta_deg: float,
                       quadrupole_scale: float = 0.0) -> float:
    """Return Phi_S for equal first-order S/T recombination, exactly."""

    energies, vectors = scipy.linalg.eigh(
        hamiltonian(condition, field_mT, theta_deg, quadrupole_scale),
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    projected = vectors.conj().T @ (SINGLET_PROJECTOR @ vectors)
    gaps = energies[:, None] - energies[None, :]
    kernel = K_RECOMBINATION_PER_US**2 / (K_RECOMBINATION_PER_US**2 + gaps**2)
    return float(np.sum(np.abs(projected) ** 2 * kernel).real / NUCLEAR_DIMENSION)


def field_grid_mt() -> np.ndarray:
    """Broad log grid plus a 0.0025-mT grid through the crossover."""

    broad = np.geomspace(0.05, 100.0, 121)
    crossover = np.arange(0.40, 0.8000001, 0.0025)
    return np.unique(np.round(np.concatenate([broad, crossover, ORIGINAL_FIELDS_MT]), 10))


@dataclass(frozen=True)
class GridConfig:
    coarse_angle_step_deg: float = 0.01
    refine_angle_step_deg: float = 0.001
    refine_half_width_deg: float = 0.025
    candidate_count: int = 4
    quadrupole_scale: float = 0.0

    def stable_dict(self) -> dict[str, object]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "quadrupole_scale" or v != 0},
            "lifetime_us": LIFETIME_US,
            "n5_tensor_mT": N5_TENSOR_MT.tolist(),
            "trp_tensor_mT": TRP_TENSOR_MT.tolist(),
            "field_grid_mT": field_grid_mt().tolist(),
            "conditions": list(CONDITIONS),
            "hamiltonian_dimension": 36,
            "J_uT": 0.0,
            "D_uT": 0.0,
            "relaxation": False,
        }


def _top_candidates(theta: np.ndarray, sensitivity: np.ndarray, count: int) -> list[float]:
    order = np.argsort(sensitivity)[::-1]
    selected: list[float] = []
    for index in order:
        angle = float(theta[index])
        if all(abs(angle - previous) >= 0.15 for previous in selected):
            selected.append(angle)
        if len(selected) >= count:
            break
    return selected


def condition_metrics(
    condition: str, field_mT: float, config: GridConfig
) -> dict[str, object]:
    # Symmetry permits the diagnostic orientation to be reported on 0-90 deg,
    # matching the historical Figure 1 convention.
    theta = np.arange(0.0, 90.0 + 0.5 * config.coarse_angle_step_deg, config.coarse_angle_step_deg)
    yields = np.array([
        exact_singlet_yield(condition, field_mT, value, config.quadrupole_scale)
        for value in theta
    ])
    gradient = np.gradient(yields, theta, edge_order=2)
    candidates = _top_candidates(theta, np.abs(gradient), config.candidate_count)

    best_sensitivity = -math.inf
    best_angle = math.nan
    for candidate in candidates:
        lower = max(0.0, candidate - config.refine_half_width_deg)
        upper = min(90.0, candidate + config.refine_half_width_deg)
        fine_theta = np.arange(
            lower,
            upper + 0.5 * config.refine_angle_step_deg,
            config.refine_angle_step_deg,
        )
        fine_yields = np.array(
            [exact_singlet_yield(condition, field_mT, value, config.quadrupole_scale)
             for value in fine_theta]
        )
        fine_gradient = np.gradient(fine_yields, fine_theta, edge_order=2)
        index = int(np.argmax(np.abs(fine_gradient)))
        sensitivity = float(abs(fine_gradient[index]))
        if sensitivity > best_sensitivity:
            best_sensitivity = sensitivity
            best_angle = float(fine_theta[index])

    return {
        "condition": condition,
        "field_mT": float(field_mT),
        "max_abs_dphi_dtheta_per_deg": best_sensitivity,
        "theta_max_sensitivity_deg": best_angle,
        "coarse_theta_deg": theta,
        "coarse_phi_s": yields,
    }


def solve_field(field_mT: float, config: GridConfig) -> dict[str, object]:
    started = time.perf_counter()
    results = {condition: condition_metrics(condition, field_mT, config) for condition in CONDITIONS}
    full = results["full"]
    ablated = results["transverse_ablated"]
    gamma_full = float(full["max_abs_dphi_dtheta_per_deg"])
    gamma_ablated = float(ablated["max_abs_dphi_dtheta_per_deg"])
    return {
        "field_mT": float(field_mT),
        "full": full,
        "transverse_ablated": ablated,
        "ablation_fraction": 1.0 - gamma_ablated / gamma_full,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _checkpoint_path(output_dir: Path, field_mT: float) -> Path:
    label = f"{field_mT:010.6f}".replace(".", "p")
    return output_dir / "checkpoints" / f"field_{label}.npz"


def save_checkpoint(output_dir: Path, result: dict[str, object]) -> None:
    path = _checkpoint_path(output_dir, float(result["field_mT"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    full = result["full"]
    ablated = result["transverse_ablated"]
    np.savez_compressed(
        temporary,
        field_mT=result["field_mT"],
        ablation_fraction=result["ablation_fraction"],
        elapsed_seconds=result["elapsed_seconds"],
        theta_deg=full["coarse_theta_deg"],
        full_phi_s=full["coarse_phi_s"],
        ablated_phi_s=ablated["coarse_phi_s"],
        full_gamma=full["max_abs_dphi_dtheta_per_deg"],
        ablated_gamma=ablated["max_abs_dphi_dtheta_per_deg"],
        full_theta_max=full["theta_max_sensitivity_deg"],
        ablated_theta_max=ablated["theta_max_sensitivity_deg"],
    )
    temporary.replace(path)


def load_checkpoint(output_dir: Path, field_mT: float) -> dict[str, object]:
    with np.load(_checkpoint_path(output_dir, field_mT)) as data:
        return {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data.files}


def run_grid(output_dir: Path, config: GridConfig, workers: int, resume: bool) -> None:
    fields = field_grid_mt()
    pending = [
        float(field)
        for field in fields
        if not (resume and _checkpoint_path(output_dir, float(field)).exists())
    ]
    print(f"fields={len(fields)} pending={len(pending)} workers={workers}", flush=True)
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(solve_field, field, config): field for field in pending}
        completed = len(fields) - len(pending)
        for future in as_completed(futures):
            result = future.result()
            save_checkpoint(output_dir, result)
            completed += 1
            print(
                f"[{completed}/{len(fields)}] B={result['field_mT']:.6g} mT "
                f"f={result['ablation_fraction']:.6f} "
                f"theta={result['full']['theta_max_sensitivity_deg']:.4f} deg "
                f"({result['elapsed_seconds']:.1f} s)",
                flush=True,
            )


def write_outputs(output_dir: Path, config: GridConfig) -> None:
    fields = field_grid_mt()
    cells = [load_checkpoint(output_dir, float(field)) for field in fields]
    rows = [
        {
            "field_mT": float(cell["field_mT"]),
            "ablation_fraction": float(cell["ablation_fraction"]),
            "full_max_abs_dphi_dtheta_per_deg": float(cell["full_gamma"]),
            "ablated_max_abs_dphi_dtheta_per_deg": float(cell["ablated_gamma"]),
            "full_theta_max_sensitivity_deg": float(cell["full_theta_max"]),
            "ablated_theta_max_sensitivity_deg": float(cell["ablated_theta_max"]),
            "elapsed_seconds": float(cell["elapsed_seconds"]),
        }
        for cell in cells
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "regime_shift_refined_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "configuration": config.stable_dict(),
        "field_count": len(fields),
        "completed": len(cells),
        "mean_field_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])),
        "total_worker_seconds": float(np.sum([row["elapsed_seconds"] for row in rows])),
        "yield_method": "exact equal-rate eigenbasis integral on full 36-dimensional Hamiltonian",
    }
    field = np.array([row["field_mT"] for row in rows])
    fraction = np.array([row["ablation_fraction"] for row in rows])
    orientation = np.array([row["full_theta_max_sensitivity_deg"] for row in rows])
    zero_indices = np.flatnonzero(np.signbit(fraction[:-1]) != np.signbit(fraction[1:]))
    if zero_indices.size:
        zero_index = int(zero_indices[0])
        fraction_zero_mT = float(
            np.interp(
                0.0,
                fraction[zero_index : zero_index + 2][::-1],
                field[zero_index : zero_index + 2][::-1],
            )
        )
    else:
        fraction_zero_mT = math.nan
    jump_index = int(np.argmax(np.abs(np.diff(orientation))))
    orientation_jump_bracket_mT = [float(field[jump_index]), float(field[jump_index + 1])]
    transition_low_mT = min(fraction_zero_mT, orientation_jump_bracket_mT[0])
    transition_high_mT = max(fraction_zero_mT, orientation_jump_bracket_mT[1])
    transition_center_mT = 0.5 * (transition_low_mT + transition_high_mT)
    metadata["refined_transition"] = {
        "ablation_fraction_zero_mT": fraction_zero_mT,
        "orientation_jump_bracket_mT": orientation_jump_bracket_mT,
        "orientation_before_jump_deg": float(orientation[jump_index]),
        "orientation_after_jump_deg": float(orientation[jump_index + 1]),
    }
    (output_dir / "regime_shift_refined_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)
    axes[0].plot(field, fraction, color="#343434", linewidth=1.5)
    axes[0].scatter(field, fraction, color="#343434", s=8, zorder=3)
    axes[0].axhline(0.0, color="0.25", linestyle=":", linewidth=0.9)
    axes[0].set(
        xscale="log",
        xlabel=r"external field $B_0$ (mT)",
        ylabel="ablation fraction  f",
        title="(a) Transverse-hyperfine ablation",
    )
    axes[1].plot(field, orientation, color="#343434", linewidth=1.5)
    axes[1].scatter(field, orientation, color="#343434", s=8, zorder=3)
    axes[1].axhline(90.0, color="0.25", linestyle=":", linewidth=0.9)
    axes[1].set(
        xscale="log",
        xlabel=r"external field $B_0$ (mT)",
        ylabel=r"orientation of maximum sensitivity $\theta$ (degrees)",
        title="(b) Maximum-sensitivity orientation",
        ylim=(0.0, 92.0),
    )
    for axis in axes:
        axis.axvspan(0.05, transition_center_mT, color="#d9edf7", alpha=0.50, zorder=0)
        axis.axvspan(transition_center_mT, 100.0, color="#f8dddd", alpha=0.45, zorder=0)
        axis.axvspan(transition_low_mT, transition_high_mT, color="0.55", alpha=0.18, zorder=1)
        axis.axvline(transition_center_mT, color="0.15", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.17)
    label_style = {
        "transform": axes[0].get_xaxis_transform(),
        "ha": "center",
        "va": "bottom",
        "fontsize": 9,
        "fontweight": "semibold",
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 1.5},
        "zorder": 5,
    }
    axes[0].text(
        math.sqrt(0.05 * transition_center_mT),
        0.055,
        "transverse driven",
        color="#2879a9",
        **label_style,
    )
    axes[0].text(
        math.sqrt(transition_center_mT * 100.0),
        0.055,
        "axial driven",
        color="#cf3f52",
        **label_style,
    )
    fig.suptitle(
        r"Two-nucleus transverse-to-axial regime shift; $\tau=10\,\mu$s, $J=D=0$"
        + f", Q scale={config.quadrupole_scale:g}"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"regime_shift_refined_figure1.{suffix}", dpi=300)
    plt.close(fig)


def validate_exact_solver() -> None:
    for condition in CONDITIONS:
        value = exact_singlet_yield(condition, 0.05, 90.0)
        if not (0.0 <= value <= 1.0):
            raise AssertionError(f"unphysical singlet yield for {condition}: {value}")
    zero_field = exact_singlet_yield("full", 0.0, 37.0)
    if not (0.0 <= zero_field <= 1.0):
        raise AssertionError(f"unphysical zero-field yield: {zero_field}")
    if not np.allclose(HYPERFINE["full"], HYPERFINE["full"].conj().T, atol=1e-13):
        raise AssertionError("full Hamiltonian is not Hermitian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coarse-angle-step", type=float, default=0.01)
    parser.add_argument("--refine-angle-step", type=float, default=0.001)
    parser.add_argument("--refine-half-width", type=float, default=0.025)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    add_quadrupole_argument(parser, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GridConfig(
        coarse_angle_step_deg=args.coarse_angle_step,
        refine_angle_step_deg=args.refine_angle_step,
        refine_half_width_deg=args.refine_half_width,
        candidate_count=args.candidate_count,
        quadrupole_scale=args.quadrupole_scale,
    )
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    validate_exact_solver()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    stable = config.stable_dict()
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing != stable:
            raise ValueError("output manifest is incompatible with this configuration")
    else:
        manifest.write_text(json.dumps(stable, indent=2) + "\n")
    if not args.report_only:
        run_grid(args.output_dir, config, args.workers, args.resume)
    missing = [
        float(field)
        for field in field_grid_mt()
        if not _checkpoint_path(args.output_dir, float(field)).exists()
    ]
    if missing:
        print(f"incomplete: {len(missing)} fields remain", flush=True)
        return
    write_outputs(args.output_dir, config)
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
