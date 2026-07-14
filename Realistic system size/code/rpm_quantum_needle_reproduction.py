"""Reproduce Fig. 1B of Hiscock et al., PNAS 2016 (the quantum needle).

This implementation follows Sections S1 and S5 of the supporting information:

* FAD radical anion and TrpH radical cation, seven magnetic nuclei each;
* the published full hyperfine tensors in the FAD molecular frame;
* the physical nuclear spin quantum numbers (14N: I=1, 1H: I=1/2);
* a 50 microtesla field rotated in the flavin zx plane;
* a pure electronic singlet with an unpolarized nuclear ensemble;
* no electron-electron exchange/dipolar coupling and no spin relaxation; and
* equal singlet/triplet first-order recombination rates, k = 1/tau.

The two radicals are dynamically independent under those assumptions.  We use
the paper's radical-factorized electron-spin correlation formula, avoiding the
full 221,184-dimensional radical-pair Hilbert space.  Correlation spectra are
placed on a fine frequency grid and convolved by FFT.  ``--self-test`` checks
that numerical convolution against a direct line-by-line calculation for the
paper's two-nitrogen toy radical pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.fft import fft, ifft, next_fast_len

from rpm_quantum_needle_model import (
    DEFAULT_LIFETIMES_US,
    FAD_NUCLEI,
    GAMMA_E_MHZ_PER_MT,
    TRP_NUCLEI,
    TWOPI,
    Nucleus,
    field_vector_mT,
    spin_operators,
    theta_grid,
)


@dataclass
class RadicalModel:
    name: str
    nuclei: tuple[Nucleus, ...]
    h0: sp.csr_matrix
    electron_ops: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
    nuclear_dimension: int

    @property
    def dimension(self) -> int:
        return self.h0.shape[0]


@dataclass(frozen=True)
class Spectrum:
    eigenvalues: np.ndarray
    electron_ops_eigenbasis: tuple[np.ndarray, np.ndarray, np.ndarray]
    nuclear_dimension: int

    @property
    def max_gap(self) -> float:
        return float(np.ptp(self.eigenvalues))


def embed(operator: np.ndarray, position: int, dimensions: tuple[int, ...]) -> sp.csr_matrix:
    result = sp.csr_matrix([[1.0 + 0.0j]])
    for index, dimension in enumerate(dimensions):
        factor = sp.csr_matrix(operator) if index == position else sp.identity(dimension, dtype=complex, format="csr")
        result = sp.kron(result, factor, format="csr")
    return result


def build_radical(name: str, nuclei: Iterable[Nucleus]) -> RadicalModel:
    nuclei = tuple(nuclei)
    dimensions = (2, *(int(round(2.0 * nucleus.spin + 1.0)) for nucleus in nuclei))
    electron_ops = tuple(embed(operator, 0, dimensions) for operator in spin_operators(0.5))
    h0 = sp.csr_matrix((math.prod(dimensions), math.prod(dimensions)), dtype=complex)
    for nucleus_index, nucleus in enumerate(nuclei, start=1):
        nuclear_ops = tuple(embed(operator, nucleus_index, dimensions) for operator in spin_operators(nucleus.spin))
        for electron_axis in range(3):
            for nuclear_axis in range(3):
                coupling = nucleus.tensor_mT[electron_axis, nuclear_axis]
                if abs(coupling) > 1e-15:
                    h0 += (
                        TWOPI
                        * GAMMA_E_MHZ_PER_MT
                        * coupling
                        * (electron_ops[electron_axis] @ nuclear_ops[nuclear_axis])
                    )
    return RadicalModel(
        name=name,
        nuclei=nuclei,
        h0=h0.tocsr(),
        electron_ops=electron_ops,
        nuclear_dimension=math.prod(dimensions[1:]),
    )


def diagonalize_radical(model: RadicalModel, theta_deg: float, field_microT: float) -> Spectrum:
    field = field_vector_mT(theta_deg, field_microT)
    hamiltonian = model.h0.copy()
    for component, operator in zip(field, model.electron_ops):
        hamiltonian += TWOPI * GAMMA_E_MHZ_PER_MT * component * operator
    dense = hamiltonian.toarray()
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        dense,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    transformed = tuple(
        eigenvectors.conj().T @ (operator @ eigenvectors) for operator in model.electron_ops
    )
    return Spectrum(eigenvalues, transformed, model.nuclear_dimension)


def transition_lines(spectrum: Spectrum, p: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    frequencies = (spectrum.eigenvalues[:, None] - spectrum.eigenvalues[None, :]).reshape(-1)
    coefficients = (
        spectrum.electron_ops_eigenbasis[p]
        * spectrum.electron_ops_eigenbasis[q].T
        / spectrum.nuclear_dimension
    ).reshape(-1)
    return frequencies, coefficients


def binned_spectrum(
    frequencies: np.ndarray,
    coefficients: np.ndarray,
    delta_omega: float,
    half_width_bins: int,
) -> np.ndarray:
    """Deposit spectral lines with linear interpolation onto a centered grid."""
    position = frequencies / delta_omega + half_width_bins
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    size = 2 * half_width_bins + 1
    if lower.min() < 0 or lower.max() + 1 >= size:
        raise ValueError("Frequency grid is too small for transition spectrum.")

    result = np.zeros(size, dtype=complex)
    np.add.at(result, lower, coefficients * (1.0 - fraction))
    np.add.at(result, lower + 1, coefficients * fraction)
    return result


def convolved_correlation_spectrum(
    spectrum_a: Spectrum,
    spectrum_b: Spectrum,
    delta_omega: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    margin = 3
    half_a = int(np.ceil(spectrum_a.max_gap / delta_omega)) + margin
    half_b = int(np.ceil(spectrum_b.max_gap / delta_omega)) + margin
    length_a = 2 * half_a + 1
    length_b = 2 * half_b + 1
    output_length = length_a + length_b - 1
    fft_length = next_fast_len(output_length)
    combined_fft = np.zeros(fft_length, dtype=complex)

    for p in range(3):
        for q in range(3):
            freq_a, coeff_a = transition_lines(spectrum_a, p, q)
            freq_b, coeff_b = transition_lines(spectrum_b, p, q)
            grid_a = binned_spectrum(freq_a, coeff_a, delta_omega, half_a)
            grid_b = binned_spectrum(freq_b, coeff_b, delta_omega, half_b)
            combined_fft += fft(grid_a, fft_length, workers=1) * fft(grid_b, fft_length, workers=1)

    combined = ifft(combined_fft, workers=1)[:output_length]
    origin = half_a + half_b
    omega = (np.arange(output_length, dtype=float) - origin) * delta_omega
    initial_singlet = float((0.25 + np.sum(combined)).real)
    return omega, combined, initial_singlet


def yields_from_spectrum(
    omega: np.ndarray,
    coefficients: np.ndarray,
    lifetimes_us: Iterable[float],
) -> dict[float, float]:
    yields = {}
    for lifetime in lifetimes_us:
        k = 1.0 / float(lifetime)
        kernel = k / (k - 1.0j * omega)
        yields[float(lifetime)] = float((0.25 + np.sum(coefficients * kernel)).real)
    return yields


def solve_orientation(
    theta_deg: float,
    field_microT: float,
    lifetimes_us: tuple[float, ...],
    bins_per_linewidth: float,
) -> dict[str, float]:
    fad = diagonalize_radical(_FAD_MODEL, theta_deg, field_microT)
    trp = diagonalize_radical(_TRP_MODEL, theta_deg, field_microT)
    min_k = 1.0 / max(lifetimes_us)
    delta_omega = min_k / bins_per_linewidth
    omega, coefficients, initial_singlet = convolved_correlation_spectrum(fad, trp, delta_omega)
    result: dict[str, float] = {
        "theta_deg": float(theta_deg),
        "initial_singlet": initial_singlet,
        "delta_omega_rad_per_us": delta_omega,
    }
    for lifetime, value in yields_from_spectrum(omega, coefficients, lifetimes_us).items():
        result[f"phi_s_tau_{lifetime:g}_us"] = value
    return result


def direct_line_yields(
    spectrum_a: Spectrum,
    spectrum_b: Spectrum,
    lifetimes_us: Iterable[float],
) -> dict[float, float]:
    """Exact O(d_A^2 d_B^2) reference, used only for the tiny toy system."""
    output = {float(lifetime): 0.25 for lifetime in lifetimes_us}
    for p in range(3):
        for q in range(3):
            omega_a, coeff_a = transition_lines(spectrum_a, p, q)
            omega_b, coeff_b = transition_lines(spectrum_b, p, q)
            combined_omega = omega_a[:, None] + omega_b[None, :]
            combined_coeff = coeff_a[:, None] * coeff_b[None, :]
            for lifetime in output:
                k = 1.0 / lifetime
                output[lifetime] += float(np.sum(combined_coeff * k / (k - 1.0j * combined_omega)).real)
    return output


def run_self_test() -> None:
    toy_a = build_radical(
        "X",
        (Nucleus("N", 1.0, np.diag([-0.0989, -0.0989, 1.7569])),),
    )
    toy_b = build_radical(
        "Y",
        (Nucleus("N", 1.0, np.diag([0.0, 0.0, 1.0812])),),
    )
    lifetimes = (1.0, 10.0, 100.0)
    spectrum_a = diagonalize_radical(toy_a, 89.0, 50.0)
    spectrum_b = diagonalize_radical(toy_b, 89.0, 50.0)
    exact = direct_line_yields(spectrum_a, spectrum_b, lifetimes)
    delta = (1.0 / max(lifetimes)) / 16.0
    omega, coefficients, initial = convolved_correlation_spectrum(spectrum_a, spectrum_b, delta)
    approximate = yields_from_spectrum(omega, coefficients, lifetimes)
    if not np.isclose(initial, 1.0, atol=2e-12):
        raise AssertionError(f"Initial singlet population is {initial}, expected 1.")
    worst_error = max(abs(exact[tau] - approximate[tau]) for tau in lifetimes)
    print("Toy-pair spectral convolution self-test")
    for lifetime in lifetimes:
        print(
            f"tau={lifetime:g} us exact={exact[lifetime]:.10f} "
            f"FFT={approximate[lifetime]:.10f} error={abs(exact[lifetime] - approximate[lifetime]):.3e}"
        )
    print(f"maximum absolute error: {worst_error:.3e}")
    if worst_error > 2e-4:
        raise AssertionError("FFT spectral convolution did not meet the self-test tolerance.")


def save_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_hyperfines(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["radical", "nucleus", "spin", "Axx_mT", "Axy_mT", "Axz_mT", "Ayx_mT", "Ayy_mT", "Ayz_mT", "Azx_mT", "Azy_mT", "Azz_mT"]
        )
        for radical, nuclei in (("FAD", FAD_NUCLEI), ("TrpH", TRP_NUCLEI)):
            for nucleus in nuclei:
                writer.writerow([radical, nucleus.name, nucleus.spin, *nucleus.tensor_mT.reshape(-1)])


def plot_results(rows: list[dict[str, float]], lifetimes_us: tuple[float, ...], path: Path) -> None:
    theta = np.array([row["theta_deg"] for row in rows])
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(lifetimes_us)))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    for lifetime, color in zip(lifetimes_us, colors):
        values = np.array([row[f"phi_s_tau_{lifetime:g}_us"] for row in rows])
        axes[0].plot(theta, values, color=color, linewidth=1.35, label=f"{lifetime:g} us")
        axes[1].plot(theta, values - np.mean(values), color=color, linewidth=1.35)
    axes[0].set_ylabel("singlet yield Phi_S")
    axes[0].set_title("Published yields (Fig. 1B model)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_ylabel("Phi_S - angular mean")
    axes[1].set_title("Anisotropic component")
    for axis in axes:
        axis.set_xlabel("field angle theta (degrees, flavin zx plane)")
        axis.set_xlim(0.0, 180.0)
        axis.grid(alpha=0.2)
    fig.suptitle("Hiscock et al. 2016 quantum-needle reproduction, B = 50 microtesla")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize(rows: list[dict[str, float]], lifetimes_us: tuple[float, ...]) -> list[dict[str, float]]:
    output = []
    theta = np.array([row["theta_deg"] for row in rows])
    for lifetime in lifetimes_us:
        values = np.array([row[f"phi_s_tau_{lifetime:g}_us"] for row in rows])
        minimum = int(np.argmin(values))
        maximum = int(np.argmax(values))
        output.append(
            {
                "lifetime_us": lifetime,
                "min_phi_s": float(values[minimum]),
                "theta_min_deg": float(theta[minimum]),
                "max_phi_s": float(values[maximum]),
                "theta_max_deg": float(theta[maximum]),
                "absolute_anisotropy": float(values[maximum] - values[minimum]),
            }
        )
    return output


def parse_lifetimes(value: str) -> tuple[float, ...]:
    lifetimes = tuple(float(item) for item in value.split(",") if item.strip())
    if not lifetimes or any(lifetime <= 0.0 for lifetime in lifetimes):
        raise argparse.ArgumentTypeError("Lifetimes must be a comma-separated list of positive values.")
    return lifetimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Validate FFT convolution on the paper's toy model and exit.")
    parser.add_argument("--quick", action="store_true", help="Run five angles and lifetimes 1, 10, and 100 us.")
    parser.add_argument("--field-microt", type=float, default=50.0)
    parser.add_argument("--lifetimes-us", type=parse_lifetimes, default=DEFAULT_LIFETIMES_US)
    parser.add_argument("--theta-step", type=float, default=1.0)
    parser.add_argument("--fine-theta-step", type=float, default=0.25)
    parser.add_argument("--fine-half-width", type=float, default=3.0)
    parser.add_argument("--bins-per-linewidth", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "quantum_needle_2016")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.bins_per_linewidth < 2.0:
        raise ValueError("--bins-per-linewidth must be at least 2.")
    if args.quick:
        theta_values = np.array([0.0, 45.0, 89.0, 90.0, 180.0])
        lifetimes_us = (1.0, 10.0, 100.0)
        output_dir = args.output_dir / "quick"
    else:
        theta_values = theta_grid(args.theta_step, args.fine_theta_step, args.fine_half_width)
        lifetimes_us = tuple(args.lifetimes_us)
        output_dir = args.output_dir

    print("Hiscock et al. 2016 quantum-needle reproduction")
    print(f"FAD dimension: {_FAD_MODEL.dimension}; TrpH dimension: {_TRP_MODEL.dimension}")
    print(f"field: {args.field_microt:g} microtesla; angles: {len(theta_values)}")
    print(f"lifetimes (us): {', '.join(f'{value:g}' for value in lifetimes_us)}")
    print(f"workers: {args.workers}; bins per narrowest linewidth: {args.bins_per_linewidth:g}")

    task_args = [
        (float(theta), args.field_microt, lifetimes_us, args.bins_per_linewidth)
        for theta in theta_values
    ]
    if args.workers == 1:
        rows = []
        for index, task in enumerate(task_args, start=1):
            print(f"[{index:03d}/{len(task_args):03d}] theta={task[0]:g}", flush=True)
            rows.append(solve_orientation(*task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_solve_orientation_tuple, task_args, chunksize=1))
    rows.sort(key=lambda row: row["theta_deg"])

    initial_error = max(abs(row["initial_singlet"] - 1.0) for row in rows)
    if initial_error > 5e-9:
        raise RuntimeError(f"Correlation normalization failed: max |P_S(0)-1|={initial_error:.3e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "quantum_needle_fig1b.csv"
    summary_path = output_dir / "quantum_needle_summary.csv"
    plot_path = output_dir / "quantum_needle_fig1b.png"
    metadata_path = output_dir / "quantum_needle_metadata.json"
    hyperfine_path = output_dir / "quantum_needle_hyperfine_tensors.csv"
    save_csv(rows, results_path)
    summary_rows = summarize(rows, lifetimes_us)
    save_csv(summary_rows, summary_path)
    save_hyperfines(hyperfine_path)
    plot_results(rows, lifetimes_us, plot_path)
    metadata_path.write_text(
        json.dumps(
            {
                "paper": "Hiscock et al., PNAS 113, 4634-4639 (2016)",
                "doi": "10.1073/pnas.1600341113",
                "field_microT": args.field_microt,
                "lifetimes_us": lifetimes_us,
                "theta_values_deg": theta_values.tolist(),
                "bins_per_linewidth": args.bins_per_linewidth,
                "max_initial_singlet_error": initial_error,
                "fad_dimension": _FAD_MODEL.dimension,
                "trp_dimension": _TRP_MODEL.dimension,
                "full_pair_dimension": _FAD_MODEL.dimension * _TRP_MODEL.dimension,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nLifetime summary")
    for row in summary_rows:
        print(
            f"tau={row['lifetime_us']:g} us: min={row['min_phi_s']:.8f} "
            f"at {row['theta_min_deg']:g} deg; max={row['max_phi_s']:.8f}; "
            f"anisotropy={row['absolute_anisotropy']:.3e}"
        )
    print(f"saved results: {results_path}")
    print(f"saved plot: {plot_path}")


def _solve_orientation_tuple(args: tuple[float, float, tuple[float, ...], float]) -> dict[str, float]:
    theta, field, lifetimes, bins_per_linewidth = args
    return solve_orientation(theta, field, lifetimes, bins_per_linewidth)


_FAD_MODEL = build_radical("FAD", FAD_NUCLEI)
_TRP_MODEL = build_radical("TrpH", TRP_NUCLEI)


if __name__ == "__main__":
    main()
