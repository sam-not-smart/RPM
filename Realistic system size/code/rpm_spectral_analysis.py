"""Baseline spectral analysis for the FAD-Trp radical-pair model.

This script reuses the full FAD-Trp Hamiltonian implementation and runs an
exact-diagonalization spectral test on a reduced, explicitly labeled nucleus
subset. The default subset is a balanced 10-nucleus FAD-Trp-like selection,
small enough for dense diagonalization while retaining strong, medium, and weak
hyperfine couplings from both radicals.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import rpm_full_fad_trp_magnetosensitivity as rpm


BALANCED10_NUCLEI = [
    "WCHB1",
    "FN5",
    "WCHE1",
    "WCNE1",
    "WCHE3",
    "FH82",
    "FH6",
    "WCHZ2",
    "FH9",
    "WCHZ3",
]

BALANCED8_NUCLEI = BALANCED10_NUCLEI[:8]


@dataclass
class SpectralResult:
    theta_deg: float
    gaps_MHz: np.ndarray
    raw_count_pm: dict[float, float]
    singlet_density_pm: dict[float, float]
    response_density_pm: dict[float, float]
    singlet_weights: np.ndarray | None = None
    response_weights: np.ndarray | None = None


def theta_unit_vector(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.array(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ],
        dtype=float,
    )


def parse_angles(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("theta step must be positive.")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def parse_windows(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one spectral window is required.")
    if any(value <= 0.0 for value in values):
        raise ValueError("Spectral windows must be positive.")
    return values


def window_key(window: float) -> str:
    return f"{window:g}".replace(".", "p")


def all_nuclei_by_name() -> dict[str, rpm.Nucleus]:
    return {
        nucleus.name: nucleus
        for nucleus in rpm.selected_nuclei(len(rpm.PUBLISHED_NUCLEUS_ORDER))
    }


def select_nuclei(args: argparse.Namespace) -> tuple[list[rpm.Nucleus], str]:
    by_name = all_nuclei_by_name()
    if args.quick:
        names = BALANCED8_NUCLEI
        label = "balanced8_quick"
    elif args.selection == "balanced10":
        names = BALANCED10_NUCLEI
        label = "balanced10"
    elif args.selection == "ranked":
        if args.max_nuclei < 1:
            raise ValueError("--max-nuclei must be positive for ranked selection.")
        return rpm.selected_nuclei(args.max_nuclei), f"ranked{args.max_nuclei}"
    elif args.selection == "custom":
        if not args.nucleus_names:
            raise ValueError("--nucleus-names is required for custom selection.")
        names = [name.strip() for name in args.nucleus_names.split(",") if name.strip()]
        label = "custom"
    else:
        raise ValueError(f"Unsupported selection mode: {args.selection}")

    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown nucleus name(s): {', '.join(missing)}")
    return [by_name[name] for name in names], label


def dense_hamiltonian(params: rpm.Params, nuclei: list[rpm.Nucleus], theta_deg: float) -> np.ndarray:
    n_qubits = 2 + len(nuclei)
    terms = rpm.build_terms(params, nuclei, theta_deg)
    hamiltonian_sparse = rpm.make_sparse_operator(terms, n_qubits)
    hermitian_error = hamiltonian_sparse - hamiltonian_sparse.getH()
    max_error = 0.0 if hermitian_error.nnz == 0 else float(np.max(np.abs(hermitian_error.data)))
    if max_error > 1e-9:
        raise ValueError(f"Hamiltonian is not Hermitian at theta={theta_deg:g} deg; max error={max_error:.3e}")
    return hamiltonian_sparse.toarray()


def singlet_projector(n_nuclei: int) -> sp.csr_matrix:
    singlet = rpm.singlet_electron_state()
    projector_e = np.outer(singlet, singlet.conj())
    identity_n = sp.identity(2**n_nuclei, dtype=complex, format="csr")
    return sp.kron(sp.csr_matrix(projector_e), identity_n, format="csr")


def response_operator(n_nuclei: int, theta_deg: float, phi_deg: float) -> sp.csr_matrix:
    n_qubits = 2 + n_nuclei
    e_theta = theta_unit_vector(theta_deg, phi_deg)
    terms: list[tuple[complex, tuple[tuple[str, int], ...]]] = []
    rpm.add_single_spin_terms(terms, 0, e_theta)
    rpm.add_single_spin_terms(terms, 1, e_theta)
    return rpm.make_sparse_operator(terms, n_qubits)


def upper_triangle_gaps(eigenvalues_rad_per_us: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    dim = len(eigenvalues_rad_per_us)
    pair_indices = np.triu_indices(dim, k=1)
    gaps = (eigenvalues_rad_per_us[pair_indices[1]] - eigenvalues_rad_per_us[pair_indices[0]]) / rpm.TWOPI
    if np.any(gaps < -1e-10):
        raise ValueError("Encountered negative energy gaps after sorting eigenvalues.")
    return np.maximum(gaps, 0.0), pair_indices


def pair_weights(operator: sp.csr_matrix, eigenvectors: np.ndarray, pair_indices: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    transformed = eigenvectors.conj().T @ (operator @ eigenvectors)
    weights = np.abs(transformed[pair_indices]) ** 2
    if not np.all(np.isfinite(weights)):
        raise ValueError("Encountered non-finite spectral weights.")
    return weights.astype(float, copy=False)


def integrated_density(gaps_MHz: np.ndarray, weights: np.ndarray | None, target_MHz: float, window_MHz: float) -> float:
    mask = np.abs(gaps_MHz - target_MHz) <= window_MHz
    if weights is None:
        return float(np.count_nonzero(mask))
    return float(np.sum(weights[mask]))


def analyze_spectrum(
    params: rpm.Params,
    nuclei: list[rpm.Nucleus],
    theta_deg: float,
    target_MHz: float,
    windows_MHz: Iterable[float],
    keep_weights: bool,
) -> SpectralResult:
    hamiltonian = dense_hamiltonian(params, nuclei, theta_deg)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    gaps_MHz, pair_indices = upper_triangle_gaps(eigenvalues)

    ps = singlet_projector(len(nuclei))
    h_perp = response_operator(len(nuclei), theta_deg, params.phi_deg)
    singlet_weights = pair_weights(ps, eigenvectors, pair_indices)
    response_weights = pair_weights(h_perp, eigenvectors, pair_indices)

    if not np.all(np.isfinite(gaps_MHz)):
        raise ValueError("Encountered non-finite energy gaps.")

    windows = list(windows_MHz)
    return SpectralResult(
        theta_deg=float(theta_deg),
        gaps_MHz=gaps_MHz,
        raw_count_pm={window: integrated_density(gaps_MHz, None, target_MHz, window) for window in windows},
        singlet_density_pm={window: integrated_density(gaps_MHz, singlet_weights, target_MHz, window) for window in windows},
        response_density_pm={window: integrated_density(gaps_MHz, response_weights, target_MHz, window) for window in windows},
        singlet_weights=singlet_weights if keep_weights else None,
        response_weights=response_weights if keep_weights else None,
    )


def run_yield(params: rpm.Params, nuclei: list[rpm.Nucleus], theta_deg: float) -> dict[str, float]:
    rng = np.random.default_rng(params.seed)
    result = rpm.run_orientation(params, nuclei, theta_deg, rng)
    return {
        "singlet_yield": float(result["singlet_yield"]),
        "singlet_yield_std": float(result["singlet_yield_std"]),
    }


def build_params(args: argparse.Namespace) -> rpm.Params:
    params = rpm.Params()
    params.B_microT = args.b_microt
    params.phi_deg = args.phi
    params.seed = args.seed
    params.n_samples = args.samples
    params.recombination_rate_per_us = args.k
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.J_MHz = args.J_MHz
    params.include_dipole = not args.no_dipole
    params.optimize_dipole_axis = not args.physical_dipole
    params.D_MHz_if_isotropic = args.D_MHz
    params.hyperfine_mode = args.hyperfine_mode
    params.operator_mode = "sparse"
    params.max_sparse_qubits = 99
    params.evolver = args.evolver
    params.output_dir = args.output_dir
    if args.quick:
        params.theta_values_deg = np.array([0.0, 90.0, 180.0], dtype=float)
        params.n_samples = min(params.n_samples, 1)
        params.t_max_us = min(params.t_max_us, 1.0)
        params.n_steps = min(params.n_steps, 61)
    else:
        params.theta_values_deg = parse_angles(args.theta_start, args.theta_stop, args.theta_step)
    if params.n_samples < 1:
        raise ValueError("--samples must be at least 1.")
    if params.n_steps < 2:
        raise ValueError("--n-steps must be at least 2.")
    return params


def save_theta_csv(rows: list[dict[str, float]], windows_MHz: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["theta_deg", "singlet_yield", "dphi_dtheta_abs"]
    for prefix in ("raw_density", "singlet_density", "response_density"):
        for window in windows_MHz:
            columns.append(f"{prefix}_pm_{window_key(window)}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def plot_gap_histogram(
    gaps_MHz: np.ndarray,
    path: Path,
    target_MHz: float,
    max_MHz: float,
    bin_width_MHz: float,
    weights: np.ndarray | None,
    ylabel: str,
    title: str,
    show: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.arange(0.0, max_MHz + bin_width_MHz, bin_width_MHz)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(gaps_MHz, bins=bins, weights=weights, color="#356f8f", edgecolor="none")
    ax.axvline(target_MHz, color="#c43b3b", linestyle="--", linewidth=1.8, label=f"{target_MHz:g} MHz")
    ax.set_xlim(0.0, max_MHz)
    ax.set_xlabel("gap frequency (MHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_phi_theta(rows: list[dict[str, float]], path: Path, show: bool) -> None:
    theta = np.array([row["theta_deg"] for row in rows], dtype=float)
    yields = np.array([row["singlet_yield"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(theta, yields, marker="o", linewidth=2.0)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("singlet yield Phi_S")
    ax.set_title("Singlet yield vs field angle")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_density_theta(rows: list[dict[str, float]], windows_MHz: list[float], path: Path, show: bool) -> None:
    theta = np.array([row["theta_deg"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = ["#356f8f", "#b55d2a", "#4f8a4b", "#7551a8"]
    for index, window in enumerate(windows_MHz):
        key = f"singlet_density_pm_{window_key(window)}"
        density = np.array([row[key] for row in rows], dtype=float)
        ax.plot(theta, density, marker="o", linewidth=2.0, color=colors[index % len(colors)], label=f"+/- {window:g} MHz")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("singlet-weighted density near target")
    ax.set_title("Low-frequency singlet-active spectral density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_scatter(
    rows: list[dict[str, float]],
    windows_MHz: list[float],
    y_key: str,
    y_label: str,
    path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = ["o", "s", "^", "D"]
    for index, window in enumerate(windows_MHz):
        x_key = f"singlet_density_pm_{window_key(window)}"
        x = np.array([row[x_key] for row in rows], dtype=float)
        y = np.array([row[y_key] for row in rows], dtype=float)
        ax.scatter(x, y, s=54, marker=markers[index % len(markers)], label=f"+/- {window:g} MHz")
    ax.set_xlabel("singlet-weighted density near target")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def save_summary(
    args: argparse.Namespace,
    params: rpm.Params,
    nuclei: list[rpm.Nucleus],
    selection_label: str,
    rows: list[dict[str, float]],
    windows_MHz: list[float],
    output_paths: dict[str, str],
    path: Path,
) -> None:
    yields = np.array([row["singlet_yield"] for row in rows], dtype=float)
    theta = np.array([row["theta_deg"] for row in rows], dtype=float)
    mean_yield = float(np.mean(yields))
    min_idx = int(np.argmin(yields))
    max_idx = int(np.argmax(yields))
    larmor_MHz = params.gamma_e_MHz_per_mT * 1e-3 * params.B_microT

    correlations: dict[str, dict[str, float]] = {}
    for window in windows_MHz:
        density_key = f"singlet_density_pm_{window_key(window)}"
        density = np.array([row[density_key] for row in rows], dtype=float)
        dphi = np.array([row["dphi_dtheta_abs"] for row in rows], dtype=float)
        correlations[f"pm_{window_key(window)}"] = {
            "pearson_density_vs_phi_s": pearson_or_nan(density, yields),
            "pearson_density_vs_abs_dphi_dtheta": pearson_or_nan(density, dphi),
        }

    summary = {
        "analysis_label": "balanced reduced exact-diagonalization spectral baseline",
        "selection": selection_label,
        "n_nuclei": len(nuclei),
        "n_qubits": 2 + len(nuclei),
        "nuclei": [{"index": idx, "name": nucleus.name, "radical": nucleus.radical} for idx, nucleus in enumerate(nuclei, start=1)],
        "B_microT": params.B_microT,
        "electron_larmor_MHz": larmor_MHz,
        "target_MHz": args.target_MHz,
        "windows_MHz": windows_MHz,
        "J_MHz": params.J_MHz,
        "include_dipole": params.include_dipole,
        "dipole_model": "published_physical" if params.include_dipole and args.physical_dipole else ("optimistic_axial" if params.include_dipole else "disabled"),
        "hyperfine_mode": params.hyperfine_mode,
        "reference_theta_deg": args.reference_theta,
        "theta_values_deg": theta.tolist(),
        "yield_summary": {
            "min_yield": float(yields[min_idx]),
            "theta_min_yield_deg": float(theta[min_idx]),
            "max_yield": float(yields[max_idx]),
            "theta_max_yield_deg": float(theta[max_idx]),
            "absolute_contrast": float(yields[max_idx] - yields[min_idx]),
            "relative_contrast": float((yields[max_idx] - yields[min_idx]) / mean_yield) if mean_yield != 0.0 else float("nan"),
            "mean_yield": mean_yield,
        },
        "correlations": correlations,
        "density_units": {
            "raw": "pair count integrated inside target +/- window",
            "singlet": "sum |<i|P_S|j>|^2 inside target +/- window",
            "response": "sum |<i|H_perp|j>|^2 inside target +/- window",
        },
        "outputs": output_paths,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a smaller balanced8, three-angle smoke test.")
    parser.add_argument("--selection", choices=["balanced10", "ranked", "custom"], default="balanced10")
    parser.add_argument("--max-nuclei", type=int, default=10, help="Used with --selection ranked.")
    parser.add_argument("--nucleus-names", type=str, default="", help="Comma-separated names used with --selection custom.")
    parser.add_argument("--max-dense-dim", type=int, default=4096, help="Refuse dense diagonalization above this Hilbert-space dimension.")
    parser.add_argument("--theta-step", type=float, default=45.0)
    parser.add_argument("--theta-start", type=float, default=0.0)
    parser.add_argument("--theta-stop", type=float, default=180.0)
    parser.add_argument("--reference-theta", type=float, default=90.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--b-microt", type=float, default=50.0)
    parser.add_argument("--target-MHz", type=float, default=1.4)
    parser.add_argument("--windows-MHz", type=str, default="0.2,0.5")
    parser.add_argument("--hist-bin-width-MHz", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--t-max-us", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=101)
    parser.add_argument("--J-MHz", type=float, default=3.0)
    parser.add_argument("--physical-dipole", action="store_true", help="Use the published Cry4a dipolar tensor.")
    parser.add_argument("--no-dipole", action="store_true")
    parser.add_argument("--D-MHz", type=float, default=-19.10703069)
    parser.add_argument("--hyperfine-mode", choices=["full", "diagonal", "z-only"], default="full")
    parser.add_argument("--evolver", choices=["expm", "rk4"], default="expm")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "baseline_spectral_analysis")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows_MHz = parse_windows(args.windows_MHz)
    params = build_params(args)
    nuclei, selection_label = select_nuclei(args)
    n_qubits = 2 + len(nuclei)
    dimension = 2**n_qubits
    if dimension > args.max_dense_dim:
        raise ValueError(
            f"Selected system has dimension {dimension}, above --max-dense-dim={args.max_dense_dim}. "
            "Use a smaller subset or explicitly raise the limit."
        )

    params.output_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("RPM spectral analysis")
    print("---------------------")
    print(f"selection: {selection_label}")
    print(f"nuclei: {len(nuclei)} ({n_qubits} qubits, dim={dimension})")
    print(f"theta values: {', '.join(f'{theta:g}' for theta in params.theta_values_deg)}")
    print(f"target: {args.target_MHz:g} MHz; windows: {', '.join(f'+/- {w:g}' for w in windows_MHz)} MHz")
    print()

    rows: list[dict[str, float]] = []
    reference_result: SpectralResult | None = None
    total = len(params.theta_values_deg)
    for index, theta_deg in enumerate(params.theta_values_deg, start=1):
        keep_weights = bool(np.isclose(theta_deg, args.reference_theta))
        print(f"[{index:02d}/{total:02d}] theta={theta_deg:g} deg: diagonalizing and evolving yield")
        spectral = analyze_spectrum(params, nuclei, float(theta_deg), args.target_MHz, windows_MHz, keep_weights)
        if keep_weights:
            reference_result = spectral
        yield_row = run_yield(params, nuclei, float(theta_deg))
        row: dict[str, float] = {
            "theta_deg": float(theta_deg),
            "singlet_yield": yield_row["singlet_yield"],
            "singlet_yield_std": yield_row["singlet_yield_std"],
            "dphi_dtheta_abs": 0.0,
        }
        for window in windows_MHz:
            suffix = window_key(window)
            row[f"raw_density_pm_{suffix}"] = spectral.raw_count_pm[window]
            row[f"singlet_density_pm_{suffix}"] = spectral.singlet_density_pm[window]
            row[f"response_density_pm_{suffix}"] = spectral.response_density_pm[window]
        rows.append(row)

    if reference_result is None:
        print(f"[ref] theta={args.reference_theta:g} deg: diagonalizing reference histograms")
        reference_result = analyze_spectrum(params, nuclei, float(args.reference_theta), args.target_MHz, windows_MHz, True)

    theta_rad = np.deg2rad(np.array([row["theta_deg"] for row in rows], dtype=float))
    yields = np.array([row["singlet_yield"] for row in rows], dtype=float)
    gradients = np.abs(np.gradient(yields, theta_rad)) if len(rows) > 1 else np.zeros_like(yields)
    if not np.all(np.isfinite(gradients)):
        raise ValueError("Encountered non-finite |dPhi_S/dtheta| values.")
    for row, gradient in zip(rows, gradients):
        row["dphi_dtheta_abs"] = float(gradient)

    csv_path = params.output_dir / "spectral_theta_scan.csv"
    nuclei_path = params.output_dir / "selected_hyperfine_tensors.csv"
    summary_path = params.output_dir / "spectral_summary.json"
    save_theta_csv(rows, windows_MHz, csv_path)
    rpm.save_nuclei_csv(nuclei, nuclei_path)

    output_paths = {
        "theta_csv": str(csv_path),
        "selected_hyperfine_tensors": str(nuclei_path),
        "summary_json": str(summary_path),
    }

    raw_20 = params.output_dir / "raw_gap_histogram_0_20MHz.png"
    raw_5 = params.output_dir / "raw_gap_histogram_0_5MHz.png"
    singlet_20 = params.output_dir / "singlet_weighted_gap_histogram_0_20MHz.png"
    singlet_5 = params.output_dir / "singlet_weighted_gap_histogram_0_5MHz.png"
    phi_plot = params.output_dir / "phi_s_theta.png"
    density_plot = params.output_dir / "spectral_density_theta.png"
    scatter_phi = params.output_dir / "scatter_density_vs_phi_s.png"
    scatter_dphi = params.output_dir / "scatter_density_vs_abs_dphi_dtheta.png"

    plot_gap_histogram(reference_result.gaps_MHz, raw_20, args.target_MHz, 20.0, args.hist_bin_width_MHz, None, "pair count", "Raw energy-gap histogram", args.show_plot)
    plot_gap_histogram(reference_result.gaps_MHz, raw_5, args.target_MHz, 5.0, args.hist_bin_width_MHz, None, "pair count", "Raw energy-gap histogram, zoom", args.show_plot)
    plot_gap_histogram(reference_result.gaps_MHz, singlet_20, args.target_MHz, 20.0, args.hist_bin_width_MHz, reference_result.singlet_weights, "sum |<i|P_S|j>|^2", "Singlet-weighted gap histogram", args.show_plot)
    plot_gap_histogram(reference_result.gaps_MHz, singlet_5, args.target_MHz, 5.0, args.hist_bin_width_MHz, reference_result.singlet_weights, "sum |<i|P_S|j>|^2", "Singlet-weighted gap histogram, zoom", args.show_plot)
    plot_phi_theta(rows, phi_plot, args.show_plot)
    plot_density_theta(rows, windows_MHz, density_plot, args.show_plot)
    plot_scatter(rows, windows_MHz, "singlet_yield", "singlet yield Phi_S", scatter_phi, args.show_plot)
    plot_scatter(rows, windows_MHz, "dphi_dtheta_abs", "|dPhi_S/dtheta|", scatter_dphi, args.show_plot)

    output_paths.update(
        {
            "raw_gap_histogram_0_20MHz": str(raw_20),
            "raw_gap_histogram_0_5MHz": str(raw_5),
            "singlet_weighted_gap_histogram_0_20MHz": str(singlet_20),
            "singlet_weighted_gap_histogram_0_5MHz": str(singlet_5),
            "phi_s_theta": str(phi_plot),
            "spectral_density_theta": str(density_plot),
            "scatter_density_vs_phi_s": str(scatter_phi),
            "scatter_density_vs_abs_dphi_dtheta": str(scatter_dphi),
        }
    )
    save_summary(args, params, nuclei, selection_label, rows, windows_MHz, output_paths, summary_path)

    print()
    print("Saved spectral analysis outputs")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Plots: {params.output_dir}")


if __name__ == "__main__":
    main()
