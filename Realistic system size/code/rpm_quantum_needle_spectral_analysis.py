"""Spectral representation for the Quantum Needle baseline.

The full [FAD*- TrpH*+] Hilbert space is about 221k states, so this script uses
the J = D = 0 factorization from the angular-sensitivity runner. It diagonalizes
the two radical-local Hamiltonians, represents the full pair energy spectrum as
all sums of local radical energies, and builds the singlet-active frequency
spectrum from the same correlation-line histograms used for the yield integral.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

import rpm_quantum_needle_angular_sensitivity as qn


def build_local_spectra(theta_deg: float, b_microT: float, bin_width_rad_per_us: float, selection: str):
    nuclei = qn.quantum_needle_nuclei(False, selection)
    fad_nuclei = qn.radical_nuclei(nuclei, "FAD")
    trp_nuclei = qn.radical_nuclei(nuclei, "Trp")

    h_fad, ops_fad, z_fad = qn.build_radical_hamiltonian(fad_nuclei, theta_deg, b_microT)
    h_trp, ops_trp, z_trp = qn.build_radical_hamiltonian(trp_nuclei, theta_deg, b_microT)
    evals_fad = np.linalg.eigvalsh(h_fad) / qn.TWOPI
    evals_trp = np.linalg.eigvalsh(h_trp) / qn.TWOPI

    hist_fad, min_fad, width = qn.spectral_histograms(h_fad, ops_fad, z_fad, bin_width_rad_per_us)
    hist_trp, min_trp, _width = qn.spectral_histograms(h_trp, ops_trp, z_trp, bin_width_rad_per_us)
    return evals_fad, evals_trp, hist_fad, min_fad, hist_trp, min_trp, width


def singlet_line_spectrum(hist_fad, min_fad: int, hist_trp, min_trp: int, bin_width: float):
    total = None
    for p in range(3):
        for q in range(3):
            line = fftconvolve(hist_fad[p][q], hist_trp[p][q], mode="full")
            total = line if total is None else total + line
    if total is None:
        raise AssertionError("No spectral lines were generated.")
    indices = np.arange(len(total), dtype=float)
    omega = (min_fad + min_trp + indices) * bin_width
    return omega, total


def positive_frequency_summary(omega: np.ndarray, coeff: np.ndarray, bin_width: float):
    abs_bins = np.rint(np.abs(omega) / bin_width).astype(np.int64)
    n_bins = int(np.max(abs_bins)) + 1
    abs_weight = np.bincount(abs_bins, weights=np.abs(coeff), minlength=n_bins)
    signed_real = np.bincount(abs_bins, weights=coeff.real, minlength=n_bins)
    freq_MHz = np.arange(n_bins, dtype=float) * bin_width / qn.TWOPI
    return freq_MHz, abs_weight, signed_real


def density_near(freq_MHz: np.ndarray, weight: np.ndarray, target_MHz: float, window_MHz: float) -> float:
    mask = np.abs(freq_MHz - target_MHz) <= window_MHz
    return float(np.sum(weight[mask]))


def pair_energy_sums(evals_fad_MHz: np.ndarray, evals_trp_MHz: np.ndarray) -> np.ndarray:
    pair = (evals_fad_MHz[:, None] + evals_trp_MHz[None, :]).reshape(-1)
    return pair - float(np.mean(pair))


def save_frequency_csv(freq_MHz: np.ndarray, abs_weight: np.ndarray, signed_real: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frequency_MHz", "singlet_abs_weight", "singlet_signed_real_weight"])
        for freq, abs_value, signed_value in zip(freq_MHz, abs_weight, signed_real):
            writer.writerow([freq, abs_value, signed_value])


def plot_local_energy(evals_fad: np.ndarray, evals_trp: np.ndarray, path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hist(evals_fad - np.mean(evals_fad), bins=90, alpha=0.62, label="FAD radical")
    ax.hist(evals_trp - np.mean(evals_trp), bins=90, alpha=0.62, label="Trp radical")
    ax.set_xlabel("local radical energy relative to mean (MHz)")
    ax.set_ylabel("state count")
    ax.set_title("Quantum Needle local radical spectra")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_pair_energy(pair_energies: np.ndarray, path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hist(pair_energies, bins=160, color="#346a8a")
    ax.set_xlabel("pair energy relative to mean (MHz)")
    ax.set_ylabel("state count")
    ax.set_title("Full pair energy spectrum from local sums")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_frequency(freq_MHz: np.ndarray, weight: np.ndarray, target_MHz: float, max_MHz: float, path: Path, show: bool) -> None:
    mask = freq_MHz <= max_MHz
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(freq_MHz[mask], weight[mask], linewidth=1.4, color="#3c6d8e")
    ax.axvline(target_MHz, color="#c43b3b", linestyle="--", linewidth=1.6, label=f"{target_MHz:g} MHz")
    ax.set_xlim(0.0, max_MHz)
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("absolute singlet-active line weight")
    ax.set_title(f"Singlet-active frequency spectrum, 0-{max_MHz:g} MHz")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_theta_density(rows: list[dict[str, float]], path: Path, show: bool) -> None:
    theta = np.array([row["theta_deg"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for window in (0.2, 0.5):
        key = f"abs_weight_pm_{str(window).replace('.', 'p')}"
        values = np.array([row[key] for row in rows], dtype=float)
        ax.plot(theta, values, marker="o", linewidth=1.8, label=f"+/- {window:g} MHz")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("absolute singlet-active weight near 1.4 MHz")
    ax.set_title("Low-frequency spectral weight vs angle")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def save_theta_density_csv(rows: list[dict[str, float]], path: Path) -> None:
    columns = ["theta_deg", "abs_weight_pm_0p2", "abs_weight_pm_0p5", "signed_real_pm_0p2", "signed_real_pm_0p5"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def theta_values(args: argparse.Namespace) -> np.ndarray:
    if args.quick:
        return np.array([0.0, 80.0, 85.0, 89.0, 90.0, 91.0, 95.0, 100.0, 180.0])
    return np.array(sorted({*np.arange(0.0, 181.0, 15.0), *np.arange(84.0, 96.1, 1.0), 90.0}), dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--b-microt", type=float, default=50.0)
    parser.add_argument("--reference-theta", type=float, default=90.0)
    parser.add_argument("--selection", choices=["paper14", "balanced-tail14"], default="paper14")
    parser.add_argument("--target-MHz", type=float, default=1.4)
    parser.add_argument("--frequency-bin-width-rad-per-us", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "quantum_needle_spectral_analysis")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frequency_bin_width_rad_per_us <= 0.0:
        raise ValueError("--frequency-bin-width-rad-per-us must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Quantum Needle spectral analysis")
    print("--------------------------------")
    print(f"reference theta: {args.reference_theta:g} deg")
    print(f"selection: {args.selection}")
    print("assumption: full pair energies are represented as all FAD+Trp local energy sums because J=D=0")
    print()

    evals_fad, evals_trp, hist_fad, min_fad, hist_trp, min_trp, width = build_local_spectra(
        args.reference_theta,
        args.b_microt,
        args.frequency_bin_width_rad_per_us,
        args.selection,
    )
    pair_energies = pair_energy_sums(evals_fad, evals_trp)
    omega, coeff = singlet_line_spectrum(hist_fad, min_fad, hist_trp, min_trp, width)
    freq_MHz, abs_weight, signed_real = positive_frequency_summary(omega, coeff, width)

    freq_csv = args.output_dir / "singlet_active_frequency_spectrum_theta_90deg.csv"
    theta_csv = args.output_dir / "theta_low_frequency_weight.csv"
    summary_path = args.output_dir / "quantum_needle_spectral_summary.json"
    local_plot = args.output_dir / "local_radical_energy_spectrum_theta_90deg.png"
    pair_plot = args.output_dir / "pair_energy_spectrum_theta_90deg.png"
    freq_20_plot = args.output_dir / "singlet_active_frequency_spectrum_0_20MHz_theta_90deg.png"
    freq_5_plot = args.output_dir / "singlet_active_frequency_spectrum_0_5MHz_theta_90deg.png"
    theta_density_plot = args.output_dir / "low_frequency_weight_vs_theta.png"

    save_frequency_csv(freq_MHz, abs_weight, signed_real, freq_csv)
    plot_local_energy(evals_fad, evals_trp, local_plot, args.show_plot)
    plot_pair_energy(pair_energies, pair_plot, args.show_plot)
    plot_frequency(freq_MHz, abs_weight, args.target_MHz, 20.0, freq_20_plot, args.show_plot)
    plot_frequency(freq_MHz, abs_weight, args.target_MHz, 5.0, freq_5_plot, args.show_plot)

    rows: list[dict[str, float]] = []
    for index, theta in enumerate(theta_values(args), start=1):
        print(f"[{index:02d}] theta={theta:g} deg spectral weights")
        _evals_fad, _evals_trp, h_fad, m_fad, h_trp, m_trp, w = build_local_spectra(theta, args.b_microt, args.frequency_bin_width_rad_per_us, args.selection)
        theta_omega, theta_coeff = singlet_line_spectrum(h_fad, m_fad, h_trp, m_trp, w)
        theta_freq, theta_abs, theta_signed = positive_frequency_summary(theta_omega, theta_coeff, w)
        rows.append(
            {
                "theta_deg": float(theta),
                "abs_weight_pm_0p2": density_near(theta_freq, theta_abs, args.target_MHz, 0.2),
                "abs_weight_pm_0p5": density_near(theta_freq, theta_abs, args.target_MHz, 0.5),
                "signed_real_pm_0p2": density_near(theta_freq, theta_signed, args.target_MHz, 0.2),
                "signed_real_pm_0p5": density_near(theta_freq, theta_signed, args.target_MHz, 0.5),
            }
        )
    save_theta_density_csv(rows, theta_csv)
    plot_theta_density(rows, theta_density_plot, args.show_plot)

    summary = {
        "analysis_label": "Quantum Needle spectral representation",
        "assumptions": [
            "J = D = 0, so pair energies are sums of FAD and Trp radical-local eigenvalues.",
            "Singlet-active frequency spectrum is the absolute positive-frequency weight of the pS(t) correlation-line coefficients.",
            "Line frequencies are binned at the configured rad/us resolution for tractability and consistency with the yield runner.",
        ],
        "B_microT": args.b_microt,
        "selection": args.selection,
        "reference_theta_deg": args.reference_theta,
        "frequency_bin_width_rad_per_us": args.frequency_bin_width_rad_per_us,
        "frequency_bin_width_MHz": args.frequency_bin_width_rad_per_us / qn.TWOPI,
        "dimensions": {
            "FAD_radical": int(len(evals_fad)),
            "Trp_radical": int(len(evals_trp)),
            "full_pair_sum_states": int(len(pair_energies)),
        },
        "energy_ranges_MHz": {
            "FAD_local_span": float(np.max(evals_fad) - np.min(evals_fad)),
            "Trp_local_span": float(np.max(evals_trp) - np.min(evals_trp)),
            "pair_sum_span": float(np.max(pair_energies) - np.min(pair_energies)),
        },
        "target_MHz": args.target_MHz,
        "reference_theta_weights": {
            "abs_weight_pm_0p2": density_near(freq_MHz, abs_weight, args.target_MHz, 0.2),
            "abs_weight_pm_0p5": density_near(freq_MHz, abs_weight, args.target_MHz, 0.5),
            "signed_real_pm_0p2": density_near(freq_MHz, signed_real, args.target_MHz, 0.2),
            "signed_real_pm_0p5": density_near(freq_MHz, signed_real, args.target_MHz, 0.5),
        },
        "outputs": {
            "frequency_csv": str(freq_csv),
            "theta_density_csv": str(theta_csv),
            "local_radical_energy_spectrum": str(local_plot),
            "pair_energy_spectrum": str(pair_plot),
            "singlet_active_frequency_0_20MHz": str(freq_20_plot),
            "singlet_active_frequency_0_5MHz": str(freq_5_plot),
            "low_frequency_weight_vs_theta": str(theta_density_plot),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("Saved spectral outputs")
    print(f"Summary: {summary_path}")
    print(f"Plots: {args.output_dir}")


if __name__ == "__main__":
    main()
