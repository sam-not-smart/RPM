"""Quantum Needle angular sensitivity baseline for [FAD*- TrpH*+].

This standalone runner follows the model described in the Quantum Needle paper:
14 selected nuclei, true spin-1 nitrogens, spin-1/2 protons, B = 50 uT,
initial electronic singlet, maximally mixed nuclei, equal singlet/triplet
recombination rates, and no exchange or electron-electron dipolar coupling.

Because J = D = 0 and kS = kT, the radical-pair yield factorizes into local
radical electron-spin correlation tensors. The implementation evaluates the
Laplace integral in the frequency domain using binned spectral lines; this is
deterministic and avoids long time grids for the 100 us lifetime case.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.signal import fftconvolve

TWOPI = 2.0 * np.pi
GAMMA_E_MHZ_PER_MT = 28.02495164
DEFAULT_LIFETIMES_US = [1.0, 2.0, 5.0, 10.0, 20.0, 100.0]
PAPER14_SELECTION = [
    "FAD:N5",
    "FAD:N10",
    "FAD:H6",
    "FAD:H8a",
    "FAD:H8b",
    "FAD:H8c",
    "FAD:Hbeta",
    "Trp:N1",
    "Trp:H1",
    "Trp:H2",
    "Trp:H4",
    "Trp:H5",
    "Trp:H7",
    "Trp:Hbeta1",
]
BALANCED_TAIL14_SELECTION = [
    "FAD:N5",
    "FAD:N10",
    "FAD:H6",
    "FAD:H8a",
    "FAD:Hbeta",
    "FAD:H7a",
    "FAD:H7b",
    "Trp:N1",
    "Trp:H1",
    "Trp:H4",
    "Trp:H7",
    "Trp:Hbeta1",
    "Trp:Nstar",
    "Trp:Hbeta2",
]


@dataclass(frozen=True)
class NeedleNucleus:
    radical: str
    name: str
    spin: float
    tensor_mT: np.ndarray


def tensor(values: list[list[float]]) -> np.ndarray:
    out = np.array(values, dtype=float)
    if out.shape != (3, 3):
        raise ValueError("Hyperfine tensor must be 3x3.")
    if not np.allclose(out, out.T, atol=1e-12):
        raise ValueError("Hyperfine tensor must be symmetric.")
    return out


def all_quantum_needle_nuclei(zero_hyperfine: bool = False) -> list[NeedleNucleus]:
    nuclei = [
        NeedleNucleus("FAD", "N5", 1.0, tensor([[-0.0989, 0.0039, 0.0], [0.0039, -0.0881, 0.0], [0.0, 0.0, 1.7569]])),
        NeedleNucleus("FAD", "N10", 1.0, tensor([[-0.0190, -0.0048, 0.0], [-0.0048, -0.0196, 0.0], [0.0, 0.0, 0.6046]])),
        NeedleNucleus("FAD", "H6", 0.5, tensor([[-0.2569, -0.1273, 0.0], [-0.1273, -0.4711, 0.0], [0.0, 0.0, -0.4336]])),
        NeedleNucleus("FAD", "H8a", 0.5, 0.4399 * np.identity(3)),
        NeedleNucleus("FAD", "H8b", 0.5, 0.4399 * np.identity(3)),
        NeedleNucleus("FAD", "H8c", 0.5, 0.4399 * np.identity(3)),
        NeedleNucleus("FAD", "Hbeta", 0.5, 0.4070 * np.identity(3)),
        NeedleNucleus("FAD", "Hbeta2", 0.5, 0.4070 * np.identity(3)),
        NeedleNucleus("FAD", "H7a", 0.5, -0.1416 * np.identity(3)),
        NeedleNucleus("FAD", "H7b", 0.5, -0.1416 * np.identity(3)),
        NeedleNucleus("FAD", "H7c", 0.5, -0.1416 * np.identity(3)),
        NeedleNucleus("Trp", "N1", 1.0, tensor([[-0.0336, 0.0924, -0.1354], [0.0924, 0.3303, -0.5318], [-0.1354, -0.5318, 0.6680]])),
        NeedleNucleus("Trp", "H1", 0.5, tensor([[-0.9920, -0.2091, -0.2003], [-0.2091, -0.2631, 0.2803], [-0.2003, 0.2803, -0.5398]])),
        NeedleNucleus("Trp", "H2", 0.5, tensor([[-0.2843, 0.1757, 0.1525], [0.1757, -0.2798, 0.0975], [0.1525, 0.0975, -0.2699]])),
        NeedleNucleus("Trp", "H4", 0.5, tensor([[-0.5596, -0.1956, -0.1657], [-0.1956, -0.4020, 0.0762], [-0.1657, 0.0762, -0.5021]])),
        NeedleNucleus("Trp", "H5", 0.5, tensor([[0.0051, 0.0616, 0.0694], [0.0616, -0.0665, 0.0391], [0.0694, 0.0391, -0.0586]])),
        NeedleNucleus("Trp", "H7", 0.5, tensor([[-0.4355, -0.1541, -0.1239], [-0.1541, -0.2777, 0.0864], [-0.1239, 0.0864, -0.3770]])),
        NeedleNucleus("Trp", "Hbeta1", 0.5, tensor([[1.5808, -0.0453, -0.0506], [-0.0453, 1.5575, 0.0988], [-0.0506, 0.0988, 1.6752]])),
        NeedleNucleus("Trp", "Hbeta", 0.5, tensor([[0.0601, 0.0037, 0.0331], [0.0037, 0.0251, 0.0111], [0.0331, 0.0111, 0.1940]])),
        NeedleNucleus("Trp", "Nstar", 1.0, tensor([[0.1295, 0.0134, 0.0075], [0.0134, 0.1729, 0.0249], [0.0075, 0.0249, 0.1371]])),
        NeedleNucleus("Trp", "Hbeta2", 0.5, tensor([[0.1634, 0.0230, 0.0064], [0.0230, -0.0082, 0.0158], [0.0064, 0.0158, -0.0182]])),
        NeedleNucleus("Trp", "H6", 0.5, tensor([[0.0506, 0.0622, 0.0889], [0.0622, 0.3100, 0.0297], [0.0889, 0.0297, 0.2642]])),
    ]
    if not zero_hyperfine:
        return nuclei
    return [
        NeedleNucleus(nucleus.radical, nucleus.name, nucleus.spin, np.zeros((3, 3), dtype=float))
        for nucleus in nuclei
    ]


def quantum_needle_nuclei(zero_hyperfine: bool = False, selection: str = "paper14") -> list[NeedleNucleus]:
    all_nuclei = all_quantum_needle_nuclei(zero_hyperfine)
    by_key = {f"{nucleus.radical}:{nucleus.name}": nucleus for nucleus in all_nuclei}
    if selection == "paper14":
        keys = PAPER14_SELECTION
    elif selection == "balanced-tail14":
        keys = BALANCED_TAIL14_SELECTION
    else:
        raise ValueError("selection must be one of: paper14, balanced-tail14.")
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"Internal selection contains unknown nuclei: {', '.join(missing)}")
    return [by_key[key] for key in keys]


def spin_operators(spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = int(round(2.0 * spin + 1.0))
    m_values = np.arange(spin, -spin - 0.5, -1.0, dtype=float)
    sp_op = np.zeros((dim, dim), dtype=complex)
    for col, m in enumerate(m_values):
        m_plus = m + 1.0
        if m_plus <= spin + 1e-12:
            row_matches = np.where(np.isclose(m_values, m_plus))[0]
            if len(row_matches) == 1:
                row = int(row_matches[0])
                sp_op[row, col] = math.sqrt(spin * (spin + 1.0) - m * (m + 1.0))
    sm_op = sp_op.conj().T
    sx = 0.5 * (sp_op + sm_op)
    sy = -0.5j * (sp_op - sm_op)
    sz = np.diag(m_values).astype(complex)
    return sx, sy, sz


def kron_all_dense(operators: Iterable[np.ndarray]) -> np.ndarray:
    iterator = iter(operators)
    result = np.asarray(next(iterator), dtype=complex)
    for operator in iterator:
        result = np.kron(result, np.asarray(operator, dtype=complex))
    return result


def op_on_site(dims: list[int], site_index: int, operator: np.ndarray) -> np.ndarray:
    factors = [np.eye(dim, dtype=complex) for dim in dims]
    factors[site_index] = operator
    return kron_all_dense(factors)


def field_vector_mT(theta_deg: float, b_microT: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    b_mT = 1e-3 * b_microT
    return b_mT * np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=float)


def radical_nuclei(nuclei: list[NeedleNucleus], radical: str) -> list[NeedleNucleus]:
    return [nucleus for nucleus in nuclei if nucleus.radical == radical]


def nuclear_dimension(nuclei: list[NeedleNucleus]) -> int:
    dim = 1
    for nucleus in nuclei:
        dim *= int(round(2.0 * nucleus.spin + 1.0))
    return dim


def build_radical_hamiltonian(
    nuclei: list[NeedleNucleus],
    theta_deg: float,
    b_microT: float,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    dims = [2] + [int(round(2.0 * nucleus.spin + 1.0)) for nucleus in nuclei]
    dimension = int(np.prod(dims))
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    electron_ops = [op_on_site(dims, 0, op) for op in spin_operators(0.5)]
    b_MHz = GAMMA_E_MHZ_PER_MT * field_vector_mT(theta_deg, b_microT)
    for axis in range(3):
        hamiltonian += TWOPI * b_MHz[axis] * electron_ops[axis]

    for nucleus_index, nucleus in enumerate(nuclei, start=1):
        nuclear_ops = [op_on_site(dims, nucleus_index, op) for op in spin_operators(nucleus.spin)]
        tensor_MHz = GAMMA_E_MHZ_PER_MT * nucleus.tensor_mT
        for a in range(3):
            for b in range(3):
                coefficient = TWOPI * tensor_MHz[a, b]
                if abs(coefficient) > 1e-15:
                    hamiltonian += coefficient * (electron_ops[a] @ nuclear_ops[b])

    hermitian_error = np.max(np.abs(hamiltonian - hamiltonian.conj().T))
    if hermitian_error > 1e-9:
        raise ValueError(f"Radical Hamiltonian is not Hermitian; max error={hermitian_error:.3e}")
    return hamiltonian, electron_ops, nuclear_dimension(nuclei)


def spectral_histograms(
    hamiltonian: np.ndarray,
    electron_ops: list[np.ndarray],
    z_nuc: int,
    bin_width_rad_per_us: float,
) -> tuple[list[list[np.ndarray]], int, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    deltas = (eigenvalues[:, None] - eigenvalues[None, :]).reshape(-1)
    rounded_bins = np.rint(deltas / bin_width_rad_per_us).astype(np.int64)
    min_bin = int(np.min(rounded_bins))
    max_bin = int(np.max(rounded_bins))
    n_bins = max_bin - min_bin + 1
    bin_indices = rounded_bins - min_bin

    ops_eigen = [eigenvectors.conj().T @ operator @ eigenvectors for operator in electron_ops]
    histograms: list[list[np.ndarray]] = []
    for p in range(3):
        row = []
        for q in range(3):
            coeff = (ops_eigen[p] * ops_eigen[q].T).reshape(-1) / float(z_nuc)
            real_part = np.bincount(bin_indices, weights=coeff.real, minlength=n_bins)
            imag_part = np.bincount(bin_indices, weights=coeff.imag, minlength=n_bins)
            row.append(real_part + 1.0j * imag_part)
        histograms.append(row)
    return histograms, min_bin, bin_width_rad_per_us


def integrate_histogram_product(
    hist_a: list[list[np.ndarray]],
    min_bin_a: int,
    hist_b: list[list[np.ndarray]],
    min_bin_b: int,
    bin_width: float,
    lifetime_us: float,
) -> float:
    k = 1.0 / lifetime_us
    value = 0.25 + 0.0j
    for p in range(3):
        for q in range(3):
            convolved = fftconvolve(hist_a[p][q], hist_b[p][q], mode="full")
            indices = np.arange(len(convolved), dtype=float)
            omega = (min_bin_a + min_bin_b + indices) * bin_width
            value += np.sum(convolved * (k / (k - 1.0j * omega)))
    return float(value.real)


def integrate_histogram_products(
    hist_a: list[list[np.ndarray]],
    min_bin_a: int,
    hist_b: list[list[np.ndarray]],
    min_bin_b: int,
    bin_width: float,
    lifetimes_us: list[float],
) -> dict[float, float]:
    spectra: list[tuple[np.ndarray, np.ndarray]] = []
    for p in range(3):
        for q in range(3):
            convolved = fftconvolve(hist_a[p][q], hist_b[p][q], mode="full")
            indices = np.arange(len(convolved), dtype=float)
            omega = (min_bin_a + min_bin_b + indices) * bin_width
            spectra.append((convolved, omega))

    values: dict[float, float] = {}
    for lifetime_us in lifetimes_us:
        k = 1.0 / lifetime_us
        value = 0.25 + 0.0j
        for convolved, omega in spectra:
            value += np.sum(convolved * (k / (k - 1.0j * omega)))
        values[lifetime_us] = float(value.real)
    return values


def exact_p_s_zero_time(
    hamiltonian_a: np.ndarray,
    electron_ops_a: list[np.ndarray],
    z_a: int,
    hamiltonian_b: np.ndarray,
    electron_ops_b: list[np.ndarray],
    z_b: int,
) -> float:
    del hamiltonian_a, hamiltonian_b
    r_a = np.array([[np.trace(op_p @ op_q) / z_a for op_q in electron_ops_a] for op_p in electron_ops_a], dtype=complex)
    r_b = np.array([[np.trace(op_p @ op_q) / z_b for op_q in electron_ops_b] for op_p in electron_ops_b], dtype=complex)
    return float((0.25 + np.sum(r_a * r_b)).real)


def theta_grid(args: argparse.Namespace) -> np.ndarray:
    if args.quick:
        return np.array([0.0, 85.0, 89.0, 90.0, 91.0, 95.0, 180.0], dtype=float)
    values = [
        np.arange(0.0, 180.0 + 0.5 * args.coarse_step_deg, args.coarse_step_deg),
        np.arange(80.0, 100.0 + 0.5 * args.fine_step_deg, args.fine_step_deg),
        np.arange(88.0, 92.0 + 0.5 * args.ultrafine_step_deg, args.ultrafine_step_deg),
    ]
    return np.array(sorted({round(float(theta), 10) for grid in values for theta in grid}), dtype=float)


def parse_lifetimes(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one lifetime is required.")
    if any(value <= 0.0 for value in values):
        raise ValueError("Lifetimes must be positive.")
    return values


def lifetime_key(lifetime: float) -> str:
    return f"{lifetime:g}".replace(".", "p")


def run_theta(
    nuclei: list[NeedleNucleus],
    theta_deg: float,
    lifetimes_us: list[float],
    b_microT: float,
    bin_width_rad_per_us: float,
) -> dict[str, float]:
    fad = radical_nuclei(nuclei, "FAD")
    trp = radical_nuclei(nuclei, "Trp")
    h_fad, ops_fad, z_fad = build_radical_hamiltonian(fad, theta_deg, b_microT)
    h_trp, ops_trp, z_trp = build_radical_hamiltonian(trp, theta_deg, b_microT)

    p_s_0 = exact_p_s_zero_time(h_fad, ops_fad, z_fad, h_trp, ops_trp, z_trp)
    if abs(p_s_0 - 1.0) > 1e-9:
        raise ValueError(f"pS(0) validation failed at theta={theta_deg:g}: {p_s_0:.12f}")

    hist_fad, min_fad, width_fad = spectral_histograms(h_fad, ops_fad, z_fad, bin_width_rad_per_us)
    hist_trp, min_trp, width_trp = spectral_histograms(h_trp, ops_trp, z_trp, bin_width_rad_per_us)
    if abs(width_fad - width_trp) > 1e-15:
        raise AssertionError("Internal bin-width mismatch.")

    row = {"theta_deg": float(theta_deg)}
    phi_by_lifetime = integrate_histogram_products(hist_fad, min_fad, hist_trp, min_trp, width_fad, lifetimes_us)
    for lifetime in lifetimes_us:
        phi_s = phi_by_lifetime[lifetime]
        if not np.isfinite(phi_s) or phi_s < -1e-8 or phi_s > 1.0 + 1e-8:
            raise ValueError(f"Invalid Phi_S={phi_s:.12f} at theta={theta_deg:g}, tau={lifetime:g}")
        row[f"phi_s_tau_{lifetime_key(lifetime)}us"] = min(max(phi_s, 0.0), 1.0)
    return row


def save_tensor_csv(nuclei: list[NeedleNucleus], path: Path, units: str) -> None:
    multiplier = 1.0 if units == "mT" else GAMMA_E_MHZ_PER_MT
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["radical", "name", "spin", "Axx", "Axy", "Axz", "Ayx", "Ayy", "Ayz", "Azx", "Azy", "Azz"]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for nucleus in nuclei:
            writer.writerow([nucleus.radical, nucleus.name, nucleus.spin, *(multiplier * nucleus.tensor_mT).reshape(-1)])


def add_delta_columns(rows: list[dict[str, float]], lifetimes_us: list[float]) -> None:
    for lifetime in lifetimes_us:
        phi_key = f"phi_s_tau_{lifetime_key(lifetime)}us"
        delta_key = f"delta_phi_s_tau_{lifetime_key(lifetime)}us"
        mean_phi = float(np.mean([row[phi_key] for row in rows]))
        for row in rows:
            row[delta_key] = row[phi_key] - mean_phi


def save_theta_csv(rows: list[dict[str, float]], lifetimes_us: list[float], path: Path) -> None:
    columns = ["theta_deg"]
    columns.extend(f"phi_s_tau_{lifetime_key(tau)}us" for tau in lifetimes_us)
    columns.extend(f"delta_phi_s_tau_{lifetime_key(tau)}us" for tau in lifetimes_us)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def plot_yield(rows: list[dict[str, float]], lifetimes_us: list[float], path: Path, delta: bool, zoom: bool, show: bool) -> None:
    theta = np.array([row["theta_deg"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for lifetime in lifetimes_us:
        prefix = "delta_phi_s" if delta else "phi_s"
        key = f"{prefix}_tau_{lifetime_key(lifetime)}us"
        values = np.array([row[key] for row in rows], dtype=float)
        ax.plot(theta, values, linewidth=1.8, label=f"{lifetime:g} us")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("Delta Phi_S" if delta else "Phi_S")
    ax.set_title(("Anisotropic singlet yield" if delta else "Singlet yield") + (" near 90 deg" if zoom else ""))
    if zoom:
        ax.set_xlim(80.0, 100.0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_dip_depth(summary_by_tau: dict[str, dict[str, float]], lifetimes_us: list[float], path: Path, show: bool) -> None:
    tau = np.array(lifetimes_us, dtype=float)
    dip = np.array([summary_by_tau[lifetime_key(value)]["dip_depth_near_90"] for value in tau], dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(tau, dip, marker="o", linewidth=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("lifetime tau (us)")
    ax.set_ylabel("near-90 deg dip depth")
    ax.set_title("Quantum Needle feature depth")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def summarize(rows: list[dict[str, float]], lifetimes_us: list[float]) -> dict[str, dict[str, float]]:
    theta = np.array([row["theta_deg"] for row in rows], dtype=float)
    near_90 = np.abs(theta - 90.0) <= 2.0
    out: dict[str, dict[str, float]] = {}
    for lifetime in lifetimes_us:
        key = f"phi_s_tau_{lifetime_key(lifetime)}us"
        values = np.array([row[key] for row in rows], dtype=float)
        min_index = int(np.argmin(values))
        max_index = int(np.argmax(values))
        near_min = float(np.min(values[near_90])) if np.any(near_90) else float("nan")
        shoulder_mask = (np.abs(theta - 90.0) >= 5.0) & (np.abs(theta - 90.0) <= 10.0)
        shoulder_mean = float(np.mean(values[shoulder_mask])) if np.any(shoulder_mask) else float(np.mean(values))
        out[lifetime_key(lifetime)] = {
            "tau_us": float(lifetime),
            "min_phi_s": float(values[min_index]),
            "theta_min_phi_s_deg": float(theta[min_index]),
            "max_phi_s": float(values[max_index]),
            "theta_max_phi_s_deg": float(theta[max_index]),
            "mean_phi_s": float(np.mean(values)),
            "dip_depth_near_90": float(shoulder_mean - near_min),
        }
    return out


def save_summary(
    args: argparse.Namespace,
    nuclei: list[NeedleNucleus],
    rows: list[dict[str, float]],
    lifetimes_us: list[float],
    output_paths: dict[str, str],
    path: Path,
) -> dict[str, dict[str, float]]:
    summary_by_tau = summarize(rows, lifetimes_us)
    payload = {
        "analysis_label": "Quantum Needle angular sensitivity baseline",
        "tensor_source": "Quantum Needle SI Tables S1/S2, signs cross-checked against readable mirrored text.",
        "method": "factorized radical-local electron-spin correlation tensors with binned frequency-domain infinite-time Laplace integral",
        "selection": args.selection,
        "B_microT": args.b_microt,
        "field_plane": "flavin zx plane, B = (B sin theta, 0, B cos theta)",
        "J_MHz": 0.0,
        "D_MHz": 0.0,
        "kS_equals_kT": True,
        "lifetimes_us": lifetimes_us,
        "frequency_bin_width_rad_per_us": args.frequency_bin_width_rad_per_us,
        "recombination_tail": 0.0,
        "zero_hyperfine_sanity": bool(args.zero_hyperfine_sanity),
        "nuclei": [
            {
                "radical": nucleus.radical,
                "name": nucleus.name,
                "spin": nucleus.spin,
                "dimension": int(round(2.0 * nucleus.spin + 1.0)),
            }
            for nucleus in nuclei
        ],
        "theta_count": len(rows),
        "theta_min_deg": min(row["theta_deg"] for row in rows),
        "theta_max_deg": max(row["theta_deg"] for row in rows),
        "summary_by_lifetime": summary_by_tau,
        "outputs": output_paths,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_by_tau


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small seven-angle smoke test.")
    parser.add_argument("--selection", choices=["paper14", "balanced-tail14"], default="paper14")
    parser.add_argument("--zero-hyperfine-sanity", action="store_true", help="Set all hyperfine tensors to zero; Phi_S should stay at one.")
    parser.add_argument("--b-microt", type=float, default=50.0)
    parser.add_argument("--lifetimes-us", type=str, default=",".join(f"{value:g}" for value in DEFAULT_LIFETIMES_US))
    parser.add_argument("--coarse-step-deg", type=float, default=2.0)
    parser.add_argument("--fine-step-deg", type=float, default=0.1)
    parser.add_argument("--ultrafine-step-deg", type=float, default=0.02)
    parser.add_argument("--frequency-bin-width-rad-per-us", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "quantum_needle_angular_sensitivity")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frequency_bin_width_rad_per_us <= 0.0:
        raise ValueError("--frequency-bin-width-rad-per-us must be positive.")
    lifetimes_us = [1.0, 10.0] if args.quick else parse_lifetimes(args.lifetimes_us)
    nuclei = quantum_needle_nuclei(args.zero_hyperfine_sanity, args.selection)
    thetas = theta_grid(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("Quantum Needle angular sensitivity")
    print("----------------------------------")
    print(f"selection: {args.selection}")
    print(f"nuclei: {len(nuclei)} total; FAD={len(radical_nuclei(nuclei, 'FAD'))}, Trp={len(radical_nuclei(nuclei, 'Trp'))}")
    print(f"theta values: {len(thetas)}")
    print(f"lifetimes: {', '.join(f'{tau:g}' for tau in lifetimes_us)} us")
    print(f"frequency bin width: {args.frequency_bin_width_rad_per_us:g} rad/us")
    print()

    rows: list[dict[str, float]] = []
    for index, theta_deg in enumerate(thetas, start=1):
        print(f"[{index:03d}/{len(thetas):03d}] theta={theta_deg:g} deg")
        rows.append(
            run_theta(
                nuclei,
                float(theta_deg),
                lifetimes_us,
                args.b_microt,
                args.frequency_bin_width_rad_per_us,
            )
        )
    add_delta_columns(rows, lifetimes_us)

    if args.zero_hyperfine_sanity:
        for row in rows:
            for lifetime in lifetimes_us:
                key = f"phi_s_tau_{lifetime_key(lifetime)}us"
                if abs(row[key] - 1.0) > 5e-4:
                    raise ValueError(f"Zero-hyperfine sanity failed at theta={row['theta_deg']}: {key}={row[key]}")

    theta_csv = args.output_dir / "quantum_needle_theta_scan.csv"
    tensors_mT = args.output_dir / "quantum_needle_hyperfine_tensors_mT.csv"
    tensors_MHz = args.output_dir / "quantum_needle_hyperfine_tensors_MHz.csv"
    summary_json = args.output_dir / "quantum_needle_summary.json"
    phi_plot = args.output_dir / "phi_s_vs_theta_all_tau.png"
    delta_plot = args.output_dir / "delta_phi_s_vs_theta_all_tau.png"
    zoom_plot = args.output_dir / "phi_s_zoom_90deg_all_tau.png"
    dip_plot = args.output_dir / "dip_depth_vs_tau.png"

    save_theta_csv(rows, lifetimes_us, theta_csv)
    save_tensor_csv(nuclei, tensors_mT, "mT")
    save_tensor_csv(nuclei, tensors_MHz, "MHz")
    plot_yield(rows, lifetimes_us, phi_plot, delta=False, zoom=False, show=args.show_plot)
    plot_yield(rows, lifetimes_us, delta_plot, delta=True, zoom=False, show=args.show_plot)
    plot_yield(rows, lifetimes_us, zoom_plot, delta=False, zoom=True, show=args.show_plot)

    output_paths = {
        "theta_csv": str(theta_csv),
        "hyperfine_tensors_mT": str(tensors_mT),
        "hyperfine_tensors_MHz": str(tensors_MHz),
        "summary_json": str(summary_json),
        "phi_s_vs_theta_all_tau": str(phi_plot),
        "delta_phi_s_vs_theta_all_tau": str(delta_plot),
        "phi_s_zoom_90deg_all_tau": str(zoom_plot),
        "dip_depth_vs_tau": str(dip_plot),
    }
    summary_by_tau = save_summary(args, nuclei, rows, lifetimes_us, output_paths, summary_json)
    plot_dip_depth(summary_by_tau, lifetimes_us, dip_plot, args.show_plot)

    print()
    print("Saved Quantum Needle outputs")
    print(f"CSV: {theta_csv}")
    print(f"Summary: {summary_json}")
    print(f"Plots: {args.output_dir}")


if __name__ == "__main__":
    main()
