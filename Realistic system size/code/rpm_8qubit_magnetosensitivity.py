"""Standalone 8-qubit radical-pair magnetosensitivity baseline.

This file intentionally does not import or depend on the thesis-era RPM scripts.
It builds a sparse Hamiltonian for:

    2 electron spins + 6 spin-1/2 nuclear spins

and tests magnetic-field-angle sensitivity using the recombination-weighted
singlet yield

    Phi_S = integral_0^inf k exp(-k t) P_S(t) dt.

The numerical integral is truncated at t_max_us. Increase t_max_us until the
reported tail probability exp(-k t_max_us) is small enough for the comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

TWOPI = 2.0 * np.pi
SQRT2 = np.sqrt(2.0)


def mhz_to_rad_per_us(value):
    """Convert ordinary frequency in MHz to angular frequency in rad/us."""
    return TWOPI * np.asarray(value, dtype=float)


def unit_vector_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Return spherical unit vector, with theta from +z and phi from +x."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )


def symmetrize(tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor, dtype=float)
    return 0.5 * (tensor + tensor.T)


@dataclass
class Params:
    """Physical and numerical parameters for the 8-qubit baseline."""

    # Magnetic field.
    B_microT: float = 50.0
    gamma_e_MHz_per_mT: float = 28.02495164
    theta_values_deg: np.ndarray = field(default_factory=lambda: np.arange(0.0, 181.0, 10.0))
    phi_deg: float = 0.0

    # Six spin-1/2 nuclei: three coupled to electron 1, three to electron 2.
    n_e1_nuclei: int = 3
    n_e2_nuclei: int = 3
    A_tensors_MHz: list[np.ndarray] = field(
        default_factory=lambda: [
            np.array(
                [[-2.44, 0.00, 0.00], [0.00, -2.80, 0.00], [0.00, 0.00, 49.24]],
                dtype=float,
            ),
            np.array(
                [[12.00, 0.00, 0.00], [0.00, 12.00, 0.00], [0.00, 0.00, 12.00]],
                dtype=float,
            ),
            np.array(
                [[-0.72, 0.08, 0.00], [0.08, -0.64, 0.00], [0.00, 0.00, 1.40]],
                dtype=float,
            ),
            np.array(
                [[-1.70, 0.00, 0.00], [0.00, -1.50, 0.00], [0.00, 0.00, 30.00]],
                dtype=float,
            ),
            np.array(
                [[-15.0, 0.00, 0.00], [0.0, -11.0, 0.00], [0.0, 0.00, -3.00]],
                dtype=float,
            ),
            np.array(
                [[-1.0, 0.00, 0.00], [0.0, -0.90, 0.00], [0.0, 0.00, 2.40]],
                dtype=float,
            ),
        ]
    )

    # Exchange and full non-secular dipolar interaction.
    r_nm: float = 2.0
    D_sign: float = -1.0
    ratio_D_over_J: float = -6.0
    dipole_axis_theta_deg: float = 0.0
    dipole_axis_phi_deg: float = 0.0

    # Closed-system evolution and yield.
    t_max_us: float = 10.0
    n_steps: int = 601
    recombination_rate_per_us: float = 0.5

    # Output.
    output_dir: Path = Path("outputs") / "8qubit_magnetosensitivity"
    save_plot: bool = True
    show_plot: bool = False


def copy_params(params: Params) -> Params:
    """Copy Params while avoiding shared mutable arrays."""
    return replace(
        params,
        theta_values_deg=np.array(params.theta_values_deg, dtype=float).copy(),
        A_tensors_MHz=[np.array(tensor, dtype=float).copy() for tensor in params.A_tensors_MHz],
        output_dir=Path(params.output_dir),
    )


def spin_operators() -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Dimensionless spin-1/2 operators sigma/2 and identity."""
    sx = 0.5 * sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    sy = 0.5 * sp.csr_matrix(np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex))
    sz = 0.5 * sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))
    ident = sp.identity(2, dtype=complex, format="csr")
    return sx, sy, sz, ident


def kron_all(operators: Iterable[sp.spmatrix]) -> sp.csr_matrix:
    """Kronecker product over all operators, preserving qubit order."""
    iterator = iter(operators)
    result = next(iterator).tocsr()
    for operator in iterator:
        result = sp.kron(result, operator, format="csr")
    return result


def op_on_qubit(n_qubits: int, qubit_index: int, operator: sp.spmatrix) -> sp.csr_matrix:
    """Embed a one-qubit operator in the full Hilbert space."""
    _sx, _sy, _sz, ident = spin_operators()
    ops = [ident] * n_qubits
    ops[qubit_index] = operator
    return kron_all(ops)


def build_embedded_spin_ops(n_qubits: int) -> dict[str, list[sp.csr_matrix]]:
    sx, sy, sz, _ident = spin_operators()
    return {
        "x": [op_on_qubit(n_qubits, idx, sx) for idx in range(n_qubits)],
        "y": [op_on_qubit(n_qubits, idx, sy) for idx in range(n_qubits)],
        "z": [op_on_qubit(n_qubits, idx, sz) for idx in range(n_qubits)],
    }


def dipolar_uT_from_r_nm(r_nm: float) -> float:
    """Point-dipole coupling magnitude in microtesla: |D| = 2783 / r_nm^3."""
    return 2.783e3 / (r_nm**3)


def exchange_and_dipolar_couplings_MHz(params: Params) -> tuple[float, float]:
    """Return J_MHz and D_MHz from the configured D/J ratio."""
    if abs(params.ratio_D_over_J) < 1e-15:
        raise ValueError("ratio_D_over_J cannot be zero.")
    D_uT = params.D_sign * dipolar_uT_from_r_nm(params.r_nm)
    J_uT = D_uT / params.ratio_D_over_J
    D_MHz = params.gamma_e_MHz_per_mT * 1e-3 * D_uT
    J_MHz = params.gamma_e_MHz_per_mT * 1e-3 * J_uT
    return J_MHz, D_MHz


def electron_for_nucleus(params: Params, nucleus_index: int) -> int:
    """Map nucleus index 0..5 to electron qubit 0 or 1."""
    if nucleus_index < params.n_e1_nuclei:
        return 0
    if nucleus_index < params.n_e1_nuclei + params.n_e2_nuclei:
        return 1
    raise ValueError("Nucleus index exceeds configured electron-nuclear mapping.")


def validate_params(params: Params) -> None:
    if len(params.A_tensors_MHz) != 6:
        raise ValueError("This baseline expects exactly six hyperfine tensors.")
    if params.n_e1_nuclei + params.n_e2_nuclei != 6:
        raise ValueError("This baseline expects exactly six nuclei.")
    if params.n_steps < 2:
        raise ValueError("n_steps must be at least 2.")
    if params.t_max_us <= 0.0:
        raise ValueError("t_max_us must be positive.")
    if params.recombination_rate_per_us <= 0.0:
        raise ValueError("recombination_rate_per_us must be positive.")


def build_hamiltonian(params: Params, theta_deg: float) -> tuple[sp.csr_matrix, dict[str, float]]:
    """Build H = H_Z + H_hyperfine + H_exchange + H_full_dipole."""
    validate_params(params)

    n_qubits = 8
    dim = 2**n_qubits
    ops = build_embedded_spin_ops(n_qubits)
    sx, sy, sz = ops["x"], ops["y"], ops["z"]
    identity = sp.identity(dim, dtype=complex, format="csr")
    hamiltonian = sp.csr_matrix((dim, dim), dtype=complex)

    B_mT = 1e-3 * params.B_microT * unit_vector_from_angles(theta_deg, params.phi_deg)
    omega_e = mhz_to_rad_per_us(params.gamma_e_MHz_per_mT) * B_mT
    hamiltonian += omega_e[0] * (sx[0] + sx[1])
    hamiltonian += omega_e[1] * (sy[0] + sy[1])
    hamiltonian += omega_e[2] * (sz[0] + sz[1])

    spin_axes = [sx, sy, sz]
    for nucleus_index, tensor_MHz in enumerate(params.A_tensors_MHz):
        electron_index = electron_for_nucleus(params, nucleus_index)
        nuclear_qubit_index = 2 + nucleus_index
        tensor_w = mhz_to_rad_per_us(symmetrize(tensor_MHz))
        for a in range(3):
            for b in range(3):
                coefficient = tensor_w[a, b]
                if abs(coefficient) > 0.0:
                    hamiltonian += coefficient * (
                        spin_axes[a][electron_index] @ spin_axes[b][nuclear_qubit_index]
                    )

    J_MHz, D_MHz = exchange_and_dipolar_couplings_MHz(params)
    Jw = mhz_to_rad_per_us(J_MHz)
    Dw = mhz_to_rad_per_us(D_MHz)

    s_total_x = sx[0] + sx[1]
    s_total_y = sy[0] + sy[1]
    s_total_z = sz[0] + sz[1]
    s_total_squared = s_total_x @ s_total_x + s_total_y @ s_total_y + s_total_z @ s_total_z
    hamiltonian += -Jw * (s_total_squared - identity)

    rhat = unit_vector_from_angles(params.dipole_axis_theta_deg, params.dipole_axis_phi_deg)
    s1_dot_s2 = sx[0] @ sx[1] + sy[0] @ sy[1] + sz[0] @ sz[1]
    s1_dot_r = rhat[0] * sx[0] + rhat[1] * sy[0] + rhat[2] * sz[0]
    s2_dot_r = rhat[0] * sx[1] + rhat[1] * sy[1] + rhat[2] * sz[1]
    hamiltonian += Dw * (s1_dot_s2 - 3.0 * (s1_dot_r @ s2_dot_r))

    metadata = {
        "J_MHz": float(J_MHz),
        "D_MHz": float(D_MHz),
        "B_mT_x": float(B_mT[0]),
        "B_mT_y": float(B_mT[1]),
        "B_mT_z": float(B_mT[2]),
    }
    return hamiltonian.tocsr(), metadata


def initial_state_batch() -> np.ndarray:
    """Return 64 pure-state columns representing |S><S| x I_nuc/64."""
    n_electron_states = 4
    n_nuclear_states = 64
    dim = n_electron_states * n_nuclear_states
    states = np.zeros((dim, n_nuclear_states), dtype=complex)

    for nuclear_index in range(n_nuclear_states):
        states[1 * n_nuclear_states + nuclear_index, nuclear_index] = 1.0 / SQRT2
        states[2 * n_nuclear_states + nuclear_index, nuclear_index] = -1.0 / SQRT2

    return states


def singlet_population_from_batch(states: np.ndarray) -> float:
    """Average singlet population over a batch of pure-state columns."""
    n_nuclear_states = 64
    reshaped = states.reshape((4, n_nuclear_states, -1))
    singlet_amplitudes = (reshaped[1, :, :] - reshaped[2, :, :]) / SQRT2
    singlet_probabilities = np.sum(np.abs(singlet_amplitudes) ** 2, axis=0)
    return float(np.mean(singlet_probabilities))


def singlet_yield(t_us: np.ndarray, singlet_population: np.ndarray, k_per_us: float) -> float:
    """Compute truncated Phi_S = integral k exp(-k t) P_S(t) dt."""
    weights = k_per_us * np.exp(-k_per_us * t_us)
    return float(np.trapezoid(weights * singlet_population, t_us))


def run_orientation(params: Params, theta_deg: float) -> dict[str, object]:
    """Evolve the 8-qubit system for one magnetic-field orientation."""
    hamiltonian, metadata = build_hamiltonian(params, theta_deg)
    t_us = np.linspace(0.0, params.t_max_us, params.n_steps)
    states_0 = initial_state_batch()

    evolved = expm_multiply(
        -1.0j * hamiltonian,
        states_0,
        start=0.0,
        stop=params.t_max_us,
        num=params.n_steps,
        endpoint=True,
    )
    populations = np.array([singlet_population_from_batch(states) for states in evolved], dtype=float)
    yield_value = singlet_yield(t_us, populations, params.recombination_rate_per_us)

    return {
        "theta_deg": float(theta_deg),
        "phi_deg": float(params.phi_deg),
        "t_us": t_us,
        "singlet_population": populations,
        "singlet_yield": yield_value,
        "singlet_population_initial": float(populations[0]),
        "singlet_population_final": float(populations[-1]),
        "singlet_population_min": float(np.min(populations)),
        "singlet_population_max": float(np.max(populations)),
        **metadata,
    }


def run_sweep(params: Params) -> list[dict[str, object]]:
    """Run the configured theta sweep."""
    validate_params(params)
    results = []
    total = len(params.theta_values_deg)
    for index, theta_deg in enumerate(params.theta_values_deg, start=1):
        print(f"[{index:02d}/{total:02d}] theta={theta_deg:g} deg")
        results.append(run_orientation(params, float(theta_deg)))
    return results


def summarize_results(results: list[dict[str, object]]) -> dict[str, float]:
    yields = np.array([row["singlet_yield"] for row in results], dtype=float)
    theta = np.array([row["theta_deg"] for row in results], dtype=float)
    min_index = int(np.argmin(yields))
    max_index = int(np.argmax(yields))
    contrast = float(yields[max_index] - yields[min_index])
    mean_yield = float(np.mean(yields))
    relative_contrast = contrast / mean_yield if mean_yield != 0.0 else np.nan
    return {
        "min_yield": float(yields[min_index]),
        "theta_min_yield_deg": float(theta[min_index]),
        "max_yield": float(yields[max_index]),
        "theta_max_yield_deg": float(theta[max_index]),
        "absolute_contrast": contrast,
        "relative_contrast": float(relative_contrast),
        "mean_yield": mean_yield,
    }


def save_results_csv(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "theta_deg",
        "phi_deg",
        "singlet_yield",
        "singlet_population_initial",
        "singlet_population_final",
        "singlet_population_min",
        "singlet_population_max",
        "J_MHz",
        "D_MHz",
        "B_mT_x",
        "B_mT_y",
        "B_mT_z",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({column: row[column] for column in columns})


def plot_results(results: list[dict[str, object]], path: Path, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.array([row["theta_deg"] for row in results], dtype=float)
    yields = np.array([row["singlet_yield"] for row in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].plot(theta, yields, marker="o", linewidth=2.0)
    axes[0].set_xlabel("magnetic-field polar angle theta (deg)")
    axes[0].set_ylabel("singlet yield Phi_S")
    axes[0].set_title("8-qubit RPM magnetosensitivity")

    selected_indices = sorted(set([0, len(results) // 2, len(results) - 1]))
    for idx in selected_indices:
        row = results[idx]
        axes[1].plot(row["t_us"], row["singlet_population"], linewidth=1.8, label=f"theta={row['theta_deg']:g} deg")
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("singlet population P_S(t)")
    axes[1].set_title("Representative singlet dynamics")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def print_summary(params: Params, summary: dict[str, float], csv_path: Path, plot_path: Path | None) -> None:
    tail = np.exp(-params.recombination_rate_per_us * params.t_max_us)
    J_MHz, D_MHz = exchange_and_dipolar_couplings_MHz(params)
    print()
    print("8-qubit RPM magnetosensitivity summary")
    print("--------------------------------------")
    print(f"B field: {params.B_microT:g} uT, phi={params.phi_deg:g} deg")
    print(f"D/J: {params.ratio_D_over_J:g}, J={J_MHz:.6f} MHz, D={D_MHz:.6f} MHz")
    print(f"k: {params.recombination_rate_per_us:g} us^-1, t_max={params.t_max_us:g} us")
    print(f"truncated recombination tail exp(-kT): {tail:.3e}")
    print(f"min Phi_S: {summary['min_yield']:.8f} at theta={summary['theta_min_yield_deg']:.3f} deg")
    print(f"max Phi_S: {summary['max_yield']:.8f} at theta={summary['theta_max_yield_deg']:.3f} deg")
    print(f"absolute contrast: {summary['absolute_contrast']:.8e}")
    print(f"relative contrast: {summary['relative_contrast']:.8e}")
    print(f"saved CSV: {csv_path}")
    if plot_path is not None:
        print(f"saved plot: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short five-angle sanity check.")
    parser.add_argument("--theta-step", type=float, default=10.0, help="Theta sweep step in degrees.")
    parser.add_argument("--theta-start", type=float, default=0.0, help="Starting theta in degrees.")
    parser.add_argument("--theta-stop", type=float, default=180.0, help="Final theta in degrees.")
    parser.add_argument("--phi", type=float, default=0.0, help="Magnetic-field azimuthal angle in degrees.")
    parser.add_argument("--b-microt", type=float, default=50.0, help="Magnetic-field magnitude in microtesla.")
    parser.add_argument("--ratio", type=float, default=-6.0, help="Dipolar/exchange ratio D/J.")
    parser.add_argument("--k", type=float, default=0.5, help="Recombination rate in us^-1.")
    parser.add_argument("--t-max-us", type=float, default=10.0, help="Simulation time window in microseconds.")
    parser.add_argument("--n-steps", type=int, default=601, help="Number of stored time points.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "8qubit_magnetosensitivity")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--show-plot", action="store_true", help="Show plot interactively after saving.")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> Params:
    params = Params()
    params.phi_deg = args.phi
    params.B_microT = args.b_microt
    params.ratio_D_over_J = args.ratio
    params.recombination_rate_per_us = args.k
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.output_dir = args.output_dir
    params.save_plot = not args.no_plot
    params.show_plot = args.show_plot

    if args.quick:
        params.theta_values_deg = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        params.t_max_us = min(params.t_max_us, 2.0)
        params.n_steps = min(params.n_steps, 121)
        params.output_dir = args.output_dir / "quick"
    else:
        params.theta_values_deg = np.arange(args.theta_start, args.theta_stop + 0.5 * args.theta_step, args.theta_step)

    return params


def main() -> None:
    params = copy_params(params_from_args(parse_args()))
    results = run_sweep(params)
    summary = summarize_results(results)

    params.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = params.output_dir / "8qubit_magnetosensitivity.csv"
    save_results_csv(results, csv_path)

    plot_path = None
    if params.save_plot:
        plot_path = params.output_dir / "8qubit_magnetosensitivity.png"
        plot_results(results, plot_path, show=params.show_plot)

    print_summary(params, summary, csv_path, plot_path)


if __name__ == "__main__":
    main()
