"""Entanglement dynamics comparison for the scalable FAD-Trp RPM model.

This recreates the structure of thesis Figures 6.11 and 6.12 for the current
larger FAD-Trp model by comparing subsystem von Neumann entropies with and
without electron-electron exchange/dipolar interactions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm_multiply

import rpm_full_fad_trp_magnetosensitivity as rpm


def entropy_bits_from_probabilities(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 1e-14]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def pure_state_entropy_bits(state: np.ndarray, n_qubits: int, subsystem: list[int]) -> float:
    """Von Neumann entropy of a subsystem for a pure full-system state."""
    subsystem = sorted(subsystem)
    complement = [idx for idx in range(n_qubits) if idx not in subsystem]
    # Use the smaller side of the bipartition for cheaper SVD.
    active = subsystem if len(subsystem) <= len(complement) else complement
    rest = [idx for idx in range(n_qubits) if idx not in active]

    tensor = state.reshape((2,) * n_qubits)
    matrix = np.transpose(tensor, active + rest).reshape(2 ** len(active), -1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    probabilities = np.square(singular_values)
    probabilities /= np.sum(probabilities)
    return entropy_bits_from_probabilities(probabilities)


def selected_partitions(nuclei: list[rpm.Nucleus]) -> dict[str, list[int] | None]:
    fad_nuclei = [2 + idx for idx, nucleus in enumerate(nuclei) if nucleus.radical == "FAD"]
    return {
        "total_state": None,
        "electron_pair": [0, 1],
        "fad_radical": [0] + fad_nuclei,
    }


def product_up_initial_state(n_nuclei: int) -> np.ndarray:
    nuclear = np.zeros(2**n_nuclei, dtype=complex)
    nuclear[0] = 1.0
    return np.kron(rpm.singlet_electron_state(), nuclear)


def product_x_initial_state(n_nuclei: int) -> np.ndarray:
    nuclear = np.ones(2**n_nuclei, dtype=complex) / np.sqrt(2**n_nuclei)
    return np.kron(rpm.singlet_electron_state(), nuclear)


def initial_state_for_entropy(n_nuclei: int, rng: np.random.Generator, nuclear_state: str) -> np.ndarray:
    if nuclear_state == "random-phase":
        return rpm.initial_state(n_nuclei, rng)
    if nuclear_state == "product-up":
        return product_up_initial_state(n_nuclei)
    if nuclear_state == "product-x":
        return product_x_initial_state(n_nuclei)
    raise ValueError("nuclear_state must be one of: product-up, product-x, random-phase.")


def evolve_states(
    params: rpm.Params,
    nuclei: list[rpm.Nucleus],
    theta_deg: float,
    seed: int,
    nuclear_state: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_qubits = 2 + len(nuclei)
    terms = rpm.build_terms(params, nuclei, theta_deg)
    hamiltonian = rpm.make_hamiltonian_operator(terms, n_qubits, params)
    t_us = np.linspace(0.0, params.t_max_us, params.n_steps)
    rng = np.random.default_rng(seed)
    psi0 = initial_state_for_entropy(len(nuclei), rng, nuclear_state)

    if params.evolver == "rk4":
        states = rpm.evolve_rk4(hamiltonian, psi0, t_us)
    else:
        states = expm_multiply(
            -1.0j * hamiltonian,
            psi0,
            start=0.0,
            stop=params.t_max_us,
            num=params.n_steps,
            endpoint=True,
            traceA=0.0,
        )
    return t_us, states


def analyze_case(
    base_params: rpm.Params,
    nuclei: list[rpm.Nucleus],
    theta_deg: float,
    seed: int,
    interactions_on: bool,
    nuclear_state: str,
) -> dict[str, np.ndarray]:
    params = rpm.Params()
    params.__dict__.update(base_params.__dict__)
    params.J_MHz = base_params.J_MHz if interactions_on else 0.0
    params.include_dipole = interactions_on

    t_us, states = evolve_states(params, nuclei, theta_deg, seed, nuclear_state)
    n_qubits = 2 + len(nuclei)
    partitions = selected_partitions(nuclei)

    data: dict[str, np.ndarray] = {
        "t_us": t_us,
        "singlet_population": np.array([rpm.singlet_population(state, len(nuclei)) for state in states]),
        "total_state": np.zeros(len(t_us), dtype=float),
    }

    for label, subsystem in partitions.items():
        if subsystem is None:
            continue
        data[label] = np.array([pure_state_entropy_bits(state, n_qubits, subsystem) for state in states], dtype=float)

    return data


def save_case_csv(case: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["t_us", "total_state", "electron_pair", "fad_radical", "singlet_population"]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for index in range(len(case["t_us"])):
            writer.writerow([case[column][index] for column in columns])


def plot_entropy_case(case: dict[str, np.ndarray], title: str, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(case["t_us"], case["total_state"], linewidth=1.8, label="total state")
    ax.plot(case["t_us"], case["electron_pair"], linewidth=1.8, label="electron pair")
    ax.plot(case["t_us"], case["fad_radical"], linewidth=1.8, label="FAD radical")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("von Neumann entropy (bits)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_singlet_comparison(off_case: dict[str, np.ndarray], on_case: dict[str, np.ndarray], path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(off_case["t_us"], off_case["singlet_population"], linewidth=1.8, label="J/D off")
    ax.plot(on_case["t_us"], on_case["singlet_population"], linewidth=1.8, label="J/D on")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("singlet population P_S(t)")
    ax.set_title("Singlet dynamics with and without exchange/dipole")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nuclei", type=int, default=16)
    parser.add_argument("--theta", type=float, default=90.0)
    parser.add_argument("--t-max-us", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=101)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--nuclear-state", choices=["product-up", "product-x", "random-phase"], default="product-x")
    parser.add_argument("--J-MHz", type=float, default=3.0)
    parser.add_argument("--hyperfine-mode", choices=["full", "diagonal", "z-only"], default="full")
    parser.add_argument("--operator-mode", choices=["auto", "sparse", "matrix-free"], default="sparse")
    parser.add_argument("--evolver", choices=["expm", "rk4"], default="expm")
    parser.add_argument("--physical-dipole", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "entanglement_dynamics")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> rpm.Params:
    params = rpm.Params()
    params.max_nuclei = args.max_nuclei
    params.n_samples = 1
    params.seed = args.seed
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.J_MHz = args.J_MHz
    params.hyperfine_mode = args.hyperfine_mode
    params.operator_mode = args.operator_mode
    params.evolver = args.evolver
    params.optimize_dipole_axis = not args.physical_dipole
    return params


def main() -> None:
    args = parse_args()
    params = params_from_args(args)
    nuclei = rpm.selected_nuclei(args.max_nuclei)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Entanglement dynamics comparison")
    print("--------------------------------")
    print(f"nuclei: {len(nuclei)} ({2 + len(nuclei)} qubits)")
    print(f"theta: {args.theta:g} deg")
    print(f"t_max: {args.t_max_us:g} us, n_steps: {args.n_steps}")
    print(f"hyperfine mode: {args.hyperfine_mode}")
    print(f"nuclear initial state: {args.nuclear_state}")
    print(f"operator mode: {args.operator_mode}, evolver: {args.evolver}")
    print(f"J/D on case: J={args.J_MHz:g} MHz, dipole={'physical' if args.physical_dipole else 'optimistic axial'}")

    print("\nRunning J/D off case...")
    off_case = analyze_case(params, nuclei, args.theta, args.seed, interactions_on=False, nuclear_state=args.nuclear_state)
    print("Running J/D on case...")
    on_case = analyze_case(params, nuclei, args.theta, args.seed, interactions_on=True, nuclear_state=args.nuclear_state)

    save_case_csv(off_case, args.output_dir / "entropy_without_exchange_dipole.csv")
    save_case_csv(on_case, args.output_dir / "entropy_with_exchange_dipole.csv")
    plot_entropy_case(off_case, "Subsystem entropies without exchange/dipole", args.output_dir / "fig_6_11_style_without_exchange_dipole.png", args.show_plot)
    plot_entropy_case(on_case, "Subsystem entropies with exchange/dipole", args.output_dir / "fig_6_12_style_with_exchange_dipole.png", args.show_plot)
    plot_singlet_comparison(off_case, on_case, args.output_dir / "singlet_population_exchange_dipole_comparison.png", args.show_plot)

    print("\nSummary")
    print("-------")
    print(f"max S electron pair, off: {np.max(off_case['electron_pair']):.6f} bits")
    print(f"max S electron pair, on : {np.max(on_case['electron_pair']):.6f} bits")
    print(f"max S FAD radical, off : {np.max(off_case['fad_radical']):.6f} bits")
    print(f"max S FAD radical, on  : {np.max(on_case['fad_radical']):.6f} bits")
    print(f"saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
