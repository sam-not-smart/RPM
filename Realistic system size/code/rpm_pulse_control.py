"""Closed-system pulse-control screening for RPM magnetosensitivity.

The pulses are ideal instantaneous global pi rotations on both electron spins.
This is a coherent-control screening tool, not an RF bandwidth or open-system
model.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm_multiply

import rpm_full_fad_trp_magnetosensitivity as rpm


@dataclass(frozen=True)
class PulseProtocol:
    name: str
    axis: str | None
    pulse_times_us: tuple[float, ...]


def parse_protocol(name: str, t_max_us: float) -> PulseProtocol:
    normalized = name.strip().lower()
    if normalized == "none":
        return PulseProtocol(name="none", axis=None, pulse_times_us=())

    match = re.fullmatch(r"hahn_([xy])", normalized)
    if match:
        axis = match.group(1)
        return PulseProtocol(name=f"hahn_{axis}", axis=axis, pulse_times_us=(0.5 * t_max_us,))

    match = re.fullmatch(r"cpmg(\d+)_([xy])", normalized)
    if match:
        count = int(match.group(1))
        if count < 1:
            raise ValueError("CPMG pulse count must be positive.")
        axis = match.group(2)
        pulse_times = tuple((j - 0.5) * t_max_us / count for j in range(1, count + 1))
        return PulseProtocol(name=f"cpmg{count}_{axis}", axis=axis, pulse_times_us=pulse_times)

    raise ValueError(f"Unsupported protocol '{name}'. Use none, hahn_x/y, or cpmgN_x/y.")


def global_pi_pulse(state: np.ndarray, axis: str, n_qubits: int) -> np.ndarray:
    """Apply an ideal pi pulse to both electron spins around x or y.

    The physical unitary differs from sigma_axis(1) sigma_axis(2) by a global
    phase, which has no observable effect here.
    """
    return rpm.apply_pauli_product(state, ((axis, 0), (axis, 1)), n_qubits)


def evolve_interval(hamiltonian, state: np.ndarray, start_us: float, stop_us: float, output_times: np.ndarray) -> np.ndarray:
    if len(output_times) == 0:
        return np.empty((0, len(state)), dtype=np.complex128)
    if abs(stop_us - start_us) < 1e-15:
        return np.repeat(state[None, :], len(output_times), axis=0)
    shifted_times = np.asarray(output_times, dtype=float) - float(start_us)
    end = float(stop_us - start_us)
    if len(output_times) == 1:
        evolved = expm_multiply(-1.0j * hamiltonian * shifted_times[0], state, traceA=0.0)
        return evolved[None, :]
    # expm_multiply interval returns states at evenly spaced points. Our output
    # grid segments are evenly spaced whenever pulse times align with grid
    # points; otherwise fall back to one-point calls to avoid interpolation.
    diffs = np.diff(shifted_times)
    if len(diffs) > 0 and np.allclose(diffs, diffs[0], rtol=1e-10, atol=1e-12):
        return expm_multiply(
            -1.0j * hamiltonian,
            state,
            start=float(shifted_times[0]),
            stop=float(shifted_times[-1]),
            num=len(output_times),
            endpoint=True,
            traceA=0.0,
        )
    return np.asarray([expm_multiply(-1.0j * hamiltonian * t, state, traceA=0.0) for t in shifted_times])


def evolve_with_protocol(
    hamiltonian,
    psi0: np.ndarray,
    t_us: np.ndarray,
    protocol: PulseProtocol,
    n_qubits: int,
) -> np.ndarray:
    if len(protocol.pulse_times_us) == 0:
        return expm_multiply(
            -1.0j * hamiltonian,
            psi0,
            start=float(t_us[0]),
            stop=float(t_us[-1]),
            num=len(t_us),
            endpoint=True,
            traceA=0.0,
        )

    states = np.empty((len(t_us), len(psi0)), dtype=np.complex128)
    current_state = psi0
    current_time = float(t_us[0])
    cursor = 0

    boundaries = list(protocol.pulse_times_us) + [float(t_us[-1])]
    for boundary in boundaries:
        segment_mask = (t_us >= current_time - 1e-12) & (t_us <= boundary + 1e-12)
        segment_indices = np.where(segment_mask)[0]
        segment_indices = segment_indices[segment_indices >= cursor]
        segment_times = t_us[segment_indices]

        segment_states = evolve_interval(hamiltonian, current_state, current_time, boundary, segment_times)
        for idx, state in zip(segment_indices, segment_states):
            states[idx] = state
        if len(segment_states) > 0:
            current_state = segment_states[-1]
            current_time = float(segment_times[-1])

        if boundary < t_us[-1] - 1e-12:
            if current_time < boundary - 1e-12:
                current_state = expm_multiply(-1.0j * hamiltonian * (boundary - current_time), current_state, traceA=0.0)
                current_time = boundary
            current_state = global_pi_pulse(current_state, protocol.axis or "y", n_qubits)
            current_state /= np.linalg.norm(current_state)
            cursor = int(np.searchsorted(t_us, boundary, side="right"))
        else:
            break

    return states


def run_protocol_orientation(
    params: rpm.Params,
    nuclei: list[rpm.Nucleus],
    protocol: PulseProtocol,
    theta_deg: float,
    seed: int,
) -> dict[str, object]:
    n_nuclei = len(nuclei)
    n_qubits = 2 + n_nuclei
    terms = rpm.build_terms(params, nuclei, theta_deg)
    hamiltonian = rpm.make_hamiltonian_operator(terms, n_qubits, params)
    t_us = np.linspace(0.0, params.t_max_us, params.n_steps)
    rng = np.random.default_rng(seed)
    psi0 = rpm.initial_state(n_nuclei, rng)

    states = evolve_with_protocol(hamiltonian, psi0, t_us, protocol, n_qubits)
    norms = np.linalg.norm(states, axis=1)
    populations = np.array([rpm.singlet_population(state, n_nuclei) for state in states], dtype=float)
    yield_value = rpm.singlet_yield(t_us, populations, params.recombination_rate_per_us)

    return {
        "protocol": protocol.name,
        "theta_deg": float(theta_deg),
        "phi_deg": float(params.phi_deg),
        "singlet_yield": float(yield_value),
        "singlet_population_initial": float(populations[0]),
        "singlet_population_final": float(populations[-1]),
        "singlet_population_min": float(np.min(populations)),
        "singlet_population_max": float(np.max(populations)),
        "state_norm_min": float(np.min(norms)),
        "state_norm_max": float(np.max(norms)),
        "t_us": t_us,
        "singlet_population": populations,
    }


def run_screen(params: rpm.Params, protocol_names: list[str]) -> tuple[list[dict[str, object]], list[rpm.Nucleus]]:
    nuclei = rpm.selected_nuclei(params.max_nuclei)
    protocols = [parse_protocol(name, params.t_max_us) for name in protocol_names]
    results: list[dict[str, object]] = []
    total = len(protocols) * len(params.theta_values_deg)

    print("Closed-system pulse-control screen")
    print("----------------------------------")
    print(f"nuclei: {len(nuclei)} ({2 + len(nuclei)} qubits)")
    print(f"protocols: {', '.join(protocol.name for protocol in protocols)}")
    print(f"theta values: {', '.join(f'{theta:g}' for theta in params.theta_values_deg)}")
    print(f"t_max: {params.t_max_us:g} us, n_steps: {params.n_steps}")
    print(f"J: {params.J_MHz:g} MHz, hyperfine mode: {params.hyperfine_mode}")
    print()

    counter = 0
    for protocol in protocols:
        for theta_deg in params.theta_values_deg:
            counter += 1
            print(f"[{counter:02d}/{total:02d}] protocol={protocol.name}, theta={theta_deg:g} deg")
            results.append(
                run_protocol_orientation(
                    params,
                    nuclei,
                    protocol,
                    float(theta_deg),
                    # Match the production sweep convention: reuse the same
                    # bath sample across theta and protocols.
                    seed=params.seed,
                )
            )
    return results, nuclei


def summarize_by_protocol(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    protocols = sorted({str(row["protocol"]) for row in results})
    for protocol in protocols:
        rows = [row for row in results if row["protocol"] == protocol]
        theta = np.array([row["theta_deg"] for row in rows], dtype=float)
        yields = np.array([row["singlet_yield"] for row in rows], dtype=float)
        min_idx = int(np.argmin(yields))
        max_idx = int(np.argmax(yields))
        mean_yield = float(np.mean(yields))
        contrast = float(yields[max_idx] - yields[min_idx])
        summary[protocol] = {
            "min_yield": float(yields[min_idx]),
            "theta_min_yield_deg": float(theta[min_idx]),
            "max_yield": float(yields[max_idx]),
            "theta_max_yield_deg": float(theta[max_idx]),
            "absolute_contrast": contrast,
            "relative_contrast": float(contrast / mean_yield) if mean_yield != 0.0 else float("nan"),
            "mean_yield": mean_yield,
        }
    return summary


def save_results_csv(results: list[dict[str, object]], summary: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "protocol",
        "theta_deg",
        "phi_deg",
        "singlet_yield",
        "singlet_population_initial",
        "singlet_population_final",
        "singlet_population_min",
        "singlet_population_max",
        "state_norm_min",
        "state_norm_max",
        "protocol_min_yield",
        "protocol_theta_min_yield_deg",
        "protocol_max_yield",
        "protocol_theta_max_yield_deg",
        "protocol_absolute_contrast",
        "protocol_relative_contrast",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in results:
            protocol_summary = summary[str(row["protocol"])]
            writer.writerow(
                {
                    **{column: row[column] for column in columns[:10]},
                    "protocol_min_yield": protocol_summary["min_yield"],
                    "protocol_theta_min_yield_deg": protocol_summary["theta_min_yield_deg"],
                    "protocol_max_yield": protocol_summary["max_yield"],
                    "protocol_theta_max_yield_deg": protocol_summary["theta_max_yield_deg"],
                    "protocol_absolute_contrast": protocol_summary["absolute_contrast"],
                    "protocol_relative_contrast": protocol_summary["relative_contrast"],
                }
            )


def save_summary_csv(summary: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "protocol",
        "min_yield",
        "theta_min_yield_deg",
        "max_yield",
        "theta_max_yield_deg",
        "absolute_contrast",
        "relative_contrast",
        "mean_yield",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for protocol, values in summary.items():
            writer.writerow({"protocol": protocol, **values})


def plot_yield_vs_angle(results: list[dict[str, object]], path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for protocol in sorted({str(row["protocol"]) for row in results}):
        rows = sorted([row for row in results if row["protocol"] == protocol], key=lambda row: row["theta_deg"])
        ax.plot(
            [row["theta_deg"] for row in rows],
            [row["singlet_yield"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=protocol,
        )
    ax.set_xlabel("magnetic-field polar angle theta (deg)")
    ax.set_ylabel("singlet yield Phi_S")
    ax.set_title("Pulse-control magnetosensitivity screen")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_representative_traces(results: list[dict[str, object]], theta_deg: float, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for protocol in sorted({str(row["protocol"]) for row in results}):
        candidates = [row for row in results if row["protocol"] == protocol]
        row = min(candidates, key=lambda candidate: abs(float(candidate["theta_deg"]) - theta_deg))
        ax.plot(row["t_us"], row["singlet_population"], linewidth=1.8, label=f"{protocol}, theta={row['theta_deg']:g}")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("singlet population P_S(t)")
    ax.set_title("Representative pulse-control singlet dynamics")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def print_summary(summary: dict[str, dict[str, float]], output_dir: Path) -> None:
    print()
    print("Pulse-control summary")
    print("---------------------")
    for protocol, values in sorted(summary.items(), key=lambda item: item[1]["relative_contrast"], reverse=True):
        print(
            f"{protocol:10s} contrast={values['absolute_contrast']:.8e} "
            f"relative={values['relative_contrast']:.8e} "
            f"min={values['min_yield']:.8f}@{values['theta_min_yield_deg']:.1f} "
            f"max={values['max_yield']:.8f}@{values['theta_max_yield_deg']:.1f}"
        )
    print(f"saved outputs: {output_dir}")


def verify_no_pulse_matches_direct(params: rpm.Params) -> None:
    nuclei = rpm.selected_nuclei(params.max_nuclei)
    theta = float(params.theta_values_deg[0])
    protocol = parse_protocol("none", params.t_max_us)
    rng = np.random.default_rng(params.seed)
    direct_populations, direct_yield = rpm.evolve_one_sample(params, nuclei, theta, rng)
    pulsed = run_protocol_orientation(params, nuclei, protocol, theta, params.seed)
    population_rms = float(np.sqrt(np.mean((direct_populations - pulsed["singlet_population"]) ** 2)))
    yield_delta = abs(float(direct_yield) - float(pulsed["singlet_yield"]))
    if population_rms > 1e-10 or yield_delta > 1e-10:
        raise AssertionError(
            "No-pulse pulse-control path does not match direct evolution: "
            f"population_rms={population_rms:.3e}, yield_delta={yield_delta:.3e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nuclei", type=int, default=16)
    parser.add_argument("--protocols", default="none,hahn_y,cpmg2_y,cpmg4_y")
    parser.add_argument("--theta-start", type=float, default=0.0)
    parser.add_argument("--theta-stop", type=float, default=180.0)
    parser.add_argument("--theta-step", type=float, default=45.0)
    parser.add_argument("--trace-theta", type=float, default=90.0)
    parser.add_argument("--t-max-us", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=101)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--J-MHz", type=float, default=3.0)
    parser.add_argument("--hyperfine-mode", choices=["full", "diagonal", "z-only"], default="full")
    parser.add_argument("--operator-mode", choices=["auto", "sparse", "matrix-free"], default="sparse")
    parser.add_argument("--physical-dipole", action="store_true")
    parser.add_argument("--no-dipole", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "pulse_control")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--verify-no-pulse", action="store_true")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> rpm.Params:
    params = rpm.Params()
    params.max_nuclei = args.max_nuclei
    params.n_samples = 1
    params.seed = args.seed
    params.theta_values_deg = rpm.parse_angles(args.theta_start, args.theta_stop, args.theta_step)
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.J_MHz = args.J_MHz
    params.hyperfine_mode = args.hyperfine_mode
    params.operator_mode = args.operator_mode
    params.include_dipole = not args.no_dipole
    params.optimize_dipole_axis = not args.physical_dipole
    return params


def main() -> None:
    args = parse_args()
    params = params_from_args(args)
    protocol_names = [name.strip() for name in args.protocols.split(",") if name.strip()]
    if args.verify_no_pulse:
        verify_no_pulse_matches_direct(params)
        print("Verified no-pulse pulse-control path matches direct evolution.")

    results, nuclei = run_screen(params, protocol_names)
    summary = summarize_by_protocol(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_results_csv(results, summary, args.output_dir / "pulse_control_results.csv")
    save_summary_csv(summary, args.output_dir / "pulse_control_summary.csv")
    rpm.save_nuclei_csv(nuclei, args.output_dir / "selected_hyperfine_tensors.csv")
    plot_yield_vs_angle(results, args.output_dir / "pulse_yield_vs_angle.png", args.show_plot)
    plot_representative_traces(results, args.trace_theta, args.output_dir / "pulse_singlet_population_vs_time.png", args.show_plot)
    print_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
