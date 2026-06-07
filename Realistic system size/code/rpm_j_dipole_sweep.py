"""J and dipole sweep for selected FAD-Trp RPM nucleus subsets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rpm_full_fad_trp_magnetosensitivity as rpm


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def parse_mode_list(value: str) -> list[str]:
    modes = [part.strip().lower() for part in value.split(",") if part.strip()]
    allowed = {"none", "physical", "optimistic"}
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise ValueError(f"unknown dipole mode(s): {', '.join(unknown)}")
    return modes


def base_params_from_args(args: argparse.Namespace) -> rpm.Params:
    params = rpm.Params()
    params.max_nuclei = args.max_nuclei
    preset_names = rpm.NUCLEUS_PRESETS[args.nucleus_preset]
    if args.nuclei:
        params.nucleus_names = rpm.validate_nucleus_names(rpm.parse_nucleus_names(args.nuclei))
    elif preset_names is not None:
        params.nucleus_names = preset_names
    if params.nucleus_names is not None:
        params.max_nuclei = len(params.nucleus_names)
    params.n_samples = args.samples
    params.seed = args.seed
    params.theta_values_deg = rpm.parse_angles(args.theta_start, args.theta_stop, args.theta_step)
    params.phi_deg = args.phi
    params.B_microT = args.b_microt
    params.recombination_rate_per_us = args.k
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.hyperfine_mode = args.hyperfine_mode
    params.operator_mode = args.operator_mode
    params.max_sparse_qubits = args.max_sparse_qubits
    params.evolver = args.evolver
    params.save_plot = False
    params.show_plot = False
    return params


def configure_dipole(params: rpm.Params, mode: str) -> None:
    if mode == "none":
        params.include_dipole = False
        params.optimize_dipole_axis = False
    elif mode == "physical":
        params.include_dipole = True
        params.optimize_dipole_axis = False
    elif mode == "optimistic":
        params.include_dipole = True
        params.optimize_dipole_axis = True
    else:
        raise ValueError(f"unsupported dipole mode: {mode}")


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "dipole_mode",
        "J_MHz",
        "min_yield",
        "theta_min_yield_deg",
        "max_yield",
        "theta_max_yield_deg",
        "absolute_contrast",
        "relative_contrast",
        "mean_yield",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_detail(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "dipole_mode",
        "J_MHz",
        "theta_deg",
        "phi_deg",
        "singlet_yield",
        "singlet_yield_std",
        "singlet_population_initial",
        "singlet_population_final",
        "singlet_population_min",
            "singlet_population_max",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows(output_dir: Path) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    summary_path = output_dir / "j_dipole_sweep_summary.csv"
    detail_path = output_dir / "j_dipole_sweep_results.csv"
    summary_rows: list[dict[str, float | str]] = []
    detail_rows: list[dict[str, float | str]] = []
    if summary_path.exists():
        with summary_path.open(newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
    if detail_path.exists():
        with detail_path.open(newline="") as handle:
            detail_rows = list(csv.DictReader(handle))
    return summary_rows, detail_rows


def completed_keys(rows: list[dict[str, float | str]]) -> set[tuple[str, float]]:
    return {(str(row["dipole_mode"]), float(row["J_MHz"])) for row in rows}


def plot_summary(rows: list[dict[str, float | str]], path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    modes = sorted({str(row["dipole_mode"]) for row in rows})
    for mode in modes:
        mode_rows = [row for row in rows if row["dipole_mode"] == mode]
        mode_rows.sort(key=lambda row: float(row["J_MHz"]))
        ax.plot(
            [float(row["J_MHz"]) for row in mode_rows],
            [float(row["relative_contrast"]) for row in mode_rows],
            marker="o",
            linewidth=1.8,
            label=mode,
        )
    ax.set_xlabel("J (MHz)")
    ax.set_ylabel("relative singlet-yield contrast")
    ax.set_title("J/dipole sweep")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nuclei", type=int, default=16)
    parser.add_argument("--nucleus-preset", choices=sorted(rpm.NUCLEUS_PRESETS), default="core-tail-16")
    parser.add_argument("--nuclei", help="Comma-separated explicit nucleus list. Overrides --nucleus-preset and --max-nuclei.")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--theta-start", type=float, default=0.0)
    parser.add_argument("--theta-stop", type=float, default=180.0)
    parser.add_argument("--theta-step", type=float, default=45.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--b-microt", type=float, default=50.0)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--t-max-us", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=101)
    parser.add_argument("--J-values", default="-10,-5,-3,0,3,5,10")
    parser.add_argument("--dipole-modes", default="none,physical,optimistic")
    parser.add_argument("--hyperfine-mode", choices=["full", "diagonal", "z-only"], default="full")
    parser.add_argument("--operator-mode", choices=["auto", "sparse", "matrix-free"], default="auto")
    parser.add_argument("--max-sparse-qubits", type=int, default=18)
    parser.add_argument("--evolver", choices=["expm", "rk4"], default="expm")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "baseline_core_tail_16" / "j_dipole_sweep")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows already present in the output CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_params = base_params_from_args(args)
    j_values = parse_float_list(args.J_values)
    dipole_modes = parse_mode_list(args.dipole_modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        summary_rows, detail_rows = load_existing_rows(args.output_dir)
        done = completed_keys(summary_rows)
        print(f"Resuming with {len(done)} completed J/dipole settings.")
    else:
        summary_rows = []
        detail_rows = []
        done = set()
    total = len(j_values) * len(dipole_modes)
    counter = 0
    nuclei_for_save: list[rpm.Nucleus] | None = None

    for mode in dipole_modes:
        for j_value in j_values:
            counter += 1
            if (mode, float(j_value)) in done:
                print(f"\n[{counter:02d}/{total:02d}] skipping completed dipole={mode}, J={j_value:g} MHz")
                continue
            print(f"\n[{counter:02d}/{total:02d}] dipole={mode}, J={j_value:g} MHz")
            params = rpm.Params(**vars(base_params))
            params.J_MHz = float(j_value)
            configure_dipole(params, mode)
            results, nuclei = rpm.run_sweep(params)
            nuclei_for_save = nuclei
            summary = rpm.summarize_results(results)
            summary_rows.append({"dipole_mode": mode, "J_MHz": float(j_value), **summary})
            for result in results:
                detail_rows.append(
                    {
                        "dipole_mode": mode,
                        "J_MHz": float(j_value),
                        "theta_deg": float(result["theta_deg"]),
                        "phi_deg": float(result["phi_deg"]),
                        "singlet_yield": float(result["singlet_yield"]),
                        "singlet_yield_std": float(result["singlet_yield_std"]),
                        "singlet_population_initial": float(result["singlet_population_initial"]),
                        "singlet_population_final": float(result["singlet_population_final"]),
                        "singlet_population_min": float(result["singlet_population_min"]),
                        "singlet_population_max": float(result["singlet_population_max"]),
                    }
                )
            write_summary(args.output_dir / "j_dipole_sweep_summary.csv", summary_rows)
            write_detail(args.output_dir / "j_dipole_sweep_results.csv", detail_rows)

    if nuclei_for_save is None:
        nuclei_for_save = rpm.selected_nuclei(base_params.max_nuclei, base_params.nucleus_names)
    rpm.save_nuclei_csv(nuclei_for_save, args.output_dir / "selected_hyperfine_tensors.csv")
    if summary_rows:
        plot_summary(summary_rows, args.output_dir / "j_dipole_sweep_relative_contrast.png", args.show_plot)

    best = max(summary_rows, key=lambda row: float(row["relative_contrast"]))
    print("\nBest setting")
    print("------------")
    print(
        f"dipole={best['dipole_mode']}, J={float(best['J_MHz']):g} MHz, "
        f"relative contrast={float(best['relative_contrast']):.8e}, "
        f"absolute contrast={float(best['absolute_contrast']):.8e}"
    )
    print(f"saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
