"""Earth-field rescue screen for weak angular sensitivity.

This runner keeps the selected nuclear set fixed, then scans global hyperfine
tensor scale, exchange J, and electron-electron dipole mode at B = 50 uT by
default. It is meant as a targeted diagnostic for cases where the full quantum
model is nearly angle-independent because the internal interaction scales are
large compared with the Earth-field electron Larmor frequency.
"""

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


def base_names_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.nuclei:
        return rpm.validate_nucleus_names(rpm.parse_nucleus_names(args.nuclei))
    preset_names = rpm.NUCLEUS_PRESETS[args.nucleus_preset]
    if preset_names is not None:
        return rpm.validate_nucleus_names(preset_names)
    return rpm.validate_nucleus_names(rpm.PUBLISHED_NUCLEUS_ORDER[: args.max_nuclei])


def scaled_nuclei(names: tuple[str, ...], scale: float) -> list[rpm.Nucleus]:
    nuclei = rpm.selected_nuclei(len(names), names)
    return [rpm.Nucleus(nucleus.name, nucleus.radical, scale * nucleus.tensor_MHz) for nucleus in nuclei]


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


def params_from_args(args: argparse.Namespace, names: tuple[str, ...], j_MHz: float, dipole_mode: str) -> rpm.Params:
    params = rpm.Params()
    params.nucleus_names = names
    params.max_nuclei = len(names)
    params.n_samples = args.samples
    params.seed = args.seed
    params.model = "quantum"
    params.theta_values_deg = rpm.parse_angles(args.theta_start, args.theta_stop, args.theta_step)
    params.phi_deg = args.phi
    params.B_microT = args.b_microt
    params.recombination_rate_per_us = args.k
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.J_MHz = j_MHz
    params.hyperfine_mode = "full"
    params.operator_mode = args.operator_mode
    params.max_sparse_qubits = args.max_sparse_qubits
    params.evolver = args.evolver
    params.save_plot = False
    params.show_plot = False
    configure_dipole(params, dipole_mode)
    return params


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_fieldnames() -> list[str]:
    return [
        "global_hyperfine_scale",
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


def detail_fieldnames() -> list[str]:
    return [
        "global_hyperfine_scale",
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


def completed_keys(rows: list[dict[str, object]]) -> set[tuple[float, str, float]]:
    return {
        (float(row["global_hyperfine_scale"]), str(row["dipole_mode"]), float(row["J_MHz"]))
        for row in rows
    }


def plot_summary(rows: list[dict[str, object]], path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    labels = sorted({(str(row["dipole_mode"]), float(row["J_MHz"])) for row in rows})
    for dipole_mode, j_MHz in labels:
        label_rows = [
            row for row in rows if str(row["dipole_mode"]) == dipole_mode and float(row["J_MHz"]) == j_MHz
        ]
        label_rows.sort(key=lambda row: float(row["global_hyperfine_scale"]))
        ax.plot(
            [float(row["global_hyperfine_scale"]) for row in label_rows],
            [float(row["relative_contrast"]) for row in label_rows],
            marker="o",
            linewidth=1.8,
            label=f"{dipole_mode}, J={j_MHz:g} MHz",
        )
    ax.set_xscale("log")
    ax.set_xlabel("global hyperfine tensor scale")
    ax.set_ylabel("relative singlet-yield contrast")
    ax.set_title("Earth-field rescue screen")
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
    parser.add_argument("--scales", default="0.02,0.05,0.1,0.2,0.5,1,2")
    parser.add_argument("--J-values", default="0,3")
    parser.add_argument("--dipole-modes", default="none,optimistic")
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
    parser.add_argument("--operator-mode", choices=["auto", "sparse", "matrix-free"], default="auto")
    parser.add_argument("--max-sparse-qubits", type=int, default=18)
    parser.add_argument("--evolver", choices=["expm", "rk4"], default="expm")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "baseline_core_tail_16" / "earth_field_rescue_screen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = base_names_from_args(args)
    scales = parse_float_list(args.scales)
    j_values = parse_float_list(args.J_values)
    dipole_modes = parse_mode_list(args.dipole_modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "earth_field_rescue_screen_summary.csv"
    detail_path = args.output_dir / "earth_field_rescue_screen_results.csv"

    if args.plot_only:
        summary_rows = read_csv(summary_path)
        plot_summary(summary_rows, args.output_dir / "earth_field_rescue_screen_relative_contrast.png", args.show_plot)
        print(f"regenerated plot from existing CSV output: {args.output_dir}")
        return

    summary_rows: list[dict[str, object]] = read_csv(summary_path) if args.resume else []
    detail_rows: list[dict[str, object]] = read_csv(detail_path) if args.resume else []
    done = completed_keys(summary_rows)
    total = len(scales) * len(j_values) * len(dipole_modes)
    counter = 0

    for scale in scales:
        nuclei = scaled_nuclei(names, scale)
        for dipole_mode in dipole_modes:
            for j_MHz in j_values:
                counter += 1
                key = (float(scale), dipole_mode, float(j_MHz))
                if key in done:
                    print(f"\n[{counter:02d}/{total:02d}] skipping scale={scale:g}, dipole={dipole_mode}, J={j_MHz:g}")
                    continue
                print(f"\n[{counter:02d}/{total:02d}] scale={scale:g}, dipole={dipole_mode}, J={j_MHz:g} MHz")
                params = params_from_args(args, names, float(j_MHz), dipole_mode)
                results = []
                for theta_deg in params.theta_values_deg:
                    print(f"  theta={theta_deg:g} deg")
                    rng = np.random.default_rng(params.seed)
                    results.append(rpm.run_orientation(params, nuclei, float(theta_deg), rng))

                summary = rpm.summarize_results(results)
                summary_rows.append(
                    {
                        "global_hyperfine_scale": float(scale),
                        "dipole_mode": dipole_mode,
                        "J_MHz": float(j_MHz),
                        **summary,
                    }
                )
                for result in results:
                    scalar_result = {field: result[field] for field in detail_fieldnames() if field in result}
                    detail_rows.append(
                        {
                            "global_hyperfine_scale": float(scale),
                            "dipole_mode": dipole_mode,
                            "J_MHz": float(j_MHz),
                            **scalar_result,
                        }
                    )
                write_csv(summary_path, summary_rows, summary_fieldnames())
                write_csv(detail_path, detail_rows, detail_fieldnames())
                plot_summary(summary_rows, args.output_dir / "earth_field_rescue_screen_relative_contrast.png", args.show_plot)

    best = max(summary_rows, key=lambda row: float(row["relative_contrast"]))
    print("\nBest setting")
    print("------------")
    print(
        f"scale={float(best['global_hyperfine_scale']):g}, dipole={best['dipole_mode']}, "
        f"J={float(best['J_MHz']):g} MHz, relative contrast={float(best['relative_contrast']):.8e}, "
        f"absolute contrast={float(best['absolute_contrast']):.8e}"
    )
    print(f"saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
