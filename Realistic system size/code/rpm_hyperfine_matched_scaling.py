"""Matched-field hyperfine tensor scaling study for RPM magnetosensitivity.

This keeps the same nuclear qubits but scales the full anisotropic hyperfine
tensors for nuclei in selected hyperfine-strength bands. It is meant to test
whether a band helps sensitivity because of its coupling scale, without changing
Hilbert-space size by deleting nuclei.
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


def field_microT_for_frequency(frequency_MHz: float) -> float:
    return 1000.0 * frequency_MHz / rpm.GAMMA_E_MHZ_PER_MT


def strength_MHz(name: str) -> float:
    tensor = rpm.literature_tensor_MHz(name)
    symmetric = 0.5 * (tensor + tensor.T)
    return float(np.max(np.abs(np.linalg.eigvalsh(symmetric))))


def base_names_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.nuclei:
        return rpm.validate_nucleus_names(rpm.parse_nucleus_names(args.nuclei))
    preset_names = rpm.NUCLEUS_PRESETS[args.nucleus_preset]
    if preset_names is not None:
        return rpm.validate_nucleus_names(preset_names)
    return rpm.validate_nucleus_names(rpm.PUBLISHED_NUCLEUS_ORDER[: args.max_nuclei])


def band_names(base_names: tuple[str, ...], center_MHz: float, relative_half_width: float) -> tuple[str, ...]:
    lower = center_MHz * (1.0 - relative_half_width)
    upper = center_MHz * (1.0 + relative_half_width)
    return tuple(name for name in base_names if lower <= strength_MHz(name) <= upper)


def scaled_nuclei(base_names: tuple[str, ...], names_to_scale: tuple[str, ...], scale: float) -> list[rpm.Nucleus]:
    scale_set = set(names_to_scale)
    nuclei = []
    for name in base_names:
        _atom_key, radical = rpm.canonical_atom_key(name)
        tensor = rpm.literature_tensor_MHz(name)
        if name in scale_set:
            tensor = scale * tensor
        nuclei.append(rpm.Nucleus(name=name, radical=radical, tensor_MHz=tensor))
    return nuclei


def params_from_args(args: argparse.Namespace, field_frequency_MHz: float, n_nuclei: int) -> rpm.Params:
    params = rpm.Params()
    params.max_nuclei = n_nuclei
    params.nucleus_names = None
    params.n_samples = 1
    params.seed = args.seed
    params.model = "quantum"
    params.theta_values_deg = rpm.parse_angles(args.theta_start, args.theta_stop, args.theta_step)
    params.phi_deg = args.phi
    params.B_microT = field_microT_for_frequency(field_frequency_MHz)
    params.recombination_rate_per_us = args.k
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.J_MHz = args.J_MHz
    params.include_dipole = not args.no_dipole
    params.optimize_dipole_axis = not args.physical_dipole
    params.hyperfine_mode = "full"
    params.operator_mode = args.operator_mode
    params.max_sparse_qubits = args.max_sparse_qubits
    params.evolver = args.evolver
    params.save_plot = False
    params.show_plot = False
    return params


def run_custom_sweep(params: rpm.Params, nuclei: list[rpm.Nucleus]) -> list[dict[str, object]]:
    results = []
    total = len(params.theta_values_deg)
    rpm.print_model_header(params, nuclei)
    for index, theta_deg in enumerate(params.theta_values_deg, start=1):
        print(f"[{index:02d}/{total:02d}] theta={theta_deg:g} deg, samples=1")
        rng = np.random.default_rng(params.seed)
        results.append(rpm.run_orientation(params, nuclei, float(theta_deg), rng))
    return results


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summary_fieldnames() -> list[str]:
    return [
        "band_center_MHz",
        "band_lower_MHz",
        "band_upper_MHz",
        "field_factor",
        "field_frequency_MHz",
        "B_microT",
        "band_scale",
        "scaled_nuclei",
        "n_scaled",
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
        "band_center_MHz",
        "field_factor",
        "field_frequency_MHz",
        "B_microT",
        "band_scale",
        "scaled_nuclei",
        "theta_deg",
        "phi_deg",
        "singlet_yield",
        "singlet_yield_std",
        "singlet_population_initial",
        "singlet_population_final",
        "singlet_population_min",
        "singlet_population_max",
    ]


def completed_keys(rows: list[dict[str, object]]) -> set[tuple[float, float, float]]:
    return {(float(row["band_center_MHz"]), float(row["field_factor"]), float(row["band_scale"])) for row in rows}


def plot_scale_sweep(summary_rows: list[dict[str, object]], output_dir: Path, show: bool) -> None:
    centers = sorted({float(row["band_center_MHz"]) for row in summary_rows})
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for center in centers:
        rows = [row for row in summary_rows if float(row["band_center_MHz"]) == center]
        rows.sort(key=lambda row: float(row["band_scale"]))
        ax.plot(
            [float(row["band_scale"]) for row in rows],
            [float(row["relative_contrast"]) for row in rows],
            marker="o",
            linewidth=1.8,
            label=f"{center:g} MHz",
        )
    ax.set_xlabel("scaled-band hyperfine tensor multiplier")
    ax.set_ylabel("relative singlet-yield contrast")
    ax.set_title("Matched-field tensor scaling")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "matched_field_tensor_scale_contrast.png", dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_yield_shapes(detail_rows: list[dict[str, object]], output_dir: Path, show: bool) -> None:
    centers = sorted({float(row["band_center_MHz"]) for row in detail_rows})
    shape_dir = output_dir / "yield_vs_angle_by_scale"
    shape_dir.mkdir(parents=True, exist_ok=True)
    for center in centers:
        rows = [row for row in detail_rows if float(row["band_center_MHz"]) == center]
        scales = sorted({float(row["band_scale"]) for row in rows})
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for scale in scales:
            scale_rows = [row for row in rows if float(row["band_scale"]) == scale]
            scale_rows.sort(key=lambda row: float(row["theta_deg"]))
            ax.plot(
                [float(row["theta_deg"]) for row in scale_rows],
                [float(row["singlet_yield"]) for row in scale_rows],
                marker="o",
                linewidth=1.8,
                label=f"scale {scale:g}",
            )
        ax.set_xlabel("magnetic-field polar angle theta (deg)")
        ax.set_ylabel("singlet yield Phi_S")
        ax.set_title(f"Band {center:g} MHz, matched field")
        ax.legend(frameon=False)
        fig.tight_layout()
        label = f"{center:g}".replace(".", "p")
        fig.savefig(shape_dir / f"yield_vs_angle_band_{label}MHz_scale_sweep.png", dpi=180)
        if show:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nuclei", type=int, default=16)
    parser.add_argument("--nucleus-preset", choices=sorted(rpm.NUCLEUS_PRESETS), default="core-tail-16")
    parser.add_argument("--nuclei", help="Comma-separated explicit nucleus list. Overrides --nucleus-preset and --max-nuclei.")
    parser.add_argument("--centers-MHz", default="1.4,15")
    parser.add_argument("--relative-half-width", type=float, default=0.30)
    parser.add_argument("--scales", default="0,0.5,1,2")
    parser.add_argument("--field-factors", default="1")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--theta-start", type=float, default=0.0)
    parser.add_argument("--theta-stop", type=float, default=180.0)
    parser.add_argument("--theta-step", type=float, default=45.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--t-max-us", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=101)
    parser.add_argument("--J-MHz", type=float, default=3.0)
    parser.add_argument("--physical-dipole", action="store_true")
    parser.add_argument("--no-dipole", action="store_true")
    parser.add_argument("--operator-mode", choices=["auto", "sparse", "matrix-free"], default="auto")
    parser.add_argument("--max-sparse-qubits", type=int, default=18)
    parser.add_argument("--evolver", choices=["expm", "rk4"], default="expm")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "hyperfine_matched_scaling" / "pilot_B_core_tail_16_J3_2us")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_names = base_names_from_args(args)
    centers = parse_float_list(args.centers_MHz)
    scales = parse_float_list(args.scales)
    field_factors = parse_float_list(args.field_factors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "hyperfine_matched_scaling_summary.csv"
    detail_path = args.output_dir / "hyperfine_matched_scaling_results.csv"

    if args.plot_only:
        summary_rows = read_csv(summary_path)
        detail_rows = read_csv(detail_path)
        plot_scale_sweep(summary_rows, args.output_dir, args.show_plot)
        plot_yield_shapes(detail_rows, args.output_dir, args.show_plot)
        print(f"regenerated plots from existing CSV outputs: {args.output_dir}")
        return

    if args.resume:
        summary_rows: list[dict[str, object]] = read_csv(summary_path)
        detail_rows: list[dict[str, object]] = read_csv(detail_path)
        done = completed_keys(summary_rows)
        print(f"Resuming with {len(done)} completed conditions.")
    else:
        summary_rows = []
        detail_rows = []
        done = set()

    total = len(centers) * len(field_factors) * len(scales)
    counter = 0
    for center in centers:
        lower = center * (1.0 - args.relative_half_width)
        upper = center * (1.0 + args.relative_half_width)
        scaled_names = band_names(base_names, center, args.relative_half_width)
        for field_factor in field_factors:
            for scale in scales:
                counter += 1
                key = (center, field_factor, scale)
                if key in done:
                    print(f"\n[{counter:02d}/{total:02d}] skipping center={center:g}, field_factor={field_factor:g}, scale={scale:g}")
                    continue
                field_frequency = center * field_factor
                print(
                    f"\n[{counter:02d}/{total:02d}] center={center:g} MHz, "
                    f"field_factor={field_factor:g}, scale={scale:g}, "
                    f"scaled={','.join(scaled_names) or 'none'}"
                )
                params = params_from_args(args, field_frequency, len(base_names))
                nuclei = scaled_nuclei(base_names, scaled_names, scale)
                results = run_custom_sweep(params, nuclei)
                summary = rpm.summarize_results(results)
                summary_rows.append(
                    {
                        "band_center_MHz": center,
                        "band_lower_MHz": lower,
                        "band_upper_MHz": upper,
                        "field_factor": field_factor,
                        "field_frequency_MHz": field_frequency,
                        "B_microT": field_microT_for_frequency(field_frequency),
                        "band_scale": scale,
                        "scaled_nuclei": ";".join(scaled_names),
                        "n_scaled": len(scaled_names),
                        **summary,
                    }
                )
                for row in results:
                    detail_rows.append(
                        {
                            "band_center_MHz": center,
                            "field_factor": field_factor,
                            "field_frequency_MHz": field_frequency,
                            "B_microT": field_microT_for_frequency(field_frequency),
                            "band_scale": scale,
                            "scaled_nuclei": ";".join(scaled_names),
                            "theta_deg": float(row["theta_deg"]),
                            "phi_deg": float(row["phi_deg"]),
                            "singlet_yield": float(row["singlet_yield"]),
                            "singlet_yield_std": float(row["singlet_yield_std"]),
                            "singlet_population_initial": float(row["singlet_population_initial"]),
                            "singlet_population_final": float(row["singlet_population_final"]),
                            "singlet_population_min": float(row["singlet_population_min"]),
                            "singlet_population_max": float(row["singlet_population_max"]),
                        }
                    )
                write_csv(summary_path, summary_rows, summary_fieldnames())
                write_csv(detail_path, detail_rows, detail_fieldnames())

    plot_scale_sweep(summary_rows, args.output_dir, args.show_plot)
    plot_yield_shapes(detail_rows, args.output_dir, args.show_plot)
    best = max(summary_rows, key=lambda row: float(row["relative_contrast"]))
    print("\nBest setting")
    print("------------")
    print(
        f"center={float(best['band_center_MHz']):g} MHz, scale={float(best['band_scale']):g}, "
        f"relative contrast={float(best['relative_contrast']):.8e}"
    )
    print(f"saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
