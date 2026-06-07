"""Hyperfine-band ablation test for RPM magnetosensitivity.

For each hyperfine-frequency band, this runner:

1. Sets the static field strength so gamma_e B matches the band center.
2. Runs the selected full-anisotropic nucleus set.
3. Removes nuclei whose largest absolute principal hyperfine value falls in
   that band.
4. Runs the ablated set and compares singlet-yield angular contrast.

This is intended to test whether nuclei with hyperfine frequencies comparable
to an external-field electron Larmor scale are important for magnetosensitivity.
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


def strength_MHz(name: str) -> float:
    tensor = rpm.literature_tensor_MHz(name)
    symmetric = 0.5 * (tensor + tensor.T)
    return float(np.max(np.abs(np.linalg.eigvalsh(symmetric))))


def field_microT_for_frequency(frequency_MHz: float) -> float:
    return 1000.0 * frequency_MHz / rpm.GAMMA_E_MHZ_PER_MT


def nucleus_strength_rows(names: tuple[str, ...]) -> list[dict[str, float | str]]:
    rows = []
    for index, name in enumerate(names, start=1):
        _key, radical = rpm.canonical_atom_key(name)
        tensor = rpm.literature_tensor_MHz(name)
        symmetric = 0.5 * (tensor + tensor.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        rows.append(
            {
                "index": index,
                "name": name,
                "radical": radical,
                "iso_MHz": float(np.trace(tensor) / 3.0),
                "max_abs_principal_MHz": float(np.max(np.abs(eigenvalues))),
                "frobenius_MHz": float(np.linalg.norm(tensor)),
                "principal_1_MHz": float(eigenvalues[0]),
                "principal_2_MHz": float(eigenvalues[1]),
                "principal_3_MHz": float(eigenvalues[2]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def base_names_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.nuclei:
        return rpm.validate_nucleus_names(rpm.parse_nucleus_names(args.nuclei))
    preset_names = rpm.NUCLEUS_PRESETS[args.nucleus_preset]
    if preset_names is not None:
        return rpm.validate_nucleus_names(preset_names)
    return rpm.validate_nucleus_names(rpm.PUBLISHED_NUCLEUS_ORDER[: args.max_nuclei])


def params_for_names(args: argparse.Namespace, names: tuple[str, ...], field_frequency_MHz: float) -> rpm.Params:
    params = rpm.Params()
    params.nucleus_names = names
    params.max_nuclei = len(names)
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


def summarize_case(
    label: str,
    center_MHz: float,
    lower_MHz: float,
    upper_MHz: float,
    removed: tuple[str, ...],
    remaining: tuple[str, ...],
    summary: dict[str, float],
) -> dict[str, object]:
    return {
        "case": label,
        "band_center_MHz": center_MHz,
        "band_lower_MHz": lower_MHz,
        "band_upper_MHz": upper_MHz,
        "B_microT": field_microT_for_frequency(center_MHz),
        "removed_nuclei": ";".join(removed),
        "n_removed": len(removed),
        "remaining_nuclei": ";".join(remaining),
        "n_remaining": len(remaining),
        **summary,
    }


def plot_contrast(summary_rows: list[dict[str, object]], path: Path, show: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    cases = ["full", "ablated"]
    for case in cases:
        rows = [row for row in summary_rows if row["case"] == case]
        rows.sort(key=lambda row: float(row["band_center_MHz"]))
        axes[0].plot(
            [float(row["band_center_MHz"]) for row in rows],
            [float(row["relative_contrast"]) for row in rows],
            marker="o",
            linewidth=1.8,
            label=case,
        )
    centers = sorted({float(row["band_center_MHz"]) for row in summary_rows})
    losses = []
    for center in centers:
        full = next(row for row in summary_rows if row["case"] == "full" and float(row["band_center_MHz"]) == center)
        ablated = next(row for row in summary_rows if row["case"] == "ablated" and float(row["band_center_MHz"]) == center)
        losses.append(float(full["relative_contrast"]) - float(ablated["relative_contrast"]))

    axes[0].set_xlabel("matched frequency gamma_e B (MHz)")
    axes[0].set_ylabel("relative singlet-yield contrast")
    axes[0].set_title("Full vs band-ablated contrast")
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="0.4", linewidth=1.0)
    axes[1].bar([str(center) for center in centers], losses)
    axes[1].set_xlabel("removed hyperfine band center (MHz)")
    axes[1].set_ylabel("contrast loss: full - ablated")
    axes[1].set_title("Positive means removed band helped sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def safe_frequency_label(center: float) -> str:
    return f"{center:g}".replace(".", "p").replace("-", "m")


def plot_angular_shapes(detail_rows: list[dict[str, object]], output_dir: Path, show: bool) -> None:
    centers = sorted({float(row["band_center_MHz"]) for row in detail_rows})
    if not centers:
        return

    shape_dir = output_dir / "yield_vs_angle_by_band"
    shape_dir.mkdir(parents=True, exist_ok=True)

    n_cols = min(3, len(centers))
    n_rows = int(np.ceil(len(centers) / n_cols))
    overview_fig, overview_axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), squeeze=False)
    overview_axes_flat = list(overview_axes.flat)

    for panel_index, center in enumerate(centers):
        rows = [row for row in detail_rows if float(row["band_center_MHz"]) == center]
        removed = next((str(row["removed_nuclei"]) for row in rows if str(row["case"]) == "ablated"), "")
        b_microT = float(rows[0]["B_microT"])

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        overview_ax = overview_axes_flat[panel_index]
        for case, linestyle in (("full", "-"), ("ablated", "--")):
            case_rows = [row for row in rows if str(row["case"]) == case]
            case_rows.sort(key=lambda row: float(row["theta_deg"]))
            theta = [float(row["theta_deg"]) for row in case_rows]
            yield_values = [float(row["singlet_yield"]) for row in case_rows]
            label = "with band" if case == "full" else "without band"
            ax.plot(theta, yield_values, marker="o", linewidth=1.8, linestyle=linestyle, label=label)
            overview_ax.plot(theta, yield_values, marker="o", linewidth=1.5, linestyle=linestyle, label=label)

        title = f"Band {center:g} MHz, B={b_microT:.1f} uT"
        subtitle = f"removed: {removed or 'none'}"
        for target_ax in (ax, overview_ax):
            target_ax.set_xlabel("magnetic-field polar angle theta (deg)")
            target_ax.set_ylabel("singlet yield Phi_S")
            target_ax.set_title(title)
            target_ax.legend(frameon=False)
        ax.text(0.0, -0.20, subtitle, transform=ax.transAxes, fontsize=9, va="top")
        fig.tight_layout()
        fig.savefig(shape_dir / f"yield_vs_angle_band_{safe_frequency_label(center)}MHz.png", dpi=180)
        if show:
            plt.show()
        plt.close(fig)

    for ax in overview_axes_flat[len(centers) :]:
        ax.axis("off")
    overview_fig.suptitle("Hyperfine-band ablation: singlet yield vs angle", y=0.995)
    overview_fig.tight_layout()
    overview_fig.savefig(output_dir / "hyperfine_band_ablation_yield_vs_angle_grid.png", dpi=180)
    if show:
        plt.show()
    plt.close(overview_fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nuclei", type=int, default=16)
    parser.add_argument("--nucleus-preset", choices=sorted(rpm.NUCLEUS_PRESETS), default="core-tail-16")
    parser.add_argument("--nuclei", help="Comma-separated explicit nucleus list. Overrides --nucleus-preset and --max-nuclei.")
    parser.add_argument("--centers-MHz", default="1.4,2.5,4,15,30,49")
    parser.add_argument("--relative-half-width", type=float, default=0.30)
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "hyperfine_band_ablation")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate plots from existing CSV outputs without rerunning simulations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.relative_half_width <= 0.0:
        raise ValueError("relative-half-width must be positive.")
    base_names = base_names_from_args(args)
    centers = parse_float_list(args.centers_MHz)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        detail_rows = read_csv(args.output_dir / "hyperfine_band_ablation_results.csv")
        summary_rows = read_csv(args.output_dir / "hyperfine_band_ablation_summary.csv")
        plot_contrast(summary_rows, args.output_dir / "hyperfine_band_ablation_contrast.png", args.show_plot)
        plot_angular_shapes(detail_rows, args.output_dir, args.show_plot)
        print(f"regenerated plots from existing CSV outputs: {args.output_dir}")
        return

    strength_rows = nucleus_strength_rows(base_names)
    write_csv(
        args.output_dir / "nucleus_hyperfine_strengths.csv",
        strength_rows,
        [
            "index",
            "name",
            "radical",
            "iso_MHz",
            "max_abs_principal_MHz",
            "frobenius_MHz",
            "principal_1_MHz",
            "principal_2_MHz",
            "principal_3_MHz",
        ],
    )

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    total = 2 * len(centers)
    counter = 0
    for center in centers:
        lower = center * (1.0 - args.relative_half_width)
        upper = center * (1.0 + args.relative_half_width)
        strengths = {name: strength_MHz(name) for name in base_names}
        removed = tuple(name for name in base_names if lower <= strengths[name] <= upper)
        remaining = tuple(name for name in base_names if name not in removed)
        if not removed:
            print(f"\nBand {center:g} MHz removes no nuclei; ablated case equals full case.")

        for label, names, removed_for_row in (
            ("full", base_names, ()),
            ("ablated", remaining, removed),
        ):
            counter += 1
            print(
                f"\n[{counter:02d}/{total:02d}] {label}: center={center:g} MHz, "
                f"B={field_microT_for_frequency(center):.3f} uT, nuclei={len(names)}, removed={','.join(removed) or 'none'}"
            )
            params = params_for_names(args, names, center)
            results, _nuclei = rpm.run_sweep(params)
            summary = rpm.summarize_results(results)
            summary_rows.append(summarize_case(label, center, lower, upper, removed_for_row, names, summary))
            for row in results:
                detail_rows.append(
                    {
                        "case": label,
                        "band_center_MHz": center,
                        "band_lower_MHz": lower,
                        "band_upper_MHz": upper,
                        "B_microT": field_microT_for_frequency(center),
                        "removed_nuclei": ";".join(removed_for_row),
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
            write_csv(
                args.output_dir / "hyperfine_band_ablation_summary.csv",
                summary_rows,
                [
                    "case",
                    "band_center_MHz",
                    "band_lower_MHz",
                    "band_upper_MHz",
                    "B_microT",
                    "removed_nuclei",
                    "n_removed",
                    "remaining_nuclei",
                    "n_remaining",
                    "min_yield",
                    "theta_min_yield_deg",
                    "max_yield",
                    "theta_max_yield_deg",
                    "absolute_contrast",
                    "relative_contrast",
                    "mean_yield",
                ],
            )
            write_csv(
                args.output_dir / "hyperfine_band_ablation_results.csv",
                detail_rows,
                [
                    "case",
                    "band_center_MHz",
                    "band_lower_MHz",
                    "band_upper_MHz",
                    "B_microT",
                    "removed_nuclei",
                    "theta_deg",
                    "phi_deg",
                    "singlet_yield",
                    "singlet_yield_std",
                    "singlet_population_initial",
                    "singlet_population_final",
                    "singlet_population_min",
                    "singlet_population_max",
                ],
            )

    plot_contrast(summary_rows, args.output_dir / "hyperfine_band_ablation_contrast.png", args.show_plot)
    plot_angular_shapes(detail_rows, args.output_dir, args.show_plot)
    print(f"\nsaved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
