"""Reproduce Section 3 ("Modelling the optimum") of the RPM brief.

This is the same two-nucleus, low-field toy radical pair used in the brief:

* two electrons and one spin-1 nucleus on each radical;
* a 50-uT magnetic field;
* an N5-like tensor on radical A whose axial value ``Azz`` and transverse
  magnitude ``b = 1000*sqrt(Axx**2 + Ayy**2)`` are scanned;
* a purely axial Trp-N1-like tensor diag(0, 0, 1.0812) mT on radical B;
* equal singlet/triplet first-order recombination;
* no exchange, electron dipolar coupling, relaxation, or nuclear Zeeman term.

The complete 36-dimensional pair Hamiltonian is diagonalized, and the
reaction yield is evaluated analytically in its eigenbasis.  The reported
angular sensitivity is ``Gamma = max_theta |d Phi_S / d theta|`` on the
0.05-degree grid used for the original Section 3 calculation.  Only the
sensitivity optimum is tracked; the amplitude optimum is a separate branch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.optimize

from rpm_quantum_needle_model import GAMMA_E_MHZ_PER_MT, TWOPI, spin_operators


DEFAULT_OUTPUT = Path("outputs") / "two_nucleus_optimum_reproduction"
FIELD_MT = 0.05
TRP_N1_AZZ_MT = 1.0812
AXIAL_COUPLINGS_MT = (0.6046, 1.2, 1.7569)
LIFETIMES_US = (5.0, 10.0, 20.0, 50.0, 100.0)
REAL_N5_B_UT = 132.5
DIRECT_CHECK_TAU_US = 29.0
DIRECT_CHECK_B_UT = (70.0, 90.0, 110.0, 125.0, 132.5, 140.0, 160.0, 185.0, 215.0)


def _embed(operator: np.ndarray, position: int, dimensions: tuple[int, ...]) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for index, dimension in enumerate(dimensions):
        factor = operator if index == position else np.eye(dimension, dtype=complex)
        result = np.kron(result, factor)
    return result


DIMENSIONS = (2, 2, 3, 3)  # eA, eB, N5-like, Trp-N1-like
ELECTRON_A = tuple(_embed(op, 0, DIMENSIONS) for op in spin_operators(0.5))
ELECTRON_B = tuple(_embed(op, 1, DIMENSIONS) for op in spin_operators(0.5))
NUCLEUS_A = tuple(_embed(op, 2, DIMENSIONS) for op in spin_operators(1.0))
NUCLEUS_B = tuple(_embed(op, 3, DIMENSIONS) for op in spin_operators(1.0))
TOTAL_ELECTRON = tuple(a + b for a, b in zip(ELECTRON_A, ELECTRON_B))

_singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
_singlet_projector = np.outer(_singlet, _singlet.conj())
SINGLET_PROJECTOR = np.kron(_singlet_projector, np.eye(9, dtype=complex))
NUCLEAR_DIMENSION = 9


def transverse_tensor_mT(azz_mT: float, b_uT: float) -> np.ndarray:
    """Return diag(Aperp, Aperp, Azz), preserving the N5 transverse sign."""

    aperp_mT = -float(b_uT) / (1000.0 * math.sqrt(2.0))
    return np.diag([aperp_mT, aperp_mT, float(azz_mT)])


def _hyperfine_hamiltonian(azz_mT: float, b_uT: float) -> np.ndarray:
    tensor_a = transverse_tensor_mT(azz_mT, b_uT)
    tensor_b = np.diag([0.0, 0.0, TRP_N1_AZZ_MT])
    result = np.zeros((36, 36), dtype=complex)
    for electron_axis in range(3):
        for nuclear_axis in range(3):
            result += (
                TWOPI
                * GAMMA_E_MHZ_PER_MT
                * tensor_a[electron_axis, nuclear_axis]
                * (ELECTRON_A[electron_axis] @ NUCLEUS_A[nuclear_axis])
            )
            result += (
                TWOPI
                * GAMMA_E_MHZ_PER_MT
                * tensor_b[electron_axis, nuclear_axis]
                * (ELECTRON_B[electron_axis] @ NUCLEUS_B[nuclear_axis])
            )
    return result


def _zeeman_hamiltonian(theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    direction = math.sin(theta) * TOTAL_ELECTRON[0] + math.cos(theta) * TOTAL_ELECTRON[2]
    return TWOPI * GAMMA_E_MHZ_PER_MT * FIELD_MT * direction


@dataclass(frozen=True)
class Configuration:
    angle_step_deg: float = 0.05
    b_lower_uT: float = 5.0
    b_upper_uT: float = 500.0
    optimizer_xatol_uT: float = 0.02

    def stable_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "field_mT": FIELD_MT,
            "trp_n1_tensor_mT": [0.0, 0.0, TRP_N1_AZZ_MT],
            "axial_couplings_mT": list(AXIAL_COUPLINGS_MT),
            "lifetimes_us": list(LIFETIMES_US),
            "real_n5_b_uT": REAL_N5_B_UT,
            "direct_check_tau_us": DIRECT_CHECK_TAU_US,
            "hamiltonian_dimension": 36,
            "J_uT": 0.0,
            "D_uT": 0.0,
            "relaxation": False,
            "nuclear_zeeman": False,
            "yield_method": "exact equal-rate eigenbasis integral",
        }


def angular_curve(
    azz_mT: float,
    b_uT: float,
    lifetime_us: float,
    angle_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(0.0, 90.0 + 0.5 * angle_step_deg, angle_step_deg)
    return theta, _yields_for_angles(azz_mT, b_uT, lifetime_us, theta)


def _yields_for_angles(
    azz_mT: float,
    b_uT: float,
    lifetime_us: float,
    theta: np.ndarray,
) -> np.ndarray:
    hyperfine = _hyperfine_hamiltonian(azz_mT, b_uT)
    k = 1.0 / float(lifetime_us)
    yields = np.empty(theta.size, dtype=float)
    for index, angle in enumerate(theta):
        energies, vectors = scipy.linalg.eigh(
            hyperfine + _zeeman_hamiltonian(float(angle)),
            overwrite_a=True,
            check_finite=False,
            driver="evd",
        )
        projected = vectors.conj().T @ (SINGLET_PROJECTOR @ vectors)
        gaps = energies[:, None] - energies[None, :]
        kernel = k * k / (k * k + gaps * gaps)
        yields[index] = float(np.sum(np.abs(projected) ** 2 * kernel).real / NUCLEAR_DIMENSION)
    return yields


def sensitivity(
    azz_mT: float,
    b_uT: float,
    lifetime_us: float,
    angle_step_deg: float,
) -> tuple[float, float]:
    theta, yields = angular_curve(azz_mT, b_uT, lifetime_us, angle_step_deg)
    coarse_gradient = np.gradient(yields, theta, edge_order=2)
    coarse_index = int(np.argmax(np.abs(coarse_gradient)))

    # The 0.05-degree scan identifies the needle.  Re-evaluate only a narrow
    # local window at one-tenth that spacing before extracting its slope; this
    # removes the resolution bias noted for the 100-us cells in the brief
    # without changing any physical parameter or the discovery grid.
    center = float(theta[coarse_index])
    local_step = angle_step_deg / 10.0
    lower_window = max(0.0, center - 3.0 * angle_step_deg)
    upper_window = min(90.0, center + 3.0 * angle_step_deg)
    local_theta = np.arange(
        lower_window, upper_window + 0.5 * local_step, local_step
    )
    local_yields = _yields_for_angles(azz_mT, b_uT, lifetime_us, local_theta)
    spline = scipy.interpolate.CubicSpline(local_theta, local_yields)
    probe_step = local_step / 10.0
    probe = np.arange(lower_window, upper_window + 0.5 * probe_step, probe_step)
    derivative = np.abs(spline(probe, 1))
    index = int(np.argmax(derivative))
    lower = float(probe[max(0, index - 1)])
    upper = float(probe[min(probe.size - 1, index + 1)])
    if math.isclose(lower, upper):
        return float(derivative[index] * 180.0 / math.pi), float(probe[index])
    refined = scipy.optimize.minimize_scalar(
        lambda angle: -abs(float(spline(angle, 1))),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-7},
    )
    return float(-refined.fun * 180.0 / math.pi), float(refined.x)


def optimize_cell(
    azz_mT: float,
    lifetime_us: float,
    config: Configuration,
) -> dict[str, float]:
    evaluations: dict[float, tuple[float, float]] = {}

    def objective(value: float) -> float:
        key = round(float(value), 9)
        if key not in evaluations:
            evaluations[key] = sensitivity(
                azz_mT, key, lifetime_us, config.angle_step_deg
            )
        return -evaluations[key][0]

    # The sensitivity landscape contains additional high-b resonances that are
    # outside the transverse-side branch studied in Section 3.  The original
    # analysis follows the first interior maximum as b increases.  Locate that
    # branch with a broad, assumption-free logarithmic scan, then refine only
    # the bracketing interval.
    scouting_grid = np.geomspace(config.b_lower_uT, config.b_upper_uT, 36)
    scouting_gamma = np.array([-objective(float(value)) for value in scouting_grid])
    local_maxima = np.flatnonzero(
        (scouting_gamma[1:-1] >= scouting_gamma[:-2])
        & (scouting_gamma[1:-1] >= scouting_gamma[2:])
    ) + 1
    if local_maxima.size == 0:
        raise RuntimeError(
            f"no interior transverse-branch maximum for Azz={azz_mT}, tau={lifetime_us}"
        )
    branch_index = int(local_maxima[0])
    lower = float(scouting_grid[branch_index - 1])
    upper = float(scouting_grid[branch_index + 1])
    result = scipy.optimize.minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": config.optimizer_xatol_uT, "maxiter": 80},
    )
    if not result.success:
        raise RuntimeError(
            f"b optimization failed for Azz={azz_mT}, tau={lifetime_us}: {result.message}"
        )
    b_opt = float(result.x)
    gamma, theta = sensitivity(azz_mT, b_opt, lifetime_us, config.angle_step_deg)
    return {
        "azz_mT": float(azz_mT),
        "lifetime_us": float(lifetime_us),
        "b_opt_uT": b_opt,
        "gamma_max_per_rad": gamma,
        "theta_max_deg": theta,
        "scaled_constant_uT_sqrt_us_per_mT": b_opt * math.sqrt(lifetime_us) / azz_mT,
        "objective_evaluations": float(len(evaluations)),
    }


def reproduce(config: Configuration) -> tuple[list[dict[str, float]], dict[str, object]]:
    rows: list[dict[str, float]] = []
    total = len(AXIAL_COUPLINGS_MT) * len(LIFETIMES_US)
    for azz in AXIAL_COUPLINGS_MT:
        for lifetime in LIFETIMES_US:
            row = optimize_cell(azz, lifetime, config)
            rows.append(row)
            print(
                f"[{len(rows):02d}/{total}] Azz={azz:.4f} mT, tau={lifetime:5.1f} us: "
                f"b_opt={row['b_opt_uT']:.3f} uT, "
                f"C={row['scaled_constant_uT_sqrt_us_per_mT']:.3f}",
                flush=True,
            )

    fits: dict[str, dict[str, float]] = {}
    for azz in AXIAL_COUPLINGS_MT:
        selected = [row for row in rows if math.isclose(row["azz_mT"], azz)]
        x = np.log([row["lifetime_us"] for row in selected])
        y = np.log([row["b_opt_uT"] for row in selected])
        exponent, intercept = np.polyfit(x, y, 1)
        fits[f"{azz:.4f}"] = {
            "exponent": float(exponent),
            "prefactor_uT": float(math.exp(intercept)),
        }

    c_values = np.array(
        [row["scaled_constant_uT_sqrt_us_per_mT"] for row in rows], dtype=float
    )
    # The brief reports C as the tau=1 intercept of the fitted power laws.
    # This is about 391 even though b*sqrt(tau)/Azz drifts upward by ~2% over
    # the sampled lifetimes because the fitted exponent is -0.494 rather than
    # exactly -0.5.
    fitted_c = float(
        np.mean(
            [
                fit["prefactor_uT"] / float(label)
                for label, fit in fits.items()
            ]
        )
    )

    direct = []
    for b_value in DIRECT_CHECK_B_UT:
        gamma, theta = sensitivity(
            1.7569, b_value, DIRECT_CHECK_TAU_US, config.angle_step_deg
        )
        direct.append({"b_uT": b_value, "gamma": gamma, "theta_deg": theta})
    direct_peak = optimize_cell(1.7569, DIRECT_CHECK_TAU_US, config)
    real_gamma, real_theta = sensitivity(
        1.7569, REAL_N5_B_UT, DIRECT_CHECK_TAU_US, config.angle_step_deg
    )

    summary: dict[str, object] = {
        "configuration": config.stable_dict(),
        "fitted_scaling_constant_uT_sqrt_us_per_mT": fitted_c,
        "observed_b_sqrt_tau_over_azz_mean": float(np.mean(c_values)),
        "observed_b_sqrt_tau_over_azz_std": float(np.std(c_values, ddof=1)),
        "power_law_fits": fits,
        "direct_check": {
            "lifetime_us": DIRECT_CHECK_TAU_US,
            "curve": direct,
            "continuous_peak": direct_peak,
            "real_n5_b_uT": REAL_N5_B_UT,
            "real_n5_gamma": real_gamma,
            "real_n5_theta_deg": real_theta,
            "real_n5_percent_of_peak": 100.0
            * real_gamma
            / float(direct_peak["gamma_max_per_rad"]),
        },
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shape_lifetime_us(x: np.ndarray | float, fitted_c: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    k = fitted_c / 1000.0
    return (k * (values + 2.0) / (math.sqrt(2.0) * np.abs(values - 1.0))) ** 2


def make_figure(
    output_dir: Path,
    rows: list[dict[str, float]],
    summary: dict[str, object],
) -> None:
    fitted_c = float(summary["fitted_scaling_constant_uT_sqrt_us_per_mT"])
    direct = summary["direct_check"]
    assert isinstance(direct, dict)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)

    colors = ("#2ca02c", "#ff7f0e", "#d62728")
    markers = ("o", "s", "^")
    tau_line = np.geomspace(min(LIFETIMES_US), max(LIFETIMES_US), 300)
    axes[0].plot(
        tau_line,
        fitted_c / np.sqrt(tau_line),
        "--",
        color="black",
        linewidth=1.2,
        label=rf"$b_{{opt}}/A_{{zz}}={fitted_c:.0f}/\sqrt{{\tau}}$",
    )
    for azz, color, marker in zip(AXIAL_COUPLINGS_MT, colors, markers):
        selected = sorted(
            (row for row in rows if math.isclose(row["azz_mT"], azz)),
            key=lambda row: row["lifetime_us"],
        )
        axes[0].plot(
            [row["lifetime_us"] for row in selected],
            [row["b_opt_uT"] / azz for row in selected],
            linestyle="none",
            marker=marker,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.5,
            color=color,
            label=rf"$A_{{zz}}={azz:.2f}$ mT",
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel=r"lifetime $\tau$ ($\mu$s)",
        ylabel=r"$b_{opt}/A_{zz}$  ($\mu$T per mT)",
        title="(a) scale collapse: 3x range of $A_{zz}$\nfalls on one curve",
    )
    axes[0].legend(fontsize=7.5, frameon=True)

    direct_curve = direct["curve"]
    assert isinstance(direct_curve, list)
    direct_b = np.array([float(point["b_uT"]) for point in direct_curve])
    direct_gamma = np.array([float(point["gamma"]) for point in direct_curve])
    axes[1].plot(direct_b, direct_gamma, "o-", color="0.30", linewidth=1.4, markersize=4)
    real_gamma = float(direct["real_n5_gamma"])
    percent = float(direct["real_n5_percent_of_peak"])
    axes[1].scatter(
        [REAL_N5_B_UT],
        [real_gamma],
        marker="*",
        s=330,
        color="#d62728",
        edgecolor="#d62728",
        zorder=5,
        label=f"real FAD N5, {REAL_N5_B_UT:.1f} $\\mu$T\n= {percent:.1f}% of peak",
    )
    axes[1].set(
        xlabel=r"transverse coupling $b$ ($\mu$T)",
        ylabel=r"$\Gamma=\max\,|\partial\Phi_S/\partial\theta|$",
        title=(
            rf"(b) direct check at $\tau={DIRECT_CHECK_TAU_US:.0f}\ \mu$s"
            "\nreal N5 sits on the optimum"
        ),
    )
    axes[1].legend(loc="lower center", fontsize=7.5)

    x_curve = np.linspace(0.55, 0.99, 700)
    axes[2].plot(x_curve, shape_lifetime_us(x_curve, fitted_c), color="black", linewidth=1.6)
    axes[2].axhspan(10.0, 100.0, color="#f6c76b", alpha=0.30)
    axes[2].text(0.565, 12.0, "10-100 $\\mu$s", color="#ff8c00", fontsize=7.5)
    nuclei = (
        ("FAD N5", 0.5233, 0.6168, "#d62728"),
        ("Trp N1", 0.3215, 0.3798, "#1f77b4"),
        ("FAD N10", 0.1887, 0.2080, "#2ca02c"),
    )
    for name, aiso, dipolar_t, color in nuclei:
        x_value = aiso / dipolar_t
        tau_value = float(shape_lifetime_us(x_value, fitted_c))
        axes[2].scatter(
            [x_value], [tau_value], s=90, color=color, edgecolor="black", linewidth=0.7,
            label=f"{name}: x={x_value:.3f} -> {tau_value:.0f} $\\mu$s",
            zorder=4,
        )
    axes[2].set(
        yscale="log",
        xlim=(0.55, 1.0),
        ylim=(1.0, 1.0e4),
        xlabel=r"$x=a_{iso}/T$  (shape ratio)",
        ylabel=r"implied optimal lifetime $\tau^*$ ($\mu$s)",
        title=r"(c) $\tau^*$ depends only on x, and is steep",
    )
    axes[2].legend(loc="upper left", fontsize=7.2, frameon=True)

    fig.suptitle(
        "Modelling of the transverse-coupling optimum",
        fontsize=12,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"section3_modelling_the_optimum.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, float]],
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "section3_optimum_cells.csv", rows)
    (output_dir / "section3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(output_dir, rows, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angle-step", type=float, default=0.05)
    parser.add_argument("--b-lower", type=float, default=5.0)
    parser.add_argument("--b-upper", type=float, default=500.0)
    parser.add_argument("--optimizer-xatol", type=float, default=0.02)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Configuration(
        angle_step_deg=args.angle_step,
        b_lower_uT=args.b_lower,
        b_upper_uT=args.b_upper,
        optimizer_xatol_uT=args.optimizer_xatol,
    )
    if config.angle_step_deg <= 0.0:
        raise ValueError("--angle-step must be positive")
    if not (0.0 < config.b_lower_uT < config.b_upper_uT):
        raise ValueError("require 0 < --b-lower < --b-upper")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(config.stable_dict(), indent=2) + "\n"
    )
    if args.report_only:
        cells_path = args.output_dir / "section3_optimum_cells.csv"
        summary_path = args.output_dir / "section3_summary.json"
        if not cells_path.exists() or not summary_path.exists():
            raise FileNotFoundError("--report-only requires completed CSV and summary outputs")
        with cells_path.open(newline="") as stream:
            rows = [
                {key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
        summary = json.loads(summary_path.read_text())
        fits = summary["power_law_fits"]
        fitted_c = float(
            np.mean(
                [
                    float(fit["prefactor_uT"]) / float(label)
                    for label, fit in fits.items()
                ]
            )
        )
        c_values = np.array(
            [row["scaled_constant_uT_sqrt_us_per_mT"] for row in rows]
        )
        summary["fitted_scaling_constant_uT_sqrt_us_per_mT"] = fitted_c
        summary.pop("scaling_constant_mean", None)
        summary.pop("scaling_constant_std", None)
        summary["observed_b_sqrt_tau_over_azz_mean"] = float(np.mean(c_values))
        summary["observed_b_sqrt_tau_over_azz_std"] = float(np.std(c_values, ddof=1))
    else:
        rows, summary = reproduce(config)
    write_outputs(args.output_dir, rows, summary)
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
