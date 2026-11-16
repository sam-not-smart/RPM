"""Replot the archived four-condition quadrupole comparison and exact reference."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/quadrupole"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_data(data_dir=RESULTS):
    run = data_dir / "full14_uncoupled_tau5_29angles"
    config = json.loads((run / "manifest.json").read_text())["configuration"]
    angles = np.array(config["angles_deg"], dtype=float)
    expected = np.array(sorted(set(range(0, 181, 10)) | set(range(85, 96))), dtype=float)
    np.testing.assert_array_equal(angles, expected)
    if config["samples"] != 4 or config["field_mT"] != .05 or config["lifetime_us"] != 5:
        raise ValueError("Unexpected archived physical settings")
    rows = read_csv(run / "raw_sample_yields.csv")
    if len(rows) != 464:
        raise ValueError("The archive must contain all 464 paired uncoupled/coupled yields")
    groups = {}
    for label in ("D=J=0", "D/J=-3"):
        for q in (0, 1):
            samples, errors = np.full((29, 4), np.nan), np.full((29, 4), np.nan)
            for row in rows:
                if row["electron_interactions"] != label or float(row["scale"]) != q:
                    continue
                a = int(np.flatnonzero(angles == float(row["theta_deg"]))[0])
                sample = int(row["sample_id"])
                if not np.isnan(samples[a, sample]):
                    raise ValueError("Duplicate sample")
                samples[a, sample] = float(row["yield"])
                errors[a, sample] = sum(float(row[key]) for key in (
                    "quadrature_error", "yield_propagation_estimate", "yield_tail_bound"))
            if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(errors)):
                raise ValueError("Missing yields or numerical errors")
            groups[label, q] = samples, errors
    for sample in range(4):
        if len({r["initial_state_sha256"] for r in rows if int(r["sample_id"]) == sample}) != 1:
            raise ValueError("Initial states are not paired across all four conditions")
    ref_dir = data_dir / "zeeman_hyperfine_reference"
    meta = json.loads((ref_dir / "quantum_needle_metadata.json").read_text())
    if meta["field_microT"] != 50 or meta.get("quadrupole_scale", 0) != 0:
        raise ValueError("Reference must be the archived Q=0, 50-microtesla calculation")
    tensors = {f"{row['radical']}:{row['nucleus']}": [
        [float(row[f"A{a}{b}_mT"]) for b in "xyz"] for a in "xyz"
    ] for row in read_csv(ref_dir / "quantum_needle_hyperfine_tensors.csv")}
    if tensors != config["hyperfine_tensors_mT"]:
        raise ValueError("Reference hyperfine tensors do not match the paired simulations")
    reference = read_csv(ref_dir / "quantum_needle_fig1b.csv")
    theta = np.array([float(row["theta_deg"]) for row in reference])
    values = np.array([float(row["phi_s_tau_5_us"]) for row in reference])
    np.testing.assert_array_equal(theta, meta["theta_values_deg"])
    return angles, groups, theta, values


def plot(output_dir, data_dir=RESULTS):
    angles, groups, ref_theta, ref_values = load_data(data_dir)
    draws = np.random.default_rng(1701).integers(0, 4, (2000, 4))
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(5.25, 5.45), facecolor="white")
    ax = fig.add_axes([.164, .216, .596, .674])
    bounds = [ref_values.min(), ref_values.max()]

    def label(text, y, color, offset=0):
        leader = dict(arrowstyle="-", lw=.65, color=color, shrinkA=2, shrinkB=2) if offset else None
        ax.annotate(text, (180, y), xytext=(10, offset), textcoords="offset points",
                    va="center", ha="left", color=color, fontsize=9.5, clip_on=False,
                    arrowprops=leader)

    styles = [
        ("D/J=-3", 0, r"$D/J=-3$" + "\n" + r"$Q=0$", "#226ba4", "o", "-", 0),
        ("D/J=-3", 1, r"$D/J=-3$" + "\nNominal Q", "#c45a1a", "o", "-", 0),
        ("D=J=0", 0, r"$D=J=0$" + "\n" + r"$Q=0$ (sampled)", "#207653", "s", "--", -12),
        ("D=J=0", 1, r"$D=J=0$" + "\nNominal Q", "#8054a6", "o", "-", 0),
    ]
    for condition, q, name, color, marker, style, offset in styles:
        y, e = groups[condition, q]
        lo, hi = np.quantile(y[:, draws].mean(axis=-1), [.025, .975], axis=1)
        lo, hi = lo - e.mean(axis=1), hi + e.mean(axis=1)
        bounds.extend([lo.min(), hi.max()])
        mean = y.mean(axis=1)
        ax.fill_between(angles, lo, hi, color=color, alpha=.14, linewidth=0)
        ax.plot(angles, mean, color=color, lw=1.3, ls=style, marker=marker,
                ms=2.6 if condition == "D=J=0" else 1.9,
                mfc="white" if style == "--" else color, mew=.8)
        label(name, mean[-1], color, offset)
    ax.plot(ref_theta, ref_values, color="#333333", lw=1.05, zorder=3)
    label("Zeeman + HF\nExact reference", ref_values[-1], "#333333", 12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.95)
    ax.set(xlim=(-2, 182), ylim=(.2638, .2812), xticks=range(0, 181, 30),
           yticks=[.265, .270, .275, .280])
    if not ax.get_ylim()[0] < min(bounds) < max(bounds) < ax.get_ylim()[1]:
        raise ValueError("Plot range would clip data or uncertainty")
    ax.tick_params(direction="in", length=4, labelsize=9.5)
    ax.set_xlabel(r"$\theta$ / degrees", fontsize=11, labelpad=5)
    ax.set_ylabel(r"$\Phi_\mathrm{S}$", fontsize=13)
    fig.text(.164, .928, "14 nuclei  ·  50 μT  ·  τ = 5 μs", fontsize=9.6)
    fig.text(.164, .034,
             "Markers: 29 angles at D = J = 0, 29 at D/J = −3 (M = 4).\n"
             "Bands: pointwise 95% intervals plus numerical estimates.\n"
             f"Exact Zeeman + hyperfine reference: {len(ref_theta)} angles, no nuclear sampling.",
             fontsize=7.75, color="#414141", linespacing=1.35)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure1b_all_conditions_absolute_yield"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/quadrupole tests/combined_29angle_plot")
    args = parser.parse_args()
    plot(args.output_dir, args.data_dir)
