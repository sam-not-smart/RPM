"""Exchange-tuned avoided-crossing reconnaissance for the quantum needle.

Stage A diagonalizes an exact 108-dimensional needle-core Hamiltonian and
ranks singlet-active avoided crossings. Stage B uses those candidates in the
complete 14-nucleus matrix-free state-vector model. The experiment fixes the
literal Efimova-Hore electron coupling convention and configurable quadrupoles.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from rpm_quantum_needle_jd_sweeps import (
    EFIMOVA_HORE_AXIS,
    QUADRUPOLE_SCALE,
    R2_D_PAPER_UT,
    EfimovaHoreCondition,
    SweepConfig,
    _complete,
    _write_csv,
    _write_json,
    checkpoint_path,
    condition_metrics,
    efimova_hore_uncoupled_control,
    equivalent_field_uT_to_MHz,
    load_cell,
    run_tasks,
)
from rpm_quantum_needle_model import (
    FAD_NUCLEI,
    GAMMA_E_MHZ_PER_MT,
    TRP_NUCLEI,
    TWOPI,
    QUADRUPOLE_TENSORS_MT,
    add_quadrupole_argument,
    field_vector_mT,
    nominal_quadrupole_metadata,
    spin_operators,
)
from rpm_statevector_engine import (
    LocalTerm,
    MatrixFreeHamiltonian,
    efimova_hore_electron_coupling_operator,
    quadrupole_operator,
    two_spin_coupling_operator,
)


SCHEMA_VERSION = 1
FULL_ANGLES_DEG = (0.0, 60.0, 80.0, 85.0, 88.0, 90.0, 92.0, 95.0, 100.0, 120.0, 180.0)
NEEDLE_REFINEMENT_ANGLES_DEG = tuple(float(value) for value in range(86, 95))
COARSE_J_UT = tuple(float(value) for value in range(-200, 201, 20))
COARSE_THETA_DEG = tuple(np.round(np.arange(85.0, 95.0 + 0.05, 0.1), 10))
REFINED_THETA_DEG = tuple(np.round(np.arange(87.0, 93.0 + 0.0125, 0.025), 10))
CANCELLATION_REFERENCES_UT = (-57.979166666666664, 115.95833333333333)
CORE_DIMENSIONS = (2, 2, 3, 3, 3)
CORE_ORDER = ("eA", "eB", "FAD:N5", "FAD:N10", "TrpH:N1")
CORE_NUCLEI = (FAD_NUCLEI[0], FAD_NUCLEI[1], TRP_NUCLEI[0])
CORE_ELECTRON_AXES = (0, 0, 1)


@dataclass(frozen=True)
class SpectralConfig:
    field_microT: float = 50.0
    d_paper_uT: float = R2_D_PAPER_UT
    dipolar_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    coarse_j_uT: tuple[float, ...] = COARSE_J_UT
    coarse_theta_deg: tuple[float, ...] = COARSE_THETA_DEG
    refined_theta_deg: tuple[float, ...] = REFINED_THETA_DEG
    refinement_half_width_uT: float = 10.0
    refinement_step_uT: float = 5.0
    refined_region_count: int = 6
    quadrupole_scale: float = QUADRUPOLE_SCALE

    def stable_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "schema_version": SCHEMA_VERSION,
            "core_order": list(CORE_ORDER),
            "core_dimensions": list(CORE_DIMENSIONS),
            **nominal_quadrupole_metadata(self.quadrupole_scale),
            "exchange_hamiltonian": "-J*(S^2-1)",
            "dipolar_hamiltonian": "D*((S.n)^2-S^2/3)",
        }

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def exchange_condition(exchange_j_uT: float) -> EfimovaHoreCondition:
    value = float(exchange_j_uT)
    token = f"{value:+010.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return EfimovaHoreCondition(
        condition_id=f"avoided_J_{token}uT",
        family="exchange_avoided_crossing",
        role="dipole_plus_exchange",
        d_paper_uT=R2_D_PAPER_UT,
        exchange_J_uT=value,
        ratio_D_over_J=None if value == 0.0 else R2_D_PAPER_UT / value,
        dipolar_axis=tuple(float(value) for value in EFIMOVA_HORE_AXIS),
    )


def _singlet_projector(dimensions: tuple[int, ...]) -> np.ndarray:
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    electronic = np.outer(singlet, singlet.conj())
    return np.kron(electronic, np.eye(math.prod(dimensions[2:]), dtype=complex))


def build_core_hamiltonian(
    theta_deg: float,
    exchange_j_uT: float,
    *,
    d_paper_uT: float = R2_D_PAPER_UT,
    field_microT: float = 50.0,
    remove_n5_n10_transverse: bool = False,
    quadrupole_scale: float = QUADRUPOLE_SCALE,
) -> np.ndarray:
    """Return the exact 108x108 needle-core Hamiltonian in rad/us."""

    electron_ops = spin_operators(0.5)
    field = field_vector_mT(theta_deg, field_microT)
    zeeman = TWOPI * GAMMA_E_MHZ_PER_MT * sum(
        component * operator for component, operator in zip(field, electron_ops)
    )
    terms: list[LocalTerm] = [
        LocalTerm((0,), zeeman, "electron A Zeeman"),
        LocalTerm((1,), zeeman, "electron B Zeeman"),
    ]
    for nucleus_index, (nucleus, electron_axis) in enumerate(
        zip(CORE_NUCLEI, CORE_ELECTRON_AXES)
    ):
        tensor = np.array(nucleus.tensor_mT, copy=True)
        if remove_n5_n10_transverse and nucleus_index < 2:
            # Hiscock's axial control sets Axx=Ayy=0. In the present shared
            # molecular frame the same transverse block also has a small Axy
            # rotation term, so remove the complete in-plane block rather than
            # leaving an artificial residual avoided coupling.
            tensor[:2, :2] = 0.0
        local = TWOPI * GAMMA_E_MHZ_PER_MT * two_spin_coupling_operator(
            0.5, nucleus.spin, tensor
        )
        terms.append(
            LocalTerm(
                (electron_axis, nucleus_index + 2),
                local,
                f"{CORE_ORDER[nucleus_index + 2]} hyperfine",
            )
        )
        if quadrupole_scale:
            site = CORE_ORDER[nucleus_index + 2]
            q = TWOPI * GAMMA_E_MHZ_PER_MT * quadrupole_scale * QUADRUPOLE_TENSORS_MT[site]
            terms.append(LocalTerm((nucleus_index + 2,), quadrupole_operator(nucleus.spin, q),
                                   f"{site} quadrupole"))
    pair = TWOPI * efimova_hore_electron_coupling_operator(
        equivalent_field_uT_to_MHz(exchange_j_uT),
        equivalent_field_uT_to_MHz(d_paper_uT),
        np.asarray(EFIMOVA_HORE_AXIS),
    )
    if np.any(pair != 0.0):
        terms.append(LocalTerm((0, 1), pair, "Efimova-Hore J+D"))
    return MatrixFreeHamiltonian(CORE_DIMENSIONS, terms).explicit_matrix()


def exchange_operator_controls() -> dict[str, float]:
    """Verify that J changes only singlet-triplet detuning."""

    projector = _singlet_projector((2, 2))
    identity = np.eye(4, dtype=complex)
    exchange = efimova_hore_electron_coupling_operator(1.0, 0.0, EFIMOVA_HORE_AXIS)
    dipolar = efimova_hore_electron_coupling_operator(0.0, 1.0, EFIMOVA_HORE_AXIS)
    return {
        "exchange_commutator_with_singlet_norm": float(
            np.linalg.norm(exchange @ projector - projector @ exchange)
        ),
        "exchange_singlet_triplet_block_norm": float(
            np.linalg.norm(projector @ exchange @ (identity - projector))
        ),
        "dipolar_commutator_with_singlet_norm": float(
            np.linalg.norm(dipolar @ projector - projector @ dipolar)
        ),
        "dipolar_singlet_triplet_block_norm": float(
            np.linalg.norm(projector @ dipolar @ (identity - projector))
        ),
    }


def _spectrum_for_j(
    payload: tuple[float, tuple[float, ...], SpectralConfig, bool, str]
) -> dict[str, object]:
    exchange_j_uT, angles_deg, config, remove_transverse, scan_kind = payload
    projector = _singlet_projector(CORE_DIMENSIONS)
    energies = np.empty((len(angles_deg), math.prod(CORE_DIMENSIONS)), dtype=float)
    singlets = np.empty_like(energies)
    for theta_index, theta_deg in enumerate(angles_deg):
        hamiltonian = build_core_hamiltonian(
            theta_deg,
            exchange_j_uT,
            d_paper_uT=config.d_paper_uT,
            field_microT=config.field_microT,
            remove_n5_n10_transverse=remove_transverse,
            quadrupole_scale=config.quadrupole_scale,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        energies[theta_index] = eigenvalues / TWOPI
        singlets[theta_index] = np.real(
            np.sum(eigenvectors.conj() * (projector @ eigenvectors), axis=0)
        )
    return {
        "J_uT": float(exchange_j_uT),
        "angles_deg": np.asarray(angles_deg),
        "energies_MHz": energies,
        "singlet_characters": singlets,
        "remove_transverse": bool(remove_transverse),
        "scan_kind": scan_kind,
        "quadrupole_scale": config.quadrupole_scale,
    }


def detect_avoided_crossings(spectrum: dict[str, object]) -> list[dict[str, object]]:
    """Find adjacent-level gap minima and score their needle relevance."""

    angles = np.asarray(spectrum["angles_deg"], dtype=float)
    energies = np.asarray(spectrum["energies_MHz"], dtype=float)
    singlets = np.asarray(spectrum["singlet_characters"], dtype=float)
    gaps = np.diff(energies, axis=1)
    rows: list[dict[str, object]] = []
    angular_step = float(np.median(np.diff(angles)))
    transfer_offset = max(1, int(round(0.3 / angular_step)))
    for pair_index in range(gaps.shape[1]):
        values = gaps[:, pair_index]
        minima = np.flatnonzero(
            (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
        ) + 1
        for theta_index in minima:
            left_index = max(0, theta_index - transfer_offset)
            right_index = min(angles.size - 1, theta_index + transfer_offset)
            lower_change = singlets[right_index, pair_index] - singlets[left_index, pair_index]
            upper_change = singlets[right_index, pair_index + 1] - singlets[left_index, pair_index + 1]
            transfer = 0.5 * (abs(lower_change) + abs(upper_change))
            exchange_balance = max(
                0.0,
                1.0
                - abs(lower_change + upper_change)
                / (abs(lower_change) + abs(upper_change) + 1e-15),
            )
            activity = float(
                np.max(
                    singlets[
                        left_index : right_index + 1,
                        pair_index : pair_index + 2,
                    ]
                )
            )
            gap = float(values[theta_index])
            threshold = gap + max(gap, 0.01)
            width_left = theta_index
            width_right = theta_index
            while width_left > 0 and values[width_left - 1] <= threshold:
                width_left -= 1
            while width_right + 1 < angles.size and values[width_right + 1] <= threshold:
                width_right += 1
            width = float(angles[width_right] - angles[width_left])
            neighbor_gaps = []
            if pair_index > 0:
                neighbor_gaps.append(gaps[theta_index, pair_index - 1])
            if pair_index + 1 < gaps.shape[1]:
                neighbor_gaps.append(gaps[theta_index, pair_index + 1])
            nearest_other = float(min(neighbor_gaps, default=gap))
            isolation = nearest_other / (nearest_other + 5.0 * gap + 1e-15)
            proximity = math.exp(-0.5 * ((float(angles[theta_index]) - 90.0) / 1.5) ** 2)
            finite_gap = gap / (gap + 1e-5) * math.exp(-gap / 0.1)
            narrowness = 1.0 / (1.0 + width / 2.0)
            score = (
                transfer
                * exchange_balance
                * math.sqrt(max(activity, 0.0))
                * proximity
                * isolation
                * finite_gap
                * narrowness
            )
            rows.append(
                {
                    "scan_kind": spectrum["scan_kind"],
                    "remove_transverse": spectrum["remove_transverse"],
                    "J_uT": spectrum["J_uT"],
                    "lower_level": pair_index,
                    "upper_level": pair_index + 1,
                    "theta_c_deg": float(angles[theta_index]),
                    "minimum_gap_MHz": gap,
                    "lower_singlet_at_crossing": float(singlets[theta_index, pair_index]),
                    "upper_singlet_at_crossing": float(singlets[theta_index, pair_index + 1]),
                    "lower_singlet_change": float(lower_change),
                    "upper_singlet_change": float(upper_change),
                    "singlet_character_transfer": float(transfer),
                    "exchange_balance": float(exchange_balance),
                    "singlet_activity": activity,
                    "angular_width_deg": width,
                    "proximity_factor": proximity,
                    "isolation_factor": float(isolation),
                    "finite_gap_factor": finite_gap,
                    "narrowness_factor": narrowness,
                    "needle_crossing_score": float(score),
                }
            )
    return rows


def _ranking_rows(crossings: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    by_j: dict[float, list[dict[str, object]]] = {}
    for row in crossings:
        if bool(row["remove_transverse"]):
            continue
        by_j.setdefault(float(row["J_uT"]), []).append(row)
    ranking: list[dict[str, object]] = []
    for exchange_j_uT, rows in by_j.items():
        best = max(rows, key=lambda row: float(row["needle_crossing_score"]))
        ranking.append(
            {
                "J_uT": exchange_j_uT,
                "best_crossing_score": best["needle_crossing_score"],
                "best_theta_c_deg": best["theta_c_deg"],
                "best_gap_MHz": best["minimum_gap_MHz"],
                "best_singlet_transfer": best["singlet_character_transfer"],
                "best_singlet_activity": best["singlet_activity"],
                "best_angular_width_deg": best["angular_width_deg"],
                "best_lower_level": best["lower_level"],
                "best_upper_level": best["upper_level"],
                "crossings_with_score_above_1e-4": sum(
                    float(row["needle_crossing_score"]) > 1e-4 for row in rows
                ),
            }
        )
    ranking.sort(key=lambda row: float(row["best_crossing_score"]), reverse=True)
    return ranking


def _nonredundant_j_values(
    ranking: Iterable[dict[str, object]], count: int, minimum_spacing_uT: float
) -> list[float]:
    selected: list[float] = []
    for row in ranking:
        value = float(row["J_uT"])
        if all(abs(value - previous) >= minimum_spacing_uT for previous in selected):
            selected.append(value)
        if len(selected) == count:
            break
    return selected


def select_full_system_j_values(ranking: Iterable[dict[str, object]]) -> tuple[float, ...]:
    # Reserve the two cancellation references and dipole-only J=0 control
    # before choosing discovery points. This prevents a nominal "top five"
    # from spending three full-state conditions on 110, 115.96, and 120 uT.
    reserved = [0.0, *CANCELLATION_REFERENCES_UT]
    candidates: list[float] = []
    for row in ranking:
        value = float(row["J_uT"])
        if any(abs(value - reference) < 15.0 for reference in reserved):
            continue
        if any(abs(value - previous) < 15.0 for previous in candidates):
            continue
        candidates.append(value)
        if len(candidates) == 5:
            break
    if len(candidates) < 5:
        raise RuntimeError("The spectral ranking did not provide five distinct J regions.")
    return tuple(candidates + reserved)


def _run_spectra(
    j_values_uT: Iterable[float],
    angles_deg: tuple[float, ...],
    config: SpectralConfig,
    workers: int,
    *,
    remove_transverse: bool = False,
    scan_kind: str,
) -> list[dict[str, object]]:
    payloads = [
        (float(value), angles_deg, config, remove_transverse, scan_kind)
        for value in j_values_uT
    ]
    if workers == 1:
        return [_spectrum_for_j(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_spectrum_for_j, payloads))


def _spectral_rows(spectra: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spectrum in spectra:
        angles = np.asarray(spectrum["angles_deg"])
        energies = np.asarray(spectrum["energies_MHz"])
        singlets = np.asarray(spectrum["singlet_characters"])
        for theta_index, theta in enumerate(angles):
            for level in range(energies.shape[1]):
                rows.append(
                    {
                        "scan_kind": spectrum["scan_kind"],
                        "remove_transverse": spectrum["remove_transverse"],
                        "J_uT": spectrum["J_uT"],
                        "theta_deg": float(theta),
                        "level": level,
                        "energy_MHz": float(energies[theta_index, level]),
                        "singlet_character": float(singlets[theta_index, level]),
                    }
                )
    return rows


def run_spectroscopy(
    output_dir: Path, config: SpectralConfig, workers: int
) -> tuple[list[dict[str, object]], tuple[float, ...]]:
    started = time.perf_counter()
    coarse = _run_spectra(
        config.coarse_j_uT,
        config.coarse_theta_deg,
        config,
        workers,
        scan_kind="coarse",
    )
    control_full = _run_spectra(
        (0.0,),
        config.coarse_theta_deg,
        replace(config, d_paper_uT=0.0),
        1,
        scan_kind="control_D0_J0",
    )
    control_axial = _run_spectra(
        (0.0,),
        config.coarse_theta_deg,
        replace(config, d_paper_uT=0.0),
        1,
        remove_transverse=True,
        scan_kind="control_D0_J0_no_N5_N10_transverse",
    )
    coarse_crossings = [row for spectrum in coarse for row in detect_avoided_crossings(spectrum)]
    coarse_ranking = _ranking_rows(coarse_crossings)
    region_centres = _nonredundant_j_values(
        coarse_ranking, config.refined_region_count, 20.0
    )
    refined_j = sorted(
        {
            round(centre + offset, 10)
            for centre in region_centres
            for offset in np.arange(
                -config.refinement_half_width_uT,
                config.refinement_half_width_uT + 0.5 * config.refinement_step_uT,
                config.refinement_step_uT,
            )
            if -200.0 <= centre + offset <= 200.0
        }
    )
    refined = _run_spectra(
        refined_j,
        config.refined_theta_deg,
        config,
        workers,
        scan_kind="refined",
    )
    refined_crossings = [
        row for spectrum in refined for row in detect_avoided_crossings(spectrum)
    ]
    control_crossings = [
        row
        for spectrum in control_full + control_axial
        for row in detect_avoided_crossings(spectrum)
    ]
    crossings = coarse_crossings + refined_crossings + control_crossings
    physical_control_crossings = [
        row for row in control_crossings if row["scan_kind"] == "control_D0_J0"
    ]
    best_physical_control = max(
        physical_control_crossings,
        key=lambda row: float(row["needle_crossing_score"]),
    )
    control_pair = int(best_physical_control["lower_level"])
    axial_energies = np.asarray(control_axial[0]["energies_MHz"])
    axial_pair_gaps = axial_energies[:, control_pair + 1] - axial_energies[:, control_pair]
    axial_minimum_index = int(np.argmin(axial_pair_gaps))
    operator_controls = exchange_operator_controls()
    transverse_control = {
        "physical_best_crossing": best_physical_control,
        "same_pair_without_transverse": {
            "lower_level": control_pair,
            "upper_level": control_pair + 1,
            "theta_c_deg": float(
                np.asarray(control_axial[0]["angles_deg"])[axial_minimum_index]
            ),
            "minimum_gap_MHz": float(axial_pair_gaps[axial_minimum_index]),
        },
        "gap_ratio_no_transverse_over_physical": float(
            axial_pair_gaps[axial_minimum_index]
            / float(best_physical_control["minimum_gap_MHz"])
        ),
        "control_A_physical_avoided_crossing_present": bool(
            abs(float(best_physical_control["theta_c_deg"]) - 90.0) <= 1.0
            and float(best_physical_control["minimum_gap_MHz"]) > 1e-8
            and float(best_physical_control["singlet_character_transfer"]) > 0.1
        ),
        "control_B_transverse_removal_closes_same_gap": bool(
            float(axial_pair_gaps[axial_minimum_index])
            < 0.1 * float(best_physical_control["minimum_gap_MHz"])
        ),
        "control_C_exchange_is_block_diagonal": bool(
            max(operator_controls.values()) < 1e-12
        ),
        "exchange_operator_controls": operator_controls,
    }
    transverse_control["all_controls_passed"] = bool(
        transverse_control["control_A_physical_avoided_crossing_present"]
        and transverse_control["control_B_transverse_removal_closes_same_gap"]
        and transverse_control["control_C_exchange_is_block_diagonal"]
    )
    _write_json(output_dir / "spectral_control_results.json", transverse_control)
    if not transverse_control["all_controls_passed"]:
        raise RuntimeError(
            "Reduced spectral controls failed; refusing to select full-system candidates."
        )
    ranking_by_j = {
        float(row["J_uT"]): row for row in _ranking_rows(coarse_crossings)
    }
    for row in _ranking_rows(refined_crossings):
        ranking_by_j[float(row["J_uT"])] = row
    ranking = sorted(
        ranking_by_j.values(),
        key=lambda row: float(row["best_crossing_score"]),
        reverse=True,
    )
    selected = select_full_system_j_values(ranking)
    _write_csv(
        output_dir / "reduced_spectral_scan.csv",
        _spectral_rows(coarse + refined + control_full + control_axial),
    )
    _write_csv(output_dir / "avoided_crossings.csv", crossings)
    _write_csv(output_dir / "exchange_candidate_ranking.csv", ranking)
    _write_json(
        output_dir / "spectral_selection.json",
        {
            "selected_J_uT": selected,
            "coarse_region_centres_uT": region_centres,
            "refined_J_uT": refined_j,
            "cancellation_references_uT": CANCELLATION_REFERENCES_UT,
            "selection_policy": (
                "five highest-scoring discovery points separated by at least 15 uT "
                "from each other and from reserved J=0,-57.9792,+115.9583 uT controls"
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    _plot_spectroscopy(
        output_dir,
        coarse + refined,
        crossings,
        ranking,
        selected,
        control_full[0],
        control_axial[0],
    )
    return ranking, selected


def _plot_spectroscopy(
    output_dir: Path,
    spectra: list[dict[str, object]],
    crossings: list[dict[str, object]],
    ranking: list[dict[str, object]],
    selected_j: tuple[float, ...],
    control_full: dict[str, object],
    control_axial: dict[str, object],
) -> None:
    primary = [
        row
        for row in crossings
        if not row["remove_transverse"] and row["scan_kind"] in ("coarse", "refined")
    ]
    scores = np.asarray([float(row["needle_crossing_score"]) for row in primary])
    keep = scores >= (np.quantile(scores, 0.98) if scores.size else math.inf)
    shown = [row for row, selected in zip(primary, keep) if selected]
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    if shown:
        score_values = np.asarray([float(row["needle_crossing_score"]) for row in shown])
        sizes = 18.0 + 180.0 * score_values / max(float(np.max(score_values)), 1e-15)
        image = axis.scatter(
            [float(row["J_uT"]) for row in shown],
            [float(row["theta_c_deg"]) for row in shown],
            s=sizes,
            c=[float(row["singlet_character_transfer"]) for row in shown],
            cmap="viridis",
            alpha=0.75,
        )
        fig.colorbar(image, ax=axis, label="Singlet-character transfer")
    for value, label in ((0.0, "J=0"), (CANCELLATION_REFERENCES_UT[0], "D/J=6"), (CANCELLATION_REFERENCES_UT[1], "D/J=-3")):
        axis.axvline(value, linestyle="--", linewidth=1.0, label=label)
    axis.set(
        xlabel=r"Exchange $J$ ($\mu$T)",
        ylabel=r"Crossing angle $\theta_c$ (degrees)",
        title="Singlet-active avoided crossings near 90 degrees",
        ylim=(85.0, 95.0),
    )
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "crossing_map_vs_J.png", dpi=180)
    plt.close(fig)

    ordered = sorted(ranking, key=lambda row: float(row["J_uT"]))
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    scatter = axis.scatter(
        [float(row["J_uT"]) for row in ordered],
        [float(row["best_gap_MHz"]) for row in ordered],
        c=[float(row["best_crossing_score"]) for row in ordered],
        cmap="plasma",
    )
    axis.set(
        xlabel=r"Exchange $J$ ($\mu$T)",
        ylabel="Best singlet-active gap (MHz)",
        title="Gap selected by the transparent needle-crossing score",
    )
    for value in (0.0, *CANCELLATION_REFERENCES_UT):
        axis.axvline(value, color="black", linestyle="--", linewidth=0.8)
    fig.colorbar(scatter, ax=axis, label="Needle-crossing score")
    fig.tight_layout()
    fig.savefig(output_dir / "singlet_active_gap_vs_J.png", dpi=180)
    plt.close(fig)

    by_j = {float(spectrum["J_uT"]): spectrum for spectrum in spectra if spectrum["scan_kind"] == "refined"}
    top_spectral_j = _nonredundant_j_values(ranking, 8, 5.0)
    panels = []
    for value in top_spectral_j:
        nearest = min(by_j, key=lambda item: abs(item - value), default=None)
        if nearest is not None:
            panels.append((value, by_j[nearest]))
    if panels:
        columns = 2
        rows_count = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(11.0, 3.8 * rows_count),
            squeeze=False,
            layout="constrained",
        )
        ranking_lookup = {float(row["J_uT"]): row for row in ranking}
        last_scatter = None
        for axis, (requested_j, spectrum) in zip(axes.flat, panels):
            nearest_rank = min(ranking_lookup, key=lambda item: abs(item - requested_j))
            best = ranking_lookup[nearest_rank]
            low = int(best["best_lower_level"])
            high = int(best["best_upper_level"]) + 1
            angles = np.asarray(spectrum["angles_deg"])
            energies = np.asarray(spectrum["energies_MHz"])
            singlets = np.asarray(spectrum["singlet_characters"])
            centre_index = int(np.argmin(np.abs(angles - float(best["best_theta_c_deg"]))))
            reference = 0.5 * (
                energies[centre_index, int(best["best_lower_level"])]
                + energies[centre_index, int(best["best_upper_level"])]
            )
            for level in range(low, high):
                axis.plot(
                    angles,
                    1e3 * (energies[:, level] - reference),
                    color="0.75",
                    linewidth=0.8,
                )
                last_scatter = axis.scatter(
                    angles,
                    1e3 * (energies[:, level] - reference),
                    c=singlets[:, level],
                    vmin=0.0,
                    vmax=1.0,
                    cmap="viridis",
                    s=5,
                )
            axis.axvline(90.0, color="black", linestyle="--", linewidth=0.7)
            axis.set(
                title=rf"$J={requested_j:g}\ \mu$T",
                xlabel="Angle (degrees)",
                ylabel="Relative energy (kHz)",
            )
        for axis in axes.flat[len(panels) :]:
            axis.set_visible(False)
        if last_scatter is not None:
            fig.colorbar(last_scatter, ax=axes.ravel().tolist(), label=r"$\langle P_S\rangle$", shrink=0.75)
        fig.suptitle("Candidate needle-core avoided crossings: color is singlet character")
        fig.savefig(output_dir / "candidate_spectra.png", dpi=180)
        plt.close(fig)

    physical_candidates = detect_avoided_crossings(control_full)
    physical_best = max(
        physical_candidates, key=lambda row: float(row["needle_crossing_score"])
    )
    pair = int(physical_best["lower_level"])
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.5), sharey=True, layout="constrained"
    )
    last_scatter = None
    for axis, spectrum, title in (
        (axes[0], control_full, "Physical transverse N5/N10 terms"),
        (axes[1], control_axial, "N5/N10 transverse block removed"),
    ):
        angles = np.asarray(spectrum["angles_deg"])
        energies = np.asarray(spectrum["energies_MHz"])
        singlets = np.asarray(spectrum["singlet_characters"])
        centre_index = int(np.argmin(np.abs(angles - 90.0)))
        reference = 0.5 * (
            energies[centre_index, pair] + energies[centre_index, pair + 1]
        )
        window = (angles >= 89.5) & (angles <= 90.5)
        for level in (pair, pair + 1):
            relative = 1e3 * (energies[:, level] - reference)
            axis.plot(angles[window], relative[window], color="0.75", linewidth=0.8)
            last_scatter = axis.scatter(
                angles[window],
                relative[window],
                c=singlets[window, level],
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                s=9,
            )
        axis.axvline(90.0, color="black", linestyle="--", linewidth=0.8)
        axis.set(title=title, xlabel="Angle (degrees)", xlim=(89.5, 90.5))
    axes[0].set_ylabel("Relative energy (kHz)")
    if last_scatter is not None:
        fig.colorbar(
            last_scatter,
            ax=axes,
            label=r"$\langle P_S\rangle$",
            shrink=0.8,
        )
    fig.suptitle(f"Transverse-hyperfine avoided-crossing control (D=J=0, Q scale={control_full.get('quadrupole_scale', 0):g})")
    fig.savefig(output_dir / "transverse_hyperfine_control.png", dpi=180)
    plt.close(fig)


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "run_metadata.json"


def ensure_manifest(
    output_dir: Path,
    spectral_config: SpectralConfig,
    full_config: SweepConfig,
    resume: bool,
) -> dict[str, object]:
    stable = {
        "spectral_configuration": spectral_config.stable_dict(),
        "full_configuration": full_config.stable_dict(),
        "full_angles_deg": list(FULL_ANGLES_DEG),
        "needle_refinement_angles_deg": list(NEEDLE_REFINEMENT_ANGLES_DEG),
        "quadrupole_scale": full_config.quadrupole_scale,
        "literal_efimova_hore": True,
    }
    digest = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = _manifest_path(output_dir)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["configuration_hash"] != digest:
            raise RuntimeError("Existing output uses an incompatible configuration.")
        if not resume:
            raise RuntimeError("Output exists; pass --resume to continue.")
        return manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "configuration_hash": digest,
        **stable,
        "status": "created",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": _git_commit(),
        "exchange_operator_controls": exchange_operator_controls(),
        "stages": {},
    }
    _write_json(path, manifest)
    return manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_selection(output_dir: Path) -> tuple[float, ...]:
    value = json.loads((output_dir / "spectral_selection.json").read_text(encoding="utf-8"))
    return tuple(float(item) for item in value["selected_J_uT"])


def _available_samples(
    output_dir: Path,
    condition: EfimovaHoreCondition,
    angles_deg: Iterable[float],
    maximum: int,
) -> int:
    count = 0
    for sample in range(maximum):
        if not _complete(output_dir, condition, angles_deg, sample + 1):
            break
        count += 1
    return count


def full_screen_metrics(
    output_dir: Path,
    conditions: Iterable[EfimovaHoreCondition],
    config: SweepConfig,
    sample_count: int,
) -> list[dict[str, object]]:
    control = efimova_hore_uncoupled_control()
    curves: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for condition in conditions:
        means = []
        for theta in FULL_ANGLES_DEG:
            values = [
                float(load_cell(output_dir, condition, theta, sample, config)["yield"])
                for sample in range(sample_count)
            ]
            means.append(float(np.mean(values)))
        curves[condition.condition_id] = np.asarray(means)
    control_curve = curves[control.condition_id]
    index = {theta: position for position, theta in enumerate(FULL_ANGLES_DEG)}
    for condition in conditions:
        curve = curves[condition.condition_id]
        near = np.array([index[value] for value in (85.0, 88.0, 90.0, 92.0, 95.0)])
        central_feature = abs(
            curve[index[90.0]] - 0.5 * (curve[index[85.0]] + curve[index[95.0]])
        )
        curvature = abs(
            curve[index[88.0]] - 2.0 * curve[index[90.0]] + curve[index[92.0]]
        )
        total_range = float(np.ptp(curve))
        central_values = curve[near]
        if curve[index[90.0]] <= np.mean((curve[index[85.0]], curve[index[95.0]])):
            extremum_local = int(np.argmin(central_values))
        else:
            extremum_local = int(np.argmax(central_values))
        extremum_angle = (85.0, 88.0, 90.0, 92.0, 95.0)[extremum_local]
        proximity = math.exp(-0.5 * ((extremum_angle - 90.0) / 2.0) ** 2)
        correlation = float(scipy.stats.spearmanr(curve, control_curve).statistic)
        if not np.isfinite(correlation):
            correlation = -1.0
        specificity = central_feature / (total_range + 1e-15)
        score = (
            central_feature
            * proximity
            * (0.5 + 0.5 * max(correlation, 0.0))
            * math.sqrt(max(specificity, 0.0))
        )
        rows.append(
            {
                **condition.stable_dict(),
                "samples": sample_count,
                "needle_feature_90_vs_85_95": central_feature,
                "curvature_88_90_92": curvature,
                "signed_0_90_contrast": float(curve[index[0.0]] - curve[index[90.0]]),
                "full_angular_range": total_range,
                "needle_specificity": specificity,
                "central_extremum_deg": extremum_angle,
                "centre_proximity_factor": proximity,
                "rank_correlation_with_uncoupled": correlation,
                "full14_needle_score": score,
            }
        )
    rows.sort(key=lambda row: float(row["full14_needle_score"]), reverse=True)
    return rows


def select_confirmation_conditions(
    metrics: Iterable[dict[str, object]],
) -> tuple[EfimovaHoreCondition, ...]:
    nonzero = [row for row in metrics if abs(float(row["exchange_J_uT"])) > 1e-12]
    selected_j = [float(row["exchange_J_uT"]) for row in nonzero[:3]]
    selected_j.append(0.0)
    return tuple(exchange_condition(value) for value in selected_j)


def _raw_full_rows(
    output_dir: Path,
    conditions: Iterable[EfimovaHoreCondition],
    angles_deg: Iterable[float],
    maximum_samples: int,
    config: SweepConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in conditions:
        for theta in angles_deg:
            for sample in range(maximum_samples):
                if not checkpoint_path(output_dir, condition, theta, sample).exists():
                    continue
                cell = load_cell(output_dir, condition, theta, sample, config)
                rows.append(
                    {
                        **condition.stable_dict(),
                        "theta_deg": theta,
                        "sample_id": sample,
                        "yield": cell["yield"],
                        "yield_quadrature_error": cell["yield_quadrature_error"],
                        "yield_propagation_bound": cell["yield_propagation_bound"],
                        "yield_tail_bound": cell["yield_tail_bound"],
                        "accepted_steps": cell["accepted_steps"],
                        "rejected_steps": cell["rejected_steps"],
                        "matvecs": cell["matvecs"],
                        "total_seconds": cell["total_seconds"],
                    }
                )
    return rows


def _write_required_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]
) -> None:
    """Write a required table even when a wall-time stop leaves it empty."""

    if rows:
        _write_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=list(fieldnames)).writeheader()
    os.replace(temporary, path)


def create_full_report(
    output_dir: Path,
    selected_j: Iterable[float],
    config: SweepConfig,
    confirmation: Iterable[EfimovaHoreCondition] = (),
) -> dict[str, object]:
    coupled = tuple(exchange_condition(value) for value in selected_j)
    screen_conditions = (efimova_hore_uncoupled_control(), *coupled)
    screen_complete = all(
        _complete(output_dir, condition, FULL_ANGLES_DEG, 1)
        for condition in screen_conditions
    )
    screen_metrics: list[dict[str, object]] = []
    if screen_complete:
        screen_metrics = full_screen_metrics(output_dir, screen_conditions, config, 1)
    screen_rows = _raw_full_rows(
        output_dir, screen_conditions, FULL_ANGLES_DEG, 1, config
    )
    required_fields = (
        "condition_id",
        "exchange_J_uT",
        "d_paper_uT",
        "theta_deg",
        "sample_id",
        "yield",
        "yield_quadrature_error",
        "yield_propagation_bound",
        "yield_tail_bound",
    )
    _write_required_csv(
        output_dir / "full14_screen.csv", screen_rows, required_fields
    )
    confirmation = tuple(confirmation)
    confirm_rows = _raw_full_rows(
        output_dir, confirmation, tuple(sorted(set(FULL_ANGLES_DEG + NEEDLE_REFINEMENT_ANGLES_DEG))), 4, config
    )
    _write_required_csv(
        output_dir / "full14_confirm.csv", confirm_rows, required_fields
    )
    if screen_metrics:
        _write_csv(output_dir / "full14_screen_metrics.csv", screen_metrics)
        _plot_full_results(output_dir, screen_conditions, screen_metrics, config)
    summary = {
        "screen_complete": screen_complete,
        "selected_J_uT": list(selected_j),
        "screen_metrics": screen_metrics,
        "confirmation_conditions": [item.stable_dict() for item in confirmation],
        "confirmation_complete_samples": {
            item.condition_id: _available_samples(output_dir, item, FULL_ANGLES_DEG, 4)
            for item in confirmation
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output_dir / "full14_summary.json", summary)
    return summary


def _plot_full_results(
    output_dir: Path,
    conditions: tuple[EfimovaHoreCondition, ...],
    metrics: list[dict[str, object]],
    config: SweepConfig,
) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 5.4))
    for condition in conditions:
        values = np.array(
            [
                float(load_cell(output_dir, condition, theta, 0, config)["yield"])
                for theta in FULL_ANGLES_DEG
            ]
        )
        modulation = 1e3 * (values[0] - values)
        label = "D=J=0" if condition.role == "uncoupled_control" else rf"$J={condition.exchange_J_uT:g}\ \mu$T"
        axis.plot(FULL_ANGLES_DEG, modulation, marker="o", linewidth=1.2, label=label)
    axis.set(
        xlabel="Field angle (degrees)",
        ylabel=r"Yield reduction from 0 degrees ($\times10^{-3}$)",
        title=rf"Full 14-nucleus exchange reconnaissance ($D=-347.875\ \mu$T, $\tau=5\ \mu$s, Q scale={config.quadrupole_scale:g})",
    )
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "full14_angular_curves.png", dpi=180)
    plt.close(fig)

    coupled = sorted(
        (row for row in metrics if row["role"] != "uncoupled_control"),
        key=lambda row: float(row["exchange_J_uT"]),
    )
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(
        [float(row["exchange_J_uT"]) for row in coupled],
        [float(row["full14_needle_score"]) for row in coupled],
        marker="o",
    )
    for value, label in ((0.0, "J=0"), (CANCELLATION_REFERENCES_UT[0], "D/J=6"), (CANCELLATION_REFERENCES_UT[1], "D/J=-3")):
        axis.axvline(value, linestyle="--", linewidth=1.0, label=label)
    axis.set(
        xlabel=r"Exchange $J$ ($\mu$T)",
        ylabel="Full-system needle score",
        title="Narrow 90-degree feature score, not total angular range",
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "needle_metric_vs_J.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("spectroscopy", "full", "auto", "report"), required=True)
    add_quadrupole_argument(parser)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--spectral-workers", type=int, default=4)
    parser.add_argument("--max-wall-hours", type=float, default=23.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "exchange_avoided_crossing_scan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.max_wall_hours <= 23.5:
        raise ValueError("--max-wall-hours must be in (0, 23.5].")
    if args.workers < 1 or args.spectral_workers < 1:
        raise ValueError("Worker counts must be positive.")
    started_wall = time.time()
    deadline = started_wall + args.max_wall_hours * 3600.0
    spectral_config = SpectralConfig(quadrupole_scale=args.quadrupole_scale)
    full_config = replace(SweepConfig(), seed=args.seed, lifetime_us=5.0,
                          quadrupole_scale=args.quadrupole_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_manifest(
        args.output_dir, spectral_config, full_config, args.resume
    )
    if args.stage in ("spectroscopy", "auto"):
        if not (args.output_dir / "spectral_selection.json").exists():
            ranking, selected_j = run_spectroscopy(
                args.output_dir, spectral_config, args.spectral_workers
            )
        else:
            ranking = _read_csv(args.output_dir / "exchange_candidate_ranking.csv")
            selected_j = _load_selection(args.output_dir)
        manifest["stages"]["spectroscopy"] = {
            "complete": True,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest["selected_J_uT"] = list(selected_j)
        _write_json(_manifest_path(args.output_dir), manifest)
        if args.stage == "spectroscopy":
            manifest["status"] = "spectroscopy_complete"
            _write_json(_manifest_path(args.output_dir), manifest)
            return
    else:
        if not (args.output_dir / "spectral_selection.json").exists():
            raise RuntimeError("Run --stage spectroscopy before the full or report stage.")
        selected_j = _load_selection(args.output_dir)

    coupled = tuple(exchange_condition(value) for value in selected_j)
    screen_conditions = (efimova_hore_uncoupled_control(), *coupled)
    if args.stage == "report":
        confirmation_ids = json.loads(
            (args.output_dir / "full14_confirmation_selection.json").read_text(encoding="utf-8")
        ) if (args.output_dir / "full14_confirmation_selection.json").exists() else []
        confirmation = tuple(exchange_condition(float(row["exchange_J_uT"])) for row in confirmation_ids)
        create_full_report(args.output_dir, selected_j, full_config, confirmation)
        return

    benchmark_path = args.output_dir / "runtime_benchmark.json"
    if not benchmark_path.exists():
        raise RuntimeError(
            "Copy the established runtime_benchmark.json into the output directory before the full stage."
        )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    matching = [row for row in benchmark["rows"] if int(row["workers"]) == args.workers]
    if not matching:
        raise RuntimeError("Requested worker count is absent from the established benchmark.")
    throughput = float(matching[0]["simulated_us_per_wall_second"])
    estimated_wave_seconds = args.workers * full_config.maximum_time_us / throughput
    manifest["selected_workers"] = args.workers
    manifest["estimated_full_horizon_wave_seconds"] = estimated_wave_seconds
    _write_json(_manifest_path(args.output_dir), manifest)
    complete = run_tasks(
        args.output_dir,
        screen_conditions,
        FULL_ANGLES_DEG,
        1,
        full_config,
        args.workers,
        deadline,
        estimated_wave_seconds,
    )
    confirmation: tuple[EfimovaHoreCondition, ...] = ()
    if complete:
        metrics = full_screen_metrics(args.output_dir, screen_conditions, full_config, 1)
        confirmation = select_confirmation_conditions(metrics)
        _write_json(
            args.output_dir / "full14_confirmation_selection.json",
            [item.stable_dict() for item in confirmation],
        )
        complete = run_tasks(
            args.output_dir,
            confirmation,
            FULL_ANGLES_DEG,
            4,
            full_config,
            args.workers,
            deadline,
            estimated_wave_seconds,
        )
    if complete:
        complete = run_tasks(
            args.output_dir,
            confirmation,
            NEEDLE_REFINEMENT_ANGLES_DEG,
            4,
            full_config,
            args.workers,
            deadline,
            estimated_wave_seconds,
        )
    summary = create_full_report(
        args.output_dir, selected_j, full_config, confirmation
    )
    manifest["stages"]["full"] = {
        "complete": complete,
        "screen_complete": summary["screen_complete"],
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["status"] = "complete" if complete else "walltime_limited_partial"
    manifest["elapsed_wall_seconds"] = time.time() - started_wall
    _write_json(_manifest_path(args.output_dir), manifest)


if __name__ == "__main__":
    main()
