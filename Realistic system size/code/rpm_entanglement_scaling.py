"""Entanglement-size scaling study for the FAD-Trp radical-pair model.

The study grows the nuclear bath from the two strongest FAD-coupled nuclei and
the two strongest Trp-coupled nuclei, then adds the remaining nuclei in the
published approximate hyperfine-strength order. For each truncation it evolves
a closed pure state and tracks log negativity for the thesis-inspired cuts:

    e1+n1 | e2+n2
    e1+e2 | n1+n2

The default nuclear preparation is a deterministic, separable
minimum-Overhauser product state. Spin-1 nitrogens are initialized in m=0 along
their quadrupolar principal axis, while spin-1/2 nuclei are assigned fixed
m=+/-1/2 states along their strongest hyperfine axes. By default, nitrogens are
treated as spin-1 sites and receive a simple axial quadrupolar term in their
hyperfine principal-axis frame.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.sparse.linalg import LinearOperator, expm_multiply

import rpm_full_fad_trp_magnetosensitivity as rpm

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class StudyNucleus:
    name: str
    radical: str
    tensor_MHz: np.ndarray
    spin: float

    @property
    def dimension(self) -> int:
        return int(round(2.0 * self.spin + 1.0))

    @property
    def element(self) -> str:
        atom_key, _radical = rpm.canonical_atom_key(self.name)
        if atom_key.startswith("F"):
            return atom_key[1]
        if atom_key.startswith("W"):
            return atom_key[1]
        return "?"


@dataclass
class LocalTerm:
    coefficient: complex
    operators: tuple[tuple[int, np.ndarray], ...]


@dataclass
class Params:
    B_microT: float = 50.0
    theta_deg: float = 90.0
    phi_deg: float = 0.0
    t_max_us: float = 2.0
    n_steps: int = 101
    samples: int = 1
    seed: int = 20260528
    J_MHz: float = 3.0
    include_dipole: bool = True
    physical_dipole: bool = False
    D_MHz: float = -19.10703069
    hyperfine_mode: str = "full"
    operator_mode: str = "auto"
    max_sparse_dim: int = 65536
    evolver: str = "expm"
    spin_model: str = "physical"
    quadrupole_coupling_MHz: float = 0.5
    quadrupole_eta: float = 0.0
    nuclear_state: str = "minimum-overhauser"


def spin_matrices(spin: float) -> dict[str, np.ndarray]:
    """Dimensionless spin operators in the |m=s>, ..., |m=-s> basis."""
    dimension = int(round(2.0 * spin + 1.0))
    m_values = np.array([spin - index for index in range(dimension)], dtype=float)
    raising = np.zeros((dimension, dimension), dtype=complex)
    for column, m_value in enumerate(m_values):
        if m_value >= spin:
            continue
        row = column - 1
        raising[row, column] = math.sqrt(spin * (spin + 1.0) - m_value * (m_value + 1.0))

    lowering = raising.conj().T
    sx = 0.5 * (raising + lowering)
    sy = (raising - lowering) / (2.0j)
    sz = np.diag(m_values).astype(complex)
    return {
        "x": sx,
        "y": sy,
        "z": sz,
        "i": np.eye(dimension, dtype=complex),
    }


def infer_nuclear_spin(name: str, spin_model: str) -> float:
    if spin_model == "spin-half":
        return 0.5
    atom_key, _radical = rpm.canonical_atom_key(name)
    element = atom_key[1]
    if element == "N":
        return 1.0
    return 0.5


def all_study_nuclei(spin_model: str) -> list[StudyNucleus]:
    nuclei = []
    for name in rpm.PUBLISHED_NUCLEUS_ORDER:
        _atom_key, radical = rpm.canonical_atom_key(name)
        nuclei.append(
            StudyNucleus(
                name=name,
                radical=radical,
                tensor_MHz=rpm.literature_tensor_MHz(name),
                spin=infer_nuclear_spin(name, spin_model),
            )
        )
    return nuclei


def nuclei_by_radical(spin_model: str) -> tuple[list[StudyNucleus], list[StudyNucleus]]:
    all_nuclei = all_study_nuclei(spin_model)
    return (
        [nucleus for nucleus in all_nuclei if nucleus.radical == "FAD"],
        [nucleus for nucleus in all_nuclei if nucleus.radical == "TRP"],
    )


def global_growth_sequence(spin_model: str, initial_per_radical: int) -> list[StudyNucleus]:
    all_nuclei = all_study_nuclei(spin_model)
    selected: list[StudyNucleus] = []
    for radical in ("FAD", "TRP"):
        selected.extend([nucleus for nucleus in all_nuclei if nucleus.radical == radical][:initial_per_radical])

    selected_names = {nucleus.name for nucleus in selected}
    selected.extend([nucleus for nucleus in all_nuclei if nucleus.name not in selected_names])
    return selected


def paired_growth_sequence(spin_model: str) -> list[StudyNucleus]:
    fad_nuclei, trp_nuclei = nuclei_by_radical(spin_model)
    sequence = []
    for index in range(max(len(fad_nuclei), len(trp_nuclei))):
        if index < len(fad_nuclei):
            sequence.append(fad_nuclei[index])
        if index < len(trp_nuclei):
            sequence.append(trp_nuclei[index])
    return sequence


def growth_sequence(spin_model: str, initial_per_radical: int, growth_mode: str) -> list[StudyNucleus]:
    if growth_mode == "paired":
        return paired_growth_sequence(spin_model)
    if growth_mode == "global":
        return global_growth_sequence(spin_model, initial_per_radical)
    raise ValueError("growth_mode must be one of: paired, global.")


def filtered_tensor_MHz(tensor: np.ndarray, mode: str) -> np.ndarray:
    return rpm.filtered_hyperfine_tensor(np.asarray(tensor, dtype=float), mode)


def site_dimensions(nuclei: list[StudyNucleus]) -> list[int]:
    return [2, 2] + [nucleus.dimension for nucleus in nuclei]


def effective_qubits(dimensions: Iterable[int]) -> float:
    return float(np.log2(np.prod(list(dimensions), dtype=np.float64)))


def state_memory_mb(dimensions: Iterable[int]) -> float:
    dimension = int(np.prod(list(dimensions), dtype=np.int64))
    return dimension * np.dtype(np.complex128).itemsize / 1024**2


def add_one_spin_terms(
    terms: list[LocalTerm],
    site: int,
    vector: np.ndarray,
    site_ops: list[dict[str, np.ndarray]],
) -> None:
    for axis, coefficient in zip(AXES, vector):
        if abs(coefficient) > 1e-15:
            terms.append(LocalTerm(complex(coefficient), ((site, site_ops[site][axis]),)))


def add_two_spin_tensor_terms(
    terms: list[LocalTerm],
    site_a: int,
    site_b: int,
    tensor: np.ndarray,
    site_ops: list[dict[str, np.ndarray]],
) -> None:
    for a_index, axis_a in enumerate(AXES):
        for b_index, axis_b in enumerate(AXES):
            coefficient = tensor[a_index, b_index]
            if abs(coefficient) > 1e-15:
                terms.append(
                    LocalTerm(
                        complex(coefficient),
                        ((site_a, site_ops[site_a][axis_a]), (site_b, site_ops[site_b][axis_b])),
                    )
                )


def quadrupole_operator_MHz(
    nucleus: StudyNucleus,
    coupling_MHz: float,
    eta: float,
    ops: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Simple N-14 quadrupole model in the hyperfine principal-axis frame."""
    if nucleus.spin <= 0.5 or abs(coupling_MHz) < 1e-15:
        return None

    tensor = 0.5 * (nucleus.tensor_MHz + nucleus.tensor_MHz.T)
    _evals, evecs = np.linalg.eigh(tensor)
    # Smallest/middle/largest absolute principal values define x/y/z axes.
    order = np.argsort(np.abs(_evals))
    ex = evecs[:, order[0]]
    ey = evecs[:, order[1]]
    ez = evecs[:, order[2]]

    def projected_spin(axis_vector: np.ndarray) -> np.ndarray:
        return axis_vector[0] * ops["x"] + axis_vector[1] * ops["y"] + axis_vector[2] * ops["z"]

    ix_p = projected_spin(ex)
    iy_p = projected_spin(ey)
    iz_p = projected_spin(ez)
    identity = ops["i"]
    spin = nucleus.spin
    hq = 3.0 * (iz_p @ iz_p) - spin * (spin + 1.0) * identity
    hq += eta * (ix_p @ ix_p - iy_p @ iy_p)
    return rpm.mhz_to_rad_per_us(coupling_MHz) * hq / 6.0


def build_terms(params: Params, nuclei: list[StudyNucleus]) -> tuple[list[LocalTerm], list[int]]:
    dimensions = site_dimensions(nuclei)
    site_spins = [0.5, 0.5] + [nucleus.spin for nucleus in nuclei]
    site_ops = [spin_matrices(spin) for spin in site_spins]
    terms: list[LocalTerm] = []

    B_mT = 1e-3 * params.B_microT * rpm.unit_vector_from_angles(params.theta_deg, params.phi_deg)
    electron_zeeman = rpm.mhz_to_rad_per_us(rpm.GAMMA_E_MHZ_PER_MT) * B_mT
    add_one_spin_terms(terms, 0, electron_zeeman, site_ops)
    add_one_spin_terms(terms, 1, electron_zeeman, site_ops)

    for nucleus_index, nucleus in enumerate(nuclei):
        electron_site = 0 if nucleus.radical == "FAD" else 1
        nuclear_site = 2 + nucleus_index
        tensor = rpm.mhz_to_rad_per_us(filtered_tensor_MHz(nucleus.tensor_MHz, params.hyperfine_mode))
        add_two_spin_tensor_terms(terms, electron_site, nuclear_site, tensor, site_ops)

        q_operator = quadrupole_operator_MHz(
            nucleus,
            params.quadrupole_coupling_MHz,
            params.quadrupole_eta,
            site_ops[nuclear_site],
        )
        if q_operator is not None:
            terms.append(LocalTerm(1.0, ((nuclear_site, q_operator),)))

    if abs(params.J_MHz) > 1e-15:
        exchange_tensor = rpm.mhz_to_rad_per_us(-2.0 * params.J_MHz) * np.eye(3)
        add_two_spin_tensor_terms(terms, 0, 1, exchange_tensor, site_ops)

    if params.include_dipole:
        if params.physical_dipole:
            dipole_tensor = rpm.published_dipolar_tensor_MHz()
        else:
            dipole_tensor = rpm.best_case_dipolar_tensor_MHz(params.D_MHz)
        add_two_spin_tensor_terms(terms, 0, 1, rpm.mhz_to_rad_per_us(dipole_tensor), site_ops)

    return terms, dimensions


def apply_local_product(state: np.ndarray, dimensions: list[int], operators: tuple[tuple[int, np.ndarray], ...]) -> np.ndarray:
    tensor = state.reshape(dimensions)
    for site, operator in operators:
        moved = np.moveaxis(tensor, site, 0).reshape(dimensions[site], -1)
        moved = operator @ moved
        restored = moved.reshape((dimensions[site],) + tuple(dim for idx, dim in enumerate(dimensions) if idx != site))
        tensor = np.moveaxis(restored, 0, site)
    return tensor.reshape(-1)


def make_linear_operator(terms: list[LocalTerm], dimensions: list[int]) -> LinearOperator:
    dimension = int(np.prod(dimensions, dtype=np.int64))

    def matvec(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=complex).reshape(-1)
        out = np.zeros_like(vector)
        for term in terms:
            out += term.coefficient * apply_local_product(vector, dimensions, term.operators)
        return out

    def matmat(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=complex)
        return np.column_stack([matvec(matrix[:, col]) for col in range(matrix.shape[1])])

    return LinearOperator((dimension, dimension), matvec=matvec, rmatvec=matvec, matmat=matmat, dtype=np.complex128)


def sparse_local_product(term: LocalTerm, dimensions: list[int]) -> sp.csr_matrix:
    by_site = {site: operator for site, operator in term.operators}
    factors = [
        sp.csr_matrix(by_site.get(site, np.eye(dimension, dtype=complex)))
        for site, dimension in enumerate(dimensions)
    ]
    result = factors[0]
    for factor in factors[1:]:
        result = sp.kron(result, factor, format="csr")
    return term.coefficient * result


def make_sparse_operator(terms: list[LocalTerm], dimensions: list[int]) -> sp.csr_matrix:
    dimension = int(np.prod(dimensions, dtype=np.int64))
    hamiltonian = sp.csr_matrix((dimension, dimension), dtype=complex)
    for term in terms:
        hamiltonian += sparse_local_product(term, dimensions)
    return hamiltonian.tocsr()


def make_hamiltonian_operator(terms: list[LocalTerm], dimensions: list[int], params: Params):
    dimension = int(np.prod(dimensions, dtype=np.int64))
    if params.operator_mode == "sparse":
        return make_sparse_operator(terms, dimensions)
    if params.operator_mode == "matrix-free":
        return make_linear_operator(terms, dimensions)
    if params.operator_mode != "auto":
        raise ValueError("operator_mode must be one of: auto, sparse, matrix-free.")
    if dimension <= params.max_sparse_dim:
        return make_sparse_operator(terms, dimensions)
    return make_linear_operator(terms, dimensions)


def random_phase_state(dimension: int, rng: np.random.Generator) -> np.ndarray:
    phases = rng.uniform(0.0, rpm.TWOPI, size=dimension)
    return np.exp(1.0j * phases) / math.sqrt(dimension)


def product_up_state(dimensions: list[int]) -> np.ndarray:
    state = np.zeros(int(np.prod(dimensions, dtype=np.int64)), dtype=complex)
    state[0] = 1.0
    return state


def product_x_state(dimensions: list[int]) -> np.ndarray:
    dimension = int(np.prod(dimensions, dtype=np.int64))
    return np.ones(dimension, dtype=complex) / math.sqrt(dimension)


def strongest_hyperfine_axis(nucleus: StudyNucleus) -> np.ndarray:
    tensor = 0.5 * (nucleus.tensor_MHz + nucleus.tensor_MHz.T)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    axis = np.asarray(eigenvectors[:, int(np.argmax(np.abs(eigenvalues)))], dtype=float)
    dominant_index = int(np.argmax(np.abs(axis)))
    if axis[dominant_index] < 0.0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def minimum_overhauser_signs(spin_model: str) -> dict[str, int]:
    signs: dict[str, int] = {}
    for radical, radical_nuclei in zip(("FAD", "TRP"), nuclei_by_radical(spin_model)):
        spin_half_nuclei = [nucleus for nucleus in radical_nuclei if np.isclose(nucleus.spin, 0.5)]
        contributions = [
            nucleus.tensor_MHz @ (0.5 * strongest_hyperfine_axis(nucleus))
            for nucleus in spin_half_nuclei
        ]
        best_signs: tuple[int, ...] | None = None
        best_norm = float("inf")
        for candidate in itertools.product((-1, 1), repeat=len(spin_half_nuclei)):
            total = sum(
                (sign * contribution for sign, contribution in zip(candidate, contributions)),
                start=np.zeros(3, dtype=float),
            )
            norm = float(np.linalg.norm(total))
            if norm < best_norm:
                best_norm = norm
                best_signs = candidate
        if best_signs is None:
            continue
        signs.update(
            {
                nucleus.name: sign
                for nucleus, sign in zip(spin_half_nuclei, best_signs)
            }
        )
    return signs


def spin_eigenstate(spin: float, axis: np.ndarray, target_m: float) -> np.ndarray:
    ops = spin_matrices(spin)
    operator = axis[0] * ops["x"] + axis[1] * ops["y"] + axis[2] * ops["z"]
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    state = eigenvectors[:, int(np.argmin(np.abs(eigenvalues - target_m)))]
    dominant_index = int(np.argmax(np.abs(state)))
    phase = np.angle(state[dominant_index])
    return state * np.exp(-1.0j * phase)


def minimum_overhauser_product_state(nuclei: list[StudyNucleus], spin_model: str) -> np.ndarray:
    signs = minimum_overhauser_signs(spin_model)
    local_states = []
    for nucleus in nuclei:
        axis = strongest_hyperfine_axis(nucleus)
        if nucleus.spin > 0.5:
            local_states.append(spin_eigenstate(nucleus.spin, axis, 0.0))
        else:
            local_states.append(spin_eigenstate(nucleus.spin, axis, 0.5 * signs[nucleus.name]))

    state = np.array([1.0 + 0.0j])
    for local_state in local_states:
        state = np.kron(state, local_state)
    return state


def initial_state(
    params: Params,
    nuclei: list[StudyNucleus],
    dimensions: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    nuclear_dimensions = dimensions[2:]
    nuclear_dimension = int(np.prod(nuclear_dimensions, dtype=np.int64))
    if params.nuclear_state == "random-phase":
        nuclear = random_phase_state(nuclear_dimension, rng)
    elif params.nuclear_state == "minimum-overhauser":
        nuclear = minimum_overhauser_product_state(nuclei, params.spin_model)
    elif params.nuclear_state == "product-up":
        nuclear = product_up_state(nuclear_dimensions)
    elif params.nuclear_state == "product-x":
        nuclear = product_x_state(nuclear_dimensions)
    else:
        raise ValueError(
            "nuclear_state must be one of: minimum-overhauser, random-phase, product-up, product-x."
        )
    return np.kron(rpm.singlet_electron_state(), nuclear)


def hamiltonian_matvec(hamiltonian, state: np.ndarray) -> np.ndarray:
    if hasattr(hamiltonian, "matvec"):
        return hamiltonian.matvec(state)
    return hamiltonian @ state


def evolve_rk4(hamiltonian, psi0: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    states = np.empty((len(t_us), len(psi0)), dtype=np.complex128)
    psi = psi0.astype(np.complex128, copy=True)
    states[0] = psi

    def rhs(state: np.ndarray) -> np.ndarray:
        return -1.0j * hamiltonian_matvec(hamiltonian, state)

    for idx in range(1, len(t_us)):
        dt = float(t_us[idx] - t_us[idx - 1])
        k1 = rhs(psi)
        k2 = rhs(psi + 0.5 * dt * k1)
        k3 = rhs(psi + 0.5 * dt * k2)
        k4 = rhs(psi + dt * k3)
        psi = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        psi /= np.linalg.norm(psi)
        states[idx] = psi
    return states


def evolve_states(params: Params, nuclei: list[StudyNucleus], sample_seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    terms, dimensions = build_terms(params, nuclei)
    hamiltonian = make_hamiltonian_operator(terms, dimensions, params)
    t_us = np.linspace(0.0, params.t_max_us, params.n_steps)
    psi0 = initial_state(params, nuclei, dimensions, np.random.default_rng(sample_seed))

    if params.evolver == "expm":
        states = expm_multiply(
            -1.0j * hamiltonian,
            psi0,
            start=0.0,
            stop=params.t_max_us,
            num=params.n_steps,
            endpoint=True,
            traceA=0.0,
        )
    elif params.evolver == "rk4":
        states = evolve_rk4(hamiltonian, psi0, t_us)
    else:
        raise ValueError("evolver must be one of: expm, rk4.")
    return t_us, states, dimensions


def pure_log_negativity(state: np.ndarray, dimensions: list[int], subsystem: list[int]) -> float:
    subsystem = sorted(subsystem)
    complement = [site for site in range(len(dimensions)) if site not in subsystem]
    subsystem_dim = int(np.prod([dimensions[site] for site in subsystem], dtype=np.int64))
    complement_dim = int(np.prod([dimensions[site] for site in complement], dtype=np.int64))
    active = subsystem if subsystem_dim <= complement_dim else complement
    rest = [site for site in range(len(dimensions)) if site not in active]

    matrix = np.transpose(state.reshape(dimensions), active + rest).reshape(
        int(np.prod([dimensions[site] for site in active], dtype=np.int64)),
        -1,
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    norm = np.linalg.norm(singular_values)
    if norm <= 0.0:
        return 0.0
    singular_values = singular_values / norm
    return float(2.0 * np.log2(np.sum(singular_values)))


def cut_subsystems(nuclei: list[StudyNucleus]) -> dict[str, list[int]]:
    fad_sites = [2 + idx for idx, nucleus in enumerate(nuclei) if nucleus.radical == "FAD"]
    return {
        "radical_cut": [0] + fad_sites,
        "electron_nuclear_cut": [0, 1],
    }


def time_average(t_us: np.ndarray, values: np.ndarray) -> float:
    if len(t_us) == 1 or np.isclose(t_us[-1], t_us[0]):
        return float(values[0])
    return float(np.trapezoid(values, t_us) / (t_us[-1] - t_us[0]))


def analyze_size(params: Params, nuclei: list[StudyNucleus]) -> tuple[dict[str, object], list[dict[str, float]]]:
    sample_curves: list[dict[str, np.ndarray]] = []
    base_seed = params.seed + 1009 * len(nuclei)
    dimensions: list[int] | None = None
    t_us: np.ndarray | None = None

    for sample_index in range(params.samples):
        t_us, states, dimensions = evolve_states(params, nuclei, base_seed + sample_index)
        cuts = cut_subsystems(nuclei)
        sample_curves.append(
            {
                "radical_cut": np.array(
                    [pure_log_negativity(state, dimensions, cuts["radical_cut"]) for state in states],
                    dtype=float,
                ),
                "electron_nuclear_cut": np.array(
                    [pure_log_negativity(state, dimensions, cuts["electron_nuclear_cut"]) for state in states],
                    dtype=float,
                ),
            }
        )

    if dimensions is None or t_us is None:
        raise RuntimeError("No samples were evolved.")

    radical = np.array([curve["radical_cut"] for curve in sample_curves])
    electron_nuclear = np.array([curve["electron_nuclear_cut"] for curve in sample_curves])
    two_cut_average = 0.5 * (radical + electron_nuclear)

    radical_mean = np.mean(radical, axis=0)
    electron_nuclear_mean = np.mean(electron_nuclear, axis=0)
    two_cut_mean = np.mean(two_cut_average, axis=0)
    two_cut_std = np.std(two_cut_average, axis=0, ddof=1) if params.samples > 1 else np.zeros_like(two_cut_mean)

    summary = {
        "n_nuclei": len(nuclei),
        "total_spins": len(dimensions),
        "hilbert_dimension": int(np.prod(dimensions, dtype=np.int64)),
        "effective_qubits": effective_qubits(dimensions),
        "state_memory_mb": state_memory_mb(dimensions),
        "fad_nuclei": sum(nucleus.radical == "FAD" for nucleus in nuclei),
        "trp_nuclei": sum(nucleus.radical == "TRP" for nucleus in nuclei),
        "spin1_nuclei": sum(nucleus.spin > 0.5 for nucleus in nuclei),
        "nucleus_names": ",".join(nucleus.name for nucleus in nuclei),
        "mean_logneg_radical_cut": time_average(t_us, radical_mean),
        "mean_logneg_electron_nuclear_cut": time_average(t_us, electron_nuclear_mean),
        "mean_logneg_two_cut_average": time_average(t_us, two_cut_mean),
        "peak_logneg_two_cut_average": float(np.max(two_cut_mean)),
        "final_logneg_two_cut_average": float(two_cut_mean[-1]),
    }
    time_rows = []
    for idx, time in enumerate(t_us):
        time_rows.append(
            {
                "n_nuclei": len(nuclei),
                "total_spins": len(dimensions),
                "effective_qubits": float(summary["effective_qubits"]),
                "t_us": float(time),
                "mean_logneg_radical_cut": float(radical_mean[idx]),
                "mean_logneg_electron_nuclear_cut": float(electron_nuclear_mean[idx]),
                "mean_logneg_two_cut_average": float(two_cut_mean[idx]),
                "std_logneg_two_cut_average": float(two_cut_std[idx]),
            }
        )
    return summary, time_rows


def fit_power_curve(
    summary_rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
) -> dict[str, object]:
    x = np.array([row[x_key] for row in summary_rows], dtype=float)
    y = np.array([row[y_key] for row in summary_rows], dtype=float)
    fit: dict[str, object] = {"x_key": x_key, "y_key": y_key}

    if len(x) >= 2 and np.all(y > 0):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _pcov = curve_fit(lambda xx, a, b: a * np.power(xx, b), x, y, p0=(y[0] / x[0], 1.0), maxfev=20000)
            residuals = y - popt[0] * np.power(x, popt[1])
            fit.update({
                "formula": "a*x^b",
                "a": float(popt[0]),
                "b": float(popt[1]),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
            })
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            fit["error"] = str(exc)
    return fit


def fit_each_cut(summary_rows: list[dict[str, object]], x_key: str) -> dict[str, object]:
    return {
        "x_key": x_key,
        "radical_cut": fit_power_curve(summary_rows, x_key, "mean_logneg_radical_cut"),
        "electron_nuclear_cut": fit_power_curve(
            summary_rows,
            x_key,
            "mean_logneg_electron_nuclear_cut",
        ),
    }


def save_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "n_nuclei",
        "total_spins",
        "effective_qubits",
        "hilbert_dimension",
        "state_memory_mb",
        "fad_nuclei",
        "trp_nuclei",
        "spin1_nuclei",
        "runtime_hours",
        "mean_logneg_radical_cut",
        "mean_logneg_electron_nuclear_cut",
        "mean_logneg_two_cut_average",
        "peak_logneg_two_cut_average",
        "final_logneg_two_cut_average",
        "nucleus_names",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def save_time_series_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "n_nuclei",
        "total_spins",
        "effective_qubits",
        "t_us",
        "mean_logneg_radical_cut",
        "mean_logneg_electron_nuclear_cut",
        "mean_logneg_two_cut_average",
        "std_logneg_two_cut_average",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_nucleus_order_csv(sequence: list[StudyNucleus], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["growth_index", "name", "radical", "element", "spin", "A_frobenius_MHz"])
        for index, nucleus in enumerate(sequence, start=1):
            writer.writerow(
                [
                    index,
                    nucleus.name,
                    nucleus.radical,
                    nucleus.element,
                    nucleus.spin,
                    float(np.linalg.norm(nucleus.tensor_MHz)),
                ]
            )


def plot_size_dynamics(
    rows: list[dict[str, float]],
    n_nuclei: int,
    path: Path,
    show: bool,
) -> None:
    selected = [row for row in rows if int(row["n_nuclei"]) == n_nuclei]
    if not selected:
        raise ValueError(f"No time-series rows found for {n_nuclei} nuclei.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    ax.plot(
        [row["t_us"] for row in selected],
        [row["mean_logneg_radical_cut"] for row in selected],
        linewidth=1.6,
        label=r"$e_1+n_{FAD}\,|\,e_2+n_{Trp}$",
    )
    ax.plot(
        [row["t_us"] for row in selected],
        [row["mean_logneg_electron_nuclear_cut"] for row in selected],
        linewidth=1.6,
        label=r"$e_1+e_2\,|\,n_{FAD}+n_{Trp}$",
    )
    ax.set_xlabel("time (us)")
    ax.set_ylabel("log negativity")
    ax.set_title(f"{n_nuclei}-nucleus controlled-state entanglement dynamics")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_each_cut_scaling(
    summary_rows: list[dict[str, object]],
    fits: dict[str, object],
    extrapolate_to_nuclei: float,
    path: Path,
    show: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array([row["n_nuclei"] for row in summary_rows], dtype=float)
    series = [
        (
            "radical_cut",
            "mean_logneg_radical_cut",
            r"$e_1+n_{FAD}\,|\,e_2+n_{Trp}$",
            "tab:blue",
        ),
        (
            "electron_nuclear_cut",
            "mean_logneg_electron_nuclear_cut",
            r"$e_1+e_2\,|\,n_{FAD}+n_{Trp}$",
            "tab:orange",
        ),
    ]
    x_grid = np.linspace(float(np.min(x)), float(extrapolate_to_nuclei), 500)
    explicit_max = float(np.max(x))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for fit_key, y_key, label, color in series:
        y = np.array([row[y_key] for row in summary_rows], dtype=float)
        ax.scatter(x, y, color=color, s=55, zorder=3, label=f"{label} explicit")
        fit = fits[fit_key]
        if "a" not in fit:
            continue
        y_grid = fit["a"] * np.power(x_grid, fit["b"])
        observed = x_grid <= explicit_max
        extrapolated = x_grid >= explicit_max
        ax.plot(x_grid[observed], y_grid[observed], color=color, linewidth=2.0)
        ax.plot(
            x_grid[extrapolated],
            y_grid[extrapolated],
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{label} extrapolated",
        )
        ax.scatter(
            [extrapolate_to_nuclei],
            [fit["a"] * extrapolate_to_nuclei ** fit["b"]],
            color=color,
            marker="s",
            s=65,
            zorder=4,
        )

    ax.axvline(explicit_max, color="0.45", linestyle=":", linewidth=1.3, label="explicit-data boundary")
    ax.axhline(
        2.0,
        color="tab:orange",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="electron-nuclear cut maximum",
    )
    ax.set_xlim(float(np.min(x)) - 0.5, float(extrapolate_to_nuclei) + 0.5)
    ax.set_xlabel("number of nuclei")
    ax.set_ylabel("time-averaged log negativity")
    ax.set_title("Controlled-state entanglement scaling by subsystem cut")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def write_outputs(
    summary_rows: list[dict[str, object]],
    time_rows: list[dict[str, float]],
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, dict[str, object]]:
    summary_path = args.output_dir / "entanglement_scaling_summary.csv"
    time_path = args.output_dir / "entanglement_scaling_time_series.csv"
    fits_path = args.output_dir / "entanglement_scaling_fits.json"
    save_summary_csv(summary_rows, summary_path)
    save_time_series_csv(time_rows, time_path)
    fits = fit_each_cut(summary_rows, args.fit_x)
    for fit_key in ("radical_cut", "electron_nuclear_cut"):
        fit = fits[fit_key]
        if "a" in fit:
            prediction = float(fit["a"] * args.extrapolate_to_nuclei ** fit["b"])
            fit[f"power_prediction_at_{args.extrapolate_to_nuclei:g}_nuclei"] = prediction
            if fit_key == "electron_nuclear_cut":
                fit["physical_maximum"] = 2.0
                fit[f"bounded_projection_at_{args.extrapolate_to_nuclei:g}_nuclei"] = min(
                    prediction,
                    2.0,
                )
    fits_path.write_text(json.dumps(fits, indent=2), encoding="utf-8")

    if not args.no_plot:
        plot_each_cut_scaling(
            summary_rows,
            fits,
            args.extrapolate_to_nuclei,
            args.output_dir / f"entanglement_each_cut_power_fit_to_{args.extrapolate_to_nuclei:g}_nuclei.png",
            args.show_plot,
        )

    return summary_path, time_path, fits_path, fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument("--growth-mode", choices=["paired", "global"], default="paired")
    parser.add_argument("--initial-per-radical", type=int, default=2)
    parser.add_argument("--max-nuclei", type=int, default=18, help="Largest nuclear truncation to attempt.")
    parser.add_argument("--max-effective-qubits", type=float, default=22.0, help="Stop before exceeding this log2 Hilbert dimension.")
    parser.add_argument("--stop-after-size-hours", type=float, default=0.0, help="Stop after a completed size if that size took longer than this many hours. Zero disables the guard.")
    parser.add_argument("--theta", type=float, default=90.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--b-microt", type=float, default=50.0)
    parser.add_argument("--t-max-us", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=101)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--J-MHz", type=float, default=3.0)
    parser.add_argument("--D-MHz", type=float, default=-19.10703069)
    parser.add_argument("--physical-dipole", action="store_true")
    parser.add_argument("--no-dipole", action="store_true")
    parser.add_argument("--hyperfine-mode", choices=["full", "diagonal", "z-only"], default="full")
    parser.add_argument("--spin-model", choices=["physical", "spin-half"], default="physical")
    parser.add_argument("--quadrupole-coupling-MHz", type=float, default=0.5)
    parser.add_argument("--quadrupole-eta", type=float, default=0.0)
    parser.add_argument(
        "--nuclear-state",
        choices=["minimum-overhauser", "random-phase", "product-up", "product-x"],
        default="minimum-overhauser",
    )
    parser.add_argument("--operator-mode", choices=["auto", "sparse", "matrix-free"], default="auto")
    parser.add_argument("--max-sparse-dim", type=int, default=65536)
    parser.add_argument("--evolver", choices=["expm", "rk4"], default="expm")
    parser.add_argument("--fit-x", choices=["effective_qubits", "total_spins", "n_nuclei"], default="n_nuclei")
    parser.add_argument("--extrapolate-to-nuclei", type=float, default=26.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "entanglement_scaling")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> Params:
    params = Params()
    params.B_microT = args.b_microt
    params.theta_deg = args.theta
    params.phi_deg = args.phi
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.samples = args.samples
    params.seed = args.seed
    params.J_MHz = args.J_MHz
    params.include_dipole = not args.no_dipole
    params.physical_dipole = args.physical_dipole
    params.D_MHz = args.D_MHz
    params.hyperfine_mode = args.hyperfine_mode
    params.operator_mode = args.operator_mode
    params.max_sparse_dim = args.max_sparse_dim
    params.evolver = args.evolver
    params.spin_model = args.spin_model
    params.quadrupole_coupling_MHz = args.quadrupole_coupling_MHz
    params.quadrupole_eta = args.quadrupole_eta
    params.nuclear_state = args.nuclear_state

    if args.quick:
        params.t_max_us = min(params.t_max_us, 0.1)
        params.n_steps = min(params.n_steps, 5)
        params.samples = 1
    if params.samples < 1:
        raise ValueError("samples must be at least 1.")
    if params.n_steps < 2:
        raise ValueError("n-steps must be at least 2.")
    if not 0.0 <= params.quadrupole_eta <= 1.0:
        raise ValueError("quadrupole eta should be between 0 and 1.")
    return params


def planned_global_truncations(args: argparse.Namespace, sequence: list[StudyNucleus]) -> list[list[StudyNucleus]]:
    start = 2 * args.initial_per_radical
    max_nuclei = min(args.max_nuclei, len(sequence))
    if args.quick:
        max_nuclei = min(max_nuclei, start + 1)
    truncations = []
    for n_nuclei in range(start, max_nuclei + 1):
        nuclei = sequence[:n_nuclei]
        eq = effective_qubits(site_dimensions(nuclei))
        if eq > args.max_effective_qubits:
            print(f"Stopping before {n_nuclei} nuclei: effective qubits {eq:.2f} exceeds limit {args.max_effective_qubits:.2f}.")
            break
        truncations.append(nuclei)
    if not truncations:
        raise ValueError("No truncations selected. Increase --max-effective-qubits or reduce --initial-per-radical.")
    return truncations


def planned_paired_truncations(args: argparse.Namespace) -> list[list[StudyNucleus]]:
    fad_nuclei, trp_nuclei = nuclei_by_radical(args.spin_model)
    max_per_radical = min(len(fad_nuclei), len(trp_nuclei), args.max_nuclei // 2)
    if args.quick:
        max_per_radical = min(max_per_radical, args.initial_per_radical + 1)

    truncations = []
    for per_radical in range(args.initial_per_radical, max_per_radical + 1):
        nuclei = fad_nuclei[:per_radical] + trp_nuclei[:per_radical]
        eq = effective_qubits(site_dimensions(nuclei))
        if eq > args.max_effective_qubits:
            print(
                f"Stopping before {2 * per_radical} nuclei "
                f"({per_radical}+{per_radical}): effective qubits {eq:.2f} "
                f"exceeds limit {args.max_effective_qubits:.2f}."
            )
            break
        truncations.append(nuclei)
    if not truncations:
        raise ValueError("No truncations selected. Increase --max-effective-qubits or reduce --initial-per-radical.")
    return truncations


def planned_truncations(args: argparse.Namespace, sequence: list[StudyNucleus]) -> list[list[StudyNucleus]]:
    if args.growth_mode == "paired":
        return planned_paired_truncations(args)
    return planned_global_truncations(args, sequence)


def main() -> None:
    args = parse_args()
    params = params_from_args(args)
    sequence = growth_sequence(args.spin_model, args.initial_per_radical, args.growth_mode)
    truncations = planned_truncations(args, sequence)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_nucleus_order_csv(sequence, args.output_dir / "nucleus_growth_order.csv")
    run_metadata = {
        "spin_model": args.spin_model,
        "nuclear_state": args.nuclear_state,
        "growth_mode": args.growth_mode,
        "initial_per_radical": args.initial_per_radical,
        "theta_deg": params.theta_deg,
        "phi_deg": params.phi_deg,
        "B_microT": params.B_microT,
        "t_max_us": params.t_max_us,
        "n_steps": params.n_steps,
        "J_MHz": params.J_MHz,
        "dipole": "off" if not params.include_dipole else ("physical" if params.physical_dipole else "optimistic axial"),
        "quadrupole_coupling_MHz": params.quadrupole_coupling_MHz,
        "quadrupole_eta": params.quadrupole_eta,
        "extrapolate_to_nuclei": args.extrapolate_to_nuclei,
        "minimum_overhauser_signs": minimum_overhauser_signs(args.spin_model)
        if args.nuclear_state == "minimum-overhauser"
        else None,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print("Entanglement scaling study")
    print("--------------------------")
    print(f"spin model: {args.spin_model}")
    print(f"nuclear state: {args.nuclear_state}")
    print(f"theta={params.theta_deg:g} deg, t_max={params.t_max_us:g} us, steps={params.n_steps}, samples={params.samples}")
    print(f"J={params.J_MHz:g} MHz, dipole={'off' if not params.include_dipole else ('physical' if params.physical_dipole else 'optimistic axial')}")
    if args.spin_model == "physical":
        print(f"quadrupole: Cq={params.quadrupole_coupling_MHz:g} MHz, eta={params.quadrupole_eta:g}")
    print(f"growth mode: {args.growth_mode}")
    print(f"growth starts with {args.initial_per_radical} FAD + {args.initial_per_radical} Trp nuclei")
    if args.stop_after_size_hours > 0.0:
        print(f"runtime guard: stop after any completed size taking more than {args.stop_after_size_hours:g} hours")
    print()

    summary_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, float]] = []
    for index, nuclei in enumerate(truncations, start=1):
        dims = site_dimensions(nuclei)
        print(
            f"[{index:02d}/{len(truncations):02d}] {len(nuclei)} nuclei, "
            f"dim={int(np.prod(dims, dtype=np.int64))}, effective qubits={effective_qubits(dims):.2f}, "
            f"nuclei={','.join(nucleus.name for nucleus in nuclei)}"
        )
        size_start = time.perf_counter()
        summary, rows = analyze_size(params, nuclei)
        size_hours = (time.perf_counter() - size_start) / 3600.0
        summary["runtime_hours"] = size_hours
        summary_rows.append(summary)
        time_rows.extend(rows)
        print(f"    mean radical-cut log negativity: {summary['mean_logneg_radical_cut']:.6f}")
        print(f"    mean electron-nuclear log negativity: {summary['mean_logneg_electron_nuclear_cut']:.6f}")
        print(f"    size runtime: {size_hours:.4f} hours")
        if not args.no_plot:
            plot_size_dynamics(
                rows,
                len(nuclei),
                args.output_dir / "per_size_dynamics" / f"{len(nuclei):02d}_nuclei_two_cuts.png",
                args.show_plot,
            )
        summary_path, time_path, fits_path, fits = write_outputs(summary_rows, time_rows, args)
        print(f"    checkpointed cumulative outputs through {len(nuclei)} nuclei")
        if args.stop_after_size_hours > 0.0 and size_hours > args.stop_after_size_hours:
            print(f"Stopping after {len(nuclei)} nuclei because that size exceeded the runtime guard.")
            break

    summary_path, time_path, fits_path, fits = write_outputs(summary_rows, time_rows, args)

    print()
    print("Saved outputs")
    print("-------------")
    print(f"summary: {summary_path}")
    print(f"time series: {time_path}")
    print(f"fits: {fits_path}")
    for cut in ("radical_cut", "electron_nuclear_cut"):
        fit = fits[cut]
        prediction_key = f"power_prediction_at_{args.extrapolate_to_nuclei:g}_nuclei"
        if prediction_key in fit:
            print(f"{cut} power prediction at {args.extrapolate_to_nuclei:g} nuclei: {fit[prediction_key]:.6f}")
        bounded_key = f"bounded_projection_at_{args.extrapolate_to_nuclei:g}_nuclei"
        if bounded_key in fit:
            print(f"{cut} bounded projection at {args.extrapolate_to_nuclei:g} nuclei: {fit[bounded_key]:.6f}")


if __name__ == "__main__":
    main()
