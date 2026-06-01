"""Scalable FAD-Trp radical-pair magnetosensitivity model.

This is a standalone implementation for moving beyond the 8-qubit baseline.
It uses state vectors and a matrix-free sparse Hamiltonian action, so the model
can be pushed toward larger FAD-Trp truncations without constructing dense
density matrices.

Default model
-------------
The default production baseline uses a 16-nucleus FAD-Trp truncation with full
anisotropic hyperfine tensors, J = 3 MHz, optimistic axial electron-electron
dipole, a 2 us time window, and the coarse theta grid 0, 45, 90, 135, 180 deg.
The published complete list contains 27 magnetic nuclei, which would be 29
spin-1/2 qubits including the two electrons. Exact density-matrix evolution is
not viable there; this script approximates the initially mixed nuclear bath by
averaging over random-phase pure nuclear states when multiple samples are used.

Parameter provenance
--------------------
Hyperfine constants, tensor axes, FAD/Trp rotations, and Cry4a FAD-TrpC geometry
are transcribed from the public ActionSpectrumHistograms parameter source for:

    Leberecht et al., PNAS 2023, doi:10.1073/pnas.2301153120

The Hamiltonian includes electron Zeeman, anisotropic hyperfine, exchange, and
full electron-electron dipolar interactions. The physical dipolar tensor is
Table S3 from the supplementary information. Exchange defaults to 3 MHz for the
current baseline runs, but can be changed with ``--J-MHz``.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, expm_multiply

TWOPI = 2.0 * np.pi
SQRT2 = np.sqrt(2.0)
GAMMA_E_MHZ_PER_MT = 28.02495164


# Literature ordering by approximate hyperfine strength.
PUBLISHED_NUCLEUS_ORDER = [
    "FN5",
    "WCNE1",
    "WCHE1",
    "FN10",
    "WCHB1",
    "WCHE3",
    "FH82",
    "FH83",
    "FH81",
    "FH1'",
    "FH1''",
    "FH6",
    "WCHZ2",
    "WCHD1",
    "WCHH2",
    "WCN",
    "FH73",
    "FH72",
    "FH71",
    "WCHA",
    "FH9",
    "WCHB2",
    "WCHZ3",
    "FN3",
    "WCHNm",
    "FH3",
    "FN1",
]


GAUSSIAN_FAD_ROTATION = np.array(
    [
        [0.8603464852717276, -0.5097096480159905, 0.0],
        [0.5097096480159905, 0.8603464852717276, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

GAUSSIAN_TRP_ROTATION = np.array(
    [
        [-0.6285147275607671, 0.7655969456514179, 0.13722446592512128],
        [0.7349868475114238, 0.642329156028544, -0.2172730754163464],
        [-0.2544868782951958, -0.035701150189851295, -0.9664170200543283],
    ],
    dtype=float,
)

CLCRY_FAD_TO_CRYSTAL_ROTATION = np.array(
    [
        [-0.16589802507880655, 0.9440869552950132, -0.284916945997853],
        [0.08311958567422897, -0.2745056082497105, -0.9579863284602765],
        [-0.9826336956121015, -0.18261021844696496, -0.032932178282882314],
    ],
    dtype=float,
)

CLCRY_WC_TO_CRYSTAL_ROTATION = np.array(
    [
        [0.7332285393631011, 0.012948875903588501, 0.6798589821987255],
        [0.008596141712293476, 0.9995622569522814, -0.028309023722545124],
        [-0.679927948690974, 0.02660114827023299, 0.7327962632953238],
    ],
    dtype=float,
)

CLCRY_WC_DIPOLE_VECTOR_ANGSTROM = np.array(
    [-9.561276096275549, -13.691239246983221, 5.540701853737576],
    dtype=float,
)


ISO_MHZ = {
    "FN5": 14.66545,
    "FN10": 5.28776,
    "FN1": -0.09682,
    "FN3": -1.07389,
    "FH3": -0.52839,
    "FH1'": 11.40703,
    "FH1''": 11.40703,
    "FH9": 1.58379,
    "FH81": 12.32934,
    "FH82": 12.32934,
    "FH83": 12.32934,
    "FH71": -3.96765,
    "FH72": -3.96765,
    "FH73": -3.96765,
    "FH6": -10.85125,
    "WNE1": 9.01091,
    "WHE1": -16.76857,
    "WHD1": -7.79138,
    "WHZ2": -10.19254,
    "WHH2": -5.83710,
    "WHZ3": -1.12131,
    "WHE3": -13.67631,
    "WHB1": 44.96815,
    "WHB2": 1.28070,
    "WHA": -2.60835,
    "WHNa": -0.29119,
    "WHNb": 0.65336,
    "WN": 4.10515,
}
ISO_MHZ["WHNm"] = 0.5 * (ISO_MHZ["WHNa"] + ISO_MHZ["WHNb"])


ANIS_MHZ = {
    "FN5": np.array([-17.472, -17.099, 34.571], dtype=float),
    "FN10": np.array([-5.965, -5.691, 11.656], dtype=float),
    "FN1": np.array([-0.667, -0.495, 1.162], dtype=float),
    "FN3": np.array([-0.135, -0.013, 0.149], dtype=float),
    "FH3": np.array([-1.280, -0.317, 1.596], dtype=float),
    "FH1'": np.array([0.0, 0.0, 0.0], dtype=float),
    "FH1''": np.array([0.0, 0.0, 0.0], dtype=float),
    "FH9": np.array([-1.723, -0.045, 1.768], dtype=float),
    "FH81": np.array([0.0, 0.0, 0.0], dtype=float),
    "FH82": np.array([0.0, 0.0, 0.0], dtype=float),
    "FH83": np.array([0.0, 0.0, 0.0], dtype=float),
    "FH71": np.array([0.0, 0.0, 0.0], dtype=float),
    "FH72": np.array([0.0, 0.0, 0.0], dtype=float),
    "FH73": np.array([0.0, 0.0, 0.0], dtype=float),
    "FH6": np.array([-4.014, -1.300, 5.314], dtype=float),
    "WNE1": np.array([-10.793, -10.496, 21.288], dtype=float),
    "WHE1": np.array([-13.571, -3.001, 16.573], dtype=float),
    "WHD1": np.array([-5.426, -2.575, 8.001], dtype=float),
    "WHZ2": np.array([-5.452, -1.665, 7.117], dtype=float),
    "WHH2": np.array([-4.161, -1.385, 5.546], dtype=float),
    "WHZ3": np.array([-1.771, -1.727, 3.498], dtype=float),
    "WHE3": np.array([-7.064, -1.346, 8.410], dtype=float),
    "WHB1": np.array([-2.986, -1.277, 4.262], dtype=float),
    "WHB2": np.array([-2.124, -1.270, 3.393], dtype=float),
    "WHA": np.array([-3.060, 1.105, 1.955], dtype=float),
    "WHNa": np.array([-2.197, -0.709, 2.906], dtype=float),
    "WHNb": np.array([-1.733, -0.392, 2.125], dtype=float),
    "WN": np.array([-0.627, -0.581, 1.208], dtype=float),
}
ANIS_MHZ["WHNm"] = 0.5 * (ANIS_MHZ["WHNa"] + ANIS_MHZ["WHNb"])


AXES = {
    "FN5": np.array([[0.9518, -0.3068, 0.0], [0.3068, 0.9518, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    "FN10": np.array([[0.6845, 0.7291, 0.0], [0.7291, -0.6845, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    "FN1": np.array([[-0.467, 0.8843, 0.0], [0.8843, 0.467, 0.0], [0.0, 0.0, 1.0]], dtype=float),
    "FN3": np.array([[0.8526, -0.5226, 0.0], [0.0, 0.0, 1.0], [0.5226, 0.8526, 0.0]], dtype=float),
    "FH3": np.array([[0.0, 0.0, 1.0], [0.879, -0.4769, 0.0], [0.4769, 0.879, 0.0]], dtype=float),
    "FH1'": np.array([[-0.3016, 0.0392, 0.9526], [0.5871, 0.7949, 0.1532], [0.7512, -0.6055, 0.2627]], dtype=float),
    "FH1''": np.array([[0.3016, -0.0392, 0.9526], [0.5871, 0.7949, -0.1532], [0.7512, -0.6055, -0.2627]], dtype=float),
    "FH9": np.array([[0.0, 0.0, 1.0], [0.9961, -0.0882, 0.0], [0.0882, 0.9961, 0.0]], dtype=float),
    "FH6": np.array([[0.422, 0.9066, 0.0], [0.0, 0.0, 1.0], [0.9066, -0.422, 0.0]], dtype=float),
    "WNE1": np.array([[0.3221, 0.9353, -0.1463], [0.9173, -0.3466, -0.1963], [0.2343, 0.071, 0.9696]], dtype=float),
    "WHE1": np.array([[0.754, 0.6139, -0.2336], [0.2344, 0.0808, 0.9688], [-0.6136, 0.7853, 0.083]], dtype=float),
    "WHD1": np.array([[0.919, -0.3028, -0.2525], [0.2782, 0.0442, 0.9595], [0.2793, 0.952, -0.1249]], dtype=float),
    "WHZ2": np.array([[0.5779, 0.7959, -0.1803], [0.2498, 0.0378, 0.9676], [0.7769, -0.6042, -0.1769]], dtype=float),
    "WHH2": np.array([[-0.629, 0.7659, 0.1335], [0.2463, 0.0335, 0.9686], [0.7373, 0.6421, -0.2097]], dtype=float),
    "WHZ3": np.array([[-0.552, 0.4129, 0.7245], [0.6965, -0.2494, 0.6728], [0.4585, 0.8759, -0.1499]], dtype=float),
    "WHE3": np.array([[0.5359, 0.8279, -0.1653], [0.2657, 0.0205, 0.9638], [0.8014, -0.5604, -0.209]], dtype=float),
    "WHB1": np.array([[0.2968, -0.3935, 0.8701], [0.818, 0.5749, -0.019], [-0.4927, 0.7174, 0.4925]], dtype=float),
    "WHB2": np.array([[0.3333, 0.0783, 0.9396], [-0.243, 0.97, 0.0053], [0.911, 0.2301, -0.3423]], dtype=float),
    "WHA": np.array([[-0.0934, 0.5118, 0.854], [0.9368, 0.3356, -0.0987], [-0.3371, 0.7908, -0.5108]], dtype=float),
    "WHNa": np.array([[0.2189, 0.9709, -0.0971], [-0.6701, 0.2219, 0.7083], [0.7093, -0.0899, 0.6992]], dtype=float),
    "WHNb": np.array([[-0.4432, 0.1417, 0.8852], [-0.2267, 0.9376, -0.2636], [0.8673, 0.3175, 0.3834]], dtype=float),
    "WN": np.array([[-0.4082, 0.6826, 0.6062], [0.6009, 0.7008, -0.3844], [0.6872, -0.2073, 0.6962]], dtype=float),
}
for methyl_index in ("1", "2", "3"):
    AXES[f"FH8{methyl_index}"] = np.identity(3)
    AXES[f"FH7{methyl_index}"] = np.identity(3)
AXES["WHNm"] = np.identity(3)


@dataclass(frozen=True)
class Nucleus:
    name: str
    radical: str
    tensor_MHz: np.ndarray


@dataclass
class Params:
    B_microT: float = 50.0
    gamma_e_MHz_per_mT: float = GAMMA_E_MHZ_PER_MT
    theta_values_deg: np.ndarray = field(default_factory=lambda: np.arange(0.0, 181.0, 45.0))
    phi_deg: float = 0.0

    max_nuclei: int = 16
    n_samples: int = 1
    seed: int = 20260528
    model: str = "quantum"

    t_max_us: float = 2.0
    n_steps: int = 101
    recombination_rate_per_us: float = 0.5

    J_MHz: float = 3.0
    include_dipole: bool = True
    use_published_dipolar_tensor: bool = True
    D_MHz_if_isotropic: float = -19.10703069
    optimize_dipole_axis: bool = True
    hyperfine_mode: str = "full"
    operator_mode: str = "auto"
    max_sparse_qubits: int = 18
    evolver: str = "expm"

    output_dir: Path = Path("outputs") / "full_fad_trp_magnetosensitivity"
    save_plot: bool = True
    show_plot: bool = False


def mhz_to_rad_per_us(value):
    return TWOPI * np.asarray(value, dtype=float)


def unit_vector_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        dtype=float,
    )


def tensor_from_gaussian_axis_system(atom_key: str) -> np.ndarray:
    axes = AXES[atom_key]
    diagonal = ISO_MHZ[atom_key] + ANIS_MHZ[atom_key]
    return axes.T @ np.diag(diagonal) @ axes


def fad_tensor_in_fad_frame(atom_key: str) -> np.ndarray:
    tensor_gaussian = tensor_from_gaussian_axis_system(atom_key)
    return GAUSSIAN_FAD_ROTATION @ tensor_gaussian @ GAUSSIAN_FAD_ROTATION.T


def trp_tensor_in_trp_frame(atom_key: str) -> np.ndarray:
    tensor_gaussian = tensor_from_gaussian_axis_system(atom_key)
    return GAUSSIAN_TRP_ROTATION @ tensor_gaussian @ GAUSSIAN_TRP_ROTATION.T


def rotate_trp_to_fad_frame(tensor_trp_frame: np.ndarray) -> np.ndarray:
    tensor_crystal = np.linalg.inv(CLCRY_WC_TO_CRYSTAL_ROTATION) @ tensor_trp_frame @ CLCRY_WC_TO_CRYSTAL_ROTATION
    return CLCRY_FAD_TO_CRYSTAL_ROTATION @ tensor_crystal @ np.linalg.inv(CLCRY_FAD_TO_CRYSTAL_ROTATION)


def canonical_atom_key(published_name: str) -> tuple[str, str]:
    if published_name.startswith("F"):
        return published_name, "FAD"
    if published_name.startswith("WC"):
        return "W" + published_name[2:], "TRP"
    raise ValueError(f"Unrecognized nucleus name: {published_name}")


def literature_tensor_MHz(published_name: str) -> np.ndarray:
    atom_key, radical = canonical_atom_key(published_name)
    if radical == "FAD":
        return fad_tensor_in_fad_frame(atom_key)
    if atom_key == "WHNm":
        tensors = [rotate_trp_to_fad_frame(trp_tensor_in_trp_frame(key)) for key in ("WHNa", "WHNb")]
        return 0.5 * (tensors[0] + tensors[1])
    return rotate_trp_to_fad_frame(trp_tensor_in_trp_frame(atom_key))


def selected_nuclei(max_nuclei: int) -> list[Nucleus]:
    if max_nuclei < 0 or max_nuclei > len(PUBLISHED_NUCLEUS_ORDER):
        raise ValueError(f"max_nuclei must be between 0 and {len(PUBLISHED_NUCLEUS_ORDER)}.")
    nuclei = []
    for name in PUBLISHED_NUCLEUS_ORDER[:max_nuclei]:
        _atom_key, radical = canonical_atom_key(name)
        nuclei.append(Nucleus(name=name, radical=radical, tensor_MHz=literature_tensor_MHz(name)))
    return nuclei


def spin_count(max_nuclei: int) -> int:
    return 2 + max_nuclei


def estimate_state_memory_mb(max_nuclei: int) -> float:
    return (2 ** spin_count(max_nuclei)) * np.dtype(np.complex128).itemsize / 1024**2


def pauli_on_axis(state: np.ndarray, axis: str, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    out = np.empty_like(moved)

    if axis == "x":
        out[0] = moved[1]
        out[1] = moved[0]
    elif axis == "y":
        out[0] = -1.0j * moved[1]
        out[1] = 1.0j * moved[0]
    elif axis == "z":
        out[0] = moved[0]
        out[1] = -moved[1]
    else:
        raise ValueError(f"Unknown Pauli axis: {axis}")

    restored = np.moveaxis(out.reshape((2,) + (2,) * (n_qubits - 1)), 0, qubit)
    return restored.reshape(-1)


def apply_pauli_product(state: np.ndarray, operations: tuple[tuple[str, int], ...], n_qubits: int) -> np.ndarray:
    out = state
    for axis, qubit in operations:
        out = pauli_on_axis(out, axis, qubit, n_qubits)
    return out


def add_single_spin_terms(terms: list[tuple[complex, tuple[tuple[str, int], ...]]], qubit: int, vector: np.ndarray) -> None:
    for axis, coefficient in zip(("x", "y", "z"), vector):
        if abs(coefficient) > 1e-15:
            terms.append((0.5 * coefficient, ((axis, qubit),)))


def add_two_spin_tensor_terms(
    terms: list[tuple[complex, tuple[tuple[str, int], ...]]],
    q1: int,
    q2: int,
    tensor: np.ndarray,
) -> None:
    axes = ("x", "y", "z")
    for a, axis_a in enumerate(axes):
        for b, axis_b in enumerate(axes):
            coefficient = tensor[a, b]
            if abs(coefficient) > 1e-15:
                terms.append((0.25 * coefficient, ((axis_a, q1), (axis_b, q2))))


def filtered_hyperfine_tensor(tensor: np.ndarray, mode: str) -> np.ndarray:
    if mode == "full":
        return tensor
    if mode == "diagonal":
        return np.diag(np.diag(tensor))
    if mode == "z-only":
        out = np.zeros_like(tensor)
        out[2, 2] = tensor[2, 2]
        return out
    raise ValueError("hyperfine_mode must be one of: full, diagonal, z-only.")


def published_dipolar_tensor_MHz() -> np.ndarray:
    """Return Table S3 FAD-TrpCH dipolar tensor in the flavin principal-axis frame."""
    return np.array(
        [
            [1.09, -12.12, 4.91],
            [-12.12, -7.80, 7.02],
            [4.91, 7.02, 6.71],
        ],
        dtype=float,
    )


def best_case_dipolar_tensor_MHz(D_MHz: float) -> np.ndarray:
    """Optimistic axial tensor with its unique principal axis aligned to FAD-frame z."""
    return np.diag([-0.5 * D_MHz, -0.5 * D_MHz, D_MHz])


def build_terms(params: Params, nuclei: list[Nucleus], theta_deg: float) -> list[tuple[complex, tuple[tuple[str, int], ...]]]:
    n_qubits = 2 + len(nuclei)
    terms: list[tuple[complex, tuple[tuple[str, int], ...]]] = []

    B_mT = 1e-3 * params.B_microT * unit_vector_from_angles(theta_deg, params.phi_deg)
    electron_zeeman = mhz_to_rad_per_us(params.gamma_e_MHz_per_mT) * B_mT
    add_single_spin_terms(terms, 0, electron_zeeman)
    add_single_spin_terms(terms, 1, electron_zeeman)

    for nucleus_index, nucleus in enumerate(nuclei):
        electron_qubit = 0 if nucleus.radical == "FAD" else 1
        nuclear_qubit = 2 + nucleus_index
        tensor_MHz = filtered_hyperfine_tensor(nucleus.tensor_MHz, params.hyperfine_mode)
        add_two_spin_tensor_terms(terms, electron_qubit, nuclear_qubit, mhz_to_rad_per_us(tensor_MHz))

    if abs(params.J_MHz) > 1e-15:
        exchange_tensor = mhz_to_rad_per_us(-2.0 * params.J_MHz) * np.identity(3)
        add_two_spin_tensor_terms(terms, 0, 1, exchange_tensor)

    if params.include_dipole:
        if params.use_published_dipolar_tensor and not params.optimize_dipole_axis:
            dipolar_tensor_MHz = published_dipolar_tensor_MHz()
        else:
            dipolar_tensor_MHz = best_case_dipolar_tensor_MHz(params.D_MHz_if_isotropic)
        add_two_spin_tensor_terms(terms, 0, 1, mhz_to_rad_per_us(dipolar_tensor_MHz))

    # Keep qubit count captured for validation while keeping terms compact.
    if n_qubits != spin_count(len(nuclei)):
        raise AssertionError("Internal qubit count mismatch.")
    return terms


def make_linear_operator(
    terms: list[tuple[complex, tuple[tuple[str, int], ...]]],
    n_qubits: int,
) -> LinearOperator:
    dimension = 2**n_qubits
    basis_indices = np.arange(dimension, dtype=np.uint64)
    compiled_terms = []
    for coefficient, operations in terms:
        flip_mask = np.uint64(0)
        phase_ops = []
        for axis, qubit in operations:
            bit_mask = np.uint64(1 << (n_qubits - 1 - qubit))
            if axis in ("x", "y"):
                flip_mask ^= bit_mask
            if axis in ("y", "z"):
                phase_ops.append((axis, bit_mask))
        compiled_terms.append((coefficient, flip_mask, tuple(phase_ops)))

    def phase_for_ops(phase_ops: tuple[tuple[str, np.uint64], ...]) -> np.ndarray:
        phase = np.ones(dimension, dtype=np.complex128)
        for axis, bit_mask in phase_ops:
            bit_is_one = (basis_indices & bit_mask) != 0
            if axis == "z":
                phase[bit_is_one] *= -1.0
            elif axis == "y":
                phase[bit_is_one] *= 1.0j
                phase[~bit_is_one] *= -1.0j
            else:
                raise ValueError(f"Unsupported phase axis: {axis}")
        return phase

    enable_phase_cache = dimension <= 2**18
    phase_cache: dict[tuple[tuple[str, np.uint64], ...], np.ndarray] = {}

    def matvec(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=complex).reshape(-1)
        out = np.zeros_like(vector)
        for coefficient, flip_mask, phase_ops in compiled_terms:
            src = basis_indices ^ flip_mask
            if phase_ops:
                phase = phase_cache.get(phase_ops) if enable_phase_cache else None
                if phase is None:
                    phase = phase_for_ops(phase_ops)
                    if enable_phase_cache:
                        phase_cache[phase_ops] = phase
                out += coefficient * phase * vector[src]
            else:
                out += coefficient * vector[src]
        return out

    def matmat(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=complex)
        return np.column_stack([matvec(matrix[:, col]) for col in range(matrix.shape[1])])

    return LinearOperator((dimension, dimension), matvec=matvec, rmatvec=matvec, matmat=matmat, dtype=np.complex128)


def sparse_pauli_product(
    operations: tuple[tuple[str, int], ...],
    n_qubits: int,
) -> sp.csr_matrix:
    pauli = {
        "i": sp.identity(2, dtype=complex, format="csr"),
        "x": sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)),
        "y": sp.csr_matrix(np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)),
        "z": sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)),
    }
    by_qubit = {qubit: axis for axis, qubit in operations}
    factors = [pauli[by_qubit.get(qubit, "i")] for qubit in range(n_qubits)]
    result = factors[0]
    for factor in factors[1:]:
        result = sp.kron(result, factor, format="csr")
    return result


def make_sparse_operator(
    terms: list[tuple[complex, tuple[tuple[str, int], ...]]],
    n_qubits: int,
) -> sp.csr_matrix:
    dimension = 2**n_qubits
    hamiltonian = sp.csr_matrix((dimension, dimension), dtype=complex)
    for coefficient, operations in terms:
        hamiltonian += coefficient * sparse_pauli_product(operations, n_qubits)
    return hamiltonian.tocsr()


def make_hamiltonian_operator(
    terms: list[tuple[complex, tuple[tuple[str, int], ...]]],
    n_qubits: int,
    params: Params,
):
    if params.operator_mode == "sparse":
        return make_sparse_operator(terms, n_qubits)
    if params.operator_mode == "matrix-free":
        return make_linear_operator(terms, n_qubits)
    if params.operator_mode != "auto":
        raise ValueError("operator_mode must be one of: auto, sparse, matrix-free.")
    if n_qubits <= params.max_sparse_qubits:
        return make_sparse_operator(terms, n_qubits)
    return make_linear_operator(terms, n_qubits)


def random_phase_nuclear_state(n_nuclei: int, rng: np.random.Generator) -> np.ndarray:
    dimension = 2**n_nuclei
    phases = rng.uniform(0.0, TWOPI, size=dimension)
    return np.exp(1.0j * phases) / math.sqrt(dimension)


def singlet_electron_state() -> np.ndarray:
    state = np.zeros(4, dtype=complex)
    state[1] = 1.0 / SQRT2
    state[2] = -1.0 / SQRT2
    return state


def initial_state(n_nuclei: int, rng: np.random.Generator) -> np.ndarray:
    return np.kron(singlet_electron_state(), random_phase_nuclear_state(n_nuclei, rng))


def singlet_population(state: np.ndarray, n_nuclei: int) -> float:
    nuclear_dimension = 2**n_nuclei
    reshaped = state.reshape(4, nuclear_dimension)
    singlet_amplitudes = (reshaped[1] - reshaped[2]) / SQRT2
    return float(np.vdot(singlet_amplitudes, singlet_amplitudes).real)


def singlet_yield(t_us: np.ndarray, populations: np.ndarray, k_per_us: float) -> float:
    weights = k_per_us * np.exp(-k_per_us * t_us)
    return float(np.trapezoid(weights * populations, t_us))


def evolve_one_sample(
    params: Params,
    nuclei: list[Nucleus],
    theta_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    n_nuclei = len(nuclei)
    n_qubits = 2 + n_nuclei
    terms = build_terms(params, nuclei, theta_deg)
    hamiltonian = make_hamiltonian_operator(terms, n_qubits, params)
    t_us = np.linspace(0.0, params.t_max_us, params.n_steps)
    psi0 = initial_state(n_nuclei, rng)

    if params.evolver == "expm":
        evolved = expm_multiply(
            -1.0j * hamiltonian,
            psi0,
            start=0.0,
            stop=params.t_max_us,
            num=params.n_steps,
            endpoint=True,
            traceA=0.0,
        )
    elif params.evolver == "rk4":
        evolved = evolve_rk4(hamiltonian, psi0, t_us)
    else:
        raise ValueError("evolver must be one of: expm, rk4.")
    populations = np.array([singlet_population(state, n_nuclei) for state in evolved], dtype=float)
    return populations, singlet_yield(t_us, populations, params.recombination_rate_per_us)


def hamiltonian_matvec(hamiltonian, state: np.ndarray) -> np.ndarray:
    if hasattr(hamiltonian, "matvec"):
        return hamiltonian.matvec(state)
    return hamiltonian @ state


def evolve_rk4(hamiltonian, psi0: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    """Fixed-step RK4 evolution for exploratory large LinearOperator runs."""
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


def run_orientation(params: Params, nuclei: list[Nucleus], theta_deg: float, rng: np.random.Generator) -> dict[str, object]:
    if params.model == "overhauser":
        return run_overhauser_orientation(params, nuclei, theta_deg, rng)
    if params.model != "quantum":
        raise ValueError("model must be one of: quantum, overhauser.")

    sample_populations = []
    sample_yields = []
    for _sample_index in range(params.n_samples):
        populations, phi_s = evolve_one_sample(params, nuclei, theta_deg, rng)
        sample_populations.append(populations)
        sample_yields.append(phi_s)

    mean_populations = np.mean(sample_populations, axis=0)
    t_us = np.linspace(0.0, params.t_max_us, params.n_steps)
    return {
        "theta_deg": float(theta_deg),
        "phi_deg": float(params.phi_deg),
        "singlet_yield": float(np.mean(sample_yields)),
        "singlet_yield_std": float(np.std(sample_yields, ddof=1)) if len(sample_yields) > 1 else 0.0,
        "singlet_population_initial": float(mean_populations[0]),
        "singlet_population_final": float(mean_populations[-1]),
        "singlet_population_min": float(np.min(mean_populations)),
        "singlet_population_max": float(np.max(mean_populations)),
        "t_us": t_us,
        "singlet_population": mean_populations,
    }


def electron_spin_operators() -> dict[str, np.ndarray]:
    ident = np.eye(2, dtype=complex)
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return {
        "S1x": np.kron(sx, ident),
        "S1y": np.kron(sy, ident),
        "S1z": np.kron(sz, ident),
        "S2x": np.kron(ident, sx),
        "S2y": np.kron(ident, sy),
        "S2z": np.kron(ident, sz),
    }


def overhauser_fields_MHz(params: Params, nuclei: list[Nucleus], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample classical Overhauser fields in MHz for the two electrons."""
    field_1 = np.zeros(3, dtype=float)
    field_2 = np.zeros(3, dtype=float)
    for nucleus in nuclei:
        tensor = filtered_hyperfine_tensor(nucleus.tensor_MHz, params.hyperfine_mode)
        nuclear_projection = rng.choice([-0.5, 0.5])
        contribution = tensor @ np.array([0.0, 0.0, nuclear_projection])
        if nucleus.radical == "FAD":
            field_1 += contribution
        else:
            field_2 += contribution
    return field_1, field_2


def electronic_singlet_state() -> np.ndarray:
    state = np.zeros(4, dtype=complex)
    state[1] = 1.0 / SQRT2
    state[2] = -1.0 / SQRT2
    return state


def build_overhauser_hamiltonian(
    params: Params,
    theta_deg: float,
    field_1_MHz: np.ndarray,
    field_2_MHz: np.ndarray,
) -> np.ndarray:
    ops = electron_spin_operators()
    hamiltonian = np.zeros((4, 4), dtype=complex)
    B_mT = 1e-3 * params.B_microT * unit_vector_from_angles(theta_deg, params.phi_deg)
    zeeman_MHz = params.gamma_e_MHz_per_mT * B_mT

    total_1 = mhz_to_rad_per_us(zeeman_MHz + field_1_MHz)
    total_2 = mhz_to_rad_per_us(zeeman_MHz + field_2_MHz)
    for axis_index, axis in enumerate(("x", "y", "z")):
        hamiltonian += total_1[axis_index] * ops[f"S1{axis}"]
        hamiltonian += total_2[axis_index] * ops[f"S2{axis}"]

    if abs(params.J_MHz) > 1e-15:
        exchange_coeff = mhz_to_rad_per_us(-2.0 * params.J_MHz)
        for axis in ("x", "y", "z"):
            hamiltonian += exchange_coeff * (ops[f"S1{axis}"] @ ops[f"S2{axis}"])

    if params.include_dipole:
        if params.use_published_dipolar_tensor and not params.optimize_dipole_axis:
            dipolar_tensor_MHz = published_dipolar_tensor_MHz()
        else:
            dipolar_tensor_MHz = best_case_dipolar_tensor_MHz(params.D_MHz_if_isotropic)
        dipolar_tensor = mhz_to_rad_per_us(dipolar_tensor_MHz)
        for a, axis_a in enumerate(("x", "y", "z")):
            for b, axis_b in enumerate(("x", "y", "z")):
                hamiltonian += dipolar_tensor[a, b] * (ops[f"S1{axis_a}"] @ ops[f"S2{axis_b}"])

    return hamiltonian


def electron_singlet_population(state: np.ndarray) -> float:
    singlet = electronic_singlet_state()
    amp = np.vdot(singlet, state)
    return float((amp.conjugate() * amp).real)


def evolve_electron_hamiltonian(hamiltonian: np.ndarray, t_us: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(hamiltonian)
    psi0 = electronic_singlet_state()
    coeffs = evecs.conj().T @ psi0
    populations = np.empty(len(t_us), dtype=float)
    for index, time in enumerate(t_us):
        psi_t = evecs @ (np.exp(-1.0j * evals * time) * coeffs)
        populations[index] = electron_singlet_population(psi_t)
    return populations


def run_overhauser_orientation(
    params: Params,
    nuclei: list[Nucleus],
    theta_deg: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    t_us = np.linspace(0.0, params.t_max_us, params.n_steps)
    sample_populations = []
    sample_yields = []
    for _sample_index in range(params.n_samples):
        field_1, field_2 = overhauser_fields_MHz(params, nuclei, rng)
        hamiltonian = build_overhauser_hamiltonian(params, theta_deg, field_1, field_2)
        populations = evolve_electron_hamiltonian(hamiltonian, t_us)
        sample_populations.append(populations)
        sample_yields.append(singlet_yield(t_us, populations, params.recombination_rate_per_us))

    mean_populations = np.mean(sample_populations, axis=0)
    return {
        "theta_deg": float(theta_deg),
        "phi_deg": float(params.phi_deg),
        "singlet_yield": float(np.mean(sample_yields)),
        "singlet_yield_std": float(np.std(sample_yields, ddof=1)) if len(sample_yields) > 1 else 0.0,
        "singlet_population_initial": float(mean_populations[0]),
        "singlet_population_final": float(mean_populations[-1]),
        "singlet_population_min": float(np.min(mean_populations)),
        "singlet_population_max": float(np.max(mean_populations)),
        "t_us": t_us,
        "singlet_population": mean_populations,
    }


def run_sweep(params: Params) -> tuple[list[dict[str, object]], list[Nucleus]]:
    nuclei = selected_nuclei(params.max_nuclei)
    results = []
    total = len(params.theta_values_deg)
    print_model_header(params, nuclei)
    for index, theta_deg in enumerate(params.theta_values_deg, start=1):
        print(f"[{index:02d}/{total:02d}] theta={theta_deg:g} deg, samples={params.n_samples}")
        # Reuse the same random-phase/Overhauser sample ensemble across theta
        # values so angular contrast is not inflated by sample-to-sample noise.
        rng = np.random.default_rng(params.seed)
        results.append(run_orientation(params, nuclei, float(theta_deg), rng))
    return results, nuclei


def summarize_results(results: list[dict[str, object]]) -> dict[str, float]:
    yields = np.array([row["singlet_yield"] for row in results], dtype=float)
    theta = np.array([row["theta_deg"] for row in results], dtype=float)
    min_index = int(np.argmin(yields))
    max_index = int(np.argmax(yields))
    mean_yield = float(np.mean(yields))
    contrast = float(yields[max_index] - yields[min_index])
    return {
        "min_yield": float(yields[min_index]),
        "theta_min_yield_deg": float(theta[min_index]),
        "max_yield": float(yields[max_index]),
        "theta_max_yield_deg": float(theta[max_index]),
        "absolute_contrast": contrast,
        "relative_contrast": float(contrast / mean_yield) if mean_yield != 0.0 else float("nan"),
        "mean_yield": mean_yield,
    }


def save_results_csv(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "theta_deg",
        "phi_deg",
        "singlet_yield",
        "singlet_yield_std",
        "singlet_population_initial",
        "singlet_population_final",
        "singlet_population_min",
        "singlet_population_max",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({column: row[column] for column in columns})


def save_nuclei_csv(nuclei: list[Nucleus], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "name", "radical", "Axx_MHz", "Axy_MHz", "Axz_MHz", "Ayx_MHz", "Ayy_MHz", "Ayz_MHz", "Azx_MHz", "Azy_MHz", "Azz_MHz"])
        for index, nucleus in enumerate(nuclei, start=1):
            writer.writerow([index, nucleus.name, nucleus.radical, *nucleus.tensor_MHz.reshape(-1)])


def plot_results(results: list[dict[str, object]], path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.array([row["theta_deg"] for row in results], dtype=float)
    yields = np.array([row["singlet_yield"] for row in results], dtype=float)
    yield_std = np.array([row["singlet_yield_std"] for row in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].errorbar(theta, yields, yerr=yield_std, marker="o", linewidth=2.0, capsize=3)
    axes[0].set_xlabel("magnetic-field polar angle theta (deg)")
    axes[0].set_ylabel("singlet yield Phi_S")
    axes[0].set_title("FAD-Trp magnetosensitivity")

    selected_indices = sorted(set([0, len(results) // 2, len(results) - 1]))
    for idx in selected_indices:
        row = results[idx]
        axes[1].plot(row["t_us"], row["singlet_population"], linewidth=1.6, label=f"theta={row['theta_deg']:g} deg")
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("singlet population P_S(t)")
    axes[1].set_title("Representative sampled dynamics")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def print_model_header(params: Params, nuclei: list[Nucleus]) -> None:
    n_qubits = 2 + len(nuclei)
    print()
    print("FAD-Trp magnetosensitivity model")
    print("--------------------------------")
    print(f"nuclei: {len(nuclei)} ({n_qubits} qubits total)")
    print(f"model: {params.model}")
    print(f"state-vector size: {estimate_state_memory_mb(len(nuclei)):.2f} MB per vector")
    print(f"nuclear samples: {params.n_samples}")
    print(f"hyperfine mode: {params.hyperfine_mode}")
    if params.operator_mode == "auto":
        active_mode = "sparse" if n_qubits <= params.max_sparse_qubits else "matrix-free"
    else:
        active_mode = params.operator_mode
    print(f"operator mode: {active_mode}")
    print(f"evolver: {params.evolver}")
    print(f"B field: {params.B_microT:g} uT, phi={params.phi_deg:g} deg")
    print(f"J: {params.J_MHz:g} MHz")
    if not params.include_dipole:
        print("dipolar tensor: disabled")
    elif params.use_published_dipolar_tensor and not params.optimize_dipole_axis:
        print("dipolar tensor: Table S3 Cry4a FAD-TrpCH geometry")
    else:
        print(f"dipolar tensor: optimistic axial z orientation, Dzz={params.D_MHz_if_isotropic:g} MHz")
    print(f"nucleus order: {', '.join(n.name for n in nuclei)}")
    print()


def print_summary(params: Params, summary: dict[str, float], csv_path: Path, nuclei_path: Path, plot_path: Path | None) -> None:
    tail = np.exp(-params.recombination_rate_per_us * params.t_max_us)
    print()
    print("Sweep summary")
    print("-------------")
    print(f"k: {params.recombination_rate_per_us:g} us^-1, t_max={params.t_max_us:g} us")
    print(f"truncated recombination tail exp(-kT): {tail:.3e}")
    print(f"min Phi_S: {summary['min_yield']:.8f} at theta={summary['theta_min_yield_deg']:.3f} deg")
    print(f"max Phi_S: {summary['max_yield']:.8f} at theta={summary['theta_max_yield_deg']:.3f} deg")
    print(f"absolute contrast: {summary['absolute_contrast']:.8e}")
    print(f"relative contrast: {summary['relative_contrast']:.8e}")
    print(f"saved CSV: {csv_path}")
    print(f"saved nuclei tensors: {nuclei_path}")
    if plot_path is not None:
        print(f"saved plot: {plot_path}")


def parse_angles(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("theta step must be positive.")
    return np.arange(start, stop + 0.5 * step, step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short five-angle sanity check.")
    parser.add_argument("--max-nuclei", type=int, default=16, help="Number of nuclei from the published ordering to include.")
    parser.add_argument("--samples", type=int, default=1, help="Random-phase nuclear samples per orientation.")
    parser.add_argument("--seed", type=int, default=20260528, help="Random seed for nuclear samples.")
    parser.add_argument("--model", choices=["quantum", "overhauser"], default="quantum")
    parser.add_argument("--theta-step", type=float, default=45.0)
    parser.add_argument("--theta-start", type=float, default=0.0)
    parser.add_argument("--theta-stop", type=float, default=180.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--b-microt", type=float, default=50.0)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--t-max-us", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=101)
    parser.add_argument("--J-MHz", type=float, default=3.0)
    parser.add_argument("--physical-dipole", action="store_true", help="Use the published Cry4a dipolar tensor instead of the optimistic z-axis tensor.")
    parser.add_argument("--no-dipole", action="store_true", help="Disable the electron-electron dipolar interaction.")
    parser.add_argument("--D-MHz", type=float, default=-19.10703069, help="Optimistic axial tensor Dzz used when --physical-dipole is absent.")
    parser.add_argument("--hyperfine-mode", choices=["full", "diagonal", "z-only"], default="full")
    parser.add_argument("--operator-mode", choices=["auto", "sparse", "matrix-free"], default="auto")
    parser.add_argument("--max-sparse-qubits", type=int, default=18)
    parser.add_argument("--evolver", choices=["expm", "rk4"], default="expm")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "full_fad_trp_magnetosensitivity")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> Params:
    params = Params()
    params.max_nuclei = args.max_nuclei
    params.n_samples = args.samples
    params.seed = args.seed
    params.model = args.model
    params.phi_deg = args.phi
    params.B_microT = args.b_microt
    params.recombination_rate_per_us = args.k
    params.t_max_us = args.t_max_us
    params.n_steps = args.n_steps
    params.J_MHz = args.J_MHz
    params.include_dipole = not args.no_dipole
    params.optimize_dipole_axis = not args.physical_dipole
    params.D_MHz_if_isotropic = args.D_MHz
    params.hyperfine_mode = args.hyperfine_mode
    params.operator_mode = args.operator_mode
    params.max_sparse_qubits = args.max_sparse_qubits
    params.evolver = args.evolver
    params.output_dir = args.output_dir
    params.save_plot = not args.no_plot
    params.show_plot = args.show_plot

    if args.quick:
        params.max_nuclei = min(params.max_nuclei, 7)
        params.n_samples = min(params.n_samples, 1)
        params.theta_values_deg = np.array([0.0, 45.0, 90.0, 135.0, 180.0], dtype=float)
        params.t_max_us = min(params.t_max_us, 2.0)
        params.n_steps = min(params.n_steps, 101)
        params.output_dir = args.output_dir / "quick"
    else:
        params.theta_values_deg = parse_angles(args.theta_start, args.theta_stop, args.theta_step)

    if params.n_samples < 1:
        raise ValueError("samples must be at least 1.")
    if params.n_steps < 2:
        raise ValueError("n_steps must be at least 2.")
    return params


def main() -> None:
    params = params_from_args(parse_args())
    results, nuclei = run_sweep(params)
    summary = summarize_results(results)

    params.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = params.output_dir / "full_fad_trp_magnetosensitivity.csv"
    nuclei_path = params.output_dir / "selected_hyperfine_tensors.csv"
    save_results_csv(results, csv_path)
    save_nuclei_csv(nuclei, nuclei_path)

    plot_path = None
    if params.save_plot:
        plot_path = params.output_dir / "full_fad_trp_magnetosensitivity.png"
        plot_results(results, plot_path, params.show_plot)

    print_summary(params, summary, csv_path, nuclei_path, plot_path)


if __name__ == "__main__":
    main()
