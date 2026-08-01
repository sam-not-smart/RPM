"""Shared physics, sampling, statistics, and validation for poster reruns.

This module deliberately imports the production D/J condition and propagation
settings from ``rpm_quantum_needle_jd_sweeps``.  The poster experiment must not
quietly rederive or retune the electron couplings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.integrate
import scipy.linalg

from rpm_quantum_needle_jd_sweeps import (
    QUADRUPOLE_SCALE,
    SweepConfig,
    _seed as production_sample_seed,
    efimova_hore_ratio_condition,
    integrate_yield,
)
from rpm_quantum_needle_model import (
    ALL_NUCLEI,
    FAD_NUCLEI,
    GAMMA_E_MHZ_PER_MT,
    NUCLEAR_RADICALS,
    QUADRUPOLE_TENSORS_MT,
    SPIN_ORDER,
    SYSTEM_DIMENSIONS,
    TWOPI,
    Nucleus,
    field_vector_mT,
    spin_operators,
)
from rpm_statevector_engine import (
    AdaptiveLanczosPropagator,
    LocalTerm,
    MatrixFreeHamiltonian,
    SingletPopulationObservable,
    efimova_hore_electron_coupling_operator,
    quadrupole_operator,
    singlet_population,
    two_spin_coupling_operator,
)


MASTER_SEED = 20260731
POSTER_FIELDS_MT = (0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 400.0)
POSTER_THETA_DEG = tuple(float(value) for value in range(0, 181, 10))
OVERHAUSER_FIELDS_MT = (0.05, 2.0, 20.0)
ENTANGLEMENT_NUCLEAR_COUNTS = (4, 6, 8, 10, 12, 14)
N10_THETA_DEG = tuple(float(value) for value in np.round(np.arange(85.0, 95.0 + 0.025, 0.05), 10))
N10_PERIODS_US = (0.02, 0.05)
PRODUCTION_SAMPLE_COUNT = 16
BOOTSTRAP_REPLICATES = 2000


@dataclass(frozen=True)
class ProductionModel:
    """The exact currently validated Danny model used by this experiment."""

    master_seed: int = MASTER_SEED
    samples: int = PRODUCTION_SAMPLE_COUNT
    lifetime_us: float = 5.0
    field_plane: str = "molecular zx plane, B=[sin(theta),0,cos(theta)]"
    quadrupole_scale: float = QUADRUPOLE_SCALE
    ratio_D_over_J: float = -3.0

    @property
    def condition(self):
        # This is the production condition used by the latest Danny runner.
        return efimova_hore_ratio_condition(self.ratio_D_over_J)

    @property
    def sweep_config(self) -> SweepConfig:
        return replace(SweepConfig(), seed=self.master_seed, lifetime_us=self.lifetime_us)

    def stable_dict(self) -> dict[str, object]:
        config = self.sweep_config
        return {
            **asdict(self),
            "condition": self.condition.stable_dict(),
            "propagation": config.stable_dict(),
        }

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


PRODUCTION_MODEL = ProductionModel()


def sample_seed(master_seed: int, sample_id: int) -> int:
    """Return the repository production seed for a nuclear sample ID."""

    return production_sample_seed(int(master_seed), int(sample_id))


def selected_nuclei(nuclear_count: int) -> tuple[Nucleus, ...]:
    """Select equal strongest-to-weakest prefixes from the two radicals."""

    count = int(nuclear_count)
    if count not in ENTANGLEMENT_NUCLEAR_COUNTS:
        raise ValueError(f"nuclear_count must be one of {ENTANGLEMENT_NUCLEAR_COUNTS}.")
    per_radical = count // 2
    return tuple(FAD_NUCLEI[:per_radical] + ALL_NUCLEI[len(FAD_NUCLEI):][:per_radical])


def selected_radicals(nuclear_count: int) -> tuple[str, ...]:
    per_radical = int(nuclear_count) // 2
    return ("FAD",) * per_radical + ("TrpH",) * per_radical


def dimensions_for(nuclei: Sequence[Nucleus]) -> tuple[int, ...]:
    return (2, 2) + tuple(nucleus.dimension for nucleus in nuclei)


def _electron_coupling_term(model: ProductionModel) -> LocalTerm:
    condition = model.condition
    operator = TWOPI * efimova_hore_electron_coupling_operator(
        condition.exchange_J_MHz,
        condition.d_paper_MHz,
        np.asarray(condition.dipolar_axis, dtype=float),
    )
    return LocalTerm((0, 1), operator, "Efimova-Hore exchange + full nonsecular dipolar")


def n10_transverse_deleted_tensor() -> np.ndarray:
    """Recover the poster definition: delete molecular-frame nuclear x/y columns."""

    tensor = np.asarray(FAD_NUCLEI[1].tensor_mT, dtype=float).copy()
    tensor[:, :2] = 0.0
    return tensor


def build_hamiltonian(
    theta_deg: float,
    field_mT: float,
    *,
    model: ProductionModel = PRODUCTION_MODEL,
    nuclei: Sequence[Nucleus] = ALL_NUCLEI,
    radicals: Sequence[str] = NUCLEAR_RADICALS,
    hyperfine_overrides_mT: dict[str, np.ndarray] | None = None,
) -> MatrixFreeHamiltonian:
    """Build the current nonfactorized Hamiltonian for any paired bath prefix."""

    if len(nuclei) != len(radicals):
        raise ValueError("nuclei and radicals must have the same length.")
    dimensions = dimensions_for(nuclei)
    electron_ops = spin_operators(0.5)
    field = field_vector_mT(theta_deg, float(field_mT) * 1000.0)
    zeeman = TWOPI * GAMMA_E_MHZ_PER_MT * sum(
        component * operator for component, operator in zip(field, electron_ops)
    )
    terms: list[LocalTerm] = [
        LocalTerm((0,), zeeman, "electron A Zeeman"),
        LocalTerm((1,), zeeman, "electron B Zeeman"),
    ]
    overrides = hyperfine_overrides_mT or {}
    used_names: dict[str, int] = {}
    for nucleus_index, (nucleus, radical) in enumerate(zip(nuclei, radicals)):
        electron_axis = 0 if radical == "FAD" else 1
        key = f"{radical}:{nucleus.name}"
        tensor = np.asarray(overrides.get(key, nucleus.tensor_mT), dtype=float)
        local = TWOPI * GAMMA_E_MHZ_PER_MT * two_spin_coupling_operator(
            0.5, nucleus.spin, tensor
        )
        terms.append(LocalTerm((electron_axis, nucleus_index + 2), local, f"{key} hyperfine"))
        quadrupole_tensor = QUADRUPOLE_TENSORS_MT.get(key)
        if quadrupole_tensor is not None and model.quadrupole_scale != 0.0:
            q_local = quadrupole_operator(
                nucleus.spin,
                TWOPI * GAMMA_E_MHZ_PER_MT * model.quadrupole_scale * quadrupole_tensor,
            )
            terms.append(LocalTerm((nucleus_index + 2,), q_local, f"{key} quadrupole"))
        used_names[key] = used_names.get(key, 0) + 1
    if any(value != 1 for value in used_names.values()):
        raise ValueError("Selected nuclei contain duplicate radical/name keys.")
    terms.append(_electron_coupling_term(model))
    return MatrixFreeHamiltonian(dimensions, terms)


def make_propagator(hamiltonian: MatrixFreeHamiltonian, model: ProductionModel = PRODUCTION_MODEL):
    config = model.sweep_config
    return AdaptiveLanczosPropagator(
        hamiltonian,
        minimum_dimension=config.lanczos_minimum_dimension,
        maximum_dimension=config.lanczos_maximum_dimension,
        dimension_increment=config.lanczos_dimension_increment,
        embedded_offset=config.lanczos_embedded_offset,
        safety_factor=config.lanczos_safety_factor,
        state_error_budget=config.state_error_budget,
        initial_step_us=config.lanczos_initial_step_us,
        maximum_step_us=config.lanczos_maximum_step_us,
    )


def random_phase_initial_state(
    dimensions: Sequence[int], master_seed: int, sample_id: int
) -> np.ndarray:
    rng = np.random.default_rng(sample_seed(master_seed, sample_id))
    nuclear_dimension = math.prod(dimensions[2:])
    phases = rng.uniform(0.0, TWOPI, nuclear_dimension)
    nuclei = np.exp(1.0j * phases) / math.sqrt(nuclear_dimension)
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    return np.kron(singlet, nuclei)


def full_product_basis_indices(master_seed: int, sample_id: int) -> tuple[int, ...]:
    """One reproducible 14-nucleus computational product-state sample."""

    rng = np.random.default_rng(sample_seed(master_seed, sample_id))
    return tuple(int(rng.integers(dimension)) for dimension in SYSTEM_DIMENSIONS[2:])


def product_basis_initial_state(
    dimensions: Sequence[int], indices: Sequence[int]
) -> np.ndarray:
    if len(indices) != len(dimensions) - 2:
        raise ValueError("One product-basis index is required per nuclear axis.")
    flat_index = np.ravel_multi_index(tuple(indices), tuple(dimensions[2:]))
    nuclei = np.zeros(math.prod(dimensions[2:]), dtype=complex)
    nuclei[int(flat_index)] = 1.0
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    return np.kron(singlet, nuclei)


def paired_product_indices(nuclear_count: int, full_indices: Sequence[int]) -> tuple[int, ...]:
    per_radical = int(nuclear_count) // 2
    return tuple(full_indices[:per_radical]) + tuple(full_indices[7 : 7 + per_radical])


def local_unitary_apply(
    state: np.ndarray, dimensions: Sequence[int], axis: int, unitary: np.ndarray
) -> np.ndarray:
    tensor = np.asarray(state, dtype=complex).reshape(tuple(dimensions))
    moved = np.moveaxis(tensor, int(axis), 0)
    applied = np.asarray(unitary, dtype=complex) @ moved.reshape(dimensions[axis], -1)
    restored = applied.reshape((dimensions[axis],) + moved.shape[1:])
    return np.moveaxis(restored, 0, axis).reshape(-1)


def n10_principal_axis() -> np.ndarray:
    values, axes = np.linalg.eigh(0.5 * (FAD_NUCLEI[1].tensor_mT + FAD_NUCLEI[1].tensor_mT.T))
    axis = axes[:, int(np.argmax(np.abs(values)))]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    return axis


def spin_pi_pulse(spin: float, molecular_axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(molecular_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    generator = sum(value * op for value, op in zip(axis, spin_operators(spin)))
    return scipy.linalg.expm(-1.0j * math.pi * generator)


def pure_log_negativity(
    state: np.ndarray, dimensions: Sequence[int], left_axes: Sequence[int]
) -> float:
    dims = tuple(int(value) for value in dimensions)
    left = tuple(int(value) for value in left_axes)
    if len(set(left)) != len(left) or any(axis < 0 or axis >= len(dims) for axis in left):
        raise ValueError("Invalid subsystem axes.")
    right = tuple(axis for axis in range(len(dims)) if axis not in left)
    tensor = np.asarray(state, dtype=complex).reshape(dims)
    matrix = np.transpose(tensor, left + right).reshape(
        math.prod(dims[axis] for axis in left), -1
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    norm = float(np.linalg.norm(singular_values))
    if norm == 0.0:
        raise ValueError("Cannot evaluate entanglement of the zero state.")
    singular_values /= norm
    return float(2.0 * np.log2(np.sum(singular_values)))


def reduced_electron_density(state: np.ndarray, dimensions: Sequence[int]) -> np.ndarray:
    matrix = np.asarray(state, dtype=complex).reshape(4, -1)
    rho = matrix @ matrix.conj().T
    return 0.5 * (rho + rho.conj().T)


def partial_transpose_two_qubits(rho: np.ndarray, subsystem: int = 0) -> np.ndarray:
    tensor = np.asarray(rho, dtype=complex).reshape(2, 2, 2, 2)
    if subsystem == 0:
        tensor = tensor.transpose(2, 1, 0, 3)
    elif subsystem == 1:
        tensor = tensor.transpose(0, 3, 2, 1)
    else:
        raise ValueError("subsystem must be 0 or 1.")
    return tensor.reshape(4, 4)


def mixed_two_electron_log_negativity(rho: np.ndarray) -> float:
    hermitian = 0.5 * (np.asarray(rho, dtype=complex) + np.asarray(rho, dtype=complex).conj().T)
    eigenvalues = np.linalg.eigvalsh(partial_transpose_two_qubits(hermitian))
    return float(max(0.0, np.log2(np.sum(np.abs(eigenvalues)))))


def overhauser_covariances_mhz2() -> dict[str, np.ndarray]:
    covariances = {"FAD": np.zeros((3, 3)), "TrpH": np.zeros((3, 3))}
    for radical, nucleus in zip(NUCLEAR_RADICALS, ALL_NUCLEI):
        tensor_mhz = GAMMA_E_MHZ_PER_MT * np.asarray(nucleus.tensor_mT, dtype=float)
        factor = nucleus.spin * (nucleus.spin + 1.0) / 3.0
        covariances[radical] += factor * tensor_mhz @ tensor_mhz.T
    return covariances


def overhauser_sample_mhz(master_seed: int, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(sample_seed(master_seed ^ 0x0A11CE, sample_id))
    covariances = overhauser_covariances_mhz2()
    samples = []
    for radical in ("FAD", "TrpH"):
        values, axes = np.linalg.eigh(covariances[radical])
        factor = axes @ np.diag(np.sqrt(np.maximum(values, 0.0)))
        samples.append(factor @ rng.standard_normal(3))
    return samples[0], samples[1]


def atomwise_overhauser_sample_mhz(master_seed: int, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(sample_seed(master_seed ^ 0xA70A, sample_id))
    result = {"FAD": np.zeros(3), "TrpH": np.zeros(3)}
    for radical, nucleus in zip(NUCLEAR_RADICALS, ALL_NUCLEI):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        spin_vector = math.sqrt(nucleus.spin * (nucleus.spin + 1.0)) * direction
        tensor_mhz = GAMMA_E_MHZ_PER_MT * np.asarray(nucleus.tensor_mT, dtype=float)
        result[radical] += tensor_mhz @ spin_vector
    return result["FAD"], result["TrpH"]


def electron_only_hamiltonian(
    theta_deg: float,
    field_mT: float,
    local_fields_mhz: tuple[np.ndarray, np.ndarray] | None = None,
    model: ProductionModel = PRODUCTION_MODEL,
) -> np.ndarray:
    ops = spin_operators(0.5)
    identity = np.eye(2, dtype=complex)
    field = GAMMA_E_MHZ_PER_MT * field_vector_mT(theta_deg, field_mT * 1000.0)
    local = local_fields_mhz or (np.zeros(3), np.zeros(3))
    result = np.zeros((4, 4), dtype=complex)
    for index, electron_field in enumerate((field + local[0], field + local[1])):
        operator = TWOPI * sum(value * op for value, op in zip(electron_field, ops))
        result += np.kron(operator, identity) if index == 0 else np.kron(identity, operator)
    result += _electron_coupling_term(model).operator
    return 0.5 * (result + result.conj().T)


def dense_singlet_dynamics(hamiltonian: np.ndarray, times_us: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(hamiltonian, dtype=complex))
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    coefficients = vectors.conj().T @ singlet
    amplitudes = coefficients[:, None] * np.exp(-1.0j * values[:, None] * times_us[None, :])
    states = vectors @ amplitudes
    return np.abs(singlet.conj() @ states) ** 2


def dense_truncated_singlet_yield(
    hamiltonian: np.ndarray, lifetime_us: float, final_time_us: float
) -> float:
    """Integrate the four-level singlet yield analytically over a finite horizon.

    This is used for the inexpensive semiclassical ensemble.  A validation test
    requires agreement with the production Simpson grid well below its reported
    quadrature error, so it does not create an Overhauser-specific integration
    bias.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(hamiltonian, dtype=complex))
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    weights = np.abs(eigenvectors.conj().T @ singlet) ** 2
    rate = 1.0 / float(lifetime_us)
    frequencies = eigenvalues[:, None] - eigenvalues[None, :]
    decay = rate + 1.0j * frequencies
    integral = rate * (1.0 - np.exp(-decay * float(final_time_us))) / decay
    return float(np.sum(weights[:, None] * weights[None, :] * integral).real)


def overhauser_yield(
    theta_deg: float,
    field_mT: float,
    local_fields_mhz: tuple[np.ndarray, np.ndarray],
    model: ProductionModel = PRODUCTION_MODEL,
    *,
    return_dynamics: bool = False,
    dynamics_times_us: np.ndarray | None = None,
) -> tuple[float, np.ndarray | None]:
    hamiltonian = electron_only_hamiltonian(theta_deg, field_mT, local_fields_mhz, model)
    value = dense_truncated_singlet_yield(
        hamiltonian, model.lifetime_us, model.sweep_config.maximum_time_us
    )
    if not return_dynamics:
        return value, None
    times = (
        np.linspace(0.0, model.sweep_config.maximum_time_us, 1001)
        if dynamics_times_us is None
        else np.asarray(dynamics_times_us, dtype=float)
    )
    return value, dense_singlet_dynamics(hamiltonian, times)


def mean_se_ci(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=float)
    mean = np.mean(values, axis=0)
    if values.shape[0] < 2:
        se = np.full_like(mean, np.nan, dtype=float)
        return mean, se, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    se = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    half = scipy.stats_t_975(values.shape[0] - 1) * se
    return mean, se, mean - half, mean + half


def scipy_stats_t_975(df: int) -> float:
    from scipy.stats import t

    return float(t.ppf(0.975, int(df)))


def bootstrap_curve_metric(
    sample_curves: np.ndarray,
    metric,
    *,
    seed: int = MASTER_SEED ^ 0xB007,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float, float, np.ndarray]:
    curves = np.asarray(sample_curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError("sample_curves must have shape (samples, angles).")
    estimate = float(metric(np.mean(curves, axis=0)))
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=float)
    for index in range(replicates):
        ids = rng.integers(0, curves.shape[0], size=curves.shape[0])
        boot[index] = float(metric(np.mean(curves[ids], axis=0)))
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high), boot


def angular_metrics(theta_deg: np.ndarray, curve: np.ndarray) -> dict[str, float]:
    theta = np.asarray(theta_deg, dtype=float)
    values = np.asarray(curve, dtype=float)
    derivative = np.gradient(values, theta, edge_order=2 if theta.size >= 3 else 1)
    minimum = int(np.argmin(values))
    maximum = int(np.argmax(values))
    slope = int(np.argmax(np.abs(derivative)))
    contrast = float(values[maximum] - values[minimum])
    return {
        "minimum_yield": float(values[minimum]),
        "maximum_yield": float(values[maximum]),
        "contrast": contrast,
        "relative_contrast": contrast / float(np.mean(values)),
        "theta_minimum_deg": float(theta[minimum]),
        "theta_maximum_deg": float(theta[maximum]),
        "maximum_abs_slope_per_deg": float(abs(derivative[slope])),
        "theta_maximum_abs_slope_deg": float(theta[slope]),
    }


def electron_spectrum_manifest(model: ProductionModel = PRODUCTION_MODEL) -> dict[str, object]:
    condition = model.condition
    operator_mhz = efimova_hore_electron_coupling_operator(
        condition.exchange_J_MHz, condition.d_paper_MHz, np.asarray(condition.dipolar_axis)
    )
    values, vectors = np.linalg.eigh(operator_mhz)
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    singlet_weights = np.abs(vectors.conj().T @ singlet) ** 2
    gaps = np.diff(np.sort(values))
    smallest = float(np.min(np.abs(gaps))) if gaps.size else math.nan
    scale = max(1.0, float(np.max(np.abs(values))))
    return {
        "eigenvalues_MHz": values.tolist(),
        "singlet_weights": singlet_weights.tolist(),
        "adjacent_gaps_MHz": gaps.tolist(),
        "minimum_adjacent_gap_MHz": smallest,
        "cancellation_exact": bool(smallest <= 1e-12 * scale),
        "cancellation_classification": "exact" if smallest <= 1e-12 * scale else "approximate",
    }


def model_manifest(model: ProductionModel = PRODUCTION_MODEL) -> dict[str, object]:
    condition = model.condition
    nuclei = []
    for index, (radical, nucleus) in enumerate(zip(NUCLEAR_RADICALS, ALL_NUCLEI)):
        key = f"{radical}:{nucleus.name}"
        nuclei.append(
            {
                "axis": index + 2,
                "name": key,
                "spin": nucleus.spin,
                "dimension": nucleus.dimension,
                "hyperfine_tensor_mT": np.asarray(nucleus.tensor_mT).tolist(),
                "quadrupole_tensor_mT": (
                    None if key not in QUADRUPOLE_TENSORS_MT else QUADRUPOLE_TENSORS_MT[key].tolist()
                ),
            }
        )
    return {
        "schema_version": 1,
        "experiment": "poster-style rerun with the current production model",
        "configuration_hash": model.configuration_hash,
        "spin_order": list(SPIN_ORDER),
        "system_dimensions": list(SYSTEM_DIMENSIONS),
        "nuclei": nuclei,
        "tensor_source": "Hiscock et al. 2016 supporting-information Tables S1-S2",
        "tensor_reference_frame": "shared flavin molecular frame",
        "field_plane": model.field_plane,
        "quadrupole_machinery": "available for all spin-1 nuclei",
        "quadrupole_scale": model.quadrupole_scale,
        "quadrupole_note": "Frozen to the latest validated Danny production value Q=0; no nonzero Q is introduced here.",
        "electron_coupling": {
            **condition.stable_dict(),
            "H_exchange": "-J (S^2 - 1)",
            "H_dipolar": "D ((S.n)^2 - S^2/3)",
            "r_nm": 2.0,
            "axis_molecular": list(condition.dipolar_axis),
        },
        "electron_only_spectrum": electron_spectrum_manifest(model),
        "recombination": {
            "model": "equal singlet/triplet first-order recombination",
            "lifetime_us": model.lifetime_us,
            "rate_per_us": 1.0 / model.lifetime_us,
            "yield": "integral_0^T exp(-t/tau) P_S(t) dt/tau",
            "tail_bound": model.sweep_config.survival_tail_bound,
        },
        "propagation_and_integration": model.sweep_config.stable_dict(),
        "sampler": {
            "quantum_yields": "global random-phase nuclear trace vectors",
            "trajectory_entanglement": "uniform computational-basis nuclear product states",
            "master_seed": model.master_seed,
            "production_samples": model.samples,
            "pairing": "same sample ID reused across all compared conditions",
        },
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
