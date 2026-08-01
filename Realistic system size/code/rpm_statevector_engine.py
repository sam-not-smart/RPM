"""Mixed-spin, matrix-free state-vector tools for radical-pair dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from rpm_quantum_needle_model import (
    ALL_NUCLEI,
    FAD_NUCLEI,
    GAMMA_E_MHZ_PER_MT,
    QUADRUPOLE_TENSORS_MT,
    SPIN_ORDER,
    SYSTEM_DIMENSIONS,
    TWOPI,
    field_vector_mT,
    spin_operators,
)


@dataclass(frozen=True)
class LocalTerm:
    """Small operator acting on one or more axes of a tensor-product state."""

    axes: tuple[int, ...]
    operator: np.ndarray
    label: str = ""


@dataclass(frozen=True)
class _CompiledTerm:
    axes: tuple[int, ...]
    active_dimensions: tuple[int, ...]
    remaining_dimensions: tuple[int, ...]
    operator: np.ndarray
    label: str


class MatrixFreeHamiltonian:
    """Sum of local tensor operators without an assembled full matrix."""

    def __init__(
        self,
        dimensions: Iterable[int],
        terms: Iterable[LocalTerm],
        *,
        hermitian_tolerance: float = 1e-12,
    ) -> None:
        self.dimensions = tuple(int(value) for value in dimensions)
        if not self.dimensions or any(value < 1 for value in self.dimensions):
            raise ValueError("Every tensor dimension must be positive.")
        self.dimension = math.prod(self.dimensions)
        self.terms: tuple[_CompiledTerm, ...] = tuple(
            self._compile_term(term, hermitian_tolerance) for term in terms
        )
        self.shape = (self.dimension, self.dimension)
        self.dtype = np.dtype(np.complex128)
        self.matvec_count = 0

    def _compile_term(self, term: LocalTerm, tolerance: float) -> _CompiledTerm:
        axes = tuple(int(axis) for axis in term.axes)
        if not axes or len(set(axes)) != len(axes):
            raise ValueError(f"{term.label or 'Local term'} must have unique active axes.")
        if any(axis < 0 or axis >= len(self.dimensions) for axis in axes):
            raise ValueError(f"{term.label or 'Local term'} contains an out-of-range axis.")
        active_dimensions = tuple(self.dimensions[axis] for axis in axes)
        local_dimension = math.prod(active_dimensions)
        operator = np.asarray(term.operator, dtype=complex)
        if operator.shape != (local_dimension, local_dimension):
            raise ValueError(
                f"{term.label or 'Local term'} has shape {operator.shape}; "
                f"expected {(local_dimension, local_dimension)}."
            )
        scale = max(1.0, float(np.linalg.norm(operator)))
        if np.linalg.norm(operator - operator.conj().T) > tolerance * scale:
            raise ValueError(f"{term.label or 'Local term'} is not Hermitian.")
        remaining_dimensions = tuple(
            dimension for index, dimension in enumerate(self.dimensions) if index not in axes
        )
        return _CompiledTerm(
            axes=axes,
            active_dimensions=active_dimensions,
            remaining_dimensions=remaining_dimensions,
            operator=np.ascontiguousarray(operator),
            label=term.label,
        )

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        flat = np.asarray(vector, dtype=complex).reshape(-1)
        if flat.size != self.dimension:
            raise ValueError(f"Vector length {flat.size} does not match {self.dimension}.")
        tensor = flat.reshape(self.dimensions)
        result = np.zeros_like(tensor, dtype=complex)
        for term in self.terms:
            moved = np.moveaxis(tensor, term.axes, range(len(term.axes)))
            matrix_view = moved.reshape(math.prod(term.active_dimensions), -1)
            applied = term.operator @ matrix_view
            applied_tensor = applied.reshape(term.active_dimensions + term.remaining_dimensions)
            result += np.moveaxis(applied_tensor, range(len(term.axes)), term.axes)
        self.matvec_count += 1
        return result.reshape(-1)

    def matmat(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=complex)
        if values.ndim != 2 or values.shape[0] != self.dimension:
            raise ValueError(f"Matrix must have shape ({self.dimension}, n_vectors).")
        return np.column_stack([self.matvec(values[:, column]) for column in range(values.shape[1])])

    def aslinearoperator(self) -> LinearOperator:
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.matvec,
            matmat=self.matmat,
            dtype=self.dtype,
        )

    def explicit_matrix(self) -> np.ndarray:
        """Assemble a dense matrix for tests and deliberately small systems."""

        identity = np.eye(self.dimension, dtype=complex)
        return self.matmat(identity)


def two_spin_coupling_operator(
    left_spin: float,
    right_spin: float,
    tensor: np.ndarray,
) -> np.ndarray:
    """Return sum_ab tensor[a,b] S_a tensor I_b."""

    coupling = np.asarray(tensor, dtype=float)
    if coupling.shape != (3, 3):
        raise ValueError("A two-spin coupling tensor must be 3x3.")
    left = spin_operators(left_spin)
    right = spin_operators(right_spin)
    dimension = left[0].shape[0] * right[0].shape[0]
    operator = np.zeros((dimension, dimension), dtype=complex)
    for a in range(3):
        for b in range(3):
            if coupling[a, b] != 0.0:
                operator += coupling[a, b] * np.kron(left[a], right[b])
    return operator


def efimova_hore_electron_coupling_operator(
    exchange_J_MHz: float,
    dipolar_D_MHz: float,
    dipolar_axis: np.ndarray,
) -> np.ndarray:
    """Return the literal Efimova-Hore electron-pair Hamiltonian in MHz.

    Implements Eqs. 9-10 of Efimova and Hore, Biophys. J. 94,
    1565-1574 (2008):

    ``H_J = -J (S^2 - 1)`` and
    ``H_D = D ((S.n)^2 - S^2 / 3)``.

    The returned 4x4 operator includes the state-independent exchange shift.
    Keeping that shift makes the implementation literally match the paper;
    it changes only the global phase of a propagated state.
    """

    axis = np.asarray(dipolar_axis, dtype=float)
    if axis.shape != (3,):
        raise ValueError("dipolar_axis must be a length-3 vector.")
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("dipolar_axis must be nonzero.")
    axis = axis / norm
    single = spin_operators(0.5)
    identity_2 = np.eye(2, dtype=complex)
    identity_4 = np.eye(4, dtype=complex)
    total = tuple(
        np.kron(operator, identity_2) + np.kron(identity_2, operator)
        for operator in single
    )
    total_squared = sum(operator @ operator for operator in total)
    along_axis = sum(component * operator for component, operator in zip(axis, total))
    exchange = -float(exchange_J_MHz) * (total_squared - identity_4)
    dipolar = float(dipolar_D_MHz) * (
        along_axis @ along_axis - total_squared / 3.0
    )
    result = exchange + dipolar
    return 0.5 * (result + result.conj().T)


def quadrupole_operator(spin: float, tensor: np.ndarray) -> np.ndarray:
    """Return the Hermitian single-spin operator sum_ab Q_ab {I_a,I_b}/2."""

    coupling = np.asarray(tensor, dtype=float)
    if coupling.shape != (3, 3):
        raise ValueError("A quadrupole tensor must be 3x3.")
    if not np.allclose(coupling, coupling.T, atol=1e-14, rtol=0.0):
        raise ValueError("A quadrupole tensor must be symmetric.")
    operators = spin_operators(spin)
    dimension = operators[0].shape[0]
    result = np.zeros((dimension, dimension), dtype=complex)
    for a in range(3):
        for b in range(3):
            if coupling[a, b] != 0.0:
                result += 0.5 * coupling[a, b] * (
                    operators[a] @ operators[b] + operators[b] @ operators[a]
                )
    return result


def build_quantum_needle_hamiltonian(
    theta_deg: float,
    field_microT: float = 50.0,
    *,
    exchange_J_MHz: float = 0.0,
    dipolar_tensor_MHz: np.ndarray | None = None,
    efimova_hore_D_MHz: float | None = None,
    efimova_hore_axis: np.ndarray | None = None,
    quadrupole_scale: float = 1.0,
) -> MatrixFreeHamiltonian:
    """Full radical-pair Hamiltonian, including optional electron couplings.

    The legacy exchange convention is ``H_ex = -2 J S_A.S_B`` and
    ``dipolar_tensor_MHz`` is a general 3x3 tensor in the shared molecular
    frame. Passing ``efimova_hore_D_MHz`` instead selects the literal
    Efimova-Hore Eqs. 9-10, with ``efimova_hore_axis`` defining the dipolar
    axis. ``quadrupole_scale=1`` applies the
    paper-based 14N tensors; zero is reserved for explicitly quadrupole-free
    validation. The Hamiltonian is always represented in the complete pair
    space.
    """

    electron_ops = spin_operators(0.5)
    field = field_vector_mT(theta_deg, field_microT)
    zeeman = TWOPI * GAMMA_E_MHZ_PER_MT * sum(
        component * operator for component, operator in zip(field, electron_ops)
    )
    terms: list[LocalTerm] = [
        LocalTerm((0,), zeeman, "electron A Zeeman"),
        LocalTerm((1,), zeeman, "electron B Zeeman"),
    ]
    for nucleus_index, nucleus in enumerate(ALL_NUCLEI):
        electron_axis = 0 if nucleus_index < len(FAD_NUCLEI) else 1
        local = TWOPI * GAMMA_E_MHZ_PER_MT * two_spin_coupling_operator(
            0.5, nucleus.spin, nucleus.tensor_mT
        )
        terms.append(
            LocalTerm(
                (electron_axis, nucleus_index + 2),
                local,
                f"{nucleus.name} hyperfine",
            )
        )
        quadrupole_tensor = QUADRUPOLE_TENSORS_MT.get(
            SPIN_ORDER[nucleus_index + 2]
        )
        if quadrupole_tensor is not None and quadrupole_scale != 0.0:
            terms.append(
                LocalTerm(
                    (nucleus_index + 2,),
                    quadrupole_operator(
                        nucleus.spin,
                        TWOPI
                        * GAMMA_E_MHZ_PER_MT
                        * float(quadrupole_scale)
                        * quadrupole_tensor,
                    ),
                    f"{nucleus.name} nuclear quadrupole",
                )
            )
    if efimova_hore_D_MHz is not None and dipolar_tensor_MHz is not None:
        raise ValueError(
            "Choose either dipolar_tensor_MHz or efimova_hore_D_MHz, not both."
        )
    if efimova_hore_D_MHz is not None:
        axis = (
            np.array([1.0, 0.0, 0.0])
            if efimova_hore_axis is None
            else np.asarray(efimova_hore_axis, dtype=float)
        )
        electron_operator = TWOPI * efimova_hore_electron_coupling_operator(
            exchange_J_MHz, efimova_hore_D_MHz, axis
        )
        if np.any(electron_operator != 0.0):
            terms.append(
                LocalTerm(
                    (0, 1),
                    electron_operator,
                    "Efimova-Hore electron exchange + dipolar coupling",
                )
            )
        return MatrixFreeHamiltonian(SYSTEM_DIMENSIONS, terms)
    dipolar = (
        np.zeros((3, 3), dtype=float)
        if dipolar_tensor_MHz is None
        else np.asarray(dipolar_tensor_MHz, dtype=float)
    )
    if dipolar.shape != (3, 3):
        raise ValueError("dipolar_tensor_MHz must be a 3x3 tensor.")
    electron_tensor_rad_per_us = TWOPI * (
        dipolar - 2.0 * float(exchange_J_MHz) * np.eye(3)
    )
    if np.any(electron_tensor_rad_per_us != 0.0):
        terms.append(
            LocalTerm(
                (0, 1),
                two_spin_coupling_operator(
                    0.5, 0.5, electron_tensor_rad_per_us
                ),
                "electron exchange + dipolar coupling",
            )
        )
    return MatrixFreeHamiltonian(SYSTEM_DIMENSIONS, terms)


def random_phase_nuclear_state(
    nuclear_dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    phases = rng.uniform(0.0, TWOPI, int(nuclear_dimension))
    return np.exp(1.0j * phases) / math.sqrt(nuclear_dimension)


def singlet_initial_state(
    dimensions: Iterable[int],
    rng: np.random.Generator,
) -> np.ndarray:
    dimensions = tuple(int(value) for value in dimensions)
    if dimensions[:2] != (2, 2):
        raise ValueError("The first two axes must be the two electron spins.")
    singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0)
    nuclei = random_phase_nuclear_state(math.prod(dimensions[2:]), rng)
    return np.kron(singlet, nuclei)


def singlet_population(vector: np.ndarray, dimensions: Iterable[int]) -> float:
    dimensions = tuple(int(value) for value in dimensions)
    tensor = np.asarray(vector, dtype=complex).reshape(dimensions)
    electronic = tensor.reshape(2, 2, -1)
    singlet_amplitude = (electronic[0, 1] - electronic[1, 0]) / math.sqrt(2.0)
    return float(np.vdot(singlet_amplitude, singlet_amplitude).real)


@dataclass(frozen=True)
class SingletPopulationObservable:
    """Singlet population with an exact Krylov-basis projection shortcut.

    For a Krylov basis ``V`` this precomputes ``V^H P_S V``.  Populations at
    all requested times in the accepted macro-step can then be evaluated from
    the small Krylov coefficient vector without reconstructing a full state.
    The propagated endpoint state and every Lanczos decision are unchanged.
    """

    dimensions: tuple[int, ...]

    def __init__(self, dimensions: Iterable[int]) -> None:
        parsed = tuple(int(value) for value in dimensions)
        if parsed[:2] != (2, 2):
            raise ValueError("The first two axes must be the two electron spins.")
        object.__setattr__(self, "dimensions", parsed)

    def __call__(self, vector: np.ndarray) -> float:
        return singlet_population(vector, self.dimensions)

    def project_basis(self, basis: np.ndarray) -> np.ndarray:
        values = np.asarray(basis, dtype=complex)
        expected = math.prod(self.dimensions)
        if values.ndim != 2 or values.shape[0] != expected:
            raise ValueError(f"Krylov basis must have shape ({expected}, m).")
        electronic = values.reshape(2, 2, -1, values.shape[1])
        amplitudes = (electronic[0, 1] - electronic[1, 0]) / math.sqrt(2.0)
        projected = amplitudes.conj().T @ amplitudes
        return 0.5 * (projected + projected.conj().T)

    @staticmethod
    def evaluate_projected(
        projected_operator: np.ndarray,
        coefficients: np.ndarray,
        state_norm: float,
    ) -> float:
        value = np.vdot(coefficients, projected_operator @ coefficients).real
        return float(state_norm * state_norm * value)


def state_error_to_population_bound(state_error: float) -> float:
    epsilon = max(0.0, float(state_error))
    return min(1.0, 2.0 * epsilon + epsilon * epsilon)


@dataclass(frozen=True)
class PropagationStats:
    accepted_steps: int
    rejected_steps: int
    matvecs: int
    maximum_krylov_dimension: int
    minimum_step_us: float
    maximum_step_us: float
    accumulated_state_error: float
    maximum_norm_error: float


@dataclass(frozen=True)
class PropagationResult:
    times_us: np.ndarray
    observable_values: np.ndarray
    observable_error_bounds: np.ndarray
    state_error_estimates: np.ndarray
    states: np.ndarray | None
    final_state: np.ndarray
    stats: PropagationStats


def _projected_coefficients(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    time_us: float,
) -> np.ndarray:
    if diagonal.size == 1:
        return np.array([np.exp(-1.0j * diagonal[0] * time_us)])
    eigenvalues, eigenvectors = scipy.linalg.eigh_tridiagonal(
        diagonal,
        off_diagonal,
        check_finite=False,
    )
    weights = eigenvectors[0, :]
    return eigenvectors @ (np.exp(-1.0j * eigenvalues * time_us) * weights)


@dataclass(frozen=True)
class _ProjectedEvolution:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_tridiagonal(
        cls, diagonal: np.ndarray, off_diagonal: np.ndarray
    ) -> "_ProjectedEvolution":
        if diagonal.size == 1:
            return cls(
                np.asarray(diagonal, dtype=float),
                np.ones((1, 1), dtype=float),
                np.ones(1, dtype=float),
            )
        eigenvalues, eigenvectors = scipy.linalg.eigh_tridiagonal(
            diagonal, off_diagonal, check_finite=False
        )
        return cls(eigenvalues, eigenvectors, eigenvectors[0, :].copy())

    def coefficients(self, time_us: float) -> np.ndarray:
        return self.eigenvectors @ (
            np.exp(-1.0j * self.eigenvalues * time_us) * self.weights
        )


class AdaptiveLanczosPropagator:
    """Adaptive Hermitian Lanczos exponential propagation.

    Error estimates compare nested Krylov dimensions and are multiplied by a
    configurable safety factor.  They are conservative engineering estimates,
    not interval-arithmetic proofs.
    """

    def __init__(
        self,
        hamiltonian: MatrixFreeHamiltonian,
        *,
        minimum_dimension: int = 24,
        maximum_dimension: int = 64,
        dimension_increment: int = 8,
        embedded_offset: int = 4,
        safety_factor: float = 2.0,
        state_error_budget: float = 2.4e-5,
        initial_step_us: float = 0.05,
        maximum_step_us: float = 0.25,
        minimum_step_us: float = 1e-9,
        breakdown_tolerance: float = 1e-14,
    ) -> None:
        if minimum_dimension <= embedded_offset:
            raise ValueError("minimum_dimension must exceed embedded_offset.")
        if maximum_dimension < minimum_dimension:
            raise ValueError("maximum_dimension must not be smaller than minimum_dimension.")
        self.hamiltonian = hamiltonian
        self.minimum_dimension = minimum_dimension
        self.maximum_dimension = maximum_dimension
        self.dimension_increment = dimension_increment
        self.embedded_offset = embedded_offset
        self.safety_factor = safety_factor
        self.state_error_budget = state_error_budget
        self.initial_step_us = initial_step_us
        self.maximum_step_us = maximum_step_us
        self.minimum_step_us = minimum_step_us
        self.breakdown_tolerance = breakdown_tolerance

    def _basis(
        self,
        state: np.ndarray,
        maximum_dimension: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        norm = float(np.linalg.norm(state))
        if norm == 0.0:
            raise ValueError("Cannot propagate the zero vector.")
        count = min(int(maximum_dimension), state.size)
        basis = np.empty((state.size, count), dtype=complex)
        diagonal = np.empty(count, dtype=float)
        off_diagonal = np.empty(count - 1, dtype=float)
        basis[:, 0] = state / norm
        actual = count
        for index in range(count):
            work = self.hamiltonian.matvec(basis[:, index])
            diagonal[index] = float(np.vdot(basis[:, index], work).real)
            work -= diagonal[index] * basis[:, index]
            if index:
                work -= off_diagonal[index - 1] * basis[:, index - 1]
            active = basis[:, : index + 1]
            for _ in range(2):
                work -= active @ (active.conj().T @ work)
            if index == count - 1:
                break
            beta = float(np.linalg.norm(work))
            off_diagonal[index] = beta
            if beta <= self.breakdown_tolerance:
                actual = index + 1
                break
            basis[:, index + 1] = work / beta
        return basis[:, :actual], diagonal[:actual], off_diagonal[: max(0, actual - 1)]

    def _candidate(
        self,
        basis: np.ndarray,
        diagonal: np.ndarray,
        off_diagonal: np.ndarray,
        dimension: int,
        time_us: float,
        state_norm: float,
    ) -> tuple[np.ndarray, float]:
        dimension = min(dimension, diagonal.size)
        high = _projected_coefficients(
            diagonal[:dimension], off_diagonal[: max(0, dimension - 1)], time_us
        )
        low_dimension = max(1, dimension - self.embedded_offset)
        if low_dimension == dimension:
            error = 0.0
        else:
            low = _projected_coefficients(
                diagonal[:low_dimension],
                off_diagonal[: max(0, low_dimension - 1)],
                time_us,
            )
            difference = high.copy()
            difference[:low_dimension] -= low
            error = self.safety_factor * state_norm * float(np.linalg.norm(difference))
        return high, error

    def propagate(
        self,
        initial_state: np.ndarray,
        times_us: Iterable[float],
        observable: Callable[[np.ndarray], float],
        *,
        store_states: bool = False,
        progress_callback: Callable[[float, int, int, int, float], None] | None = None,
    ) -> PropagationResult:
        times = np.asarray(tuple(float(value) for value in times_us), dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("At least one observation time is required.")
        if np.any(times < 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("Observation times must be nonnegative and strictly increasing.")
        state = np.asarray(initial_state, dtype=complex).reshape(-1).copy()
        if state.size != self.hamiltonian.dimension:
            raise ValueError("Initial state dimension does not match the Hamiltonian.")

        project_basis = getattr(observable, "project_basis", None)
        evaluate_projected = getattr(observable, "evaluate_projected", None)
        use_projected_observable = (
            not store_states
            and callable(project_basis)
            and callable(evaluate_projected)
        )

        values = np.empty(times.size, dtype=float)
        bounds = np.empty(times.size, dtype=float)
        state_errors = np.empty(times.size, dtype=float)
        stored_states = (
            np.empty((times.size, state.size), dtype=complex) if store_states else None
        )
        cursor = 0
        current_time = 0.0
        accumulated_error = 0.0
        accepted = 0
        rejected = 0
        maximum_dimension = 0
        used_steps: list[float] = []
        maximum_norm_error = abs(float(np.linalg.norm(state)) - 1.0)
        starting_matvecs = self.hamiltonian.matvec_count
        if times[0] == 0.0:
            values[0] = observable(state)
            bounds[0] = 0.0
            state_errors[0] = 0.0
            if stored_states is not None:
                stored_states[0] = state
            cursor = 1
        final_time = float(times[-1])
        step = min(self.initial_step_us, final_time) if final_time else 0.0
        prefer_maximum_basis = False

        while cursor < times.size:
            remaining = final_time - current_time
            trial_step = min(step, remaining)
            if trial_step < self.minimum_step_us:
                raise RuntimeError("Lanczos step fell below the configured minimum.")
            state_norm = float(np.linalg.norm(state))
            basis_target = (
                self.maximum_dimension if prefer_maximum_basis else self.minimum_dimension
            )
            basis, diagonal, off_diagonal = self._basis(state, basis_target)
            local_budget = self.state_error_budget * trial_step / max(final_time, trial_step)
            selected_dimension = None
            endpoint_coefficients = None
            endpoint_error = math.inf
            exact_subspace = diagonal.size < basis_target
            if exact_subspace:
                # A true Lanczos breakdown closes an invariant subspace, so the
                # projected exponential is exact apart from roundoff.
                selected_dimension = diagonal.size
                endpoint_coefficients = _projected_coefficients(
                    diagonal, off_diagonal, trial_step
                )
                endpoint_error = 0.0
            elif not prefer_maximum_basis:
                endpoint_coefficients, endpoint_error = self._candidate(
                    basis,
                    diagonal,
                    off_diagonal,
                    self.minimum_dimension,
                    trial_step,
                    state_norm,
                )
                if endpoint_error <= local_budget:
                    selected_dimension = self.minimum_dimension
                elif self.maximum_dimension > self.minimum_dimension:
                    basis, diagonal, off_diagonal = self._basis(
                        state, self.maximum_dimension
                    )
                    prefer_maximum_basis = True
                    exact_subspace = diagonal.size < self.maximum_dimension
                    candidates = list(
                        range(
                            self.minimum_dimension + self.dimension_increment,
                            self.maximum_dimension + 1,
                            self.dimension_increment,
                        )
                    )
                    if not candidates or candidates[-1] != self.maximum_dimension:
                        candidates.append(self.maximum_dimension)
                    if exact_subspace:
                        selected_dimension = diagonal.size
                        endpoint_coefficients = _projected_coefficients(
                            diagonal, off_diagonal, trial_step
                        )
                        endpoint_error = 0.0
                    else:
                        for dimension in candidates:
                            endpoint_coefficients, endpoint_error = self._candidate(
                                basis, diagonal, off_diagonal, dimension, trial_step, state_norm
                            )
                            if endpoint_error <= local_budget:
                                selected_dimension = dimension
                                break
            else:
                candidates = list(
                    range(
                        self.minimum_dimension,
                        self.maximum_dimension + 1,
                        self.dimension_increment,
                    )
                )
                if candidates[-1] != self.maximum_dimension:
                    candidates.append(self.maximum_dimension)
                for dimension in candidates:
                    endpoint_coefficients, endpoint_error = self._candidate(
                        basis, diagonal, off_diagonal, dimension, trial_step, state_norm
                    )
                    if endpoint_error <= local_budget:
                        selected_dimension = dimension
                        break
            if selected_dimension is None or endpoint_error > local_budget:
                rejected += 1
                step = 0.5 * trial_step
                continue

            maximum_dimension = max(maximum_dimension, selected_dimension)
            if selected_dimension > self.minimum_dimension:
                prefer_maximum_basis = True
            active_basis = basis[:, :selected_dimension]
            projected_operator = (
                project_basis(active_basis) if use_projected_observable else None
            )
            high_evolution = _ProjectedEvolution.from_tridiagonal(
                diagonal[:selected_dimension],
                off_diagonal[: max(0, selected_dimension - 1)],
            )
            low_dimension = max(1, selected_dimension - self.embedded_offset)
            low_evolution = (
                None
                if exact_subspace or low_dimension == selected_dimension
                else _ProjectedEvolution.from_tridiagonal(
                    diagonal[:low_dimension],
                    off_diagonal[: max(0, low_dimension - 1)],
                )
            )
            step_end = current_time + trial_step
            while cursor < times.size and times[cursor] <= step_end + 1e-13:
                offset = float(times[cursor] - current_time)
                coefficients = high_evolution.coefficients(offset)
                if low_evolution is None:
                    point_error = 0.0
                else:
                    low = low_evolution.coefficients(offset)
                    difference = coefficients.copy()
                    difference[:low_dimension] -= low
                    point_error = (
                        self.safety_factor
                        * state_norm
                        * float(np.linalg.norm(difference))
                    )
                if projected_operator is not None:
                    values[cursor] = evaluate_projected(
                        projected_operator, coefficients, state_norm
                    )
                    point_state = None
                else:
                    point_state = state_norm * (active_basis @ coefficients)
                    values[cursor] = observable(point_state)
                state_errors[cursor] = accumulated_error + point_error
                bounds[cursor] = state_error_to_population_bound(state_errors[cursor])
                if stored_states is not None:
                    assert point_state is not None
                    stored_states[cursor] = point_state
                cursor += 1
            assert endpoint_coefficients is not None
            state = state_norm * (active_basis @ endpoint_coefficients)
            current_time = step_end
            accumulated_error += endpoint_error
            accepted += 1
            used_steps.append(trial_step)
            maximum_norm_error = max(maximum_norm_error, abs(float(np.linalg.norm(state)) - 1.0))
            if progress_callback is not None:
                progress_callback(
                    current_time,
                    accepted,
                    rejected,
                    self.hamiltonian.matvec_count - starting_matvecs,
                    accumulated_error,
                )
            if endpoint_error < 0.1 * local_budget:
                step = min(self.maximum_step_us, 1.5 * trial_step)
            else:
                step = trial_step

        stats = PropagationStats(
            accepted_steps=accepted,
            rejected_steps=rejected,
            matvecs=self.hamiltonian.matvec_count - starting_matvecs,
            maximum_krylov_dimension=maximum_dimension,
            minimum_step_us=min(used_steps) if used_steps else 0.0,
            maximum_step_us=max(used_steps) if used_steps else 0.0,
            accumulated_state_error=accumulated_error,
            maximum_norm_error=maximum_norm_error,
        )
        return PropagationResult(times, values, bounds, state_errors, stored_states, state, stats)
