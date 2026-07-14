from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.linalg


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rpm_quantum_needle_model import (  # noqa: E402
    NUCLEAR_DIMENSION,
    SPIN_ORDER,
    STATE_DIMENSION,
    SYSTEM_DIMENSIONS,
    spin_operators,
    theta_grid,
)
from rpm_quantum_needle_statevector import (  # noqa: E402
    CellResult,
    RunConfig,
    create_report,
    load_cell,
    reaction_time_schedule,
    save_cell,
    uncertainty_components,
)
from rpm_quantum_needle_validation import (  # noqa: E402
    FALLBACK_SWEEP,
    PREFERRED_SWEEP,
    ValidationConfig,
    choose_workload,
    reduced_validation,
)
from rpm_statevector_engine import (  # noqa: E402
    AdaptiveLanczosPropagator,
    LocalTerm,
    MatrixFreeHamiltonian,
    build_quantum_needle_hamiltonian,
    quadrupole_operator,
    random_phase_nuclear_state,
    singlet_initial_state,
    singlet_population,
)


class ModelTests(unittest.TestCase):
    def test_physical_dimensions_and_grid(self) -> None:
        self.assertEqual(SYSTEM_DIMENSIONS, (2, 2, 3, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2))
        self.assertEqual(NUCLEAR_DIMENSION, 55_296)
        self.assertEqual(STATE_DIMENSION, 221_184)
        self.assertEqual(len(SPIN_ORDER), 16)
        grid = theta_grid()
        self.assertEqual(grid.size, 199)
        self.assertIn(89.25, grid)

    def test_spin_one_basis_order(self) -> None:
        _, _, sz = spin_operators(1.0)
        np.testing.assert_allclose(np.diag(sz), [1.0, 0.0, -1.0])


class MatrixFreeTests(unittest.TestCase):
    def test_local_terms_match_dense_kronecker_products(self) -> None:
        sx = spin_operators(0.5)[0]
        sy_spin_one = spin_operators(1.0)[1]
        sz = spin_operators(0.5)[2]
        dimensions = (2, 3, 2)
        terms = (
            LocalTerm((0, 2), np.kron(sx, sz), "nonadjacent coupling"),
            LocalTerm((1,), sy_spin_one, "middle spin"),
        )
        operator = MatrixFreeHamiltonian(dimensions, terms)
        expected = np.kron(sx, np.kron(np.eye(3), sz)) + np.kron(
            np.eye(2), np.kron(sy_spin_one, np.eye(2))
        )
        np.testing.assert_allclose(operator.explicit_matrix(), expected, atol=1e-13)
        rng = np.random.default_rng(1)
        vector = rng.normal(size=12) + 1j * rng.normal(size=12)
        relative = np.linalg.norm(operator.matvec(vector) - expected @ vector) / np.linalg.norm(expected @ vector)
        self.assertLess(relative, 1e-12)

    def test_singlet_projection_matches_dense_projector(self) -> None:
        dimensions = (2, 2, 3)
        rng = np.random.default_rng(4)
        state = rng.normal(size=12) + 1j * rng.normal(size=12)
        state /= np.linalg.norm(state)
        singlet = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0)
        projector = np.kron(np.outer(singlet, singlet), np.eye(3))
        expected = float(np.vdot(state, projector @ state).real)
        self.assertAlmostEqual(singlet_population(state, dimensions), expected, places=13)

    def test_full_hamiltonian_contains_configurable_electron_coupling(self) -> None:
        zero = build_quantum_needle_hamiltonian(
            90.0,
            exchange_J_MHz=0.0,
            dipolar_tensor_MHz=np.zeros((3, 3)),
        )
        self.assertFalse(any(term.axes == (0, 1) for term in zero.terms))

        dipolar = np.diag([-0.5, -0.5, 1.0])
        coupled = build_quantum_needle_hamiltonian(
            90.0,
            exchange_J_MHz=0.25,
            dipolar_tensor_MHz=dipolar,
        )
        electron_terms = [term for term in coupled.terms if term.axes == (0, 1)]
        self.assertEqual(len(electron_terms), 1)
        sx, sy, sz = spin_operators(0.5)
        expected = 2.0 * np.pi * sum(
            coefficient * np.kron(left, right)
            for coefficient, left, right in zip(
                np.diag(dipolar) - 0.5,
                (sx, sy, sz),
                (sx, sy, sz),
            )
        )
        np.testing.assert_allclose(electron_terms[0].operator, expected, atol=1e-14)

    def test_quadrupole_operator_is_hermitian(self) -> None:
        tensor = np.array([[-0.2, 0.03, 0.0], [0.03, -0.1, 0.02], [0.0, 0.02, 0.3]])
        operator = quadrupole_operator(1.0, tensor)
        np.testing.assert_allclose(operator, operator.conj().T, atol=1e-14)
        with self.assertRaises(ValueError):
            quadrupole_operator(1.0, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


class SamplingTests(unittest.TestCase):
    def test_random_phase_normalization_and_mixed_trace(self) -> None:
        rng = np.random.default_rng(12)
        state = random_phase_nuclear_state(4, rng)
        self.assertAlmostEqual(float(np.vdot(state, state).real), 1.0, places=14)
        np.testing.assert_allclose(np.abs(state), 0.5, atol=1e-14)

        observable = np.zeros((4, 4), dtype=complex)
        observable[0, 1] = observable[1, 0] = 1.0
        estimates = []
        rng = np.random.default_rng(22)
        for _ in range(4000):
            sample = random_phase_nuclear_state(4, rng)
            estimates.append(np.vdot(sample, observable @ sample).real)
        self.assertLess(abs(float(np.mean(estimates))), 0.02)

    def test_sobol_schedule_is_nested(self) -> None:
        config = RunConfig()
        small = reaction_time_schedule(config, 32, sample_id=3)
        large = reaction_time_schedule(config, 64, sample_id=3)
        np.testing.assert_allclose(small, large[:, :, :32], atol=0.0, rtol=0.0)
        independent = reaction_time_schedule(config, 32, sample_id=4)
        self.assertFalse(np.array_equal(small, independent))

    def test_random_effects_interval_is_deterministic(self) -> None:
        values = np.arange(32, dtype=float).reshape(4, 8) / 100.0
        first = uncertainty_components(values, 200, 99)
        second = uncertainty_components(values, 200, 99)
        self.assertEqual(first, second)
        self.assertGreater(first.joint_halfwidth, 0.0)


class LanczosTests(unittest.TestCase):
    def _fixture(self) -> tuple[MatrixFreeHamiltonian, np.ndarray, np.ndarray]:
        sx, sy, sz = spin_operators(0.5)
        dimensions = (2, 2, 2)
        terms = (
            LocalTerm((0,), 0.7 * sx + 0.2 * sz),
            LocalTerm((1, 2), 0.4 * np.kron(sy, sx) + 0.3 * np.kron(sz, sz)),
        )
        hamiltonian = MatrixFreeHamiltonian(dimensions, terms)
        state = singlet_initial_state(dimensions, np.random.default_rng(6))
        return hamiltonian, state, hamiltonian.explicit_matrix()

    def test_lanczos_matches_dense_exponential(self) -> None:
        hamiltonian, state, dense = self._fixture()
        times = np.array([0.0, 0.03, 0.2, 0.7])
        propagator = AdaptiveLanczosPropagator(
            hamiltonian,
            minimum_dimension=5,
            maximum_dimension=8,
            dimension_increment=1,
            embedded_offset=2,
            state_error_budget=1e-11,
            initial_step_us=0.4,
            maximum_step_us=0.4,
        )
        result = propagator.propagate(
            state,
            times,
            lambda vector: singlet_population(vector, (2, 2, 2)),
            store_states=True,
        )
        expected = [
            singlet_population(scipy.linalg.expm(-1j * dense * time) @ state, (2, 2, 2))
            for time in times
        ]
        np.testing.assert_allclose(result.observable_values, expected, atol=2e-12)
        np.testing.assert_allclose(result.final_state, scipy.linalg.expm(-1j * dense * times[-1]) @ state, atol=2e-12)
        self.assertIsNotNone(result.states)
        np.testing.assert_allclose(result.states[-1], result.final_state, atol=2e-12)
        self.assertLess(result.stats.maximum_norm_error, 2e-12)

    def test_minimum_step_failure_is_reported(self) -> None:
        hamiltonian, state, _ = self._fixture()
        propagator = AdaptiveLanczosPropagator(
            hamiltonian,
            minimum_dimension=3,
            maximum_dimension=3,
            dimension_increment=1,
            embedded_offset=1,
            state_error_budget=1e-30,
            initial_step_us=1.0,
            maximum_step_us=1.0,
            minimum_step_us=0.2,
        )
        with self.assertRaises(RuntimeError):
            propagator.propagate(
                state, [1.0], lambda vector: singlet_population(vector, (2, 2, 2))
            )

    def test_rejected_step_can_recover(self) -> None:
        hamiltonian, state, _ = self._fixture()
        propagator = AdaptiveLanczosPropagator(
            hamiltonian,
            minimum_dimension=3,
            maximum_dimension=3,
            dimension_increment=1,
            embedded_offset=1,
            state_error_budget=1e-2,
            initial_step_us=1.0,
            maximum_step_us=1.0,
            minimum_step_us=1e-6,
        )
        result = propagator.propagate(
            state, [1.0], lambda vector: singlet_population(vector, (2, 2, 2))
        )
        self.assertGreater(result.stats.rejected_steps, 0)
        self.assertGreater(result.stats.accepted_steps, 0)

    def test_maximum_basis_is_reused_after_growth(self) -> None:
        rng = np.random.default_rng(4)
        values = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        dense = 20.0 * (values + values.conj().T) / 2.0
        hamiltonian = MatrixFreeHamiltonian(
            (2, 2, 2, 2), (LocalTerm((0, 1, 2, 3), dense),)
        )
        state = rng.normal(size=16) + 1j * rng.normal(size=16)
        state /= np.linalg.norm(state)
        result = AdaptiveLanczosPropagator(
            hamiltonian,
            minimum_dimension=4,
            maximum_dimension=12,
            dimension_increment=4,
            embedded_offset=2,
            state_error_budget=1e-4,
            initial_step_us=0.1,
            maximum_step_us=0.1,
            minimum_step_us=1e-8,
        ).propagate(state, [0.2], lambda vector: float(abs(vector[0]) ** 2))
        attempts = result.stats.accepted_steps + result.stats.rejected_steps
        self.assertEqual(result.stats.maximum_krylov_dimension, 12)
        self.assertLess(result.stats.matvecs, attempts * (4 + 12))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_hash_validation(self) -> None:
        result = CellResult(
            theta_deg=90.0,
            sample_id=3,
            q_points=32,
            yields=np.ones((8, 6)),
            propagation_bounds=np.full((8, 6), 1e-6),
            trace_times_us=np.array([0.0, 1.0]),
            trace_population=np.array([1.0, 0.5]),
            trace_bounds=np.array([0.0, 1e-6]),
            accepted_steps=4,
            rejected_steps=1,
            matvecs=80,
            maximum_krylov_dimension=24,
            minimum_step_us=0.01,
            maximum_step_us=0.1,
            accumulated_state_error=1e-6,
            maximum_norm_error=1e-12,
            elapsed_seconds=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_cell(result, output, "abc")
            loaded = load_cell(output, 90.0, 3, 32, "abc")
            np.testing.assert_array_equal(loaded.yields, result.yields)
            with self.assertRaises(RuntimeError):
                load_cell(output, 90.0, 3, 32, "different")

    def test_synthetic_report_writes_all_artifacts(self) -> None:
        config = RunConfig(sobol_scrambles=2, bootstrap_replicates=20)
        angles = np.array([0.0, 90.0])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for theta in angles:
                for sample_id in range(2):
                    base = 0.3 + theta / 1000.0 + sample_id / 100.0
                    result = CellResult(
                        theta_deg=theta,
                        sample_id=sample_id,
                        q_points=2,
                        yields=np.full((2, 6), base) + np.arange(2)[:, None] / 1000.0,
                        propagation_bounds=np.full((2, 6), 1e-6),
                        trace_times_us=np.array([0.0, 1.0]) if theta in (0.0, 90.0) else np.empty(0),
                        trace_population=np.array([1.0, base]) if theta in (0.0, 90.0) else np.empty(0),
                        trace_bounds=np.array([0.0, 1e-6]) if theta in (0.0, 90.0) else np.empty(0),
                        accepted_steps=2,
                        rejected_steps=0,
                        matvecs=48,
                        maximum_krylov_dimension=24,
                        minimum_step_us=0.01,
                        maximum_step_us=0.1,
                        accumulated_state_error=1e-6,
                        maximum_norm_error=1e-12,
                        elapsed_seconds=1.0,
                    )
                    save_cell(result, output, config.configuration_hash)
            certification = create_report(output, angles, 2, 2, config, None)
            self.assertFalse(certification["certified"])
            for name in (
                "quantum_needle_statevector_summary.csv",
                "quantum_needle_statevector_cells.csv",
                "diagnostic_traces.csv",
                "curve_summary.csv",
                "quantum_needle_statevector_fig1b.png",
                "uncertainty_components.png",
                "quantum_needle_statevector_metadata.json",
                "certification.json",
            ):
                self.assertTrue((output / name).exists(), name)


class ExploratoryValidationTests(unittest.TestCase):
    def test_budget_fallback_and_tail_horizon(self) -> None:
        config = ValidationConfig()
        self.assertAlmostEqual(
            np.exp(-config.maximum_time_us / config.lifetime_us),
            config.survival_tail_bound,
            places=14,
        )
        preferred, _ = choose_workload(config, throughput=1.0)
        self.assertEqual(preferred.sweep_angles_deg, PREFERRED_SWEEP)
        fallback, _ = choose_workload(config, throughput=0.1)
        self.assertEqual(fallback.sweep_angles_deg, FALLBACK_SWEEP)
        self.assertEqual(fallback.lifetime_us, 2.0)

    def test_reduced_nonfactorizable_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = reduced_validation(Path(directory), seed=9917)
        self.assertTrue(summary["passed"])
        self.assertLess(summary["exact_vs_exhaustive_max_error"], 1e-12)
        self.assertLess(summary["krylov_vs_dense_max_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
