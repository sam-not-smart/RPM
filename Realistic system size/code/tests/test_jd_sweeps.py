from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rpm_quantum_needle_jd_sweeps import (  # noqa: E402
    EFIMOVA_HORE_NEGATIVE_RATIOS,
    EFIMOVA_HORE_POSITIVE_RATIOS,
    PUBLISHED_DIPOLE_TENSOR_MHZ,
    QUADRUPOLE_SCALE,
    R2_D_PAPER_UT,
    SCREEN_ANGLES_DEG,
    CouplingCondition,
    SweepConfig,
    TrajectoryTask,
    checkpoint_path,
    coarse_grid_indices,
    create_ratio_focus_report,
    create_ratio_lifetime_report,
    create_report,
    dipole_condition,
    ensure_manifest,
    ensure_efimova_hore_stage_definition,
    efimova_hore_dipole_only_control,
    efimova_hore_ratio_condition,
    efimova_hore_screen_conditions,
    efimova_hore_uncoupled_control,
    equivalent_field_uT_to_MHz,
    load_cell,
    low_exchange_condition,
    paper_to_code_dipolar_uT,
    ratio_condition,
    run_tasks,
    save_cell,
    select_efimova_hore_confirm_conditions,
    scaled_published_dipolar_tensor_MHz,
    unique_principal_value,
)
from rpm_quantum_needle_model import SYSTEM_DIMENSIONS, spin_operators  # noqa: E402
from rpm_statevector_engine import (  # noqa: E402
    AdaptiveLanczosPropagator,
    LocalTerm,
    MatrixFreeHamiltonian,
    SingletPopulationObservable,
    build_quantum_needle_hamiltonian,
    efimova_hore_electron_coupling_operator,
    singlet_initial_state,
    singlet_population,
    two_spin_coupling_operator,
)


class CouplingConventionTests(unittest.TestCase):
    def test_literal_efimova_hore_condition_grid_and_centres(self) -> None:
        conditions = efimova_hore_screen_conditions()
        self.assertEqual(len(conditions), 16)
        self.assertEqual(conditions[0], efimova_hore_uncoupled_control())
        self.assertEqual(conditions[1], efimova_hore_dipole_only_control())
        self.assertIn(-3.0, EFIMOVA_HORE_NEGATIVE_RATIOS)
        self.assertIn(6.0, EFIMOVA_HORE_POSITIVE_RATIOS)
        self.assertAlmostEqual(
            efimova_hore_ratio_condition(-3.0).exchange_J_uT,
            115.95833333333333,
        )
        self.assertAlmostEqual(
            efimova_hore_ratio_condition(6.0).exchange_J_uT,
            -57.979166666666664,
        )

    def test_literal_operators_match_equivalent_pair_forms(self) -> None:
        axis = np.array([0.2, -0.3, 0.7])
        axis /= np.linalg.norm(axis)
        exchange = 0.37
        dipolar = -0.51
        literal_exchange = efimova_hore_electron_coupling_operator(
            exchange, 0.0, axis
        )
        legacy_exchange = two_spin_coupling_operator(
            0.5, 0.5, -2.0 * exchange * np.eye(3)
        )
        np.testing.assert_allclose(
            literal_exchange - legacy_exchange,
            -0.5 * exchange * np.eye(4),
            atol=1e-14,
        )
        literal_dipolar = efimova_hore_electron_coupling_operator(
            0.0, dipolar, axis
        )
        equivalent_tensor = 2.0 * dipolar * (
            np.outer(axis, axis) - np.eye(3) / 3.0
        )
        np.testing.assert_allclose(
            literal_dipolar,
            two_spin_coupling_operator(0.5, 0.5, equivalent_tensor),
            atol=1e-14,
        )

    def test_literal_paper_crossings_are_at_minus_three_and_plus_six(self) -> None:
        singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
        triplet_x_zero = np.array([1.0, 0.0, 0.0, -1.0], dtype=complex) / np.sqrt(2.0)
        plus_x = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
        triplet_x_plus = np.kron(plus_x, plus_x)

        def energy(state: np.ndarray, d_over_j: float) -> float:
            operator = efimova_hore_electron_coupling_operator(
                1.0, d_over_j, np.array([1.0, 0.0, 0.0])
            )
            return float(np.vdot(state, operator @ state).real)

        self.assertAlmostEqual(energy(singlet, -3.0), energy(triplet_x_zero, -3.0), places=13)
        self.assertAlmostEqual(energy(singlet, 6.0), energy(triplet_x_plus, 6.0), places=13)

    def test_one_proton_model_recovers_mixing_at_paper_crossings(self) -> None:
        sx, sy, sz = spin_operators(0.5)
        identity = np.eye(2, dtype=complex)
        singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
        singlet_projector = np.outer(singlet, singlet.conj())
        initial_density = np.kron(singlet_projector, identity / 2.0)
        measurement = np.kron(singlet_projector, identity)
        conversion = equivalent_field_uT_to_MHz(1.0)

        def exact_yield(d_over_j: float) -> float:
            exchange_uT = 500.0
            pair = efimova_hore_electron_coupling_operator(
                exchange_uT * conversion,
                d_over_j * exchange_uT * conversion,
                np.array([1.0, 0.0, 0.0]),
            )
            hamiltonian = 2.0 * np.pi * np.kron(pair, identity)
            for operator in (sx, sy, sz):
                hamiltonian += (
                    2.0
                    * np.pi
                    * 500.0
                    * conversion
                    * np.kron(np.kron(operator, identity), operator)
                )
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
            density_eigenbasis = eigenvectors.conj().T @ initial_density @ eigenvectors
            measurement_eigenbasis = eigenvectors.conj().T @ measurement @ eigenvectors
            rate = 0.5  # tau=2 us
            frequencies = eigenvalues[:, None] - eigenvalues[None, :]
            return float(
                np.real(
                    np.sum(
                        density_eigenbasis
                        * measurement_eigenbasis.T
                        * rate
                        / (rate + 1j * frequencies)
                    )
                )
            )

        exchange_blocked = exact_yield(0.0)
        self.assertLess(exact_yield(-3.0), exchange_blocked - 0.3)
        self.assertLess(exact_yield(6.0), exchange_blocked - 0.3)

    def test_zero_literal_coupling_matches_legacy_zero_control(self) -> None:
        literal = build_quantum_needle_hamiltonian(
            37.0,
            exchange_J_MHz=0.0,
            efimova_hore_D_MHz=0.0,
            efimova_hore_axis=np.array([1.0, 0.0, 0.0]),
            quadrupole_scale=0.0,
        )
        legacy = build_quantum_needle_hamiltonian(
            37.0,
            exchange_J_MHz=0.0,
            dipolar_tensor_MHz=np.zeros((3, 3)),
            quadrupole_scale=0.0,
        )
        rng = np.random.default_rng(71)
        vector = rng.normal(size=literal.dimension) + 1j * rng.normal(size=literal.dimension)
        np.testing.assert_allclose(literal.matvec(vector), legacy.matvec(vector), atol=0.0, rtol=0.0)

    def test_efimova_hore_conversion_and_two_nm_anchor(self) -> None:
        self.assertAlmostEqual(R2_D_PAPER_UT, -347.875, places=12)
        self.assertAlmostEqual(paper_to_code_dipolar_uT(R2_D_PAPER_UT), -463.8333333333333)
        self.assertAlmostEqual(
            equivalent_field_uT_to_MHz(paper_to_code_dipolar_uT(R2_D_PAPER_UT)),
            -12.987333333333334,
        )
        self.assertAlmostEqual(ratio_condition(-4.0).exchange_J_uT, 115.95833333333333)
        self.assertAlmostEqual(ratio_condition(8.0).exchange_J_uT, -57.979166666666664)

    def test_full_tensor_scaling_preserves_axes_trace_and_rhombicity(self) -> None:
        target = -12.987333333333334
        scaled = scaled_published_dipolar_tensor_MHz(target)
        self.assertAlmostEqual(float(np.trace(scaled)), 0.0, places=13)
        self.assertAlmostEqual(unique_principal_value(scaled), target, places=12)
        source_values, source_axes = np.linalg.eigh(PUBLISHED_DIPOLE_TENSOR_MHZ)
        scaled_values, scaled_axes = np.linalg.eigh(scaled)
        overlaps = np.abs(source_axes.conj().T @ scaled_axes)
        np.testing.assert_allclose(overlaps, np.eye(3), atol=1e-12)
        source_ratio = (source_values[2] - source_values[1]) / abs(source_values[0])
        scaled_ratio = (scaled_values[2] - scaled_values[1]) / abs(scaled_values[0])
        self.assertAlmostEqual(float(source_ratio), float(scaled_ratio), places=13)

    def test_axial_cancellation_degeneracies_in_code_convention(self) -> None:
        singlet = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
        triplet_zero = np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2.0)
        triplet_plus = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        triplet_minus = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)

        def energy(state: np.ndarray, d_parallel: float, exchange: float) -> float:
            tensor = np.diag([-0.5 * d_parallel, -0.5 * d_parallel, d_parallel])
            operator = two_spin_coupling_operator(
                0.5, 0.5, tensor - 2.0 * exchange * np.eye(3)
            )
            return float(np.vdot(state, operator @ state).real)

        exchange = 0.37
        d_parallel = -4.0 * exchange
        self.assertAlmostEqual(
            energy(singlet, d_parallel, exchange),
            energy(triplet_zero, d_parallel, exchange),
            places=13,
        )
        d_parallel = 8.0 * exchange
        singlet_energy = energy(singlet, d_parallel, exchange)
        self.assertAlmostEqual(singlet_energy, energy(triplet_plus, d_parallel, exchange), places=13)
        self.assertAlmostEqual(singlet_energy, energy(triplet_minus, d_parallel, exchange), places=13)

    def test_sweep_hamiltonian_is_explicitly_quadrupole_free(self) -> None:
        self.assertEqual(QUADRUPOLE_SCALE, 0.0)
        condition = low_exchange_condition(10.0)
        hamiltonian = build_quantum_needle_hamiltonian(
            90.0,
            exchange_J_MHz=condition.exchange_J_MHz,
            dipolar_tensor_MHz=condition.dipolar_tensor_MHz,
            quadrupole_scale=QUADRUPOLE_SCALE,
        )
        self.assertFalse(any("quadrupole" in term.label for term in hamiltonian.terms))
        self.assertEqual(len([term for term in hamiltonian.terms if term.axes == (0, 1)]), 1)


class ProjectedObservableTests(unittest.TestCase):
    def test_basis_projection_matches_full_state_to_twelve_digits(self) -> None:
        rng = np.random.default_rng(19)
        dimensions = (2, 2, 3)
        raw = rng.normal(size=(12, 7)) + 1j * rng.normal(size=(12, 7))
        basis, _ = np.linalg.qr(raw)
        coefficients = rng.normal(size=7) + 1j * rng.normal(size=7)
        state_norm = 0.83
        observable = SingletPopulationObservable(dimensions)
        projected = observable.project_basis(basis)
        fast = observable.evaluate_projected(projected, coefficients, state_norm)
        direct = observable(state_norm * (basis @ coefficients))
        self.assertAlmostEqual(fast, direct, places=12)

    def test_projected_and_direct_lanczos_observables_agree(self) -> None:
        sx, sy, sz = spin_operators(0.5)
        terms = (
            LocalTerm((0,), 0.8 * sx + 0.3 * sz),
            LocalTerm((1, 2), 0.5 * np.kron(sy, sx) + 0.2 * np.kron(sz, sz)),
        )
        initial = singlet_initial_state((2, 2, 2), np.random.default_rng(21))
        times = np.linspace(0.0, 0.8, 81)

        def propagate(observable):
            hamiltonian = MatrixFreeHamiltonian((2, 2, 2), terms)
            return AdaptiveLanczosPropagator(
                hamiltonian,
                minimum_dimension=5,
                maximum_dimension=8,
                dimension_increment=1,
                embedded_offset=2,
                state_error_budget=1e-11,
                initial_step_us=0.4,
                maximum_step_us=0.4,
            ).propagate(initial, times, observable)

        direct = propagate(lambda state: singlet_population(state, (2, 2, 2)))
        projected = propagate(SingletPopulationObservable((2, 2, 2)))
        np.testing.assert_allclose(projected.observable_values, direct.observable_values, atol=1e-12)
        np.testing.assert_allclose(projected.final_state, direct.final_state, atol=1e-12)
        self.assertEqual(projected.stats.accepted_steps, direct.stats.accepted_steps)
        self.assertEqual(projected.stats.rejected_steps, direct.stats.rejected_steps)


class CheckpointAndManifestTests(unittest.TestCase):
    def _synthetic_result(self, condition: CouplingCondition) -> dict[str, object]:
        return {
            "condition_id": condition.condition_id,
            "condition_hash": condition.configuration_hash,
            "theta_deg": 90.0,
            "sample_id": 0,
            "quadrupole_scale": 0.0,
            "initial_singlet": 1.0,
            "yield": 0.3,
            "coarse_yield": 0.300001,
            "yield_quadrature_error": 1e-6,
            "yield_propagation_bound": 2e-6,
            "yield_tail_bound": 1e-4,
            "times_us": np.array([0.0, 1.0]),
            "singlet_population": np.array([1.0, 0.2]),
            "population_bounds": np.array([0.0, 1e-6]),
            "construction_seconds": 0.1,
            "propagation_seconds": 1.0,
            "total_seconds": 1.1,
            "peak_rss_mib": 10.0,
            "accepted_steps": 2,
            "rejected_steps": 0,
            "matvecs": 48,
            "maximum_krylov_dimension": 24,
            "minimum_step_us": 0.05,
            "maximum_step_us": 0.1,
            "accumulated_state_error": 1e-6,
            "maximum_norm_error": 1e-12,
            "hamiltonian_matvecs": 48,
        }

    def test_checkpoint_roundtrip_and_condition_rejection(self) -> None:
        config = SweepConfig()
        condition = dipole_condition(50.0)
        task = TrajectoryTask(condition, 90.0, 0, config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_cell(output, task, self._synthetic_result(condition))
            self.assertTrue(checkpoint_path(output, condition, 90.0, 0).exists())
            loaded = load_cell(output, condition, 90.0, 0, config)
            self.assertAlmostEqual(float(loaded["yield"]), 0.3)
            wrong = dipole_condition(75.0)
            wrong_path = checkpoint_path(output, wrong, 90.0, 0)
            wrong_path.parent.mkdir(parents=True)
            wrong_path.write_bytes(checkpoint_path(output, condition, 90.0, 0).read_bytes())
            with self.assertRaises(RuntimeError):
                load_cell(output, wrong, 90.0, 0, config)

    def test_manifest_resume_and_configuration_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = SweepConfig()
            manifest = ensure_manifest(output, config, resume=False)
            self.assertEqual(manifest["quadrupole_scale"], 0.0)
            resumed = ensure_manifest(output, config, resume=True)
            self.assertEqual(resumed["configuration_hash"], config.configuration_hash)
            with self.assertRaises(RuntimeError):
                ensure_manifest(output, SweepConfig(seed=config.seed + 1), resume=True)

    def test_efimova_hore_stage_definition_and_screen_selection(self) -> None:
        config = SweepConfig()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            definition = ensure_efimova_hore_stage_definition(output)
            self.assertEqual(definition["new_trajectory_budget"], 366)
            self.assertEqual(
                ensure_efimova_hore_stage_definition(output)["definition_hash"],
                definition["definition_hash"],
            )
            for condition_index, condition in enumerate(efimova_hore_screen_conditions()):
                for theta in SCREEN_ANGLES_DEG:
                    task = TrajectoryTask(condition, theta, 0, config)
                    result = self._synthetic_result(condition)
                    result["theta_deg"] = theta
                    result["yield"] = (
                        0.30
                        + (condition_index + 1) * 1e-7 * abs(theta - 90.0)
                    )
                    save_cell(output, task, result)
            selected = select_efimova_hore_confirm_conditions(output, config)
            self.assertEqual(len(selected), 6)
            self.assertIn(efimova_hore_ratio_condition(-3.0), selected)
            self.assertIn(efimova_hore_ratio_condition(6.0), selected)

    def test_fine_and_coarse_grids_are_nested(self) -> None:
        config = SweepConfig()
        indices = coarse_grid_indices(config)
        fine = config.time_grid_us
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], fine.size - 1)
        np.testing.assert_allclose(
            np.diff(fine[indices[:-1]]),
            config.coarse_integration_step_us,
            atol=2e-12,
        )

    def test_walltime_guard_stops_before_starting_a_cell(self) -> None:
        config = SweepConfig()
        condition = dipole_condition(50.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            complete = run_tasks(
                output,
                (condition,),
                (90.0,),
                1,
                config,
                workers=1,
                deadline=time.time() - 1.0,
                estimated_wave_seconds=1.0,
            )
            self.assertFalse(complete)
            self.assertFalse(checkpoint_path(output, condition, 90.0, 0).exists())

    def test_synthetic_report_writes_tables_and_plots(self) -> None:
        config = SweepConfig()
        conditions = (
            dipole_condition(0.0),
            dipole_condition(50.0),
            low_exchange_condition(0.0),
            low_exchange_condition(10.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for condition_index, condition in enumerate(conditions):
                for theta in SCREEN_ANGLES_DEG:
                    task = TrajectoryTask(condition, theta, 0, config)
                    result = self._synthetic_result(condition)
                    result["theta_deg"] = theta
                    result["yield"] = 0.30 + 1e-4 * condition_index + 1e-5 * abs(theta - 90.0)
                    save_cell(output, task, result)
            candidates = create_report(output, config)
            self.assertIn("best_physical_nonzero_exchange_baseline", candidates)
            for name in (
                "yield_cells.csv",
                "angular_curves.csv",
                "condition_metrics.csv",
                "candidate_summary.json",
                "metadata.json",
                "dipole_angular_yields.png",
                "exchange_angular_yields.png",
                "anisotropy_vs_dipole.png",
                "dipole_angle_heatmap.png",
            ):
                self.assertTrue((output / name).exists(), name)

    def test_focused_ratio_report_uses_requested_sample_count(self) -> None:
        config = SweepConfig()
        condition = ratio_condition(5.0)
        control = low_exchange_condition(0.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for current in (condition, control):
                for theta in SCREEN_ANGLES_DEG:
                    for sample in range(2):
                        task = TrajectoryTask(current, theta, sample, config)
                        result = self._synthetic_result(current)
                        result["theta_deg"] = theta
                        result["sample_id"] = sample
                        result["yield"] = (
                            0.30
                            + 1e-5 * abs(theta - 90.0)
                            + 2e-6 * sample
                        )
                        save_cell(output, task, result)
            summaries = create_ratio_focus_report(
                output, (condition,), SCREEN_ANGLES_DEG, 2, config
            )
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["samples"], 2)
            self.assertEqual(summaries[0]["comparison_control_samples"], 2)
            for name in (
                "ratio_focus_yield_cells.csv",
                "ratio_focus_angular_curves.csv",
                "ratio_focus_condition_metrics.csv",
                "ratio_focus_summary.json",
                "ratio_focus_angular_yields.png",
            ):
                self.assertTrue((output / name).exists(), name)

    def test_multi_lifetime_report_uses_shared_checkpoint_yields(self) -> None:
        config = SweepConfig(lifetime_us=10.0)
        condition = ratio_condition(8.0)
        lifetimes = (5.0, 10.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for theta in SCREEN_ANGLES_DEG:
                for sample in range(2):
                    task = TrajectoryTask(condition, theta, sample, config, lifetimes)
                    result = self._synthetic_result(condition)
                    values = np.array(
                        [
                            0.29 + 1e-5 * abs(theta - 90.0) + 2e-6 * sample,
                            0.27 + 2e-5 * abs(theta - 90.0) + 3e-6 * sample,
                        ]
                    )
                    result.update(
                        {
                            "theta_deg": theta,
                            "sample_id": sample,
                            "yield": values[-1],
                            "yield_lifetimes_us": np.array(lifetimes),
                            "yields": values,
                            "coarse_yields": values - 1e-6,
                            "yield_quadrature_errors": np.full(2, 1e-6),
                            "yield_propagation_bounds": np.full(2, 2e-6),
                            "yield_tail_bounds": np.array([1e-8, 1e-4]),
                        }
                    )
                    save_cell(output, task, result)
            summaries = create_ratio_lifetime_report(
                output, (condition,), SCREEN_ANGLES_DEG, 2, lifetimes, config
            )
            self.assertEqual(len(summaries), 2)
            self.assertEqual([row["lifetime_us"] for row in summaries], [5.0, 10.0])
            self.assertTrue(
                (output / "ratio_lifetime_angular_yields.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
