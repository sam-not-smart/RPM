from __future__ import annotations

import math
import unittest

import numpy as np
import scipy.integrate

from poster_reproduction.core import (
    MASTER_SEED,
    PRODUCTION_MODEL,
    angular_metrics,
    bootstrap_curve_metric,
    build_hamiltonian,
    dense_singlet_dynamics,
    dense_truncated_singlet_yield,
    electron_only_hamiltonian,
    dimensions_for,
    electron_spectrum_manifest,
    full_product_basis_indices,
    mixed_two_electron_log_negativity,
    mean_se_ci,
    n10_principal_axis,
    n10_transverse_deleted_tensor,
    overhauser_covariances_mhz2,
    overhauser_sample_mhz,
    paired_product_indices,
    partial_transpose_two_qubits,
    product_basis_initial_state,
    pure_log_negativity,
    random_phase_initial_state,
    reduced_electron_density,
    sample_seed,
    scrambled_sobol_overhauser_samples_mhz,
    selected_nuclei,
    selected_radicals,
    spin_pi_pulse,
)
from rpm_quantum_needle_jd_sweeps import integrate_yield
from rpm_quantum_needle_model import FAD_NUCLEI, SYSTEM_DIMENSIONS, spin_operators
from rpm_statevector_engine import singlet_population
from poster_reproduction.runner import planned_tasks


class PosterReproductionTests(unittest.TestCase):
    def test_hamiltonian_branches_are_hermitian(self):
        branches = [
            build_hamiltonian(0.0, 0.05),
            build_hamiltonian(
                90.0,
                0.05,
                hyperfine_overrides_mT={"FAD:N10": n10_transverse_deleted_tensor()},
            ),
            build_hamiltonian(
                90.0,
                0.05,
                nuclei=selected_nuclei(4),
                radicals=selected_radicals(4),
            ),
        ]
        for hamiltonian in branches:
            for term in hamiltonian.terms:
                self.assertLess(np.linalg.norm(term.operator - term.operator.conj().T), 1e-11)

    def test_random_phase_initial_state_normalization_and_ps_zero(self):
        state = random_phase_initial_state(SYSTEM_DIMENSIONS, MASTER_SEED, 3)
        self.assertAlmostEqual(float(np.linalg.norm(state)), 1.0, places=12)
        self.assertAlmostEqual(singlet_population(state, SYSTEM_DIMENSIONS), 1.0, places=12)

    def test_production_parameters_are_frozen(self):
        condition = PRODUCTION_MODEL.condition
        self.assertAlmostEqual(condition.d_paper_uT, -347.875)
        self.assertAlmostEqual(condition.exchange_J_uT, 115.95833333333333)
        self.assertAlmostEqual(condition.ratio_D_over_J, -3.0)
        self.assertEqual(PRODUCTION_MODEL.quadrupole_scale, 0.0)
        self.assertEqual(PRODUCTION_MODEL.lifetime_us, 5.0)

    def test_simpson_yield_matches_constant_population_integral(self):
        config = PRODUCTION_MODEL.sweep_config
        times = config.time_grid_us
        value = integrate_yield(times, np.ones_like(times), config.lifetime_us)
        exact_truncated = 1.0 - math.exp(-times[-1] / config.lifetime_us)
        self.assertLess(abs(value - exact_truncated), 2e-11)
        decimated = scipy.integrate.simpson(
            np.exp(-times[::2] / config.lifetime_us) / config.lifetime_us,
            x=times[::2],
        )
        self.assertLess(abs(value - decimated), 2e-10)
        electron_hamiltonian = electron_only_hamiltonian(
            37.0,
            2.0,
            (np.array([1.2, -0.7, 0.3]), np.array([-0.4, 0.1, 0.9])),
        )
        populations = dense_singlet_dynamics(electron_hamiltonian, times)
        simpson = integrate_yield(times, populations, config.lifetime_us)
        analytic = dense_truncated_singlet_yield(
            electron_hamiltonian, config.lifetime_us, times[-1]
        )
        self.assertLess(abs(simpson - analytic), 2e-9)

    def test_sample_ids_and_seeds_are_reproducible(self):
        self.assertEqual(sample_seed(MASTER_SEED, 7), sample_seed(MASTER_SEED, 7))
        self.assertNotEqual(sample_seed(MASTER_SEED, 7), sample_seed(MASTER_SEED, 8))
        first = full_product_basis_indices(MASTER_SEED, 7)
        second = full_product_basis_indices(MASTER_SEED, 7)
        self.assertEqual(first, second)

    def test_mean_standard_error_and_confidence_interval(self):
        values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
        mean, se, low, high = mean_se_ci(values)
        self.assertTrue(np.allclose(mean, [2.5, 5.0]))
        self.assertTrue(np.all(se > 0.0))
        self.assertTrue(np.all(low < mean))
        self.assertTrue(np.all(high > mean))
        self.assertTrue(np.allclose(mean - low, high - mean))

    def test_empirical_overhauser_moments(self):
        count = 12000
        fad = np.empty((count, 3))
        trp = np.empty((count, 3))
        for sample_id in range(count):
            fad[sample_id], trp[sample_id] = overhauser_sample_mhz(MASTER_SEED, sample_id)
        expected = overhauser_covariances_mhz2()
        for values, radical in ((fad, "FAD"), (trp, "TrpH")):
            self.assertLess(np.linalg.norm(np.mean(values, axis=0)), 0.07 * math.sqrt(np.trace(expected[radical])))
            empirical = np.cov(values, rowvar=False, ddof=1)
            relative = np.linalg.norm(empirical - expected[radical]) / np.linalg.norm(expected[radical])
            self.assertLess(relative, 0.06)

    def test_scrambled_sobol_overhauser_replicates_are_deterministic(self):
        first = scrambled_sobol_overhauser_samples_mhz(MASTER_SEED, 0, 1024)
        second = scrambled_sobol_overhauser_samples_mhz(MASTER_SEED, 0, 1024)
        third = scrambled_sobol_overhauser_samples_mhz(MASTER_SEED, 1, 1024)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertFalse(np.array_equal(first[0], third[0]))
        expected = overhauser_covariances_mhz2()
        for values, radical in ((first[0], "FAD"), (first[1], "TrpH")):
            empirical = np.cov(values, rowvar=False, ddof=1)
            relative = np.linalg.norm(empirical - expected[radical]) / np.linalg.norm(expected[radical])
            self.assertLess(relative, 0.03)

    def test_spin_one_pulse_unitarity_and_toggling(self):
        pulse = spin_pi_pulse(1.0, np.array([0.0, 0.0, 1.0]))
        identity = np.eye(3)
        self.assertLess(np.linalg.norm(pulse.conj().T @ pulse - identity), 1e-12)
        ix, iy, iz = spin_operators(1.0)
        for operator, expected in ((ix, -ix), (iy, -iy), (iz, iz)):
            toggled = pulse.conj().T @ operator @ pulse
            self.assertLess(np.linalg.norm(toggled - expected), 1e-12)

    def test_n10_axis_and_recovered_transverse_deletion(self):
        self.assertLess(np.linalg.norm(n10_principal_axis() - np.array([0.0, 0.0, 1.0])), 1e-12)
        deleted = n10_transverse_deleted_tensor()
        self.assertTrue(np.all(deleted[:, :2] == 0.0))
        self.assertTrue(np.array_equal(deleted[:, 2], FAD_NUCLEI[1].tensor_mT[:, 2]))

    def test_known_log_negativities(self):
        product = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
        self.assertAlmostEqual(pure_log_negativity(product, (2, 2), (0,)), 0.0, places=12)
        self.assertAlmostEqual(pure_log_negativity(bell, (2, 2), (0,)), 1.0, places=12)
        rho_product = np.outer(product, product.conj())
        rho_bell = np.outer(bell, bell.conj())
        self.assertAlmostEqual(mixed_two_electron_log_negativity(rho_product), 0.0, places=12)
        self.assertAlmostEqual(mixed_two_electron_log_negativity(rho_bell), 1.0, places=12)

    def test_negativity_physical_bounds(self):
        rng = np.random.default_rng(12)
        state = rng.normal(size=4) + 1.0j * rng.normal(size=4)
        state /= np.linalg.norm(state)
        value = pure_log_negativity(state, (2, 2), (0,))
        self.assertGreaterEqual(value, -1e-12)
        self.assertLessEqual(value, 1.0 + 1e-12)
        rho = np.eye(4) / 4.0
        self.assertGreaterEqual(mixed_two_electron_log_negativity(rho), -1e-12)
        self.assertLessEqual(mixed_two_electron_log_negativity(rho), 1.0 + 1e-12)

    def test_product_nuclear_initial_entanglement(self):
        nuclei = selected_nuclei(4)
        dimensions = dimensions_for(nuclei)
        indices = paired_product_indices(4, full_product_basis_indices(MASTER_SEED, 0))
        state = product_basis_initial_state(dimensions, indices)
        radical_left = (0, 2, 3)
        self.assertAlmostEqual(pure_log_negativity(state, dimensions, radical_left), 1.0, places=12)
        self.assertAlmostEqual(pure_log_negativity(state, dimensions, (0, 1)), 0.0, places=12)

    def test_reduced_electron_density_is_physical(self):
        dimensions = dimensions_for(selected_nuclei(4))
        states = []
        for sample_id in range(8):
            indices = paired_product_indices(4, full_product_basis_indices(MASTER_SEED, sample_id))
            states.append(product_basis_initial_state(dimensions, indices))
        rho = np.mean([reduced_electron_density(state, dimensions) for state in states], axis=0)
        self.assertLess(np.linalg.norm(rho - rho.conj().T), 1e-12)
        self.assertAlmostEqual(float(np.trace(rho).real), 1.0, places=12)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(rho))), -1e-12)
        self.assertLess(np.linalg.norm(partial_transpose_two_qubits(rho) - partial_transpose_two_qubits(rho).conj().T), 1e-12)

    def test_bootstrap_uses_paired_whole_curves(self):
        base = np.array([0.2, 0.3, 0.4])
        offsets = np.array([-0.03, -0.01, 0.01, 0.03])
        curves = base[None, :] + offsets[:, None]
        metric = lambda curve: angular_metrics(np.array([0.0, 1.0, 2.0]), curve)["contrast"]
        first = bootstrap_curve_metric(curves, metric, seed=77, replicates=100)
        second = bootstrap_curve_metric(curves, metric, seed=77, replicates=100)
        self.assertTrue(np.array_equal(first[3], second[3]))
        # Paired sample offsets cancel inside every reconstructed angular curve.
        self.assertLess(np.ptp(first[3]), 1e-12)

    def test_electron_only_cancellation_is_exact(self):
        spectrum = electron_spectrum_manifest()
        self.assertTrue(spectrum["cancellation_exact"])
        self.assertLess(spectrum["minimum_adjacent_gap_MHz"], 1e-12)

    def test_complete_pilot_inventory_is_not_silently_reduced(self):
        tasks = planned_tasks("pilot", 16, 4096, MASTER_SEED)
        counts = {kind: sum(task.kind == kind for task in tasks) for kind in ("quantum", "overhauser", "entanglement", "n10")}
        self.assertEqual(counts["quantum"], 12 * 19 * 16)
        self.assertEqual(counts["overhauser"], 12 * 19)
        self.assertEqual(counts["entanglement"], 6 * 16)
        self.assertEqual(counts["n10"], 201 * 5 * 16)
        overhauser_only = [task for task in tasks if task.kind == "overhauser"]
        self.assertEqual(len(overhauser_only), 12 * 19)
        self.assertTrue(all(task.kind == "overhauser" for task in overhauser_only))


if __name__ == "__main__":
    unittest.main()
