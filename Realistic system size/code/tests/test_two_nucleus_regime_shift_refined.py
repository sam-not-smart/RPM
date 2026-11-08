from __future__ import annotations

import unittest

import numpy as np

from rpm_quantum_needle_reproduction import (
    build_radical,
    diagonalize_radical,
    direct_line_yields,
)
from rpm_quantum_needle_model import Nucleus
from rpm_two_nucleus_regime_shift_refined import (
    CONDITIONS,
    HYPERFINE,
    N5_TENSOR_MT,
    TRP_TENSOR_MT,
    exact_singlet_yield,
    field_grid_mt,
)


class TwoNucleusRegimeShiftRefinedTests(unittest.TestCase):
    def test_full_pair_yield_matches_independent_radical_reference(self) -> None:
        toy_a = build_radical("X", (Nucleus("N", 1.0, N5_TENSOR_MT),))
        toy_b = build_radical("Y", (Nucleus("N", 1.0, TRP_TENSOR_MT),))
        for field_mT, theta_deg in ((0.05, 90.0), (0.60, 38.6), (10.0, 87.7)):
            spectrum_a = diagonalize_radical(toy_a, theta_deg, 1000.0 * field_mT)
            spectrum_b = diagonalize_radical(toy_b, theta_deg, 1000.0 * field_mT)
            reference = direct_line_yields(spectrum_a, spectrum_b, (10.0,))[10.0]
            actual = exact_singlet_yield("full", field_mT, theta_deg)
            self.assertAlmostEqual(actual, reference, places=12)

    def test_ablation_removes_only_n5_transverse_terms(self) -> None:
        delta = HYPERFINE["full"] - HYPERFINE["transverse_ablated"]
        self.assertGreater(np.linalg.norm(delta), 0.0)
        for condition in CONDITIONS:
            np.testing.assert_allclose(HYPERFINE[condition], HYPERFINE[condition].conj().T, atol=1e-13)

    def test_field_grid_is_dense_through_crossover(self) -> None:
        fields = field_grid_mt()
        local = fields[(fields >= 0.40) & (fields <= 0.80)]
        self.assertLessEqual(float(np.max(np.diff(local))), 0.0025 + 1e-12)
        for value in (0.05, 0.50, 0.60, 0.70, 100.0):
            self.assertTrue(np.any(np.isclose(fields, value)))


if __name__ == "__main__":
    unittest.main()
