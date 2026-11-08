from __future__ import annotations

import math
import unittest

import numpy as np

from rpm_two_nucleus_optimum_reproduction import (
    REAL_N5_B_UT,
    shape_lifetime_us,
    transverse_tensor_mT,
)


class TwoNucleusOptimumReproductionTests(unittest.TestCase):
    def test_transverse_tensor_has_requested_physical_magnitude(self) -> None:
        tensor = transverse_tensor_mT(1.7569, REAL_N5_B_UT)
        magnitude_uT = 1000.0 * math.hypot(tensor[0, 0], tensor[1, 1])
        self.assertAlmostEqual(magnitude_uT, REAL_N5_B_UT, places=12)
        self.assertLess(tensor[0, 0], 0.0)
        self.assertAlmostEqual(tensor[2, 2], 1.7569)

    def test_shape_lifetime_matches_closed_form(self) -> None:
        fitted_c = 391.0
        x = np.array([0.848, 0.907])
        expected = (
            (fitted_c / 1000.0) * (x + 2.0) / (math.sqrt(2.0) * np.abs(x - 1.0))
        ) ** 2
        np.testing.assert_allclose(shape_lifetime_us(x, fitted_c), expected, rtol=0.0, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
