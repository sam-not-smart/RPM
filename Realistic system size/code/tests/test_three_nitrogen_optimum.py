import unittest
import numpy as np
from rpm_three_nitrogen_optimum import PairModel, B_N5_AT_ONE_UT
from rpm_two_nucleus_optimum_reproduction import _yields_for_angles

class ThreeNitrogenTests(unittest.TestCase):
    def test_two_nucleus_control_matches_existing_solver(self):
        model=PairModel("two_nucleus_control")
        for theta in (0.,38.6,89.8,90.):
            actual=model.yields_from_hamiltonian(model.hamiltonian(1.,theta),np.array([10.]))[0]
            reference=_yields_for_angles(1.7569,B_N5_AT_ONE_UT,10.,np.array([theta]))[0]
            self.assertAlmostEqual(actual,reference,places=12)

    def test_three_nucleus_dimension_hermiticity_and_no_dynamics_limit(self):
        model=PairModel("three_nitrogen")
        self.assertEqual(model.dimension,108)
        np.testing.assert_allclose(model.hamiltonian(.7,43.),model.hamiltonian(.7,43.).T,atol=1e-13)
        yields=model.yields_from_hamiltonian(np.zeros((108,108)),np.array([1.,10.,100.]))
        np.testing.assert_allclose(yields,np.ones(3),atol=1e-13)

    def test_aligned_axes_reflection_symmetry_and_yield_bounds(self):
        model=PairModel("three_nitrogen")
        left=model.curve(.7,np.array([12.,83.7]))
        right=model.curve(.7,np.array([168.,96.3]))
        np.testing.assert_allclose(left,right,atol=1e-12)
        self.assertTrue(np.all((left>=0)&(left<=1)))

if __name__=="__main__":
    unittest.main()
