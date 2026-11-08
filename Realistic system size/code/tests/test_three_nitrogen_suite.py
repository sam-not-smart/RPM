import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np

from rpm_three_nitrogen_optimum import PairModel, NITROGENS
from rpm_three_nitrogen_suite import (field_grid, transverse_grid, slope_max,
    ensure_manifest, save_cell, load_rows, local_peak_brackets, scaling_fit,
    require_complete_inventory)


class ThreeNitrogenSuiteTests(unittest.TestCase):
    def test_common_axial_scale_preserves_ratios_and_transverse_terms(self):
        full=PairModel("three_nitrogen")
        scaled=PairModel("three_nitrogen",axial_scale=.6046/1.7569)
        for old,new in zip(NITROGENS,scaled.nuclei):
            self.assertAlmostEqual(new[3]/old[3],.6046/1.7569)
            self.assertEqual(old[2],new[2])
        np.testing.assert_allclose(scaled.axial,full.axial*.6046/1.7569,atol=1e-13)
        np.testing.assert_allclose(scaled.transverse,full.transverse,atol=1e-13)

    def test_field_parameter_does_not_modify_hyperfine(self):
        low=PairModel("three_nitrogen",field_mT=.05)
        high=PairModel("three_nitrogen",field_mT=100)
        np.testing.assert_allclose(low.axial,high.axial)
        np.testing.assert_allclose(high.zeeman_x,low.zeeman_x*2000,atol=1e-11)

    def test_spectral_integral_against_independent_full_projector(self):
        import scipy.linalg
        model=PairModel("three_nitrogen",axial_scale=.7)
        h=model.hamiltonian(.4,78.3)
        e,v=scipy.linalg.eigh(h)
        s=np.array([0,1,-1,0])/math.sqrt(2)
        p=np.kron(np.outer(s,s),np.eye(27))
        w=np.abs(v.conj().T@p@v)**2/27
        gap=e[:,None]-e[None,:]
        expected=np.array([np.sum(w/(1+t*t*gap*gap)) for t in (5,29,100)])
        np.testing.assert_allclose(model.yields_from_hamiltonian(h,np.array([5,29,100])),expected,atol=1e-12)

    def test_spline_maximum_uses_radians(self):
        theta=np.linspace(0,90,901)
        phi=np.sin(2*np.deg2rad(theta))
        gamma,angle=slope_max(theta,phi)
        self.assertAlmostEqual(gamma,2,places=7)
        self.assertTrue(angle<.01 or angle>89.99)

    def test_manifest_rejects_incompatible_resume(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);ensure_manifest(root);ensure_manifest(root)
            p=root/"manifest.json";data=json.loads(p.read_text())
            data["configuration_hash"]="incompatible";p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                ensure_manifest(root)

    def test_atomic_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            save_cell(root,{"key":"test_cell","rows":[{"lambda":1.}],
                "theta_deg":np.array([0,90]),"curves":np.array([[.2,.3]]),"elapsed_seconds":1.})
            self.assertEqual(load_rows(root,"test_"),[{"lambda":1.}])
            self.assertFalse(list((root/"checkpoints").glob("*.tmp.npz")))

    def test_bounded_search_grid_and_all_large_local_peaks(self):
        self.assertIn(0,transverse_grid());self.assertIn(1,transverse_grid())
        self.assertEqual(field_grid()[0],.05);self.assertEqual(field_grid()[-1],100)
        rows=[{"lambda":i,"gamma_per_rad":g} for i,g in enumerate([1,10,2,9.5,1])]
        self.assertEqual(local_peak_brackets(rows),[(0,2),(2,4)])

    def test_scaling_diagnostic_accepts_exact_synthetic_law(self):
        rows=[{"Azz_N5_mT":a,"tau_us":t,"b_N5_uT":391*a/math.sqrt(t)}
            for a in (.6046,1.2,1.7569) for t in (5,10,20,50,100)]
        fit=scaling_fit(rows)
        self.assertTrue(fit["passes_5_percent_fixed_minus_half_law"])
        self.assertTrue(fit["passes_5_percent_collapse"])
        self.assertAlmostEqual(fit["fixed_minus_half_C"],391)

    def test_partial_results_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(RuntimeError,"incomplete suite"):
                require_complete_inventory(Path(name))


if __name__=="__main__":
    unittest.main()
