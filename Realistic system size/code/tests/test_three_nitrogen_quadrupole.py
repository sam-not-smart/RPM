import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import expm

from rpm_three_nitrogen_quadrupole import (BY_KEY,CONDITIONS,QuadrupoleModel,
    quadrupole_tensors,TAUS_US,ensure_manifest,save_cell,load_cell,require_complete)
from rpm_three_nitrogen_optimum import PairModel,embed
from rpm_quantum_needle_model import (QUADRUPOLE_PRINCIPAL_VALUES_MT,
    QUADRUPOLE_TENSORS_MT,TWOPI,GAMMA_E_MHZ_PER_MT)
from rpm_statevector_engine import quadrupole_operator


class ThreeNitrogenQuadrupoleTests(unittest.TestCase):
    def test_nine_approved_conditions_and_tensor_scaling(self):
        self.assertEqual(len(CONDITIONS),9)
        for c in CONDITIONS:
            tensors=quadrupole_tensors(c)
            for i,q in enumerate(tensors):
                self.assertAlmostEqual(np.trace(q),0.,places=14)
                np.testing.assert_allclose(q,q.T,atol=1e-15)
                self.assertEqual(np.any(q!=0),c.scale>0 and c.enabled[i])
        for half,full,double in zip(quadrupole_tensors(BY_KEY["all_half"]),
            quadrupole_tensors(BY_KEY["all_nominal"]),quadrupole_tensors(BY_KEY["all_double"])):
            np.testing.assert_allclose(half*2,full,atol=1e-15)
            np.testing.assert_allclose(double,full*2,atol=1e-15)

    def test_orientations_preserve_eigenvalues_and_n5_qz_tilt(self):
        reference=np.linalg.eigvalsh(quadrupole_tensors(BY_KEY["all_nominal"])[0])
        for key,phi in (("all_nominal",0.),("all_phi45",45.),("all_phi90",90.)):
            q=quadrupole_tensors(BY_KEY[key])[0]
            values,vectors=np.linalg.eigh(q)
            np.testing.assert_allclose(values,reference,atol=1e-15)
            unique_axis=vectors[:,values.argmin()]
            target=np.array([math.cos(math.radians(phi)),math.sin(math.radians(phi)),0.])
            self.assertAlmostEqual(abs(float(unique_axis@target)),1.,places=13)
        np.testing.assert_allclose(quadrupole_tensors(BY_KEY["all_nominal"])[1:],
            quadrupole_tensors(BY_KEY["all_phi45"])[1:],atol=1e-15)

    def test_original_centralized_tensors_are_unchanged(self):
        for key,principal in QUADRUPOLE_PRINCIPAL_VALUES_MT.items():
            expected=np.array(principal)-np.mean(principal)
            np.testing.assert_allclose(np.linalg.eigvalsh(QUADRUPOLE_TENSORS_MT[key]),
                np.sort(expected),atol=1e-15)

    def test_q_off_matches_old_model_at_both_fields(self):
        for b in (.05,3.):
            old=PairModel("three_nitrogen",field_mT=b)
            new=QuadrupoleModel(b,BY_KEY["off"])
            for angle in (0.,38.6,89.8,90.,143.):
                expected=old.curve(1.,np.array([angle]),TAUS_US)[:,0]
                np.testing.assert_allclose(new.yields(angle),expected,atol=1e-13)

    def test_physical_complex_operator_is_retained(self):
        m=QuadrupoleModel(.05,BY_KEY["all_phi45"])
        h=m.hamiltonian(83.7)
        self.assertGreater(abs(h.imag).max(),1.)
        np.testing.assert_allclose(h,h.conj().T,atol=1e-13)
        # The corrected shared solver must agree on a complex Hamiltonian.
        np.testing.assert_allclose(m.yields(83.7),m.base.yields_from_hamiltonian(h,TAUS_US),atol=1e-13)

    def test_trace_against_independent_density_matrix_exponentiation(self):
        m=QuadrupoleModel(3.,BY_KEY["all_phi45"],bridge=True)
        s=np.array([0.,1.,-1.,0.])/math.sqrt(2)
        p=np.kron(np.outer(s,s),np.eye(27));rho=p/27
        h=m.hamiltonian(81.3);t=np.array([0.,.013,.13,1.3])
        actual=m.trace(81.3,t)
        expected=[]
        for value in t:
            u=expm(-1j*h*value)
            expected.append(np.trace(p@u@rho@u.conj().T).real)
        np.testing.assert_allclose(actual,expected,atol=2e-12)

    def test_quadrupole_only_cannot_drive_electron_singlet(self):
        m=QuadrupoleModel(.05,BY_KEY["all_phi45"])
        np.testing.assert_allclose(m.base.yields_from_hamiltonian(m.extra,TAUS_US),1.,atol=1e-13)

    def test_commuting_axial_null(self):
        m=PairModel("three_nitrogen")
        h=m.hamiltonian(0.,54.)
        q=np.zeros_like(h,dtype=complex)
        for i,principal in enumerate(QUADRUPOLE_PRINCIPAL_VALUES_MT.values()):
            qz=principal[2]
            local=quadrupole_operator(1,TWOPI*GAMMA_E_MHZ_PER_MT*np.diag([-qz/2,-qz/2,qz]))
            q+=embed(local,i+2,m.dims)
        np.testing.assert_allclose(h@q,q@h,atol=1e-12)
        np.testing.assert_allclose(m.yields_from_hamiltonian(h.copy(),TAUS_US),
            m.yields_from_hamiltonian(h+q,TAUS_US),atol=1e-12)

    def test_deterministic_manifest_checkpoint_and_resume_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);ensure_manifest(root);ensure_manifest(root)
            save_cell(root,{"key":"angular_B0.05_off","theta_deg":np.array([0.,90.]),
                "curves":np.array([[.3,.2],[.2,.1]]),"rows":[{"condition":"off"}],"elapsed_seconds":1.})
            t,y,rows=load_cell(root,.05,"off")
            np.testing.assert_array_equal(t,[0,90]);self.assertEqual(y.shape,(2,2))
            self.assertEqual(rows,[{"condition":"off"}])
            with self.assertRaisesRegex(RuntimeError,"incomplete"):
                require_complete(root)
            path=root/"manifest.json";data=json.loads(path.read_text())
            data["configuration_hash"]="incompatible";path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"incompatible"):
                ensure_manifest(root)


if __name__=="__main__":unittest.main()
