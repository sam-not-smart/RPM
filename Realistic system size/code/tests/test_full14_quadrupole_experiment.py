import json
import math
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from pathlib import Path
import numpy as np

from rpm_full14_quadrupole_experiment import (Config,Task,tasks,pilot_tasks,smoke_config,
    system,propagate,ensure_manifest,atomic_npz,cell_path,run_tasks,paired_interval,
    remaining_wall_hours)
from poster_reproduction.core import build_hamiltonian,dimensions_for,random_phase_initial_state


class Full14QuadrupoleTests(unittest.TestCase):
    def test_locked_inventory_and_physics(self):
        cfg=Config();inventory=tasks(cfg)
        self.assertEqual(len(inventory),88)
        self.assertEqual(len({t.key for t in inventory}),88)
        self.assertEqual({t.sample_id for t in inventory},{0,1,2,3})
        self.assertEqual(len(pilot_tasks(cfg)),4)
        self.assertTrue(set(pilot_tasks(cfg)).issubset(set(inventory)))
        self.assertAlmostEqual(cfg.model(0).condition.d_paper_uT,-347.875)
        self.assertAlmostEqual(cfg.model(1).condition.exchange_J_uT,115.95833333333333)
        self.assertEqual(cfg.model(1).quadrupole_scale,1.)
        self.assertEqual(cfg.model(0).quadrupole_scale,0.)
        self.assertAlmostEqual(cfg.sweep.maximum_time_us,57.564627324851145)
        self.assertEqual(np.prod(dimensions_for(system(cfg)[0])),221184)

    def test_pairing_and_quadrupole_sites_in_small_fixture(self):
        cfg=smoke_config();nuclei,radicals=system(cfg);dims=dimensions_for(nuclei)
        a=random_phase_initial_state(dims,cfg.seed,0)
        b=random_phase_initial_state(dims,cfg.seed,0)
        np.testing.assert_array_equal(a,b)
        off=build_hamiltonian(90,.05,model=cfg.model(0),nuclei=nuclei,radicals=radicals)
        on=build_hamiltonian(90,.05,model=cfg.model(1),nuclei=nuclei,radicals=radicals)
        self.assertFalse(any("quadrupole" in t.label for t in off.terms))
        self.assertEqual({t.axes for t in on.terms if "quadrupole" in t.label},{(2,),(3,),(4,)})
        difference=on.explicit_matrix()-off.explicit_matrix()
        self.assertGreater(np.linalg.norm(difference),1)
        np.testing.assert_allclose(difference,difference.conj().T,atol=1e-12)

    def test_manifest_rejects_incompatible_config(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ensure_manifest(root,smoke_config());ensure_manifest(root,smoke_config())
            with self.assertRaisesRegex(ValueError,"incompatible"):
                ensure_manifest(root,Config())

    def test_expired_deadline_does_not_start_work(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=smoke_config()
            status=run_tasks(pilot_tasks(cfg),cfg,root,1,time.monotonic()-1,0,True,"test")
            self.assertEqual(status["completed"],0)
            self.assertEqual(status["status"],"wall_time_checkpoint")

    def test_small_trajectory_checkpoint_and_finite_values(self):
        cfg=smoke_config();task=Task(1.,90.,0);data=propagate(task,cfg)
        self.assertTrue(np.isfinite(float(data["yield"])))
        self.assertAlmostEqual(float(data["initial_singlet"]),1.,places=12)
        self.assertLess(abs(float(np.linalg.norm(data["final_state"]))-1),1e-10)
        self.assertTrue(np.all((data["singlet_population"]>=-1e-12)&(data["singlet_population"]<=1+1e-12)))
        with tempfile.TemporaryDirectory() as d:
            path=cell_path(Path(d),task);atomic_npz(path,data)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["final_state"],data["final_state"])
                self.assertEqual(json.loads(str(saved["metadata_json"]))["configuration_hash"],cfg.digest)

    def test_paired_intervals_are_deterministic_and_no_m1_interval(self):
        self.assertEqual(paired_interval([1]),[None,None])
        self.assertEqual(paired_interval([1,2,3,4]),paired_interval([1,2,3,4]))

    def test_shared_capacity_mode_still_honors_expired_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=smoke_config()
            with patch("rpm_full14_quadrupole_experiment.resource_blockers",return_value=[{"reason":"capacity"}]):
                status=run_tasks(pilot_tasks(cfg),cfg,root,1,time.monotonic()-1,0,True,"test","spare-capacity")
            self.assertEqual(status["completed"],0)
            self.assertEqual(status["status"],"wall_time_checkpoint")

    def test_unlimited_resume_preserves_completed_cells_and_ignores_time_guard(self):
        cfg=smoke_config();first,second=pilot_tasks(cfg)[:2]
        def values(elapsed):
            return {"yield":np.array(.25),"metadata_json":np.array(json.dumps({
                "configuration_hash":cfg.digest,"elapsed_seconds":elapsed}))}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);saved=cell_path(root,first);atomic_npz(saved,values(1e12))
            before=saved.read_bytes()
            with patch("rpm_full14_quadrupole_experiment.ProcessPoolExecutor",ThreadPoolExecutor), \
                    patch("rpm_full14_quadrupole_experiment.resource_blockers",return_value=[]), \
                    patch("rpm_full14_quadrupole_experiment.propagate",return_value=values(1.)) as work:
                result=run_tasks([first,second],cfg,root,1,math.inf,1e12,True,"test","spare-capacity")
            self.assertEqual(result["status"],"complete")
            self.assertEqual(result["completed"],2)
            self.assertEqual(saved.read_bytes(),before)
            work.assert_called_once_with(second,cfg,root,math.inf)
            status=json.loads((root/"status.json").read_text())
            self.assertFalse(status["wall_time_limit_enabled"])
            self.assertIsNone(status["remaining_wall_hours"])

    def test_remaining_wall_hours_is_json_safe_and_clamped(self):
        self.assertIsNone(remaining_wall_hours(math.inf))
        self.assertEqual(remaining_wall_hours(time.monotonic()-1),0.)
        json.dumps({"remaining_wall_hours":remaining_wall_hours(math.inf)},allow_nan=False)


if __name__=="__main__":unittest.main()
