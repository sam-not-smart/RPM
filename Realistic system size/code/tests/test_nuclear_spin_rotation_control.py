from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from poster_reproduction.core import dimensions_for
from rpm_nuclear_spin_rotation_control import (
    ControlConfig,
    ControlTask,
    confirmation_tasks,
    dominant_principal_axis,
    ensure_manifest,
    local_metrics,
    propagate_control_task,
    pulse_schedule,
    screen_tasks,
    smoke_config,
    target_components,
    transverse_deleted_tensor,
)
from rpm_quantum_needle_model import ALL_NUCLEI, FAD_NUCLEI, NUCLEAR_RADICALS, spin_operators
from poster_reproduction.core import spin_pi_pulse


class NuclearSpinRotationControlTests(unittest.TestCase):
    def test_production_inventory_matches_24_hour_design(self) -> None:
        config = ControlConfig()
        screen = screen_tasks(config)
        confirm = confirmation_tasks(config, 0.05)
        self.assertEqual(len(screen), 43)
        self.assertEqual(len(confirm), 132)
        unique = {task.key for task in screen + confirm}
        self.assertEqual(len(unique), 160)

    def test_n5_is_the_stronger_transverse_nitrogen(self) -> None:
        n5 = np.asarray(FAD_NUCLEI[0].tensor_mT)
        n10 = np.asarray(FAD_NUCLEI[1].tensor_mT)
        self.assertGreater(np.linalg.norm(n5[:, :2]), 4.0 * np.linalg.norm(n10[:, :2]))

    def test_target_axis_and_transverse_deletion(self) -> None:
        axis, nucleus, radical = target_components("FAD:N5", ALL_NUCLEI, NUCLEAR_RADICALS)
        self.assertEqual(axis, 2)
        self.assertEqual(nucleus.name, "N5")
        self.assertEqual(radical, "FAD")
        deleted = transverse_deleted_tensor(nucleus.tensor_mT)
        np.testing.assert_allclose(deleted[:, :2], 0.0, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(deleted[:, 2], nucleus.tensor_mT[:, 2], atol=0.0, rtol=0.0)

    def test_pi_pulse_toggles_transverse_spin_components(self) -> None:
        nucleus = FAD_NUCLEI[0]
        axis = dominant_principal_axis(nucleus.tensor_mT)
        np.testing.assert_allclose(axis, np.array([0.0, 0.0, 1.0]), atol=1e-12)
        pulse = spin_pi_pulse(nucleus.spin, axis)
        identity = np.eye(nucleus.dimension)
        np.testing.assert_allclose(pulse.conj().T @ pulse, identity, atol=1e-12)
        ix, iy, iz = spin_operators(nucleus.spin)
        np.testing.assert_allclose(pulse.conj().T @ ix @ pulse, -ix, atol=1e-12)
        np.testing.assert_allclose(pulse.conj().T @ iy @ pulse, -iy, atol=1e-12)
        np.testing.assert_allclose(pulse.conj().T @ iz @ pulse, iz, atol=1e-12)

    def test_pulse_schedule_excludes_final_boundary(self) -> None:
        np.testing.assert_allclose(pulse_schedule(0.02, 0.10), np.array([0.02, 0.04, 0.06, 0.08]))
        with self.assertRaises(ValueError):
            pulse_schedule(0.0, 1.0)

    def test_local_metrics_preserve_feature_sign(self) -> None:
        metrics = local_metrics([88, 89, 90, 91, 92], [0.4, 0.3, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(metrics["local_range"], 0.2)
        self.assertAlmostEqual(metrics["signed_center_feature"], -0.2)
        self.assertAlmostEqual(metrics["absolute_center_feature"], 0.2)
        self.assertEqual(metrics["theta_minimum_deg"], 90.0)

    def test_manifest_rejects_incompatible_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = smoke_config(10)
            ensure_manifest(output, first)
            ensure_manifest(output, first)
            with self.assertRaises(ValueError):
                ensure_manifest(output, smoke_config(11))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["configuration_hash"], first.configuration_hash)

    def test_reduced_smoke_cell_is_finite_and_reproducible(self) -> None:
        config = smoke_config(123)
        task = ControlTask("FAD:N5", "periodic", 90.0, 0, 0.02)
        first = propagate_control_task(task, config)
        second = propagate_control_task(task, config)
        self.assertTrue(math.isfinite(float(first["yield"])))
        self.assertAlmostEqual(float(first["initial_singlet"]), 1.0, places=12)
        self.assertAlmostEqual(float(first["yield"]), float(second["yield"]), places=13)
        self.assertEqual(int(first["pulse_count"]), 2)
        self.assertLess(float(first["quadrature_error"]), 1e-3)
        self.assertEqual(first["singlet_population"].shape, first["times_us"].shape)
        self.assertEqual(dimensions_for(FAD_NUCLEI[:2]), (2, 2, 3, 3))


if __name__ == "__main__":
    unittest.main()
