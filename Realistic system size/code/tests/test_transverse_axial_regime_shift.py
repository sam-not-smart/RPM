from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from poster_reproduction.core import PRODUCTION_MODEL
from rpm_quantum_needle_model import FAD_NUCLEI
from rpm_transverse_axial_regime_shift import (
    ABLATION_TARGETS,
    CONDITIONS,
    FIELDS_MT,
    FULL_CURVE_FIELDS_MT,
    RegimeConfig,
    RegimeTask,
    ablated_hyperfine_tensors_mT,
    cell_path,
    discovery_tasks,
    ensure_manifest,
    principal_axial_only_tensor,
    propagate_task,
    run_tasks,
    smoke_config,
)


class TransverseAxialRegimeShiftTests(unittest.TestCase):
    def test_discovery_inventory_matches_locked_design(self) -> None:
        config = RegimeConfig()
        tasks = discovery_tasks(config)
        self.assertEqual(len(tasks), 414)
        self.assertEqual(set(task.condition for task in tasks), set(CONDITIONS))
        self.assertEqual(set(task.field_mT for task in tasks), set(FIELDS_MT))
        self.assertEqual(set(config.full_curve_fields_mT), set(FULL_CURVE_FIELDS_MT))
        self.assertTrue(all(task.sample_id == 0 for task in tasks))
        self.assertEqual(len({task.key for task in tasks}), len(tasks))

    def test_production_physics_is_frozen(self) -> None:
        config = RegimeConfig()
        condition = config.model.condition
        self.assertEqual(config.lifetime_us, 5.0)
        self.assertEqual(config.model.quadrupole_scale, 0.0)
        self.assertEqual(config.model.ratio_D_over_J, -3.0)
        self.assertAlmostEqual(condition.d_paper_uT, -347.875)
        self.assertAlmostEqual(condition.exchange_J_uT, 115.95833333333333)
        self.assertEqual(config.model.field_plane, PRODUCTION_MODEL.field_plane)

    def test_principal_axis_ablation_retains_only_axial_value(self) -> None:
        for nucleus in FAD_NUCLEI[:2]:
            original = 0.5 * (nucleus.tensor_mT + nucleus.tensor_mT.T)
            deleted = principal_axial_only_tensor(original)
            original_values, original_vectors = np.linalg.eigh(original)
            deleted_values, deleted_vectors = np.linalg.eigh(deleted)
            axial = int(np.argmax(np.abs(original_values)))
            kept = int(np.argmax(np.abs(deleted_values)))
            self.assertAlmostEqual(deleted_values[kept], original_values[axial], places=12)
            other = np.delete(deleted_values, kept)
            np.testing.assert_allclose(other, 0.0, atol=1e-12, rtol=0.0)
            overlap = abs(np.vdot(original_vectors[:, axial], deleted_vectors[:, kept]))
            self.assertAlmostEqual(overlap, 1.0, places=12)
        self.assertEqual(set(ablated_hyperfine_tensors_mT()), set(ABLATION_TARGETS))

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

    def test_smoke_cells_are_finite_and_paired(self) -> None:
        config = smoke_config(123)
        full = propagate_task(RegimeTask(CONDITIONS[0], 0.05, 90.0), config)
        ablated = propagate_task(RegimeTask(CONDITIONS[1], 0.05, 90.0), config)
        for cell in (full, ablated):
            self.assertTrue(math.isfinite(float(cell["yield"])))
            self.assertAlmostEqual(float(cell["initial_singlet"]), 1.0, places=12)
            self.assertLess(float(cell["quadrature_error"]), 1e-3)
        repeated = propagate_task(RegimeTask(CONDITIONS[0], 0.05, 90.0), config)
        self.assertAlmostEqual(float(full["yield"]), float(repeated["yield"]), places=13)

    def test_elapsed_deadline_starts_no_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = smoke_config(321)
            tasks = discovery_tasks(config)
            status = run_tasks(
                tasks,
                output,
                config,
                workers=2,
                resume=False,
                deadline=time.monotonic() - 1.0,
                start_guard_seconds=0.0,
            )
            self.assertEqual(status["status"], "wall_time_checkpoint")
            self.assertEqual(status["completed"], 0)
            self.assertFalse(any(cell_path(output, task).exists() for task in tasks))


if __name__ == "__main__":
    unittest.main()
