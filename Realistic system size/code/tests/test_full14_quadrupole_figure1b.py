import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import rpm_full14_quadrupole_experiment as base
import rpm_full14_quadrupole_figure1b as sweep


class Figure1BSweepTests(unittest.TestCase):
    def test_grid_and_physics(self):
        old, new = base.Config(), sweep.configuration()
        self.assertEqual(len(new.angles_deg), 61)
        self.assertEqual(len(base.tasks(new)), 488)
        self.assertEqual(len(set(base.tasks(new)) - set(base.tasks(old))), 400)
        self.assertEqual(replace(new, angles_deg=old.angles_deg).digest, old.digest)
        self.assertEqual(new.model(0).quadrupole_scale, 0)
        self.assertEqual(new.model(1).quadrupole_scale, 1)
        self.assertAlmostEqual(new.model(1).condition.exchange_J_uT, 115.95833333333333)
        self.assertEqual(new.model(1).condition.d_paper_uT, -347.875)

    def test_reduced_grid_reuses_88_and_needs_144_new_trajectories(self):
        original, reduced = base.Config(), sweep.configuration("reduced")
        self.assertEqual(reduced.angles_deg, tuple(sorted(set(range(0, 181, 10)) | set(range(85, 96)))))
        self.assertEqual(len(reduced.angles_deg), 29)
        self.assertEqual(len(base.tasks(reduced)), 232)
        self.assertTrue(set(base.tasks(original)).issubset(base.tasks(reduced)))
        self.assertEqual(len(set(base.tasks(reduced)) - set(base.tasks(original))), 144)
        self.assertEqual(replace(reduced, angles_deg=original.angles_deg).digest, original.digest)
        self.assertNotEqual(reduced.digest, sweep.configuration().digest)
        ordered = sweep.ordered_tasks(reduced, "reduced")
        self.assertEqual(len(ordered), 232)
        self.assertEqual(set(ordered), set(base.tasks(reduced)))
        self.assertEqual({task.theta_deg for task in ordered[:19*8]}, set(range(0, 181, 10)))
        for start in range(0, len(ordered), 8):
            self.assertEqual(len({t.theta_deg for t in ordered[start:start+8]}), 1)

    def test_order_keeps_both_conditions_and_all_samples_together(self):
        config = sweep.configuration()
        tasks = sweep.ordered_tasks(config)
        chunk = len(base.SCALES) * config.samples
        self.assertEqual(set(tasks), set(base.tasks(config)))
        for start in range(0, len(tasks), chunk):
            self.assertEqual(len({t.theta_deg for t in tasks[start:start+chunk]}), 1)
        self.assertTrue(all(t.theta_deg in sweep.COARSE_ANGLES
                            for t in tasks[:len(sweep.COARSE_ANGLES)*chunk]))

    def test_import_reuses_exact_arrays_and_preserves_source(self):
        old = base.smoke_config()
        new = replace(old, angles_deg=(0., 85., 90., 180.))
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "old", Path(tmp) / "new"
            base.ensure_manifest(source, old)
            for task in base.tasks(old):
                base.atomic_npz(base.cell_path(source, task), base.propagate(task, old))
            base.atomic_json(source / "status.json", {"status": "complete", "completed": 4})
            first = base.tasks(old)[0]
            before = sweep.sha256(base.cell_path(source, first))
            with self.assertRaisesRegex(ValueError, "only the angular"):
                sweep.reuse_completed(source, output, replace(new, seed=old.seed+1), old)
            result = sweep.reuse_completed(source, output, new, old)
            self.assertEqual(result["newly_imported"], 4)
            self.assertEqual(sweep.sha256(base.cell_path(source, first)), before)
            self.assertEqual(base.count_completed(output, new), 4)
            for task in base.tasks(old):
                with np.load(base.cell_path(source, task)) as a, np.load(base.cell_path(output, task)) as b:
                    for key in a.files:
                        if key != "metadata_json":
                            np.testing.assert_array_equal(a[key], b[key])
                    meta = json.loads(str(b["metadata_json"]))
                    self.assertEqual(meta["configuration_hash"], new.digest)
                    self.assertEqual(meta["reused_from"]["configuration_hash"], old.digest)
            self.assertEqual(sweep.reuse_completed(source, output, new, old)["newly_imported"], 0)
            angles, values, errors = sweep.curve_data(output, new)
            np.testing.assert_array_equal(angles, old.angles_deg)
            self.assertEqual(values.shape, (2, 2, 1))
            self.assertTrue(np.isfinite(errors).all())
            sweep.plot_curves(output, new)
            for tag in sweep.TAGS:
                self.assertTrue((output / f"figure1b_quadrupoles_{tag}_preview.png").is_file())
                self.assertFalse((output / f"figure1b_quadrupoles_{tag}.png").exists())

    def test_curve_data_without_paired_complete_angles(self):
        with tempfile.TemporaryDirectory() as tmp:
            angles, values, errors = sweep.curve_data(Path(tmp), sweep.configuration())
            self.assertEqual(len(angles), 0)
            self.assertEqual(values.shape, (2, 0, 4))
            self.assertEqual(errors.shape, values.shape)


if __name__ == "__main__":
    unittest.main()
