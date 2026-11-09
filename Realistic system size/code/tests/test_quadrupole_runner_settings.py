"""Check quadrupole routing and compatibility with manuscript archives."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from poster_reproduction import core, report, runner
from rpm_exchange_avoided_crossing_scan import build_core_hamiltonian
from rpm_full14_quadrupole_field_pulse_screen import inventory
from rpm_nuclear_spin_rotation_control import ControlConfig, ensure_manifest
from rpm_quantum_needle_jd_sweeps import (
    SweepConfig, _propagator, efimova_hore_ratio_condition, ensure_manifest as ensure_jd,
)
from rpm_quantum_needle_model import QUADRUPOLE_TENSORS_MT, parse_quadrupole_scale
from rpm_three_nitrogen_optimum import PairModel
from rpm_two_nucleus_optimum_reproduction import _yields_for_angles
from rpm_two_nucleus_regime_shift_refined import hamiltonian


class QuadrupoleRunnerSettings(unittest.TestCase):
    def test_archived_q_off_fingerprints(self):
        model = replace(core.PRODUCTION_MODEL, samples=4, quadrupole_scale=0.0)
        self.assertEqual(model.configuration_hash,
                         "40140f31c033969da59a8ddeb1699203c885cbb6b429b3c46a3fc4dbfa814a04")
        self.assertEqual(ControlConfig(quadrupole_scale=0.0).configuration_hash,
                         "86679c318200627fec0309595e47f79745c1e0f7d0d0ee91637b79f3976e54c9")
        self.assertNotEqual(model.configuration_hash,
                            replace(model, quadrupole_scale=1.0).configuration_hash)

    def test_full_pair_adds_only_the_three_local_terms(self):
        q0 = replace(core.PRODUCTION_MODEL, quadrupole_scale=0.0)
        q1 = replace(q0, quadrupole_scale=1.0)
        off = core.build_hamiltonian(90.0, 10.0, model=q0)
        on = core.build_hamiltonian(90.0, 10.0, model=q1)
        old = {term.label: term for term in off.terms}
        new = {term.label: term for term in on.terms}
        self.assertEqual(set(new) - set(old),
                         {f"{site} quadrupole" for site in QUADRUPOLE_TENSORS_MT})
        for name, term in old.items():
            self.assertEqual(term.axes, new[name].axes)
            np.testing.assert_array_equal(term.operator, new[name].operator)

    def test_poster_workers_and_report_use_requested_scale(self):
        for kind in ("quantum", "entanglement", "n10", "overhauser"):
            with patch.object(runner, f"_{kind}_cell", return_value={}) as cell:
                task = runner.CellTask(kind, "test", samples=4)
                runner.run_cell(task, 0.75)
                model = cell.call_args.args[1]
                self.assertEqual(model.quadrupole_scale, 0.75)
                self.assertEqual(model.stable_dict()["condition"]["quadrupole_scale"], 0.75)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(report.reported_quadrupole_scale(root), 0)
            (root / "model_manifest.json").write_text(json.dumps({"quadrupole_scale": 0.75}))
            self.assertEqual(report.reported_quadrupole_scale(root), 0.75)

    def test_coupling_sweep_routes_q_to_the_hamiltonian(self):
        condition = efimova_hore_ratio_condition(-3.0)
        with patch("rpm_quantum_needle_jd_sweeps.build_quantum_needle_hamiltonian") as build:
            _propagator(condition, 90.0, SweepConfig(quadrupole_scale=1.0))
            self.assertEqual(build.call_args.kwargs["quadrupole_scale"], 1.0)

    def test_resume_keeps_q_off_and_q_on_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_manifest(root, ControlConfig())
            with self.assertRaises(ValueError):
                ensure_manifest(root, ControlConfig(quadrupole_scale=1.0))
            jd = root / "jd"
            jd.mkdir()
            ensure_jd(jd, SweepConfig(), False)
            with self.assertRaises(RuntimeError):
                ensure_jd(jd, SweepConfig(quadrupole_scale=1.0), True)

    def test_exact_toys_and_spectral_core_are_q_sensitive(self):
        for off, on in (
            (hamiltonian("full", 0.05, 90.0, 0.0), hamiltonian("full", 0.05, 90.0, 1.0)),
            (PairModel("three_nitrogen").hamiltonian(1.0, 90.0),
             PairModel("three_nitrogen", quadrupole_scale=1.0).hamiltonian(1.0, 90.0)),
            (build_core_hamiltonian(90.0, 115.95833333333333),
             build_core_hamiltonian(90.0, 115.95833333333333, quadrupole_scale=1.0)),
        ):
            self.assertGreater(np.linalg.norm(on - off), 1.0)
            np.testing.assert_allclose(on, on.conj().T, rtol=0, atol=1e-12)
        off = _yields_for_angles(1.7569, 132.5, 10., np.array([85., 90.]), 0.)
        on = _yields_for_angles(1.7569, 132.5, 10., np.array([85., 90.]), 1.)
        self.assertGreater(np.max(np.abs(on - off)), 1e-4)

    def test_paired_screen_inventory(self):
        tasks = inventory()
        self.assertEqual([t.field_mT for group, t in tasks[:8] if group == "field"], [10.0] * 8)
        self.assertEqual(sum(group == "field" for group, _ in tasks), 16)
        self.assertEqual(sum(group == "pulse" for group, _ in tasks), 24)

    def test_q_scale_validation(self):
        for invalid in ("-1", "nan", "inf"):
            with self.assertRaises(ValueError):
                parse_quadrupole_scale(invalid)
        self.assertEqual(parse_quadrupole_scale("0"), 0.0)
        self.assertEqual(parse_quadrupole_scale("1"), 1.0)
