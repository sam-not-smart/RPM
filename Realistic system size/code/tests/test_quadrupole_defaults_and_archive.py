from dataclasses import asdict, replace
import argparse
import importlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from poster_reproduction import core, runner
import rpm_full14_uncoupled_quadrupole as uncoupled
import rpm_quantum_needle_reproduction as exact
import rpm_quantum_needle_validation as validation
from rpm_quantum_needle_jd_sweeps import SweepConfig
from rpm_quantum_needle_model import FAD_NUCLEI, TRP_NUCLEI, GAMMA_E_MHZ_PER_MT, TWOPI, field_vector_mT

ROOT = Path(__file__).resolve().parents[3]


class QuadrupoleDefaultsAndArchiveTests(unittest.TestCase):
    def test_cli_defaults_and_explicit_opt_out(self):
        names = ("poster_reproduction.runner", "rpm_quantum_needle_reproduction",
                 "rpm_quantum_needle_statevector", "rpm_quantum_needle_validation",
                 "rpm_quantum_needle_jd_sweeps", "rpm_quantum_needle_refinement",
                 "rpm_exchange_avoided_crossing_scan", "rpm_nuclear_spin_rotation_control",
                 "rpm_transverse_axial_regime_shift", "rpm_two_nucleus_regime_shift_refined",
                 "rpm_two_nucleus_optimum_reproduction")
        for name in names:
            module = importlib.import_module(name)
            stage = {"poster_reproduction.runner": "production",
                     "rpm_quantum_needle_jd_sweeps": "report",
                     "rpm_exchange_avoided_crossing_scan": "report"}.get(name)
            prefix = ["--stage", stage] if stage else []
            for args, expected in (([], 1.), (["--quadrupole-scale", "0"], 0.)):
                with self.subTest(module=name, args=args), patch.object(sys, "argv", [name, *prefix, *args]):
                    self.assertEqual(module.parse_args().quadrupole_scale, expected)

    def test_three_nitrogen_cli_defaults(self):
        original_parse = argparse.ArgumentParser.parse_args
        class Parsed(Exception):
            pass
        for name in ("rpm_three_nitrogen_optimum", "rpm_three_nitrogen_suite"):
            module = importlib.import_module(name)
            for args, expected in (([], 1.), (["--quadrupole-scale", "0"], 0.)):
                captured = []
                def capture(parser, *a, **kw):
                    captured.append(original_parse(parser, *a, **kw))
                    raise Parsed()
                with self.subTest(module=name, args=args), \
                     patch.object(sys, "argv", [name, *args]), \
                     patch.object(argparse.ArgumentParser, "parse_args", capture):
                    with self.assertRaises(Parsed):
                        module.main()
                self.assertEqual(captured[0].quadrupole_scale, expected)

    def test_python_defaults_are_nominal(self):
        names = ("rpm_three_nitrogen_optimum", "rpm_three_nitrogen_suite",
                 "rpm_two_nucleus_optimum_reproduction", "rpm_two_nucleus_regime_shift_refined",
                 "rpm_nuclear_spin_rotation_control", "rpm_transverse_axial_regime_shift",
                 "rpm_quantum_needle_statevector", "rpm_quantum_needle_validation",
                 "rpm_exchange_avoided_crossing_scan", "poster_reproduction.runner")
        for name in names:
            module = importlib.import_module(name)
            for key, obj in vars(module).items():
                if not (inspect.isfunction(obj) or inspect.isclass(obj)) or obj.__module__ != name:
                    continue
                parameter = inspect.signature(obj).parameters.get("quadrupole_scale")
                if parameter is not None and parameter.default is not inspect.Parameter.empty:
                    with self.subTest(module=name, name=key):
                        self.assertEqual(parameter.default, 1.)
        self.assertEqual(SweepConfig().quadrupole_scale, 1.)
        self.assertEqual(core.PRODUCTION_MODEL.quadrupole_scale, 1.)
        for kind in ("quantum", "entanglement", "n10", "overhauser"):
            with patch.object(runner, f"_{kind}_cell", return_value={}) as cell:
                runner.run_cell(runner.CellTask(kind, "default", samples=4))
                self.assertEqual(cell.call_args.args[1].quadrupole_scale, 1.)

    def test_reference_quadrupoles_match_full_pair_operator(self):
        nuclei = FAD_NUCLEI[:1] + TRP_NUCLEI[:1]
        for q in (0., 1.):
            model = uncoupled.UncoupledModel(quadrupole_scale=q)
            full = core.build_hamiltonian(83., .05, model=model, nuclei=nuclei,
                                          radicals=("FAD", "TrpH")).explicit_matrix()
            radicals = [exact.build_radical(name, (nucleus,), quadrupole_scale=q)
                        for name, nucleus in zip(("FAD", "TrpH"), nuclei)]
            h = []
            for radical in radicals:
                zeeman = sum(component * op.toarray() for component, op in zip(
                    field_vector_mT(83., 50.), radical.electron_ops))
                h.append(radical.h0.toarray() + TWOPI * GAMMA_E_MHZ_PER_MT * zeeman)
            independent = np.kron(h[0], np.eye(6)) + np.kron(np.eye(6), h[1])
            reordered = independent.reshape((2, 3, 2, 3) * 2).transpose(
                0, 2, 1, 3, 4, 6, 5, 7).reshape(36, 36)
            np.testing.assert_allclose(full, reordered, rtol=0, atol=2e-13)

    def test_validation_reference_cache_separates_q(self):
        validation._EXACT_REFERENCE_CACHE.clear()
        with patch.object(exact, "reference_models", return_value=(None, None)) as models, \
             patch.object(exact, "diagonalize_radical"), \
             patch.object(exact, "convolved_correlation_spectrum",
                          return_value=(np.array([0.]), np.array([.75]), 1.)), \
             patch.object(exact, "yields_from_spectrum", return_value={5.: .3}):
            for q in (0., 1., 0.):
                validation.exact_reference(90., np.array([0., 1.]),
                    validation.ValidationConfig(quadrupole_scale=q))
            self.assertEqual([call.args[0] for call in models.call_args_list], [0., 1.])
        validation._EXACT_REFERENCE_CACHE.clear()

    def test_extended_archive_and_replot_inputs(self):
        config = uncoupled.configuration()
        self.assertEqual(config.digest,
                         "b0b8037cc352fed6857a4d72bad4bb7107050df712381c3711584550757fea63")
        self.assertEqual(len(uncoupled.base.tasks(config)), 232)
        plotter = runpy.run_path(str(ROOT / "scripts/plot_quadrupole_comparison.py"))
        angles, groups, theta, values = plotter["load_data"]()
        self.assertEqual(len(angles), 29)
        self.assertEqual(len(groups), 4)
        self.assertEqual(len(theta), 199)
        self.assertTrue(np.isfinite(values).all())
        for samples, errors in groups.values():
            self.assertEqual(samples.shape, (29, 4))
            self.assertTrue((errors >= 0).all())

    def test_reuse_preserves_arrays_and_rejects_changed_physics(self):
        config = replace(uncoupled.configuration(), samples=1, angles_deg=(0., 90.))
        old_config = replace(config, angles_deg=(90.,))
        base = uncoupled.base
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "old", Path(directory) / "new"
            base.ensure_manifest(source, old_config)
            base.ensure_manifest(output, config)
            for task in base.tasks(old_config):
                base.atomic_npz(base.cell_path(source, task), {
                    "yield": np.array(.3), "quadrature_error": np.array(1e-8),
                    "yield_propagation_estimate": np.array(1e-6), "yield_tail_bound": np.array(1e-5),
                    "final_state": np.arange(6, dtype=complex),
                    "metadata_json": np.array(json.dumps({
                        "configuration_hash": old_config.digest, "task": asdict(task),
                        "initial_state_sha256": uncoupled.initial_hash(old_config, 0)}))})
            self.assertEqual(uncoupled.reuse_completed(source, output, config), 2)
            self.assertEqual(uncoupled.reuse_completed(source, output, config), 2)
            for task in base.tasks(old_config):
                uncoupled.read_cell(output, task, config)
                with np.load(base.cell_path(output, task)) as z:
                    np.testing.assert_array_equal(z["final_state"], np.arange(6))
            with self.assertRaises(ValueError):
                uncoupled.reuse_completed(source, output, replace(config, field_mT=.1))
            with self.assertRaises(ValueError):
                uncoupled.reuse_completed(source, source, config)


if __name__ == "__main__":
    unittest.main()
