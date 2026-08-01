from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rpm_exchange_avoided_crossing_scan import (  # noqa: E402
    CANCELLATION_REFERENCES_UT,
    CORE_DIMENSIONS,
    SpectralConfig,
    _singlet_projector,
    _spectrum_for_j,
    build_core_hamiltonian,
    detect_avoided_crossings,
    exchange_condition,
    exchange_operator_controls,
    select_full_system_j_values,
)


class ReducedHamiltonianTests(unittest.TestCase):
    def test_core_hamiltonian_is_108_dimensional_and_hermitian(self) -> None:
        hamiltonian = build_core_hamiltonian(90.0, -83.0)
        self.assertEqual(hamiltonian.shape, (108, 108))
        np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12)

    def test_singlet_character_trace_is_nuclear_dimension(self) -> None:
        config = SpectralConfig(coarse_theta_deg=(89.9, 90.0, 90.1))
        spectrum = _spectrum_for_j(
            (0.0, config.coarse_theta_deg, config, False, "test")
        )
        singlets = np.asarray(spectrum["singlet_characters"])
        np.testing.assert_allclose(np.sum(singlets, axis=1), 27.0, atol=2e-12)
        self.assertEqual(_singlet_projector(CORE_DIMENSIONS).shape, (108, 108))

    def test_removing_transverse_nitrogen_terms_changes_only_hyperfine_model(self) -> None:
        physical = build_core_hamiltonian(
            90.0, 0.0, d_paper_uT=0.0, remove_n5_n10_transverse=False
        )
        axialized = build_core_hamiltonian(
            90.0, 0.0, d_paper_uT=0.0, remove_n5_n10_transverse=True
        )
        difference = physical - axialized
        self.assertGreater(np.linalg.norm(difference), 1.0)
        np.testing.assert_allclose(difference, difference.conj().T, atol=1e-12)

    def test_exchange_and_dipolar_terms_do_not_create_singlet_triplet_coupling(self) -> None:
        controls = exchange_operator_controls()
        for value in controls.values():
            self.assertLess(value, 1e-13)


class CrossingDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _two_level_spectrum(gap: float) -> dict[str, object]:
        angles = np.linspace(89.5, 90.5, 11)
        coordinate = angles - 90.0
        radius = np.sqrt(coordinate**2 + gap**2)
        energies = np.column_stack((-0.5 * radius, 0.5 * radius))
        lower = 0.5 * (1.0 + coordinate / np.maximum(radius, 1e-15))
        singlets = np.column_stack((lower, 1.0 - lower))
        return {
            "J_uT": -83.0,
            "angles_deg": angles,
            "energies_MHz": energies,
            "singlet_characters": singlets,
            "remove_transverse": False,
            "scan_kind": "synthetic",
        }

    def test_finite_singlet_active_avoided_crossing_scores_above_exact_crossing(self) -> None:
        avoided = detect_avoided_crossings(self._two_level_spectrum(0.01))
        exact = detect_avoided_crossings(self._two_level_spectrum(0.0))
        self.assertEqual(len(avoided), 1)
        self.assertEqual(len(exact), 1)
        self.assertGreater(avoided[0]["singlet_character_transfer"], 0.5)
        self.assertGreater(avoided[0]["needle_crossing_score"], 0.0)
        self.assertEqual(exact[0]["needle_crossing_score"], 0.0)

    def test_full_selection_includes_controls_and_cancellation_references(self) -> None:
        ranking = [
            {"J_uT": value, "best_crossing_score": 1.0 / (index + 1)}
            for index, value in enumerate((-83.0, -80.0, 40.0, 75.0, 150.0, -130.0))
        ]
        selected = select_full_system_j_values(ranking)
        self.assertEqual(len(selected), 8)
        self.assertIn(0.0, selected)
        for reference in CANCELLATION_REFERENCES_UT:
            self.assertIn(reference, selected)
        conditions = tuple(exchange_condition(value) for value in selected)
        self.assertTrue(all(item.d_paper_uT == -347.875 for item in conditions))
        discoveries = selected[:5]
        self.assertTrue(
            all(
                abs(left - right) >= 15.0
                for index, left in enumerate(discoveries)
                for right in discoveries[index + 1 :]
            )
        )


if __name__ == "__main__":
    unittest.main()
