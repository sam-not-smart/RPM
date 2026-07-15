"""Shared physical model for the Hiscock et al. quantum-needle benchmark.

All frequencies returned by this module are angular frequencies in rad/us.
Hyperfine tensors are transcribed in mT from Tables S1-S2 of the supporting
information.  The local spin basis is ordered m=I, I-1, ..., -I.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


TWOPI = 2.0 * np.pi
GAMMA_E_MHZ_PER_MT = 28.0
DEFAULT_FIELD_MICROT = 50.0
DEFAULT_LIFETIMES_US = (1.0, 2.0, 5.0, 10.0, 20.0, 100.0)
PILOT_ANGLES_DEG = (0.0, 80.0, 90.0, 100.0, 180.0)
DIAGNOSTIC_ANGLES_DEG = PILOT_ANGLES_DEG


@dataclass(frozen=True)
class Nucleus:
    """One magnetic nucleus in the molecular-frame hyperfine model."""

    name: str
    spin: float
    tensor_mT: np.ndarray

    @property
    def dimension(self) -> int:
        return int(round(2.0 * self.spin + 1.0))


FAD_NUCLEI = (
    Nucleus("N5", 1.0, np.array([[-0.0989, 0.0039, 0.0], [0.0039, -0.0881, 0.0], [0.0, 0.0, 1.7569]])),
    Nucleus("N10", 1.0, np.array([[-0.0190, -0.0048, 0.0], [-0.0048, -0.0196, 0.0], [0.0, 0.0, 0.6046]])),
    Nucleus("H6", 0.5, np.array([[-0.2569, -0.1273, 0.0], [-0.1273, -0.4711, 0.0], [0.0, 0.0, -0.4336]])),
    Nucleus("H8a", 0.5, 0.4399 * np.eye(3)),
    Nucleus("H8b", 0.5, 0.4399 * np.eye(3)),
    Nucleus("H8c", 0.5, 0.4399 * np.eye(3)),
    Nucleus("Hbeta", 0.5, 0.4070 * np.eye(3)),
)


TRP_NUCLEI = (
    Nucleus("N1", 1.0, np.array([[-0.0336, 0.0924, -0.1354], [0.0924, 0.3303, -0.5318], [-0.1354, -0.5318, 0.6680]])),
    Nucleus("H1", 0.5, np.array([[-0.9920, -0.2091, -0.2003], [-0.2091, -0.2631, 0.2803], [-0.2003, 0.2803, -0.5398]])),
    Nucleus("H2", 0.5, np.array([[-0.2843, 0.1757, 0.1525], [0.1757, -0.2798, 0.0975], [0.1525, 0.0975, -0.2699]])),
    Nucleus("H4", 0.5, np.array([[-0.5596, -0.1956, -0.1657], [-0.1956, -0.4020, 0.0762], [-0.1657, 0.0762, -0.5021]])),
    Nucleus("H5", 0.5, np.array([[0.0051, 0.0616, 0.0694], [0.0616, -0.0665, 0.0391], [0.0694, 0.0391, -0.0586]])),
    Nucleus("H7", 0.5, np.array([[-0.4355, -0.1541, -0.1239], [-0.1541, -0.2777, 0.0864], [-0.1239, 0.0864, -0.3770]])),
    Nucleus("Hbeta1", 0.5, np.array([[1.5808, -0.0453, -0.0506], [-0.0453, 1.5575, 0.0988], [-0.0506, 0.0988, 1.6752]])),
)


ALL_NUCLEI = FAD_NUCLEI + TRP_NUCLEI
NUCLEAR_RADICALS = ("FAD",) * len(FAD_NUCLEI) + ("TrpH",) * len(TRP_NUCLEI)
SPIN_ORDER = ("eA", "eB") + tuple(
    f"{radical}:{nucleus.name}" for radical, nucleus in zip(NUCLEAR_RADICALS, ALL_NUCLEI)
)
SYSTEM_DIMENSIONS = (2, 2) + tuple(nucleus.dimension for nucleus in ALL_NUCLEI)
NUCLEAR_DIMENSION = math.prod(SYSTEM_DIMENSIONS[2:])
STATE_DIMENSION = math.prod(SYSTEM_DIMENSIONS)


def _quadrupole_tensor_in_hyperfine_frame(
    hyperfine_tensor_mT: np.ndarray,
    principal_values_mT: tuple[float, float, float],
    *,
    qz_orientation: str,
) -> np.ndarray:
    """Rotate a traceless Q tensor into the working molecular frame.

    Efimova and Hore give principal values and the Q-z tilt relative to the
    aromatic-ring normal, but not the in-plane azimuth in their main paper.
    The hyperfine principal axes provide a reproducible provisional frame.
    """

    principal = np.asarray(principal_values_mT, dtype=float)
    principal -= np.mean(principal)
    hyperfine_values, hyperfine_axes = np.linalg.eigh(hyperfine_tensor_mT)
    normal_axis = int(np.argmax(np.abs(hyperfine_values)))
    in_plane_axes = [index for index in range(3) if index != normal_axis]
    assigned = np.empty(3, dtype=float)
    if qz_orientation == "ring_normal":
        assigned[normal_axis] = principal[2]
        assigned[in_plane_axes[0]] = principal[0]
        assigned[in_plane_axes[1]] = principal[1]
    elif qz_orientation == "ring_plane_provisional":
        # The unknown N5 in-plane azimuth is provisionally placed along the
        # first HFI in-plane principal axis. Qyy is placed along the ring
        # normal. This orientation must later be sensitivity-tested.
        assigned[in_plane_axes[0]] = principal[2]
        assigned[in_plane_axes[1]] = principal[0]
        assigned[normal_axis] = principal[1]
    else:
        raise ValueError(f"Unknown quadrupole orientation: {qz_orientation}")
    tensor = hyperfine_axes @ np.diag(assigned) @ hyperfine_axes.T
    return 0.5 * (tensor + tensor.T)


# Table 2 of Efimova & Hore, Molecular Physics 107, 665-671 (2009).
# Values are equivalent-field mT tensors in the shared molecular frame. All
# current I>1/2 nuclei are 14N with I=1. The N5 in-plane azimuth is provisional
# because the main paper reports only its 90-degree tilt to the ring normal.
QUADRUPOLE_TENSORS_MT = {
    "FAD:N5": _quadrupole_tensor_in_hyperfine_frame(
        FAD_NUCLEI[0].tensor_mT,
        (0.031, 0.041, -0.072),
        qz_orientation="ring_plane_provisional",
    ),
    "FAD:N10": _quadrupole_tensor_in_hyperfine_frame(
        FAD_NUCLEI[1].tensor_mT,
        (0.032, 0.041, -0.072),
        qz_orientation="ring_normal",
    ),
    "TrpH:N1": _quadrupole_tensor_in_hyperfine_frame(
        TRP_NUCLEI[0].tensor_mT,
        (0.016, 0.020, -0.036),
        qz_orientation="ring_normal",
    ),
}


def spin_operators(spin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Sx, Sy, Sz in the descending-m basis."""

    dimension = int(round(2.0 * spin + 1.0))
    m_values = np.arange(spin, -spin - 0.5, -1.0)
    raising = np.zeros((dimension, dimension), dtype=complex)
    for column in range(1, dimension):
        m = m_values[column]
        raising[column - 1, column] = math.sqrt(spin * (spin + 1.0) - m * (m + 1.0))
    lowering = raising.conj().T
    return (
        0.5 * (raising + lowering),
        (raising - lowering) / (2.0j),
        np.diag(m_values.astype(complex)),
    )


def field_vector_mT(theta_deg: float, field_microT: float = DEFAULT_FIELD_MICROT) -> np.ndarray:
    """Magnetic field rotated through the flavin molecular zx plane."""

    theta = np.deg2rad(theta_deg)
    return 1e-3 * field_microT * np.array([np.sin(theta), 0.0, np.cos(theta)])


def theta_grid(
    step_deg: float = 1.0,
    fine_step_deg: float = 0.25,
    fine_half_width_deg: float = 3.0,
) -> np.ndarray:
    coarse = np.arange(0.0, 180.0 + 0.5 * step_deg, step_deg)
    fine = np.arange(
        90.0 - fine_half_width_deg,
        90.0 + fine_half_width_deg + 0.5 * fine_step_deg,
        fine_step_deg,
    )
    return np.unique(np.round(np.concatenate([coarse, fine]), 10))


def diagnostic_time_grid_us() -> np.ndarray:
    dense = np.arange(0.0, 5.0 + 0.005, 0.01)
    logarithmic = np.geomspace(5.0, 500.0, 257)
    return np.unique(np.concatenate([dense, logarithmic]))
