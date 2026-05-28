"""
Notebook-friendly full quantum RPM finite-size scaling tests with a full non-secular
magnetic dipole-dipole interaction.

Purpose
-------
This script is based on the recent selected-ratio polar-sweep workflow, but is
refocused on a finite-size truncation study. It runs nested nuclear environments
N = 0, 2, 4, 6 at fixed D/J=-6, phi=0, and theta=0:10:180 deg. It uses the full point-dipole
operator

    H_dip = D [ S1 · S2 - 3 (S1 · rhat)(S2 · rhat) ].

The molecular/inter-radical axis rhat is fixed in the molecular frame by
`dipole_axis_theta_deg` and `dipole_axis_phi_deg`. The external magnetic field
is swept independently using polar and azimuthal angles,

    B = B0 (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)).

Conventions
-----------
* Spin operators are dimensionless spin-1/2 operators, sigma/2.
* Couplings supplied in MHz are converted internally to angular units rad/us.
* Time is in microseconds, so FFT frequencies are returned in MHz.
* The default nuclear environment and exchange/dipole magnitude conventions are
  inherited from the recent EH baseline sweep script.
* The exchange term is kept in the same EH sign convention as the previous code:

      H_J = -J (S_total^2 - I_full),

  which gives electronic eigenvalues +J for the singlet and -J for triplets in
  the internal Hamiltonian when J is positive. Keep this convention fixed while
  comparing against previous runs.

Notebook usage
--------------
    from rpm_nonsecular_DJ_ratio_sweep import Params, run_baseline_DJ_ratio_sweep

    P = Params()
    P.ratio_values = np.r_[np.linspace(-10, -0.5, 20), np.linspace(0.5, 10, 20)]
    P.theta_values_deg = np.array([50.0, 90.0])
    P.phi_values_deg = np.array([0.0])
    results, grid = run_ratio_sweep(P)

The default executable run sweeps D/J from -10 to 10 excluding zero at the two
baseline orientations theta=50 deg and theta=90 deg with phi=0 deg.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, basis, mesolve, qeye, sigmax, sigmay, sigmaz, tensor

TWOPI = 2.0 * np.pi


def mhz_to_rad_per_us(x):
    """Convert ordinary frequency in MHz to angular frequency in rad/us."""
    return TWOPI * np.asarray(x, dtype=float)


def unit_vector_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Spherical unit vector: theta from +z, phi in x-y plane from +x."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], dtype=float)


@dataclass
class Params:
    # -----------------------------
    # Magnetic field / Zeeman setup
    # -----------------------------
    B_microT: float = 50.0
    gamma_e_MHz_per_mT: float = 28.02495164

    # Full two-angle sweep of the external magnetic-field direction.
    # theta: polar angle from +z. phi: azimuthal angle from +x in x-y plane.
    theta_values_deg: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 180.0, 7))
    phi_values_deg: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 360.0, 9, endpoint=False))

    # -----------------------------
    # Nuclear environment
    # -----------------------------
    # Six spin-1/2 nuclei, split as 3 nuclei on electron 1 and 3 on electron 2.
    nuc_spins: List[float] = field(default_factory=lambda: [0.5] * 6)
    n_e1_nuc: int = 3
    n_e2_nuc: int = 3

    # Hyperfine tensors in MHz, inherited from the recent baseline script.
    A_tensors_MHz: List[np.ndarray] = field(default_factory=lambda: [
        np.array([[-2.44, 0.00, 0.00],
                  [ 0.00,-2.80, 0.00],
                  [ 0.00, 0.00,49.24]], dtype=float),
        np.array([[12.00, 0.00, 0.00],
                  [ 0.00,12.00, 0.00],
                  [ 0.00, 0.00,12.00]], dtype=float),
        np.array([[-0.72, 0.08, 0.00],
                  [ 0.08,-0.64, 0.00],
                  [ 0.00, 0.00, 1.40]], dtype=float),
        np.array([[-1.70, 0.00, 0.00],
                  [ 0.00,-1.50, 0.00],
                  [ 0.00, 0.00,30.00]], dtype=float),
        np.array([[-15.0, 0.00, 0.00],
                  [  0.0,-11.0, 0.00],
                  [  0.0, 0.00,-3.00]], dtype=float),
        np.array([[ -1.0, 0.00, 0.00],
                  [  0.0,-0.90, 0.00],
                  [  0.0, 0.00, 2.40]], dtype=float),
    ])

    # -----------------------------
    # Exchange and full non-secular dipole
    # -----------------------------
    r_nm: float = 2.0
    D_sign: float = -1.0

    # Keep the old D/J ratio convention: ratio_x = D/J, so J = D / ratio_x.
    # ratio_x is the active value for one run; ratio_values is used by the D/J sweep.
    ratio_x: float = -3.0
    ratio_values: np.ndarray = field(default_factory=lambda: np.r_[
        np.linspace(-10.0, -0.5, 20),
        np.array([-1.0 / 3.0, -2.0 / 3.0]),
        np.linspace(0.5, 10.0, 20),
    ])

    # Fixed molecular/inter-radical axis used in H_dip.
    # theta_r = 0 means rhat = +z; theta_r = 90, phi_r = 0 means rhat = +x.
    dipole_axis_theta_deg: float = 0.0
    dipole_axis_phi_deg: float = 0.0

    # Filled automatically for each run.
    J_MHz: float = 0.0
    D_MHz: float = 0.0
    B_mT: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.05]))

    # -----------------------------
    # Time evolution
    # -----------------------------
    t_max_us: float = 10.0
    n_steps: int = 2400
    init_e_state: str = "S"   # S, T0, Tp, Tm

    # Plenio/Cai/Caruso-style recombination-weighted singlet yield.
    # The paper uses Phi_S = int_0^infty k exp(-k t) f_S(t) dt.
    # Numerically we approximate the upper limit by t_max_us; choose t_max_us
    # so exp(-k*t_max_us) is small. In the paper, k = 0.5 us^-1 is a common
    # example, corresponding to a radical-pair lifetime tau = k^-1 = 2 us.
    recombination_rate_per_us: float = 0.5
    use_plenio_yield: bool = True
    normalize_truncated_plenio_yield: bool = False

    # -----------------------------
    # Plot / output controls
    # -----------------------------
    fft_fmax_MHz: float = 80.0
    plot_each_run: bool = False
    plot_heatmaps: bool = False
    plot_ratio_summary: bool = True
    save_ratio_csv: bool = True
    ratio_csv_path: str = "DJ_ratio_sweep_results.csv"
    verbose: bool = True


def copy_params(P: Params) -> Params:
    """Dataclass copy that avoids mutating the user-facing Params object."""
    return replace(
        P,
        B_mT=np.array(P.B_mT, dtype=float).copy(),
        theta_values_deg=np.array(P.theta_values_deg, dtype=float).copy(),
        phi_values_deg=np.array(P.phi_values_deg, dtype=float).copy(),
        ratio_values=np.array(P.ratio_values, dtype=float).copy(),
        A_tensors_MHz=[np.array(A, dtype=float).copy() for A in P.A_tensors_MHz],
        nuc_spins=list(P.nuc_spins),
    )


def symmetrize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def subsystem_dims(P: Params) -> List[int]:
    return [2, 2] + [2 for _ in P.nuc_spins]


def electron_for_nucleus(P: Params, k: int) -> int:
    if k < P.n_e1_nuc:
        return 0
    if k < P.n_e1_nuc + P.n_e2_nuc:
        return 1
    raise ValueError("Nucleus index exceeds mapping. Check n_e1_nuc, n_e2_nuc, and A_tensors_MHz.")


def dipolar_uT_from_r_nm(r_nm: float) -> float:
    """Magnitude of point-dipole coupling in microtesla: |D| = 2783 / r_nm^3."""
    return 2.783e3 / (r_nm ** 3)


def set_B_from_angles(P: Params, theta_deg: float, phi_deg: float) -> None:
    """Set B_mT from B_microT and full spherical angles."""
    B_mT_mag = P.B_microT * 1e-3
    P.B_mT = B_mT_mag * unit_vector_from_angles(theta_deg, phi_deg)


def set_JD_from_ratio(P: Params) -> None:
    """Set D_MHz and J_MHz from D = ratio_x * J."""
    if abs(P.ratio_x) < 1e-15:
        raise ValueError("ratio_x cannot be zero because J = D / ratio_x would diverge.")

    D_uT = P.D_sign * dipolar_uT_from_r_nm(P.r_nm)
    J_uT = D_uT / float(P.ratio_x)

    P.D_MHz = P.gamma_e_MHz_per_mT * 1e-3 * D_uT
    P.J_MHz = P.gamma_e_MHz_per_mT * 1e-3 * J_uT


def zeeman_frequency_MHz(P: Params) -> float:
    """Electron Zeeman frequency in MHz for the current field magnitude."""
    return P.gamma_e_MHz_per_mT * np.linalg.norm(P.B_mT)


def op_on(P: Params, which: int, op: Qobj) -> Qobj:
    ops = [qeye(d) for d in subsystem_dims(P)]
    ops[which] = op
    return tensor(ops)


def build_spin_ops(P: Params) -> Dict[str, List[Qobj]]:
    Sx, Sy, Sz = [], [], []
    for i, d in enumerate(subsystem_dims(P)):
        if d != 2:
            raise ValueError("This baseline assumes spin-1/2 nuclei only.")
        Sx.append(0.5 * op_on(P, i, sigmax()))
        Sy.append(0.5 * op_on(P, i, sigmay()))
        Sz.append(0.5 * op_on(P, i, sigmaz()))
    return {"Sx": Sx, "Sy": Sy, "Sz": Sz}


def electronic_projectors() -> Dict[str, Qobj]:
    up = basis(2, 0)
    dn = basis(2, 1)
    uu = tensor(up, up)
    dd = tensor(dn, dn)
    ud = tensor(up, dn)
    du = tensor(dn, up)
    singlet = (ud - du).unit()
    trip0 = (ud + du).unit()
    return {
        "S": singlet * singlet.dag(),
        "T0": trip0 * trip0.dag(),
        "Tp": uu * uu.dag(),
        "Tm": dd * dd.dag(),
    }


def full_ST_projectors(P: Params) -> Dict[str, Qobj]:
    """Electronic S/T projectors embedded in the full Hilbert space.

    Supports the N=0 electron-only case used in the finite-size scaling test.
    """
    eproj = electronic_projectors()
    if len(P.nuc_spins) == 0:
        return eproj
    In = tensor([qeye(2) for _ in P.nuc_spins])
    return {k: tensor(v, In) for k, v in eproj.items()}


def initial_electron_density(P: Params) -> Qobj:
    projs = electronic_projectors()
    if P.init_e_state not in projs:
        raise ValueError("init_e_state must be one of: S, T0, Tp, Tm.")
    return projs[P.init_e_state]


def maximally_mixed_nuclear_state(P: Params) -> Qobj | None:
    """Maximally mixed nuclear state, or None for the N=0 electron-only case."""
    if len(P.nuc_spins) == 0:
        return None
    return tensor([qeye(2) / 2.0 for _ in P.nuc_spins])


def full_identity(P: Params) -> Qobj:
    return tensor([qeye(d) for d in subsystem_dims(P)])


def build_exchange_and_full_dipole_terms(
    P: Params,
    Sx: List[Qobj],
    Sy: List[Qobj],
    Sz: List[Qobj],
) -> Tuple[Qobj, Qobj]:
    """Build exchange and full non-secular magnetic dipole-dipole terms."""
    e1, e2 = 0, 1
    I_full = full_identity(P)

    Sxt = Sx[e1] + Sx[e2]
    Syt = Sy[e1] + Sy[e2]
    Szt = Sz[e1] + Sz[e2]
    S2_total = Sxt * Sxt + Syt * Syt + Szt * Szt

    Jw = mhz_to_rad_per_us(P.J_MHz)
    Dw = mhz_to_rad_per_us(P.D_MHz)

    # Same exchange convention as the previous baseline script.
    HJ = -Jw * (S2_total - I_full)

    rhat = unit_vector_from_angles(P.dipole_axis_theta_deg, P.dipole_axis_phi_deg)
    S1_dot_S2 = Sx[e1] * Sx[e2] + Sy[e1] * Sy[e2] + Sz[e1] * Sz[e2]
    S1_dot_r = rhat[0] * Sx[e1] + rhat[1] * Sy[e1] + rhat[2] * Sz[e1]
    S2_dot_r = rhat[0] * Sx[e2] + rhat[1] * Sy[e2] + rhat[2] * Sz[e2]

    # Full point-dipole term. This is intentionally not secularized.
    HD = Dw * (S1_dot_S2 - 3.0 * S1_dot_r * S2_dot_r)

    return HJ, HD


def build_full_hamiltonian(P: Params) -> Tuple[Qobj, Dict[str, Qobj]]:
    """Build H = H_Zeeman + H_hyperfine + H_exchange + H_full_dipole."""
    if len(P.A_tensors_MHz) != len(P.nuc_spins):
        raise ValueError("A_tensors_MHz length must equal nuc_spins length.")

    ops = build_spin_ops(P)
    Sx, Sy, Sz = ops["Sx"], ops["Sy"], ops["Sz"]
    e1, e2 = 0, 1
    H = 0 * Sx[0]

    # Electron Zeeman term.
    omega_e = mhz_to_rad_per_us(P.gamma_e_MHz_per_mT) * P.B_mT
    H += omega_e[0] * (Sx[e1] + Sx[e2])
    H += omega_e[1] * (Sy[e1] + Sy[e2])
    H += omega_e[2] * (Sz[e1] + Sz[e2])

    # Full quantum electron-nuclear hyperfine terms.
    spin_axes = [Sx, Sy, Sz]
    for k, A in enumerate(P.A_tensors_MHz):
        e = electron_for_nucleus(P, k)
        ni = 2 + k
        A_w = mhz_to_rad_per_us(symmetrize(A))
        for a in range(3):
            for b in range(3):
                coeff = A_w[a, b]
                if abs(coeff) > 0.0:
                    H += coeff * (spin_axes[a][e] * spin_axes[b][ni])

    HJ, HD = build_exchange_and_full_dipole_terms(P, Sx, Sy, Sz)
    H += HJ + HD

    return H, {"HJ": HJ, "HD_full_nonsecular": HD}


def run_full_quantum(P: Params) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Evolve the full quantum model and return S/T populations."""
    rho_e0 = initial_electron_density(P)
    rho_n0 = maximally_mixed_nuclear_state(P)
    rho0 = rho_e0 if rho_n0 is None else tensor(rho_e0, rho_n0)
    H, _terms = build_full_hamiltonian(P)
    tlist = np.linspace(0.0, P.t_max_us, P.n_steps)

    projs = full_ST_projectors(P)
    labels = ["S", "T0", "Tp", "Tm"]

    out = mesolve(
        H,
        rho0,
        tlist,
        c_ops=[],
        e_ops=[projs[k] for k in labels],
        options={"store_states": False},
        progress_bar=None,
    )

    pops = {label: np.real(np.asarray(out.expect[i])) for i, label in enumerate(labels)}
    pops["T_total"] = pops["T0"] + pops["Tp"] + pops["Tm"]
    pops["ST_sum"] = pops["S"] + pops["T_total"]
    return tlist, pops


def running_time_average(t_us: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Running average ybar(t) = (1/t) int_0^t y(t') dt'."""
    t_us = np.asarray(t_us, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t_us) != len(y):
        raise ValueError("t_us and y must have the same length.")
    if len(t_us) < 2:
        return y.copy()

    dt = np.diff(t_us)
    increments = 0.5 * (y[1:] + y[:-1]) * dt
    cumulative_area = np.concatenate(([0.0], np.cumsum(increments)))
    elapsed = t_us - t_us[0]

    avg = np.empty_like(y, dtype=float)
    avg[0] = y[0]
    avg[1:] = cumulative_area[1:] / elapsed[1:]
    return avg


def finite_time_average_yield(t_us: np.ndarray, Ps: np.ndarray) -> float:
    """Final-time singlet average used in the older sweep scripts."""
    return float(running_time_average(t_us, Ps)[-1])


def plenio_singlet_yield(
    t_us: np.ndarray,
    Ps: np.ndarray,
    k_per_us: float,
    normalize_truncated: bool = False,
) -> float:
    """
    Cai-Caruso-Plenio / conventional phenomenological RPM singlet yield.

    The paper defines

        Phi_S = int_0^infty r(t) f_S(t) dt,
        r(t) = k exp(-k t),
        f_S(t) = <S|rho_s(t)|S>.

    In this code, Ps(t) is exactly f_S(t) because the singlet projector is
    evaluated on the full electron+nuclear state, equivalently after tracing
    out the nuclei. The integral is truncated at t_max_us, so choose t_max_us
    such that exp(-k*t_max_us) is small.

    Set normalize_truncated=True only for diagnostic comparisons on short time
    windows. The paper's definition is the unnormalized truncated integral
    approaching the infinite-time integral.
    """
    t_us = np.asarray(t_us, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    k = float(k_per_us)
    if k <= 0.0:
        raise ValueError('k_per_us must be positive.')
    weight = k * np.exp(-k * t_us)
    phi = float(np.trapz(weight * Ps, t_us))
    if normalize_truncated:
        captured_probability = 1.0 - np.exp(-k * (t_us[-1] - t_us[0]))
        if captured_probability > 0.0:
            phi /= captured_probability
    return phi


def recombination_weighted_yield(t_us: np.ndarray, Ps: np.ndarray, k_per_us: float) -> float:
    """Backward-compatible alias for plenio_singlet_yield(...)."""
    return plenio_singlet_yield(t_us, Ps, k_per_us, normalize_truncated=False)


def recombination_survival_tail(t_max_us: float, k_per_us: float) -> float:
    """Remaining recombination probability beyond t_max_us: exp(-k*T)."""
    return float(np.exp(-float(k_per_us) * float(t_max_us)))


def compute_fft(
    t_us: np.ndarray,
    y: np.ndarray,
    remove_mean: bool = True,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided FFT amplitude spectrum. Time in us -> frequency in MHz."""
    t_us = np.asarray(t_us, dtype=float)
    y = np.asarray(y, dtype=float).copy()

    if len(t_us) < 2:
        raise ValueError("Need at least two time points for FFT.")

    dt = t_us[1] - t_us[0]

    if remove_mean:
        y -= np.mean(y)

    if window:
        w = np.hanning(len(y))
        y *= w
        norm = np.sum(w) / 2.0
    else:
        norm = len(y) / 2.0

    Y = np.fft.rfft(y)
    f_MHz = np.fft.rfftfreq(len(y), d=dt)
    amp = np.abs(Y) / max(norm, 1e-15)
    return f_MHz, amp


def top_fft_peaks(
    f_MHz: np.ndarray,
    amp: np.ndarray,
    n_peaks: int = 5,
    f_min_MHz: float = 0.05,
    f_max_MHz: float | None = None,
) -> List[Tuple[float, float]]:
    """Return the largest simple FFT peaks as (frequency_MHz, amplitude)."""
    mask = f_MHz >= f_min_MHz
    if f_max_MHz is not None:
        mask &= f_MHz <= f_max_MHz

    f = f_MHz[mask]
    a = amp[mask]
    if len(a) == 0:
        return []

    candidate = np.zeros_like(a, dtype=bool)
    if len(a) >= 3:
        candidate[1:-1] = (a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])
    else:
        candidate[:] = True

    idx = np.where(candidate)[0]
    if len(idx) == 0:
        idx = np.arange(len(a))

    idx = idx[np.argsort(a[idx])[::-1]][:n_peaks]
    return [(float(f[i]), float(a[i])) for i in idx]


def trace_rms_difference(y: np.ndarray, y_ref: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    if y.shape != y_ref.shape:
        raise ValueError("Trace arrays must have the same shape for RMS comparison.")
    return float(np.sqrt(np.mean((y - y_ref) ** 2)))


def plot_single_run(P: Params, theta_deg: float, phi_deg: float, t_us: np.ndarray, pops: Dict[str, np.ndarray]) -> None:
    """Plot S/T dynamics, running average, recombination yield, and singlet FFT."""
    Ps_running = running_time_average(t_us, pops["S"])
    avg_yield = float(Ps_running[-1])
    rec_yield = plenio_singlet_yield(
        t_us, pops["S"], P.recombination_rate_per_us,
        normalize_truncated=P.normalize_truncated_plenio_yield,
    )
    f_MHz, amp = compute_fft(t_us, pops["S"], remove_mean=True, window=True)
    fft_mask = f_MHz <= P.fft_fmax_MHz

    title = (
        f"Full non-secular dipole RPM | D/J={P.ratio_x:g} | "
        f"D={P.D_MHz:.4f} MHz, J={P.J_MHz:.4f} MHz | "
        f"B={P.B_microT:.1f} uT, theta={theta_deg:.1f}, phi={phi_deg:.1f} deg | "
        f"rhat=({P.dipole_axis_theta_deg:.1f},{P.dipole_axis_phi_deg:.1f}) deg"
    )

    fig, axes = plt.subplots(1, 4, figsize=(25, 4.8))
    fig.suptitle(title, y=1.04, fontsize=12)

    ax = axes[0]
    ax.plot(t_us, pops["S"], label="S", linewidth=2.0)
    ax.plot(t_us, pops["T_total"], label="T total", linewidth=2.0)
    ax.set_xlabel("time (us)")
    ax.set_ylabel("population")
    ax.set_title("Singlet vs triplet")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(t_us, pops["T0"], label="T0", linewidth=2.0)
    ax.plot(t_us, pops["Tp"], label="T+", linewidth=2.0)
    ax.plot(t_us, pops["Tm"], label="T-", linewidth=2.0)
    ax.plot(t_us, pops["ST_sum"], ":", label="S+T", linewidth=1.4)
    ax.set_xlabel("time (us)")
    ax.set_ylabel("population")
    ax.set_title("Triplet sublevels")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(t_us, Ps_running, linewidth=2.0, label="running <P_S>")
    ax.axhline(avg_yield, linestyle=":", linewidth=1.5, label=f"avg = {avg_yield:.4f}")
    ax.axhline(rec_yield, linestyle="--", linewidth=1.2, label=f"rec. yield = {rec_yield:.4f}")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("singlet metric")
    ax.set_title("Singlet averages/yields")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)

    ax = axes[3]
    ax.plot(f_MHz[fft_mask], amp[fft_mask], linewidth=2.0, label="FFT of P_S")
    ax.axvline(zeeman_frequency_MHz(P), linestyle=":", linewidth=1.5,
               label=f"Zeeman = {zeeman_frequency_MHz(P):.2f} MHz")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("amplitude")
    ax.set_title("Singlet FFT")
    ax.legend(frameon=False)

    fig.tight_layout()
    plt.show()


def run_one_case(base_params: Params, theta_deg: float, phi_deg: float, plot: bool = False) -> Dict:
    """Run one magnetic-field orientation case."""
    P = copy_params(base_params)
    set_JD_from_ratio(P)
    set_B_from_angles(P, theta_deg, phi_deg)

    if P.verbose:
        print("=" * 96)
        print(f"Running full non-secular dipole RPM: theta = {theta_deg:g} deg, phi = {phi_deg:g} deg")
        print(f"D/J = {P.ratio_x:.8g}; D = {P.D_MHz:.8f} MHz; J = {P.J_MHz:.8f} MHz")
        print(f"B = {P.B_microT:.3f} uT; Zeeman = {zeeman_frequency_MHz(P):.6f} MHz")
        print(f"Dipole axis rhat angles = ({P.dipole_axis_theta_deg:g}, {P.dipole_axis_phi_deg:g}) deg")

    t_us, pops = run_full_quantum(P)
    avg_yield = finite_time_average_yield(t_us, pops["S"])
    rec_yield = plenio_singlet_yield(
        t_us, pops["S"], P.recombination_rate_per_us,
        normalize_truncated=P.normalize_truncated_plenio_yield,
    )
    f_MHz, amp = compute_fft(t_us, pops["S"], remove_mean=True, window=True)
    peaks = top_fft_peaks(f_MHz, amp, n_peaks=5, f_max_MHz=P.fft_fmax_MHz)

    result = {
        "params": P,
        "theta_deg": float(theta_deg),
        "phi_deg": float(phi_deg),
        "t_us": t_us,
        "pops": pops,
        "running_singlet_average": running_time_average(t_us, pops["S"]),
        "finite_time_average_yield": avg_yield,
        "plenio_singlet_yield": rec_yield,
        "recombination_weighted_yield": rec_yield,  # backward-compatible key
        "selected_yield": rec_yield if P.use_plenio_yield else avg_yield,
        "fft_f_MHz": f_MHz,
        "fft_amp": amp,
        "top_fft_peaks": peaks,
        "st_sum_max_error": float(np.max(np.abs(pops["ST_sum"] - 1.0))),
    }

    if P.verbose:
        print(f"Finite-time <P_S> = {avg_yield:.8f}")
        print(f"Recombination-weighted Phi_S = {rec_yield:.8f}")
        print(f"Max |P_S + P_T - 1| = {result['st_sum_max_error']:.3e}")
        if peaks:
            peak_str = ", ".join([f"{freq:.3f} MHz" for freq, _ in peaks[:3]])
            print(f"Top FFT peak frequencies = {peak_str}")

    if plot:
        plot_single_run(P, theta_deg, phi_deg, t_us, pops)

    return result


def run_angle_sweep(P: Params) -> List[Dict]:
    """Run all polar and azimuthal magnetic-field orientations."""
    results = []
    for theta_deg in np.asarray(P.theta_values_deg, dtype=float):
        for phi_deg in np.asarray(P.phi_values_deg, dtype=float):
            results.append(run_one_case(P, theta_deg, phi_deg, plot=P.plot_each_run))
    return results


def build_angle_grid(results: List[Dict]) -> Dict[str, np.ndarray]:
    """Convert list of result dictionaries into theta-phi grids."""
    if not results:
        raise ValueError("No results to grid.")

    theta_values = np.array(sorted({r["theta_deg"] for r in results}), dtype=float)
    phi_values = np.array(sorted({r["phi_deg"] for r in results}), dtype=float)

    shape = (len(theta_values), len(phi_values))
    selected_yield = np.full(shape, np.nan, dtype=float)
    finite_avg = np.full(shape, np.nan, dtype=float)
    plenio_yield = np.full(shape, np.nan, dtype=float)
    st_err = np.full(shape, np.nan, dtype=float)
    top_peak_freq = np.full(shape, np.nan, dtype=float)
    top_peak_amp = np.full(shape, np.nan, dtype=float)

    # Reference trace is the first available orientation, often theta=0, phi=0.
    ref_trace = results[0]["pops"]["S"]
    rms_from_ref = np.full(shape, np.nan, dtype=float)

    for r in results:
        i = np.where(np.isclose(theta_values, r["theta_deg"]))[0][0]
        j = np.where(np.isclose(phi_values, r["phi_deg"]))[0][0]
        selected_yield[i, j] = r["selected_yield"]
        finite_avg[i, j] = r["finite_time_average_yield"]
        plenio_yield[i, j] = r["plenio_singlet_yield"]
        st_err[i, j] = r["st_sum_max_error"]
        rms_from_ref[i, j] = trace_rms_difference(r["pops"]["S"], ref_trace)
        if r["top_fft_peaks"]:
            top_peak_freq[i, j] = r["top_fft_peaks"][0][0]
            top_peak_amp[i, j] = r["top_fft_peaks"][0][1]

    return {
        "theta_values_deg": theta_values,
        "phi_values_deg": phi_values,
        "selected_yield": selected_yield,
        "finite_time_average_yield": finite_avg,
        "plenio_singlet_yield": plenio_yield,
        "recombination_weighted_yield": plenio_yield,  # backward-compatible key
        "rms_from_reference_trace": rms_from_ref,
        "top_fft_peak_frequency_MHz": top_peak_freq,
        "top_fft_peak_amplitude": top_peak_amp,
        "st_sum_max_error": st_err,
    }


def print_angle_summary(results: List[Dict], grid: Dict[str, np.ndarray]) -> None:
    """Print compact summary of angle sweep results."""
    print("\nAngle sweep summary")
    print("-" * 96)
    metric = grid["selected_yield"]
    theta_values = grid["theta_values_deg"]
    phi_values = grid["phi_values_deg"]

    min_idx = np.unravel_index(np.nanargmin(metric), metric.shape)
    max_idx = np.unravel_index(np.nanargmax(metric), metric.shape)
    print(f"selected yield min = {metric[min_idx]:.8f} at theta={theta_values[min_idx[0]]:.3g}, phi={phi_values[min_idx[1]]:.3g}")
    print(f"selected yield max = {metric[max_idx]:.8f} at theta={theta_values[max_idx[0]]:.3g}, phi={phi_values[max_idx[1]]:.3g}")
    print(f"selected yield contrast = {metric[max_idx] - metric[min_idx]:.8e}")
    print(f"max trace RMS from reference = {np.nanmax(grid['rms_from_reference_trace']):.8e}")
    print(f"max population conservation error = {np.nanmax(grid['st_sum_max_error']):.3e}")
    if results:
        P0 = results[0]["params"]
        tail = recombination_survival_tail(P0.t_max_us, P0.recombination_rate_per_us)
        print(f"Plenio-yield recombination tail beyond t_max = exp(-kT) = {tail:.3e}")

    print("\nFirst few individual runs")
    print("-" * 96)
    print(f"{'theta':>10} {'phi':>10} {'avg<P_S>':>14} {'Phi_S rec':>14} {'RMS ref':>14} {'top FFT / MHz':>16}")
    for r in results[: min(10, len(results))]:
        theta = r["theta_deg"]
        phi = r["phi_deg"]
        i = np.where(np.isclose(theta_values, theta))[0][0]
        j = np.where(np.isclose(phi_values, phi))[0][0]
        peak = r["top_fft_peaks"][0][0] if r["top_fft_peaks"] else np.nan
        print(f"{theta:10.4g} {phi:10.4g} {r['finite_time_average_yield']:14.8f} {r['recombination_weighted_yield']:14.8f} {grid['rms_from_reference_trace'][i,j]:14.6e} {peak:16.6f}")
    print("-" * 96)


def _imshow_angle_grid(ax, phi_values, theta_values, Z, title, cbar_label):
    if len(phi_values) > 1:
        dx = np.mean(np.diff(phi_values))
    else:
        dx = 1.0
    if len(theta_values) > 1:
        dy = np.mean(np.diff(theta_values))
    else:
        dy = 1.0

    extent = [
        phi_values[0] - dx / 2.0,
        phi_values[-1] + dx / 2.0,
        theta_values[0] - dy / 2.0,
        theta_values[-1] + dy / 2.0,
    ]
    im = ax.imshow(Z, aspect="auto", origin="lower", extent=extent)
    ax.set_xlabel("azimuthal angle phi (deg)")
    ax.set_ylabel("polar angle theta (deg)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    return im


def plot_angle_heatmaps(grid: Dict[str, np.ndarray], P: Params) -> None:
    """Plot core theta-phi heatmaps for the angle sweep."""
    theta_values = grid["theta_values_deg"]
    phi_values = grid["phi_values_deg"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Full non-secular dipole RPM orientation sweep\n"
        f"D/J={P.ratio_x:g}, B={P.B_microT:g} uT, r={P.r_nm:g} nm, "
        f"rhat=({P.dipole_axis_theta_deg:g},{P.dipole_axis_phi_deg:g}) deg",
        y=1.02,
        fontsize=12,
    )

    _imshow_angle_grid(
        axes[0, 0], phi_values, theta_values, grid["selected_yield"],
        "Selected singlet yield metric", "yield",
    )
    _imshow_angle_grid(
        axes[0, 1], phi_values, theta_values, grid["rms_from_reference_trace"],
        "RMS deviation from reference P_S(t)", "RMS",
    )
    _imshow_angle_grid(
        axes[1, 0], phi_values, theta_values, grid["top_fft_peak_frequency_MHz"],
        "Dominant FFT peak frequency", "MHz",
    )
    _imshow_angle_grid(
        axes[1, 1], phi_values, theta_values, grid["top_fft_peak_amplitude"],
        "Dominant FFT peak amplitude", "amplitude",
    )

    fig.tight_layout()
    plt.show()


def plot_representative_traces(results: List[Dict], grid: Dict[str, np.ndarray]) -> None:
    """Plot traces for orientations with min, max, and largest RMS deviation."""
    metric = grid["selected_yield"]
    rms = grid["rms_from_reference_trace"]
    theta_values = grid["theta_values_deg"]
    phi_values = grid["phi_values_deg"]

    target_indices = [
        np.unravel_index(np.nanargmin(metric), metric.shape),
        np.unravel_index(np.nanargmax(metric), metric.shape),
        np.unravel_index(np.nanargmax(rms), rms.shape),
    ]
    labels = ["min yield", "max yield", "max RMS"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, label in zip(target_indices, labels):
        theta = theta_values[idx[0]]
        phi = phi_values[idx[1]]
        match = next(r for r in results if np.isclose(r["theta_deg"], theta) and np.isclose(r["phi_deg"], phi))
        ax.plot(match["t_us"], match["pops"]["S"], linewidth=2.0,
                label=f"{label}: theta={theta:g}, phi={phi:g}")

    ax.set_xlabel("time (us)")
    ax.set_ylabel("P_S(t)")
    ax.set_title("Representative orientation-dependent singlet dynamics")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()



def azimuthal_contrast_by_theta(grid: Dict[str, np.ndarray], metric_key: str = "selected_yield") -> Dict[str, np.ndarray]:
    """
    Compute Plenio-style azimuthal sensitivity metrics at each fixed theta.

    For each polar angle theta, this returns
        Delta_phi = max_phi Phi_S(theta,phi) - min_phi Phi_S(theta,phi)
        C_phi     = Delta_phi / mean_phi Phi_S(theta,phi)
    along with the phi values where the extrema occur.
    """
    theta_values = grid["theta_values_deg"]
    phi_values = grid["phi_values_deg"]
    Z = np.asarray(grid[metric_key], dtype=float)

    delta_phi = np.nanmax(Z, axis=1) - np.nanmin(Z, axis=1)
    mean_phi = np.nanmean(Z, axis=1)
    relative_contrast = np.divide(delta_phi, mean_phi, out=np.full_like(delta_phi, np.nan), where=mean_phi != 0)

    min_j = np.nanargmin(Z, axis=1)
    max_j = np.nanargmax(Z, axis=1)

    return {
        "theta_values_deg": theta_values,
        "delta_phi": delta_phi,
        "relative_contrast": relative_contrast,
        "phi_min_yield_deg": phi_values[min_j],
        "phi_max_yield_deg": phi_values[max_j],
        "min_yield": np.nanmin(Z, axis=1),
        "max_yield": np.nanmax(Z, axis=1),
        "mean_yield": mean_phi,
    }


def plot_azimuthal_yield_curves(grid: Dict[str, np.ndarray], metric_key: str = "plenio_singlet_yield") -> None:
    """Plot Phi_S(phi) for every fixed theta in the grid."""
    theta_values = grid["theta_values_deg"]
    phi_values = grid["phi_values_deg"]
    Z = np.asarray(grid[metric_key], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for i, theta in enumerate(theta_values):
        ax.plot(phi_values, Z[i, :], marker="o", linewidth=2.0, label=f"theta={theta:g} deg")
    ax.set_xlabel("azimuthal angle phi (deg)")
    ax.set_ylabel("Plenio/Cai singlet yield Phi_S")
    ax.set_title("Azimuthal singlet-yield sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_azimuthal_contrast(grid: Dict[str, np.ndarray], metric_key: str = "plenio_singlet_yield") -> None:
    """Plot Delta_phi and relative azimuthal contrast as functions of theta."""
    c = azimuthal_contrast_by_theta(grid, metric_key=metric_key)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(c["theta_values_deg"], c["delta_phi"], marker="o", linewidth=2.0, label="absolute contrast")
    ax.set_xlabel("polar angle theta (deg)")
    ax.set_ylabel("Delta_phi Phi_S")
    ax.set_title("Azimuthal contrast versus polar angle")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def print_azimuthal_contrast_table(grid: Dict[str, np.ndarray], metric_key: str = "plenio_singlet_yield") -> None:
    """Print max-min azimuthal yield contrast for each fixed polar angle."""
    c = azimuthal_contrast_by_theta(grid, metric_key=metric_key)

    print("\nAzimuthal sensitivity summary")
    print("-" * 104)
    print(
        f"{'theta / deg':>12} {'min Phi_S':>14} {'phi_min / deg':>14} "
        f"{'max Phi_S':>14} {'phi_max / deg':>14} {'Delta_phi':>14} {'rel. contrast':>14}"
    )
    print("-" * 104)

    for i, theta in enumerate(c["theta_values_deg"]):
        print(
            f"{theta:12.3f} "
            f"{c['min_yield'][i]:14.8f} {c['phi_min_yield_deg'][i]:14.3f} "
            f"{c['max_yield'][i]:14.8f} {c['phi_max_yield_deg'][i]:14.3f} "
            f"{c['delta_phi'][i]:14.8e} {c['relative_contrast'][i]:14.8e}"
        )
    print("-" * 104)


def run_baseline_azimuthal_sweep() -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Convenience run for the first azimuthal sensitivity test.

    Uses the three polar angles selected from the baseline polar sweep:
        theta = 50, 90, 130 deg
    and sweeps phi over the full 0--360 deg azimuthal range.
    """
    P = Params()

    # Selected from the baseline polar sweep: two yield maxima and the central minimum.
    P.theta_values_deg = np.array([50.0, 90.0, 130.0])

    # 5-degree azimuthal resolution. Use 145 points for 2.5-degree resolution later.
    P.phi_values_deg = np.linspace(0.0, 360.0, 73, endpoint=False)

    # Keep the baseline chemistry/yield setup.
    P.recombination_rate_per_us = 0.5
    P.t_max_us = 10.0

    # Keep output focused on the azimuthal curves and contrast table.
    P.plot_each_run = False
    P.plot_heatmaps = False
    P.verbose = True

    results, grid = main(P)
    plot_azimuthal_yield_curves(grid, metric_key="plenio_singlet_yield")
    plot_azimuthal_contrast(grid, metric_key="plenio_singlet_yield")
    print_azimuthal_contrast_table(grid, metric_key="plenio_singlet_yield")

    return results, grid



# -----------------------------------------------------------------------------
# D/J ratio sweep utilities
# -----------------------------------------------------------------------------

def cleaned_ratio_values(ratio_values: np.ndarray, zero_tol: float = 1e-12) -> np.ndarray:
    """Return sorted unique D/J values with zero removed."""
    vals = np.asarray(ratio_values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[np.abs(vals) > zero_tol]
    vals = np.unique(np.round(vals, decimals=12))
    vals.sort()
    if len(vals) == 0:
        raise ValueError("No nonzero ratio values supplied. D/J = 0 is forbidden because J = D/(D/J).")
    return vals


def run_ratio_sweep(P: Params) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Sweep D/J over P.ratio_values for the fixed theta/phi orientations in P.

    Default intended use:
        theta = [50, 90] deg, phi = [0] deg, D/J from -10 to 10 excluding 0.

    The output grid has shape (n_ratio, n_theta, n_phi) for yield-like metrics.
    """
    ratios = cleaned_ratio_values(P.ratio_values)
    results: List[Dict] = []
    total = len(ratios) * len(P.theta_values_deg) * len(P.phi_values_deg)
    counter = 0

    for ratio_x in ratios:
        P_ratio = copy_params(P)
        P_ratio.ratio_x = float(ratio_x)
        for theta_deg in np.asarray(P.theta_values_deg, dtype=float):
            for phi_deg in np.asarray(P.phi_values_deg, dtype=float):
                counter += 1
                if P.verbose:
                    print(f"\nD/J sweep case {counter}/{total}: ratio={ratio_x:g}, theta={theta_deg:g}, phi={phi_deg:g}")
                results.append(run_one_case(P_ratio, theta_deg, phi_deg, plot=P.plot_each_run))

    grid = build_ratio_grid(results)
    print_ratio_summary(grid)

    if P.save_ratio_csv:
        save_ratio_grid_csv(grid, P.ratio_csv_path)
        print(f"Saved ratio-sweep CSV to: {P.ratio_csv_path}")

    if P.plot_ratio_summary:
        plot_ratio_yields(grid, metric_key="plenio_singlet_yield")
        plot_two_orientation_ratio_contrast(grid, metric_key="plenio_singlet_yield")
        plot_ratio_fft_summary(grid)

    return results, grid


def build_ratio_grid(results: List[Dict]) -> Dict[str, np.ndarray]:
    """Convert ratio-sweep result dictionaries into ratio-theta-phi grids."""
    if not results:
        raise ValueError("No results to grid.")

    ratios = np.array(sorted({r["params"].ratio_x for r in results}), dtype=float)
    theta_values = np.array(sorted({r["theta_deg"] for r in results}), dtype=float)
    phi_values = np.array(sorted({r["phi_deg"] for r in results}), dtype=float)

    shape = (len(ratios), len(theta_values), len(phi_values))
    selected_yield = np.full(shape, np.nan, dtype=float)
    finite_avg = np.full(shape, np.nan, dtype=float)
    plenio_yield = np.full(shape, np.nan, dtype=float)
    st_err = np.full(shape, np.nan, dtype=float)
    top_peak_freq = np.full(shape, np.nan, dtype=float)
    top_peak_amp = np.full(shape, np.nan, dtype=float)
    J_MHz = np.full(shape, np.nan, dtype=float)
    D_MHz = np.full(shape, np.nan, dtype=float)

    for r in results:
        ratio_x = r["params"].ratio_x
        i = np.where(np.isclose(ratios, ratio_x))[0][0]
        j = np.where(np.isclose(theta_values, r["theta_deg"]))[0][0]
        k = np.where(np.isclose(phi_values, r["phi_deg"]))[0][0]
        selected_yield[i, j, k] = r["selected_yield"]
        finite_avg[i, j, k] = r["finite_time_average_yield"]
        plenio_yield[i, j, k] = r["plenio_singlet_yield"]
        st_err[i, j, k] = r["st_sum_max_error"]
        J_MHz[i, j, k] = r["params"].J_MHz
        D_MHz[i, j, k] = r["params"].D_MHz
        if r["top_fft_peaks"]:
            top_peak_freq[i, j, k] = r["top_fft_peaks"][0][0]
            top_peak_amp[i, j, k] = r["top_fft_peaks"][0][1]

    return {
        "ratio_values": ratios,
        "theta_values_deg": theta_values,
        "phi_values_deg": phi_values,
        "selected_yield": selected_yield,
        "finite_time_average_yield": finite_avg,
        "plenio_singlet_yield": plenio_yield,
        "recombination_weighted_yield": plenio_yield,
        "top_fft_peak_frequency_MHz": top_peak_freq,
        "top_fft_peak_amplitude": top_peak_amp,
        "st_sum_max_error": st_err,
        "J_MHz": J_MHz,
        "D_MHz": D_MHz,
    }


def ratio_orientation_labels(grid: Dict[str, np.ndarray]) -> List[str]:
    labels = []
    for theta in grid["theta_values_deg"]:
        for phi in grid["phi_values_deg"]:
            labels.append(f"theta={theta:g}, phi={phi:g}")
    return labels


def flatten_orientation_grid(Z: np.ndarray) -> np.ndarray:
    """Convert Z[ratio, theta, phi] -> Z[ratio, orientation]."""
    return np.asarray(Z, dtype=float).reshape((Z.shape[0], Z.shape[1] * Z.shape[2]))


def print_ratio_summary(grid: Dict[str, np.ndarray], metric_key: str = "plenio_singlet_yield") -> None:
    """Print a compact summary of the D/J ratio sweep."""
    ratios = grid["ratio_values"]
    Z = np.asarray(grid[metric_key], dtype=float)
    Zflat = flatten_orientation_grid(Z)
    labels = ratio_orientation_labels(grid)

    # Contrast over all supplied orientations at each ratio. For the baseline two-orientation
    # run this is |Phi(theta=50,phi=0)-Phi(theta=90,phi=0)|.
    contrast = np.nanmax(Zflat, axis=1) - np.nanmin(Zflat, axis=1)
    best_i = int(np.nanargmax(contrast))

    print("\nD/J ratio sweep summary")
    print("-" * 118)
    print(f"Best orientation contrast = {contrast[best_i]:.8e} at D/J = {ratios[best_i]:.8g}")
    print(f"Global min yield = {np.nanmin(Zflat):.8f}; global max yield = {np.nanmax(Zflat):.8f}")
    print(f"Max population conservation error = {np.nanmax(grid['st_sum_max_error']):.3e}")

    print("\nFirst few ratio rows")
    print("-" * 118)
    header = f"{'D/J':>12} {'J/MHz':>12} {'D/MHz':>12} " + " ".join([f"{lab:>22}" for lab in labels]) + f" {'contrast':>14}"
    print(header)
    print("-" * 118)
    for i in range(min(12, len(ratios))):
        J = np.nanmean(grid["J_MHz"][i])
        D = np.nanmean(grid["D_MHz"][i])
        vals = " ".join([f"{v:22.8f}" for v in Zflat[i]])
        print(f"{ratios[i]:12.6g} {J:12.6f} {D:12.6f} {vals} {contrast[i]:14.8e}")
    print("-" * 118)


def save_ratio_grid_csv(grid: Dict[str, np.ndarray], path: str | Path, metric_key: str = "plenio_singlet_yield") -> None:
    """Save the ratio sweep in a long-form CSV table."""
    path = Path(path)
    ratios = grid["ratio_values"]
    theta_values = grid["theta_values_deg"]
    phi_values = grid["phi_values_deg"]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ratio_D_over_J", "theta_deg", "phi_deg", "J_MHz", "D_MHz",
            "plenio_singlet_yield", "finite_time_average_yield", "selected_yield",
            "top_fft_peak_frequency_MHz", "top_fft_peak_amplitude", "st_sum_max_error",
        ])
        for i, ratio in enumerate(ratios):
            for j, theta in enumerate(theta_values):
                for k, phi in enumerate(phi_values):
                    writer.writerow([
                        ratio, theta, phi,
                        grid["J_MHz"][i, j, k], grid["D_MHz"][i, j, k],
                        grid["plenio_singlet_yield"][i, j, k],
                        grid["finite_time_average_yield"][i, j, k],
                        grid["selected_yield"][i, j, k],
                        grid["top_fft_peak_frequency_MHz"][i, j, k],
                        grid["top_fft_peak_amplitude"][i, j, k],
                        grid["st_sum_max_error"][i, j, k],
                    ])


def plot_ratio_yields(grid: Dict[str, np.ndarray], metric_key: str = "plenio_singlet_yield") -> None:
    """Plot Phi_S(D/J) for every supplied fixed orientation."""
    ratios = grid["ratio_values"]
    Zflat = flatten_orientation_grid(grid[metric_key])
    labels = ratio_orientation_labels(grid)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for q, label in enumerate(labels):
        ax.plot(ratios, Zflat[:, q], marker="o", linewidth=2.0, label=label)

    ax.axvline(-3.0, linestyle=":", linewidth=1.4, label="D/J = -3")
    ax.axvline(-2.0 / 3.0, linestyle="--", linewidth=1.2, label="D/J = -2/3")
    ax.axvline(3.0, linestyle=":", linewidth=1.4, label="D/J = 3")
    ax.set_xlabel("exchange-dipole ratio D/J")
    ax.set_ylabel("Plenio/Cai singlet yield Phi_S")
    ax.set_title("Singlet yield versus exchange-dipole ratio")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_two_orientation_ratio_contrast(grid: Dict[str, np.ndarray], metric_key: str = "plenio_singlet_yield") -> None:
    """Plot max-min orientation contrast versus D/J for the supplied orientations."""
    ratios = grid["ratio_values"]
    Zflat = flatten_orientation_grid(grid[metric_key])
    contrast = np.nanmax(Zflat, axis=1) - np.nanmin(Zflat, axis=1)
    best_i = int(np.nanargmax(contrast))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(ratios, contrast, marker="o", linewidth=2.0, label="orientation contrast")
    ax.axvline(ratios[best_i], linestyle="--", linewidth=1.4, label=f"max at D/J={ratios[best_i]:g}")
    ax.axvline(-3.0, linestyle=":", linewidth=1.2, label="D/J = -3")
    ax.set_xlabel("exchange-dipole ratio D/J")
    ax.set_ylabel("Delta Phi_S over selected orientations")
    ax.set_title("Exchange-dipole ratio control of magnetic sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_ratio_fft_summary(grid: Dict[str, np.ndarray]) -> None:
    """Plot dominant FFT peak frequency versus D/J for each supplied orientation."""
    ratios = grid["ratio_values"]
    Fflat = flatten_orientation_grid(grid["top_fft_peak_frequency_MHz"])
    labels = ratio_orientation_labels(grid)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for q, label in enumerate(labels):
        ax.plot(ratios, Fflat[:, q], marker="o", linewidth=2.0, label=label)
    ax.set_xlabel("exchange-dipole ratio D/J")
    ax.set_ylabel("dominant FFT peak frequency (MHz)")
    ax.set_title("Dominant singlet-dynamics frequency versus D/J")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def run_baseline_DJ_ratio_sweep() -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Convenience run for the first exchange--dipole ratio sensitivity test.

    Uses two orientations from the baseline polar sweep:
        (theta, phi) = (50, 0) deg and (90, 0) deg
    and sweeps D/J from -10 to 10, excluding zero.
    """
    P = Params()

    # Cheap magnetic sensitivity proxy: high-yield-like and low-yield-like orientations.
    P.theta_values_deg = np.array([50.0, 90.0])
    P.phi_values_deg = np.array([0.0])

    # D/J = 0 is forbidden because J = D/(D/J). This gives a 0.5-spaced sweep.
    # Landmark values such as -3, -2/3, and 3 are automatically included or added.
    coarse = np.r_[np.arange(-10.0, 0.0, 0.5), np.arange(0.5, 10.0 + 0.5, 0.5)]
    landmarks = np.array([-3.0, -2.0 / 3.0, 3.0])
    P.ratio_values = np.unique(np.r_[coarse, landmarks])

    # Keep the baseline chemistry/yield setup.
    P.recombination_rate_per_us = 0.5
    P.t_max_us = 10.0

    # Keep output focused on ratio-sweep curves and CSV.
    P.plot_each_run = False
    P.plot_heatmaps = False
    P.plot_ratio_summary = True
    P.save_ratio_csv = True
    P.ratio_csv_path = "DJ_ratio_sweep_results.csv"
    P.verbose = True

    return run_ratio_sweep(P)


# -----------------------------------------------------------------------------
# Polar confirmation sweep utilities for selected D/J ratios
# -----------------------------------------------------------------------------

def build_polar_confirmation_grid(results: List[Dict]) -> Dict[str, np.ndarray]:
    """Convert selected-ratio polar sweep results into ratio-theta grids."""
    if not results:
        raise ValueError("No results to grid.")

    ratios = np.array(sorted({r["params"].ratio_x for r in results}), dtype=float)
    theta_values = np.array(sorted({r["theta_deg"] for r in results}), dtype=float)
    phi_values = np.array(sorted({r["phi_deg"] for r in results}), dtype=float)

    if len(phi_values) != 1:
        raise ValueError("Polar confirmation grid expects exactly one phi value.")

    shape = (len(ratios), len(theta_values))
    selected_yield = np.full(shape, np.nan, dtype=float)
    plenio_yield = np.full(shape, np.nan, dtype=float)
    finite_avg = np.full(shape, np.nan, dtype=float)
    top_peak_freq = np.full(shape, np.nan, dtype=float)
    top_peak_amp = np.full(shape, np.nan, dtype=float)
    st_err = np.full(shape, np.nan, dtype=float)
    J_MHz = np.full(shape, np.nan, dtype=float)
    D_MHz = np.full(shape, np.nan, dtype=float)

    for r in results:
        i = np.where(np.isclose(ratios, r["params"].ratio_x))[0][0]
        j = np.where(np.isclose(theta_values, r["theta_deg"]))[0][0]
        selected_yield[i, j] = r["selected_yield"]
        plenio_yield[i, j] = r["plenio_singlet_yield"]
        finite_avg[i, j] = r["finite_time_average_yield"]
        st_err[i, j] = r["st_sum_max_error"]
        J_MHz[i, j] = r["params"].J_MHz
        D_MHz[i, j] = r["params"].D_MHz
        if r["top_fft_peaks"]:
            top_peak_freq[i, j] = r["top_fft_peaks"][0][0]
            top_peak_amp[i, j] = r["top_fft_peaks"][0][1]

    contrast = np.nanmax(plenio_yield, axis=1) - np.nanmin(plenio_yield, axis=1)
    theta_min = theta_values[np.nanargmin(plenio_yield, axis=1)]
    theta_max = theta_values[np.nanargmax(plenio_yield, axis=1)]

    return {
        "ratio_values": ratios,
        "theta_values_deg": theta_values,
        "phi_values_deg": phi_values,
        "selected_yield": selected_yield,
        "plenio_singlet_yield": plenio_yield,
        "recombination_weighted_yield": plenio_yield,
        "finite_time_average_yield": finite_avg,
        "top_fft_peak_frequency_MHz": top_peak_freq,
        "top_fft_peak_amplitude": top_peak_amp,
        "st_sum_max_error": st_err,
        "J_MHz": J_MHz,
        "D_MHz": D_MHz,
        "polar_contrast": contrast,
        "theta_min_yield_deg": theta_min,
        "theta_max_yield_deg": theta_max,
        "min_yield": np.nanmin(plenio_yield, axis=1),
        "max_yield": np.nanmax(plenio_yield, axis=1),
        "mean_yield": np.nanmean(plenio_yield, axis=1),
    }


def print_polar_confirmation_summary(grid: Dict[str, np.ndarray]) -> None:
    """Print max-min polar-angle yield contrast for each selected D/J ratio."""
    print("\nPolar confirmation sweep summary")
    print("-" * 112)
    print(
        f"{'D/J':>10} {'J/MHz':>12} {'D/MHz':>12} "
        f"{'min Phi_S':>14} {'theta_min':>12} {'max Phi_S':>14} {'theta_max':>12} "
        f"{'Delta_theta':>14} {'rel. contrast':>14}"
    )
    print("-" * 112)
    for i, ratio in enumerate(grid["ratio_values"]):
        mean_y = grid["mean_yield"][i]
        rel = grid["polar_contrast"][i] / mean_y if mean_y != 0 else np.nan
        print(
            f"{ratio:10.4g} {np.nanmean(grid['J_MHz'][i]):12.6f} {np.nanmean(grid['D_MHz'][i]):12.6f} "
            f"{grid['min_yield'][i]:14.8f} {grid['theta_min_yield_deg'][i]:12.3f} "
            f"{grid['max_yield'][i]:14.8f} {grid['theta_max_yield_deg'][i]:12.3f} "
            f"{grid['polar_contrast'][i]:14.8e} {rel:14.8e}"
        )
    print("-" * 112)


def plot_polar_confirmation_yields(grid: Dict[str, np.ndarray]) -> None:
    """Plot Phi_S(theta) for each selected D/J ratio."""
    theta = grid["theta_values_deg"]
    Z = grid["plenio_singlet_yield"]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, ratio in enumerate(grid["ratio_values"]):
        ax.plot(theta, Z[i, :], marker="o", linewidth=2.0, label=f"D/J={ratio:g}")
    ax.set_xlabel("polar angle theta (deg)")
    ax.set_ylabel("Plenio/Cai singlet yield Phi_S")
    ax.set_title("Polar-angle magnetic sensitivity at selected exchange-dipole ratios")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_polar_confirmation_contrast(grid: Dict[str, np.ndarray]) -> None:
    """Plot polar-angle max-min contrast for each selected D/J ratio."""
    ratios = grid["ratio_values"]
    contrast = grid["polar_contrast"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar([str(f"{r:g}") for r in ratios], contrast)
    ax.set_xlabel("exchange-dipole ratio D/J")
    ax.set_ylabel("Delta_theta Phi_S")
    ax.set_title("Polar-angle contrast for selected exchange-dipole ratios")
    fig.tight_layout()
    plt.show()


def plot_polar_confirmation_fft(grid: Dict[str, np.ndarray]) -> None:
    """Plot dominant FFT peak frequency versus theta for each selected D/J ratio."""
    theta = grid["theta_values_deg"]
    F = grid["top_fft_peak_frequency_MHz"]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, ratio in enumerate(grid["ratio_values"]):
        ax.plot(theta, F[i, :], marker="o", linewidth=2.0, label=f"D/J={ratio:g}")
    ax.set_xlabel("polar angle theta (deg)")
    ax.set_ylabel("dominant FFT peak frequency (MHz)")
    ax.set_title("Dominant singlet-dynamics frequency across polar sweeps")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def save_polar_confirmation_csv(grid: Dict[str, np.ndarray], path: str | Path) -> None:
    """Save selected-ratio polar sweep data in long-form CSV format."""
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ratio_D_over_J", "theta_deg", "phi_deg", "J_MHz", "D_MHz",
            "plenio_singlet_yield", "finite_time_average_yield", "selected_yield",
            "top_fft_peak_frequency_MHz", "top_fft_peak_amplitude", "st_sum_max_error",
        ])
        phi = float(grid["phi_values_deg"][0])
        for i, ratio in enumerate(grid["ratio_values"]):
            for j, theta in enumerate(grid["theta_values_deg"]):
                writer.writerow([
                    ratio, theta, phi,
                    grid["J_MHz"][i, j], grid["D_MHz"][i, j],
                    grid["plenio_singlet_yield"][i, j],
                    grid["finite_time_average_yield"][i, j],
                    grid["selected_yield"][i, j],
                    grid["top_fft_peak_frequency_MHz"][i, j],
                    grid["top_fft_peak_amplitude"][i, j],
                    grid["st_sum_max_error"][i, j],
                ])


def run_selected_ratio_polar_confirmation() -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Confirm the D/J-ratio sweep by running full polar-angle sweeps for a few
    selected exchange-dipole ratios.

    Ratios:
        D/J = -6  : strongest two-orientation contrast from the coarse ratio sweep
        D/J = -3  : original baseline/cancellation-style point
        D/J =  7  : positive-ratio sensitivity bump
        D/J =  2  : low-yield positive-ratio regime

    Sweep:
        theta = 0, 5, ..., 180 deg
        phi   = 0 deg
    """
    base = Params()

    selected_ratios = np.array([-6.0, -3.0, 2.0, 7.0])
    theta_values = np.linspace(0.0, 180.0, 37)
    phi_values = np.array([0.0])

    base.theta_values_deg = theta_values
    base.phi_values_deg = phi_values
    base.recombination_rate_per_us = 0.5
    base.t_max_us = 10.0
    base.plot_each_run = False
    base.plot_heatmaps = False
    base.plot_ratio_summary = False
    base.save_ratio_csv = False
    base.verbose = True

    results: List[Dict] = []
    total = len(selected_ratios) * len(theta_values) * len(phi_values)
    counter = 0

    for ratio in selected_ratios:
        P_ratio = copy_params(base)
        P_ratio.ratio_x = float(ratio)
        for theta in theta_values:
            for phi in phi_values:
                counter += 1
                if base.verbose:
                    print(f"\nPolar confirmation case {counter}/{total}: D/J={ratio:g}, theta={theta:g}, phi={phi:g}")
                results.append(run_one_case(P_ratio, theta, phi, plot=False))

    grid = build_polar_confirmation_grid(results)
    print_polar_confirmation_summary(grid)

    csv_path = "DJ_selected_ratio_polar_confirmation.csv"
    save_polar_confirmation_csv(grid, csv_path)
    print(f"Saved polar confirmation CSV to: {csv_path}")

    plot_polar_confirmation_yields(grid)
    plot_polar_confirmation_contrast(grid)
    plot_polar_confirmation_fft(grid)

    return results, grid



# -----------------------------------------------------------------------------
# Finite-size scaling utilities
# -----------------------------------------------------------------------------

def hyperfine_frobenius_norm(A: np.ndarray) -> float:
    """Frobenius norm used to rank hyperfine tensors by strength."""
    A = np.asarray(A, dtype=float)
    return float(np.sqrt(np.sum(A * A)))


def ranked_nuclei_by_radical(P: Params) -> Tuple[List[int], List[int]]:
    """Return nucleus indices on each radical sorted by decreasing hyperfine norm."""
    e1_indices = list(range(P.n_e1_nuc))
    e2_indices = list(range(P.n_e1_nuc, P.n_e1_nuc + P.n_e2_nuc))

    e1_sorted = sorted(e1_indices, key=lambda k: hyperfine_frobenius_norm(P.A_tensors_MHz[k]), reverse=True)
    e2_sorted = sorted(e2_indices, key=lambda k: hyperfine_frobenius_norm(P.A_tensors_MHz[k]), reverse=True)
    return e1_sorted, e2_sorted


def make_finite_size_params(base: Params, total_nuclei: int) -> Tuple[Params, Dict]:
    """
    Create a nested finite-size model with N=0,2,4,6 nuclei.

    For N>0, this keeps the strongest N/2 hyperfine tensors on each radical,
    ranked separately by Frobenius norm. The selected tensors are then arranged
    as [all selected e1 tensors, all selected e2 tensors] so the existing
    electron_for_nucleus mapping remains valid.
    """
    N = int(total_nuclei)
    if N < 0 or N % 2 != 0:
        raise ValueError("total_nuclei must be one of the even values 0, 2, 4, 6 for this first test.")
    if N > len(base.A_tensors_MHz):
        raise ValueError("Requested more nuclei than are available in the base hyperfine list.")

    P = copy_params(base)
    e1_sorted, e2_sorted = ranked_nuclei_by_radical(base)
    n_each = N // 2

    if n_each > len(e1_sorted) or n_each > len(e2_sorted):
        raise ValueError("This nested test requires the same number of nuclei on each radical.")

    selected_e1 = e1_sorted[:n_each]
    selected_e2 = e2_sorted[:n_each]
    selected = selected_e1 + selected_e2

    P.A_tensors_MHz = [np.array(base.A_tensors_MHz[k], dtype=float).copy() for k in selected]
    P.nuc_spins = [0.5] * N
    P.n_e1_nuc = n_each
    P.n_e2_nuc = n_each

    selection_info = {
        "N": N,
        "selected_original_indices": selected,
        "selected_e1_original_indices": selected_e1,
        "selected_e2_original_indices": selected_e2,
        "selected_norms_MHz": [hyperfine_frobenius_norm(base.A_tensors_MHz[k]) for k in selected],
    }
    return P, selection_info


def build_finite_size_grid(results: List[Dict], selection_info_by_N: Dict[int, Dict]) -> Dict[str, np.ndarray]:
    """Convert finite-size sweep results into N-theta grids."""
    if not results:
        raise ValueError("No results to grid.")

    N_values = np.array(sorted({int(r["N_nuclei"]) for r in results}), dtype=int)
    theta_values = np.array(sorted({r["theta_deg"] for r in results}), dtype=float)
    phi_values = np.array(sorted({r["phi_deg"] for r in results}), dtype=float)
    if len(phi_values) != 1:
        raise ValueError("Finite-size first test expects exactly one phi value.")

    shape = (len(N_values), len(theta_values))
    selected_yield = np.full(shape, np.nan, dtype=float)
    plenio_yield = np.full(shape, np.nan, dtype=float)
    finite_avg = np.full(shape, np.nan, dtype=float)
    top_peak_freq = np.full(shape, np.nan, dtype=float)
    top_peak_amp = np.full(shape, np.nan, dtype=float)
    st_err = np.full(shape, np.nan, dtype=float)
    J_MHz = np.full(shape, np.nan, dtype=float)
    D_MHz = np.full(shape, np.nan, dtype=float)

    for r in results:
        i = np.where(N_values == int(r["N_nuclei"]))[0][0]
        j = np.where(np.isclose(theta_values, r["theta_deg"]))[0][0]
        selected_yield[i, j] = r["selected_yield"]
        plenio_yield[i, j] = r["plenio_singlet_yield"]
        finite_avg[i, j] = r["finite_time_average_yield"]
        st_err[i, j] = r["st_sum_max_error"]
        J_MHz[i, j] = r["params"].J_MHz
        D_MHz[i, j] = r["params"].D_MHz
        if r["top_fft_peaks"]:
            top_peak_freq[i, j] = r["top_fft_peaks"][0][0]
            top_peak_amp[i, j] = r["top_fft_peaks"][0][1]

    polar_contrast = np.nanmax(plenio_yield, axis=1) - np.nanmin(plenio_yield, axis=1)
    theta_min = theta_values[np.nanargmin(plenio_yield, axis=1)]
    theta_max = theta_values[np.nanargmax(plenio_yield, axis=1)]
    mean_yield = np.nanmean(plenio_yield, axis=1)

    # Convergence against the largest-N curve available, usually N=6.
    ref_i = int(np.argmax(N_values))
    ref_curve = plenio_yield[ref_i, :]
    curve_rms_error = np.sqrt(np.nanmean((plenio_yield - ref_curve[None, :]) ** 2, axis=1))
    contrast_error = np.abs(polar_contrast - polar_contrast[ref_i])

    return {
        "N_values": N_values,
        "theta_values_deg": theta_values,
        "phi_values_deg": phi_values,
        "selected_yield": selected_yield,
        "plenio_singlet_yield": plenio_yield,
        "recombination_weighted_yield": plenio_yield,
        "finite_time_average_yield": finite_avg,
        "top_fft_peak_frequency_MHz": top_peak_freq,
        "top_fft_peak_amplitude": top_peak_amp,
        "st_sum_max_error": st_err,
        "J_MHz": J_MHz,
        "D_MHz": D_MHz,
        "polar_contrast": polar_contrast,
        "theta_min_yield_deg": theta_min,
        "theta_max_yield_deg": theta_max,
        "min_yield": np.nanmin(plenio_yield, axis=1),
        "max_yield": np.nanmax(plenio_yield, axis=1),
        "mean_yield": mean_yield,
        "curve_rms_error_vs_largest_N": curve_rms_error,
        "contrast_error_vs_largest_N": contrast_error,
        "reference_N": int(N_values[ref_i]),
        "selection_info_by_N": selection_info_by_N,
    }


def print_finite_size_selection_table(grid: Dict[str, np.ndarray]) -> None:
    """Print which original hyperfine tensors are included at each N."""
    print("\nFinite-size nested hyperfine selection")
    print("-" * 100)
    print(f"{'N':>6} {'selected original indices':>30} {'Frobenius norms / MHz':>45}")
    print("-" * 100)
    for N in grid["N_values"]:
        info = grid["selection_info_by_N"][int(N)]
        indices = info["selected_original_indices"]
        norms = info["selected_norms_MHz"]
        print(f"{int(N):6d} {str(indices):>30} {str([round(x, 4) for x in norms]):>45}")
    print("-" * 100)


def print_finite_size_summary(grid: Dict[str, np.ndarray]) -> None:
    """Print finite-size scaling metrics."""
    print("\nFinite-size scaling summary")
    print("-" * 120)
    print(
        f"{'N nuclei':>8} {'min Phi_S':>14} {'theta_min':>12} {'max Phi_S':>14} {'theta_max':>12} "
        f"{'Delta_theta':>14} {'rel. contrast':>14} {'RMS vs ref':>14} {'contrast err':>14}"
    )
    print("-" * 120)
    for i, N in enumerate(grid["N_values"]):
        mean_y = grid["mean_yield"][i]
        rel = grid["polar_contrast"][i] / mean_y if mean_y != 0 else np.nan
        print(
            f"{int(N):8d} "
            f"{grid['min_yield'][i]:14.8f} {grid['theta_min_yield_deg'][i]:12.3f} "
            f"{grid['max_yield'][i]:14.8f} {grid['theta_max_yield_deg'][i]:12.3f} "
            f"{grid['polar_contrast'][i]:14.8e} {rel:14.8e} "
            f"{grid['curve_rms_error_vs_largest_N'][i]:14.8e} {grid['contrast_error_vs_largest_N'][i]:14.8e}"
        )
    print("-" * 120)
    print(f"Reference curve for convergence metrics: N = {grid['reference_N']} nuclei")


def plot_finite_size_yield_curves(grid: Dict[str, np.ndarray]) -> None:
    """Plot Phi_S(theta) for each finite nuclear size N."""
    theta = grid["theta_values_deg"]
    Z = grid["plenio_singlet_yield"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, N in enumerate(grid["N_values"]):
        ax.plot(theta, Z[i, :], marker="o", linewidth=2.0, label=f"N={int(N)} nuclei")
    ax.set_xlabel("polar angle theta (deg)")
    ax.set_ylabel("Plenio/Cai singlet yield Phi_S")
    ax.set_title("Finite-size scaling of polar-angle magnetic sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_finite_size_contrast(grid: Dict[str, np.ndarray]) -> None:
    """Plot polar max-min contrast as a function of nuclear size."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(grid["N_values"], grid["polar_contrast"], marker="o", linewidth=2.0)
    ax.set_xlabel("number of nuclear spins N")
    ax.set_ylabel("Delta_theta Phi_S")
    ax.set_title("Polar-angle contrast versus nuclear-environment size")
    ax.set_xticks(grid["N_values"])
    fig.tight_layout()
    plt.show()


def plot_finite_size_convergence(grid: Dict[str, np.ndarray]) -> None:
    """Plot convergence errors relative to the largest-N angular-yield curve."""
    N = grid["N_values"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(N, grid["curve_rms_error_vs_largest_N"], marker="o", linewidth=2.0, label="curve RMS error")
    ax.plot(N, grid["contrast_error_vs_largest_N"], marker="s", linewidth=2.0, label="contrast error")
    ax.set_xlabel("number of nuclear spins N")
    ax.set_ylabel("error relative to largest-N result")
    ax.set_title(f"Finite-size convergence relative to N={grid['reference_N']}")
    ax.set_xticks(N)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def plot_finite_size_fft(grid: Dict[str, np.ndarray]) -> None:
    """Plot dominant FFT peak frequency versus theta for each finite size."""
    theta = grid["theta_values_deg"]
    F = grid["top_fft_peak_frequency_MHz"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for i, N in enumerate(grid["N_values"]):
        ax.plot(theta, F[i, :], marker="o", linewidth=2.0, label=f"N={int(N)} nuclei")
    ax.set_xlabel("polar angle theta (deg)")
    ax.set_ylabel("dominant FFT peak frequency (MHz)")
    ax.set_title("Dominant singlet-dynamics frequency versus finite size")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


def save_finite_size_csv(grid: Dict[str, np.ndarray], path: str | Path) -> None:
    """Save finite-size scaling data in long-form CSV format."""
    path = Path(path)
    phi = float(grid["phi_values_deg"][0])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N_nuclei", "theta_deg", "phi_deg", "ratio_D_over_J", "J_MHz", "D_MHz",
            "plenio_singlet_yield", "finite_time_average_yield", "selected_yield",
            "top_fft_peak_frequency_MHz", "top_fft_peak_amplitude", "st_sum_max_error",
            "polar_contrast_for_N", "curve_rms_error_vs_largest_N", "contrast_error_vs_largest_N",
            "selected_original_hyperfine_indices", "selected_hyperfine_norms_MHz",
        ])
        for i, N in enumerate(grid["N_values"]):
            info = grid["selection_info_by_N"][int(N)]
            indices = ";".join(str(x) for x in info["selected_original_indices"])
            norms = ";".join(f"{x:.10g}" for x in info["selected_norms_MHz"])
            for j, theta in enumerate(grid["theta_values_deg"]):
                writer.writerow([
                    int(N), theta, phi, -6.0,
                    grid["J_MHz"][i, j], grid["D_MHz"][i, j],
                    grid["plenio_singlet_yield"][i, j],
                    grid["finite_time_average_yield"][i, j],
                    grid["selected_yield"][i, j],
                    grid["top_fft_peak_frequency_MHz"][i, j],
                    grid["top_fft_peak_amplitude"][i, j],
                    grid["st_sum_max_error"][i, j],
                    grid["polar_contrast"][i],
                    grid["curve_rms_error_vs_largest_N"][i],
                    grid["contrast_error_vs_largest_N"][i],
                    indices, norms,
                ])


def run_finite_size_scaling_first_test() -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    First finite-size scaling test.

    Fixed physics:
        D/J  = -6
        phi  = 0 deg
        theta = 0, 10, ..., 180 deg
        N    = 0, 2, 4, 6 nuclei

    The N>0 models are nested by adding the strongest N/2 hyperfine tensors on
    each radical, ranked by Frobenius norm. This tests how the polar-angle yield
    curve and polar contrast converge as the nuclear environment is enlarged.
    """
    base = Params()
    base.ratio_x = -6.0
    base.theta_values_deg = np.arange(0.0, 180.0 + 10.0, 10.0)
    base.phi_values_deg = np.array([0.0])
    base.recombination_rate_per_us = 0.5
    base.t_max_us = 10.0
    base.plot_each_run = False
    base.plot_heatmaps = False
    base.plot_ratio_summary = False
    base.save_ratio_csv = False
    base.verbose = True

    N_values = np.array([0, 2, 4, 6], dtype=int)
    results: List[Dict] = []
    selection_info_by_N: Dict[int, Dict] = {}
    total = len(N_values) * len(base.theta_values_deg) * len(base.phi_values_deg)
    counter = 0

    for N in N_values:
        P_N, info = make_finite_size_params(base, int(N))
        selection_info_by_N[int(N)] = info
        for theta in base.theta_values_deg:
            for phi in base.phi_values_deg:
                counter += 1
                if base.verbose:
                    print(f"\nFinite-size case {counter}/{total}: N={int(N)}, D/J={P_N.ratio_x:g}, theta={theta:g}, phi={phi:g}")
                result = run_one_case(P_N, theta, phi, plot=False)
                result["N_nuclei"] = int(N)
                result["selected_original_hyperfine_indices"] = info["selected_original_indices"]
                result["selected_hyperfine_norms_MHz"] = info["selected_norms_MHz"]
                results.append(result)

    grid = build_finite_size_grid(results, selection_info_by_N)
    print_finite_size_selection_table(grid)
    print_finite_size_summary(grid)

    csv_path = "finite_size_scaling_results.csv"
    save_finite_size_csv(grid, csv_path)
    print(f"Saved finite-size scaling CSV to: {csv_path}")

    plot_finite_size_yield_curves(grid)
    plot_finite_size_contrast(grid)
    plot_finite_size_convergence(grid)
    plot_finite_size_fft(grid)

    return results, grid

def main(P: Params | None = None) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """Notebook-friendly entry point."""
    if P is None:
        P = Params()

    # Fill D and J in a copy for plot titles/summaries without mutating input.
    P_for_summary = copy_params(P)
    set_JD_from_ratio(P_for_summary)

    results = run_angle_sweep(P)
    grid = build_angle_grid(results)
    print_angle_summary(results, grid)

    if P.plot_heatmaps:
        plot_angle_heatmaps(grid, P_for_summary)
        plot_representative_traces(results, grid)

    return results, grid


if __name__ == "__main__":
    results, grid = run_finite_size_scaling_first_test()
