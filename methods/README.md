# Proposed methods and figure index

Figure numbers below are the manuscript draft's numbering, not the older
APS poster numbering used internally by `poster_reproduction`.
This page separates the **proposed quadrupole-inclusive methods** from the
**settings actually used for the archived figures**. Existing figures are
not relabelled as Q-on results. Q-on settings are a roadmap, not evidence
that the updated figures are complete.

## Proposed full-bath model

The [exact matrices and principal axes](hyperfine_tensors.json) are in mT
in the common flavin molecular frame. They were checked against the
production manifest. The seven FAD nuclei are N5, N10, H6, H8a, H8b, H8c
and Hβ; the seven TrpH nuclei are N1, H1, H2, H4, H5, H7 and Hβ1. The source
is [Hiscock et al. 2016, supporting information Tables S1–S2](https://doi.org/10.1073/pnas.1600341113),
not a RadicalPy import. TrpH tensors are already in the flavin frame.
The field direction in the zx plane is `[sin(theta), 0, cos(theta)]`.
Rotating to another coordinate frame requires rotating the field and all
tensors consistently; the archive does not sample arbitrary azimuths.

Using the [Efimova–Hore convention](https://doi.org/10.1529/biophysj.107.119362),
the dipolar term is `D[(S·n)^2 − S^2/3]` and the exchange term is
`−J(S^2 − 1)`, with S the total electron spin. The coupled baseline uses
molecular x for the dipolar axis and the 2 nm model:
**D = −347.875 μT, J = +115.958333 μT**, or −9.7405 MHz and +3.246833 MHz
using 28 MHz/mT. The 21.2 Å / 5.46 MHz literature comparison in the draft
is not the simulated input. The archived Figs. 2–4 use **Q = 0**.

For the proposed quantum production runs, add
`s_Q Σᵢ Iᵢ · Qᵢ · Iᵢ` on **FAD N5, FAD N10 and TrpH N1** (the three
spin-1 nuclei). Use `--quadrupole-scale 1` for the current nominal model
and `--quadrupole-scale 0` for the archived ablation/control. The
[exact nominal matrices](quadrupole_tensors.json) are traceless and in
field-equivalent mT; multiply by `2π × 28` for rad/μs. Their starting
principal values are from [Efimova & Hore (2009), Table 2](https://doi.org/10.1080/00268970902852624):

| Site | Table 2 `(Qxx, Qyy, Qzz)`, mT before trace removal | Current modeled principal direction `Q_z` |
| --- | --- | --- |
| FAD N5 | (0.031, 0.041, −0.072) | In the flavin plane, on the lower-eigenvalue in-plane hyperfine axis (about 18° from +x toward −y). This azimuth is provisional. |
| FAD N10 | (0.032, 0.041, −0.072) | Flavin z, aligned with the dominant N10 hyperfine axis. |
| TrpH N1 | (0.016, 0.020, −0.036) | Dominant N1 hyperfine axis, about 37° from flavin z; the small reported tilt from the local ring normal is neglected. |

The current implementation did not transform and import the eigenvectors
in the paper's Supplementary Table 2. Verify the mapping into our frame
and assess orientation sensitivity before treating the nominal tensor
orientations as definitive. A scale of zero leaves the rest of the
Hamiltonian unchanged. See [quadrupole studies and commands](quadrupole.md).

Equal singlet/triplet recombination gives
`Phi_S = integral exp(-t/tau) P_S(t) dt/tau`. Thus τ = 5 μs is a lifetime,
not the integration endpoint or a separately modeled dephasing time.
Nuclear Zeeman and relaxation are omitted. Quantum yield sweeps use
paired global nuclear random-phase vectors; the code derives a sample
seed with `SeedSequence([master_seed, 0xA11CE, sample_id])`.

## Proposed figure roadmap

| Figures | Proposed Q-on work |
| --- | --- |
| **2–3 · Field sweep and heatmap** | Start with 2 and 10 mT, 0° and 90°, τ = 5 μs. Compare absolute yields and the 0°–90° contrast against paired Q-off controls. Use the screen to prioritize the wider Q-on grid. A four-point screen is not a replacement for an angular sweep or heatmap. |
| **8 · N5 spin control** | At 50 μT and τ = 10 μs, test no pulses and ideal N5 π pulses every 0.2 μs at 85°, 90° and 95°. Keep Q active between pulses. Compare damping (unpulsed minus pulsed needle depth) with Q off, where depth is the mean of the two shoulders minus the 90° yield. Extend to the full 85–95° grid and to other protocols as required for the final figure. |
| **4 · Overhauser** | Start by comparing the Q-on quantum result at 2 mT against the existing static curves. The electron-only static Overhauser models have no explicit nuclear degrees of freedom; the Q term applies to the quantum reference, not to those models. Decide whether to expand the Q-on quantum grid to the other comparison fields after the screen. |
| **1, 9 · Cancellation and entanglement** | Q-on reruns are supported. Retain the archived Q-off results as controls. If retained as claims about the quadrupole-inclusive model, rerun the relevant conditions; the current screen does not establish Q insensitivity here. |
| **5–7 · Exact toy mechanisms** | The archived mechanism tests remain explicitly idealized. Q can be enabled in the available two-nucleus runners as an exploratory extension in the toy frame. Fig. 6's original generating source and raw grid are still missing. The Q-off scaling law is not assumed to hold with Q on. |
| **10 · Magic and entanglement** | Recover the original source and parameters before defining a matched Q-on run. The available reference image does not establish reproducibility. |

The [field/pulse screen configuration](configurations/quadrupole_field_pulse_screen.json)
contains 40 Q-on trajectories with four matched nuclear states per condition
and 40 archived Q-off controls (seed 20260731). It uses the same (D/J=-3)
baseline, field plane, propagation and integration settings as the two
original workflows. The separate low-field Q-on/Q-off comparison and its
29-angle extension are documented in [Quadrupole studies](quadrupole.md).

## Archived figure settings and availability

Archive identifiers below refer to separately retained simulation outputs,
not folders that are part of this repository. These entries describe the
current reference figures, not the proposed Q-on replacements.

| Figure | Configuration, code and archive |
| --- | --- |
| [1 · Cancellation](../figures/01-cancellation.png) | [Settings](configurations/figure_01.json), `rpm_quantum_needle_jd_sweeps.py`. At 50 μT and τ = 2 μs, Q = 0, 30° steps on 0–180°, M = 4, seed 20260715. D = J = 0, dipole only, D/J = −3 and D/J = 6. The 112 plotted sample values are preserved in the `quantum_needle_efimova_hore_24h` archive; exact launch commit is unrecorded. |
| [2 · Field sweep](../figures/02-field-sweep.png), [3 · Heatmap](../figures/03-directional-heatmap.png) | [Settings](configurations/figures_02_04_09.json), `poster_reproduction`. τ = 5 μs, D/J = −3, Q = 0, M = 4, seed 20260731. Twelve fields from 0.05 to 400 mT; 10° steps on 0–180°. All 912 relevant sample yields were verified in `poster_reproduction_multisample/checkpoints/quantum`. The heatmap subtracts the angular mean at each field. |
| [4 · Overhauser](../figures/04-overhauser-comparison.png) | [Settings](configurations/figures_02_04_09.json), `poster_reproduction`. Same model at 0.05, 2 and 20 mT. Quantum M = 4; eight scrambled Sobol replicates for Gaussian fields and 4,096 atomwise draws per angle. Individual draws and replicate means exist in the Overhauser archive. The plotted quantum means match Fig. 2. |
| [5 · Regime change](../figures/05-regime-shift.png) | [Settings](configurations/figure_05.json), `rpm_two_nucleus_regime_shift_refined.py`. Exact 36-state calculation with J = D = Q = 0, τ = 10 μs. 285 fields over 0.05–100 mT; 0.01° angular grid and 0.001° local refinements. The `two_nucleus_regime_shift_refined_theta001` archive contains full/ablated curves, f(B₀), and orientations. At 100 mT the plotted run gives f = −1.46717; the orientation jump is bracketed at 0.6525–0.655 mT. |
| [6 · Transverse optimum](../figures/06-transverse-optimum.png) | [Recovered settings](configurations/figure_06.json) and the [original brief](references/RPM_transverse_vs_axial_brief_v2.pdf). Exact two-nucleus result at 50 μT, 17 transverse scales and τ = 1, 10, 100 μs. The original laptop's exact scale grid, generating source and raw sweep are not in this checkout. Reference plot only. |
| [7 · Optimum scaling](../figures/07-optimum-scaling.png) | [Settings](configurations/figure_07.json), `rpm_two_nucleus_optimum_reproduction.py`. 50 μT, exact toy model, 15 cells (three Azz values × five lifetimes). The saved 15 optima and fits are in `two_nucleus_optimum_reproduction`; trial-by-trial angle arrays were not retained. The plotted reproduction gives exponents −0.494261, −0.494315 and −0.494300, not the older numbers in the draft. |
| [8 · N5 control](../figures/08-n5-spin-control.png) | [Settings](configurations/figure_08.json), `rpm_nuclear_spin_rotation_control.py`. 50 μT, τ = 10 μs, D/J = −3, Q = 0, 85–95° at 1°, M = 4, seed 20260731. The periodic arm uses ideal N5 π pulses every 0.2 μs about its dominant hyperfine axis. Detrending uses the 85° and 95° endpoints. Sample yields and control settings exist in `nuclear_spin_rotation_control_tau10`. |
| [9 · Entanglement](../figures/09-entanglement-scaling.png) | [Settings](configurations/figures_02_04_09.json), `poster_reproduction`. 50 μT at 90°, D/J = −3, Q = 0, 16 trajectories at each of 4, 6, 8, 10, 12 and 14 nuclei. Use the 96 `*_dense_early_v3.npz` files in the separate entanglement archive. Their 1,561-point 0–2 μs traces reproduce the plotted averages. |
| [10 · Magic and entanglement](../figures/10-magic-entanglement.png) | [Archive status](configurations/figure_10.json). Available plot carried over from the earlier laptop. The recovered ZIP contains images only; the generating code, full parameters and raw data are not yet present. Reference plot only. |

## Two interpretation questions

**Fig. 9:** For each trajectory the initial state is an electronic singlet
times a sampled **nuclear computational-basis product state**, not a global
random-phase vector. The two full-system cuts use the pure-state Schmidt
singular values to calculate log negativity and then average the results
over trajectories and time. They do not evaluate log negativity of a
mixed full-bath density matrix. The separate two-electron analysis instead
averages the 4×4 reduced electron density matrices first. The archives
save the pure-cut time traces and reduced electron matrices, not the full
many-body density matrices.

**Toy nitrogen:** The tensor `diag(-0.0989,-0.0989,1.7569)` mT is the
idealized **FAD N5** tensor from Hiscock et al., SI Fig. S2. The partner is
axial TrpH N1-like `diag(0,0,1.0812)` mT. The numerical coincidence of the
largest coupling with a value in the TrpD comparison row does not identify
it as TrpD.
