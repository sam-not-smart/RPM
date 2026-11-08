# Methods and figure index

Figure numbers below are the manuscript draft's numbering, not the older
APS poster numbering used internally by `poster_reproduction`.

## Model used in the full-bath figures

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
is not the simulated input. Figs. 2–4 use **Q = 0**. The nonzero-Q
calculation is a separate [low-field study](quadrupole.md).

Equal singlet/triplet recombination gives
`Phi_S = integral exp(-t/tau) P_S(t) dt/tau`. Thus τ = 5 μs is a lifetime,
not the integration endpoint or a separately modeled dephasing time.
Nuclear Zeeman and relaxation are omitted. Quantum yield sweeps use
paired global nuclear random-phase vectors; the code derives a sample
seed with `SeedSequence([master_seed, 0xA11CE, sample_id])`.

## Figure settings and availability

Archive identifiers below refer to separately retained simulation outputs,
not folders that are part of this repository.

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
