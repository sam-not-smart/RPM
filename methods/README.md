# Proposed methods and figure index

This index describes the figures we’re working toward. Each row links to
the current reference plot and records the settings used to produce it.
Figure numbers follow the manuscript draft.

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

For the quadrupolar interaction we add
`s_Q Σᵢ Iᵢ · Qᵢ · Iᵢ` on **FAD N5, FAD N10 and TrpH N1** (the three
spin-1 nuclei). Use `--quadrupole-scale 1` for the current nominal model
and `--quadrupole-scale 0` for the archived ablation/control. The
[exact nominal matrices](quadrupole_tensors.json) are traceless and in
field-equivalent mT; multiply by `2π × 28` for rad/μs. Their starting
principal values are from [Efimova & Hore (2009), Table 2](https://doi.org/10.1080/00268970902852624):

- **FAD N5:** `(Qxx, Qyy, Qzz) = (0.031, 0.041, −0.072)` mT before trace
  removal. `Q_z` follows the lower-eigenvalue in-plane hyperfine axis,
  about 18° from flavin +x toward −y. This azimuth is provisional.
- **FAD N10:** `(0.032, 0.041, −0.072)` mT before trace removal.
  `Q_z` is along flavin z, the dominant N10 hyperfine axis.
- **TrpH N1:** `(0.016, 0.020, −0.036)` mT before trace removal.
  `Q_z` follows the dominant N1 hyperfine axis, about 37° from flavin z;
  the small reported tilt from the local ring normal is neglected.

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

## Figure index

Archive names below refer to
separately retained data, not files included in this repository. The majority of the raw data used for producing these figures is saved separately on Eliot Kapit's workstation in the Quantum Theory Lab.

| Figure | Proposed method, code and reference settings |
| --- | --- |
| [1 · Cancellation](../figures/01-cancellation.png) | **Plan:** If used to make a full-model claim, repeat the relevant cancellation conditions with nominal Q using `rpm_quantum_needle_jd_sweeps.py`. <br> **Q-off reference:** [Settings](configurations/figure_01.json): 50 μT, τ = 2 μs, 0–180° at 30° steps, M = 4, seed 20260715; D = J = 0, dipole only, D/J = −3 and D/J = 6. The `quantum_needle_efimova_hore_24h` archive holds the 112 plotted sample values; exact launch commit is unrecorded. |
| [2 · Field sweep](../figures/02-field-sweep.png), [3 · Heatmap](../figures/03-directional-heatmap.png) | [Settings](configurations/figures_02_04_09.json), `poster_reproduction`: twelve fields from 0.05 to 400 mT, 0–180° at 10° steps, τ = 5 μs, D/J = −3, M = 4, seed 20260731. The 912 sample yields are in `poster_reproduction_multisample/checkpoints/quantum`. |
| [4 · Overhauser](../figures/04-overhauser-comparison.png) | `poster_reproduction`: 0.05, 2 and 20 mT, quantum M = 4, eight scrambled Sobol replicates for Gaussian fields and 4,096 atomwise draws per angle. The plotted quantum means match Fig. 2; individual draws and replicate means are in the Overhauser archive. |
| [5 · Regime change](../figures/05-regime-shift.png) | **Method:** Keep the exact two-nucleus mechanism test; `rpm_two_nucleus_regime_shift_refined.py`: J = D = Q = 0, τ = 10 μs, 285 fields over 0.05–100 mT, 0.01° grid with 0.001° local refinement. The `two_nucleus_regime_shift_refined_theta001` archive contains full/ablated curves, f(B₀) and orientations. The reference gives f(100 mT) = −1.46717 and brackets the orientation jump at 0.6525–0.655 mT. |
| [6 · Transverse optimum](../figures/06-transverse-optimum.png) | **Next:** Recover the original grid and source before defining a matched Q-on extension. <br> **Reference only:** [Recovered settings](configurations/figure_06.json) and [original brief](references/RPM_transverse_vs_axial_brief_v2.pdf): exact two-nucleus result at 50 μT, 17 transverse scales and τ = 1, 10, 100 μs. The generating source, exact scale grid and raw sweep are missing. |
| [7 · Optimum scaling](../figures/07-optimum-scaling.png) | **Method:** Keep the idealized scaling test. The exact toy runner supports exploratory Q-on scans; the Q-off scaling law is not assumed to carry over. <br> **Q-off reference:** [Settings](configurations/figure_07.json), `rpm_two_nucleus_optimum_reproduction.py`: 50 μT, three Azz values × five lifetimes. The 15 optima and fits (exponents −0.494261, −0.494315 and −0.494300) are in `two_nucleus_optimum_reproduction`; trial-by-trial angular arrays were not retained. |
| [8 · N5 control](../figures/08-n5-spin-control.png) | **In progress:** At 50 μT and τ = 10 μs, compare no pulses against ideal N5 π pulses every 0.2 μs at 85°, 90° and 95°, with nominal Q active between pulses. Measure how Q changes the pulse-induced reduction in needle depth, defined against the mean of the 85° and 95° shoulders. Extend to the full grid and other protocols as needed. <br> **Q-off reference:** [Settings](configurations/figure_08.json), `rpm_nuclear_spin_rotation_control.py`: D/J = −3, 85–95° at 1° steps, M = 4, seed 20260731. Pulses are about N5’s dominant hyperfine axis. Yields and control settings are in `nuclear_spin_rotation_control_tau10`. |
| [9 · Entanglement](../figures/09-entanglement-scaling.png) | **Plan:** If used to make a full-model claim, repeat the bath-size series with nominal Q on the spin-1 sites present in each bath. <br> **Q-off reference:** [Settings](configurations/figures_02_04_09.json), `poster_reproduction`: 50 μT at 90°, D/J = −3, 16 trajectories at each of 4, 6, 8, 10, 12 and 14 nuclei. The 96 `*_dense_early_v3.npz` files contain the 1,561-point 0–2 μs traces behind the plotted averages. |
| [10 · Magic and entanglement](../figures/10-magic-entanglement.png) | **Next:** Recover the original source and parameters before designing a Q-on run. <br> **Reference only:** [Archive status](configurations/figure_10.json). The recovered ZIP contains images only; the generating code, full settings and raw data are missing. |

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
