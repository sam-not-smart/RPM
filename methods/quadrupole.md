# Nuclear quadrupole follow-up

These are distinct from the Q = 0 high-field manuscript runs.

The quadrupole term is `I · Q · I`. The code uses the field-equivalent
tensors based on [Efimova and Hore 2009](https://doi.org/10.1080/00268970902852624),
removes the isotropic trace, and converts with 2π × 28 rad/μs per mT.
Only FAD N5, FAD N10 and TrpH N1 are spin-1 quadrupolar sites. The N5
in-plane Q-axis azimuth in the full model is provisional; it is not fitted.
The numerical Q scale is 0 for off and 1 for the three documented tensors.

## Existing experiments

- **Exact three-nitrogen toy:** `rpm_three_nitrogen_quadrupole.py`;
  [configuration](configurations/quadrupole_toy.json). Fields 0.05 and 3 mT,
  lifetimes 5 and 10 μs. The main toy runs use J = D = 0; a separate
  three-nitrogen coupling bridge uses the cancellation condition.
  [Strength sweep](../figures/quadrupole/quadrupole_strength_yields.png),
  [nucleus attribution](../figures/quadrupole/quadrupole_nucleus_attribution.png),
  [orientation sensitivity](../figures/quadrupole/quadrupole_orientation_sensitivity.png),
  [coupling bridge](../figures/quadrupole/quadrupole_coupling_bridge.png).
- **Full 14-nucleus low-field comparison:** `rpm_full14_quadrupole_experiment.py`;
  [configuration](configurations/quadrupole_full14_11_angles.json). 50 μT,
  τ = 5 μs, D/J = −3, 11 angles, four paired nuclear samples per condition,
  all 88 trajectories completed. [Angular response](../figures/quadrupole/angular_quadrupole_comparison.png)
  and [singlet dynamics](../figures/quadrupole/singlet_dynamics_comparison.png).
- **29-angle extension:** `rpm_full14_quadrupole_figure1b.py --grid reduced`;
  [configuration](configurations/quadrupole_full14_29_angles.json).
  10° steps across 0–180° and 1° steps across 85–95°. Reuses all 88
  compatible trajectories and computes 144 more. The configuration is
  included here; the running sweep's raw or final data are not.
- **Related three-nitrogen transverse/axial suite:** `rpm_three_nitrogen_suite.py`
  with [configuration](configurations/three_nitrogen_suite.json). This
  uses a common transverse scale and common scaling of all three axial
  couplings, rather than varying only one site.

The full-bath runs retain the complete nonfactorized Hamiltonian,
paired random phases, Simpson integration, and the existing Krylov
tolerances. The 29-angle run is resumable; completed trajectory saves
remain atomic. On shared compute, the capacity safeguard leaves CPU/RAM
headroom and does not interrupt other users' work.

## Entry points

The quick smoke test is in the [main README](../README.md).
For a full extension, run `--stage prepare --grid reduced --reuse-from`
with the archived completed 11-angle directory and a separate
`--output-dir`. Then run `--stage run --grid reduced --resume` on that
output directory on a Linux compute host with capacity. Do not confuse
the 61-angle dense option with the 29-angle production design.
