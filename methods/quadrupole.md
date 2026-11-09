# Nuclear quadrupole follow-up

The archived high-field and spin-control manuscript figures used Q = 0.
The [figure index](README.md#figure-index) describes
the Q-on replacements and distinguishes screens from final figures.

The quadrupole term is `I · Q · I`. The code uses the field-equivalent
tensors based on [Efimova and Hore 2009](https://doi.org/10.1080/00268970902852624),
removes the isotropic trace, and converts with 2π × 28 rad/μs per mT.
Only FAD N5, FAD N10 and TrpH N1 are spin-1 quadrupolar sites. The N5
in-plane Q-axis azimuth in the full model is provisional; it is not fitted.
The numerical Q scale is 0 for off and 1 for the [three nominal tensors](quadrupole_tensors.json).
The tensor values are field-equivalent Hamiltonian coefficients, not the
conventional quadrupole coupling constant χ. For aligned toy extensions,
Qz is toy x for N5 and toy z for N10 and N1, matching the nominal
three-nitrogen quadrupole experiment rather than the full flavin-frame geometry.

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
- **Paired field/pulse screen:** `rpm_full14_quadrupole_field_pulse_screen.py`;
  [configuration](configurations/quadrupole_field_pulse_screen.json).
  16 Q-on field trajectories (2 and 10 mT, 0° and 90°, τ = 5 μs) and
  24 Q-on spin-control trajectories (50 μT, 85°/90°/95°, τ = 10 μs,
  no pulses or N5 π pulses every 0.2 μs). All have matching Q-off archive
  controls. This is a small exploratory screen, not the full figure grid.

The full-bath runs retain the complete nonfactorized Hamiltonian,
paired random phases, Simpson integration, and the existing Krylov
tolerances. The 29-angle run is resumable; completed trajectory saves
remain atomic. On shared compute, the capacity safeguard leaves CPU/RAM
headroom and does not interrupt other users' work.

## Entry points

Set `PYTHONPATH` as in the [main README](../README.md) and run from the
repository root. Use a new output directory when changing Q scale.
These commands show how to plan or start a run; full sweeps can take days.

```bash
# Inspect the 4-sample quantum field inventory without starting trajectories.
python -m poster_reproduction.runner --stage production --figures 2 --kinds quantum --samples 4 --workers 4 --quadrupole-scale 1 --output-dir outputs/field_q1 --dry-run
# Remove --dry-run to execute. For the archived physics use scale 0 and a different directory.
# After the field sweep completes:
python -m poster_reproduction.report --components 2 --samples 4 --quadrupole-scale 1 --output-dir outputs/field_q1

# Start the N5 screening/confirmation workflow. It selects an interval from the screen.
python "Realistic system size/code/rpm_nuclear_spin_rotation_control.py" --stage auto --workers 4 --quadrupole-scale 1 --output-dir outputs/n5_q1
```

The poster runner uses **poster** numbering: `--figures 2 --kinds quantum`
produces the manuscript field sweep/heatmap inputs (Figs. 2–3);
`--figures 3 --kinds quantum,overhauser` schedules the three quantum
comparison fields and the static Overhauser calculations;
`--figures 4 --kinds entanglement --samples 16` selects manuscript Fig. 9.
For full field-dependent Overhauser reporting, retain the complete quantum
field sweep in the same output directory. The static models have no explicit
quadrupolar nucleus. Use the archived settings for the original sample counts.

For cancellation sweeps, the full-bath regime-shift and avoided-crossing
follow-ups, and the needle refinement, pass the same `--quadrupole-scale`
option to their runners. For the two- and three-nitrogen toy mechanism studies,
specify scale 1 only for an exploratory extension of the Q-off result.
The original Hiscock exact and state-vector reproduction benchmarks remain
fixed Q-off by design; use the full-bath refinement/comparison runners for
Q-on needle tests.

The field/pulse screen is resumable and needs the two archived Q-off
checkpoint directories under a common archive root. Its smoke test
does not need those archives. On a Linux compute host, first run:

```bash
python "Realistic system size/code/rpm_full14_quadrupole_field_pulse_screen.py" --stage preflight --archive-base /path/to/archived/outputs --output-dir outputs/field_pulse_q_screen --workers 24
```

Change `--stage preflight` to `--stage run` only after verifying shared
capacity. Use a new output directory for a fresh rerun from this source;
use the original snapshot to resume the already launched KWS job.

For a full extension, run `--stage prepare --grid reduced --reuse-from`
with the archived completed 11-angle directory and a separate
`--output-dir`. Then run `--stage run --grid reduced --resume` on that
output directory on a Linux compute host with capacity. Do not confuse
the 61-angle dense option with the 29-angle production design.
