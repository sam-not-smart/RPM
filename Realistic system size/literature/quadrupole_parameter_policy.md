# Nuclear quadrupole parameter policy

## Permanent model requirement

All nuclei with spin `I > 1/2` must have a nonzero, explicitly documented
nuclear quadrupole interaction in future physical RPM baselines. A zero tensor
is allowed only for a deliberately quadrupole-free validation fixture and must
be labelled as such. Spin-1/2 nuclei have no electric quadrupole moment.

The source for the current FAD-Trp estimate is:

> O. Efimova and P. J. Hore, "Evaluation of nuclear quadrupole interactions as
> a source of magnetic anisotropy in the radical pair model of the avian
> magnetic compass," Molecular Physics 107, 665-671 (2009),
> DOI: 10.1080/00268970902852624.

The repository copy is
`Efimova_Hore_2009_nuclear_quadrupole_interactions.pdf`.

## Convention and units

Use the paper's Hamiltonian convention

```text
H_NQ = I . Q . I
```

where `Q` is a symmetric quadrupole coupling tensor expressed as a frequency.
The conventional scalar coupling is `chi = 2 Qzz`, and the asymmetry is
`eta = (Qxx - Qyy) / Qzz`. For the paper's equivalent-field units, convert
principal tensor values using the model convention `28.0 MHz/mT`; multiply by
`2*pi` once when constructing a Hamiltonian in rad/us.

The nuclear quadrupole moment is an isotope property, whereas the coupling
tensor also depends on the molecular electric-field gradient. The paper uses
the 14N quadrupole moment `2.044e-30 m^2`. Couplings for other isotopes or
chemical environments require their own sourced or calculated electric-field
gradient and must not be copied from 14N by default.

## Current full-model estimates

The current 14-nucleus quantum-needle model has three nuclei with `I > 1/2`;
all are 14N with `I = 1`. For the FAD radical anion / tryptophan radical cation
pair, use the Table 2 principal values below as the baseline estimates.

| Radical/nucleus | `(Qxx,Qyy,Qzz)` (mT) | `(Qxx,Qyy,Qzz)` (MHz) | `chi=2Qzz` (MHz) | `eta` | Q z-axis vs ring normal |
|---|---:|---:|---:|---:|---:|
| TrpH radical cation N1 | `(0.016,0.020,-0.036)` | `(0.448,0.560,-1.008)` | `-2.016` | `0.100` | `2.9 deg` |
| FAD radical anion N5 | `(0.031,0.041,-0.072)` | `(0.868,1.148,-2.016)` | `-4.032` | `0.135` | `90.0 deg` |
| FAD radical anion N10 | `(0.032,0.041,-0.072)` | `(0.896,1.148,-2.016)` | `-4.032` | `0.121` | `0.0 deg` |

The tabulated values are rounded, so a small nonzero trace can appear. Remove
the isotropic trace before constructing the Hamiltonian. For a fixed nuclear
spin this only removes a constant energy offset and does not change dynamics.

## Orientation rule and remaining uncertainty

The principal values alone are insufficient: each tensor must be rotated into
the same molecular frame as its hyperfine tensor.

- TrpH radical cation N1 and FAD radical anion N10 are nearly axial with their
  quadrupole z-axes close to the aromatic-ring normal. Their quadrupole and
  hyperfine principal frames may be provisionally aligned.
- FAD radical anion N5 has its hyperfine z-axis normal to the ring but its
  quadrupole z-axis in the ring plane. The main paper specifies the 90-degree
  tilt but not the in-plane azimuth. Until a supplementary tensor or a new
  electronic-structure calculation fixes that azimuth, it must be recorded as
  a model uncertainty and varied in a later sensitivity check. It must not be
  handled by setting the N5 quadrupole coupling to zero.

These values supersede the much smaller arbitrary quadrupole tensor used only
in the reduced numerical fixture. They do not retroactively alter the completed
quadrupole-free quantum-needle validation run.
