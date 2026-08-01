# Exchange and electron-dipolar parameter policy

## Source

The sweep conventions are based on:

> O. Efimova and P. J. Hore, "Role of Exchange and Dipolar Interactions in the
> Radical Pair Model of the Avian Magnetic Compass," *Biophysical Journal*
> **94** (2008), 1565-1574. DOI: 10.1529/biophysj.107.119362.

The archived article is
`Efimova_Hore_2008_exchange_dipolar_interactions.pdf` (SHA-256
`24669b3d6a413327bda8bef9b88d26572a0d31419cf5415d57c1ac12d7dc2552`).

## Hamiltonian convention

The primary full state-vector convention is now the literal Efimova-Hore
form

\[
H_J=-J(\mathbf S^2-1),\qquad
H_D=D_{\mathrm{paper}}
\left[(\mathbf S\mathbin{\cdot}\mathbf n)^2-\mathbf S^2/3\right].
\]

with `D` meaning `D_paper`, not a dipolar-tensor principal eigenvalue. The
dipolar axis is supplied explicitly. The validation sweep uses the paper's
`epsilon = pi/2` geometry, so `n = (1,0,0)` in the molecular frame.

The earlier implementation

\[
H_{\mathrm{ex}}=-2J\,\mathbf S_A\mathbin{\cdot}\mathbf S_B,
\qquad
H_{\mathrm{dd}}=\mathbf S_A\mathbin{\cdot}\mathbf D\mathbin{\cdot}\mathbf S_B
\]

remains available only for reproducing historical runs. Its exchange term
differs from the paper by the state-independent shift `-J/2`. For an axial
tensor, the equivalent mapping is

\[
\mathbf D=\operatorname{diag}(-D_\parallel/2,-D_\parallel/2,D_\parallel),
\qquad D_\parallel=\frac{4}{3}D_{\mathrm{paper}}.
\]

The permanent ratio reported by new calculations is
`R_EH = D_paper/J`. The exact two-electron crossing centers are therefore
`R_EH = -3` and `R_EH = 6`. Ratios `-4` and `8` refer only to the historical
principal-eigenvalue convention and must not label new Efimova-Hore results.

## Two-nanometre anchor and sweep interpretation

The paper's point-dipole estimate is

\[
D_{\mathrm{paper}}/\mu\mathrm T=-2783/(r/\mathrm{nm})^3.
\]

At `r = 2 nm`, the runner consequently uses:

- `D_paper = -347.875 uT`;
- `D_paper = -9.7405 MHz`, using `28 MHz/mT`.

The equivalent historical axial tensor would have
`D_parallel = -463.833333 uT = -12.987333 MHz`; those values are retained only
to make old output interpretable.

The low-exchange controls, `J = -10,-5,0,5,10 uT`, cover the nominal
physically plausible range. The cancellation-ratio scan implies much larger
`|J|` values, approximately 29-232 uT, and must be described as a mechanistic
scan rather than a measurement-supported parameter range.

## Current quadrupole-free validation scope

The dipole/exchange sweep runner fixes `quadrupole_scale = 0`. This is a
deliberate validation scope. Quadrupoles will be revisited separately; no
nonzero quadrupole value should be inferred from these sweep results.
