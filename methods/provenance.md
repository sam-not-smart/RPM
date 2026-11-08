# Source and environment provenance

This tree consolidates the paper and current quadrupole work in a single
checkout. The source was selected from the working copy based on
`945bf7a86348064e8acb4653edce74697fa4fdcc`, including local toy and
quadrupole scripts that had not been committed. The poster runner and
report were checked against the copies on the compute host. No numerical
model changes were made during consolidation; portability changes cover
test temporary directories and the local reference-brief path.

Historical run metadata are retained where verified in
[the configuration files](configurations/). An old manifest hash
identifies that run's archived inputs, not necessarily today's file
paths or operational wrapper. The first cancellation run's exact launch
commit is unknown. Fig. 6 and Fig. 10 were carried over from a prior
laptop and are not presented as fully reproducible here. Fig. 7's
15-cell outputs and fits exist separately, but the inner trial curves
were not archived.

Additional retained entry points include the full-bath regime-shift
discovery runner, the exchange avoided-crossing scan, and the
quantum-needle validation and refinement runners. These are follow-up
or validation experiments, not the provenance of the two-nucleus Fig. 5
or the cancellation curves in Fig. 1.

## Environment

- Consolidation tests: Python 3.12.4, NumPy 2.3.5, SciPy 1.18.0,
  Matplotlib 3.10.8.
- Kapit Work Station is the compute host; archived hostnames such as
  `Preposteron` are retained as recorded.
- Archived Kapit Work Station manifests record Python 3.11.15 for the poster and
  full-bath runs. Consult each run's manifest for any recorded library
  versions; do not assume that the consolidation versions are identical
  to every historical environment.
- For production, use one numerical-library thread per worker. The
  run configurations distinguish numerical parameters from worker and
  resource policy. Smoke tests do not require production checkpoints.

## Archived material

The prior thesis-era main is preserved at
[`71acb313`](https://github.com/sam-not-smart/RPM/tree/71acb3130d1141b8b78eb1c120547c949b0593fd)
in ordinary Git history. Its reading folders, early baselines, and
temporary artifacts are not needed for the current analysis. Do not
import that material into the active tree. The separate raw-data
handoff will identify archived inputs without committing bulk outputs.
