# Radical-pair simulations

The research model is a full coupled FAD–TrpH radical pair with two electrons
and 14 physical nuclei. The production solver uses matrix-free Hamiltonian
action and adaptive Krylov propagation of the full 221,184-component state.
The historical factorized calculation is retained only as a validation
reference, not as a production propagation path.

**Start with [Methods and figure index](methods/README.md).** It records the
settings actually used for manuscript Figs. 1–10, the exact hyperfine tensors,
where the data belong, and the reproducibility gaps. [Quadrupole studies](methods/quadrupole.md)
are separate from the manuscript's Q-off field sweep.

- `Realistic system size/code/`: shared physics, experiment entry points and tests.
- `methods/`: written methods, figure-specific configurations, and provenance.
- `figures/`: selected reference plots. These are not the raw simulation arrays.
- `scripts/danny_smoke_test.py`: a small environment check; it starts no research job.

## Quick verification

Use a Python 3.11+ environment with NumPy, SciPy and Matplotlib. Install from
the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/danny_smoke_test.py
```

Set the import path and run the small test suite:

```bash
export PYTHONPATH="Realistic system size/code"
python -m unittest discover -s "Realistic system size/code/tests" -v
python "Realistic system size/code/rpm_full14_quadrupole_experiment.py" --stage smoke
```

In PowerShell, use `$env:PYTHONPATH = 'Realistic system size/code'` instead
of `export`. The tests and smoke stage do not need saved production data or
access to Danny. Run only the `--stage smoke` command above for a quick
numerical check; full experiments are separate, expensive jobs.

See [environment and run provenance](methods/provenance.md) for tested
versions and the source history. Raw trajectory archives, bulk results, and
temporary files are intentionally not checked in. The older thesis and reading
folders remain available in Git history.
