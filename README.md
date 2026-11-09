# Radical-pair simulations

The research model is a full coupled FAD–TrpH radical pair with two electrons
and 14 physical nuclei. The production solver uses matrix-free Hamiltonian
action and adaptive Krylov propagation of the full 221,184-component state.
The historical factorized calculation is retained only as a validation
reference, not as a production propagation path.

**Start with the [Proposed methods and figure index](methods/README.md).**
It discusses the parameters used, including hyperfine tables, dipolar and exchange values and quadrupolar tensors with references as well
as all other parameters and methods used for the current body of figures and results. 

- `Realistic system size/code/`: shared physics, experiment entry points and tests.
- `methods/`: written methods, figure-specific configurations, and provenance.
- `figures/`: selected reference plots. These are not the raw simulation arrays.
- `scripts/workstation_smoke_test.py`: a small environment check; it starts no research job.

## Quick verification

Use a Python 3.11+ environment with NumPy, SciPy and Matplotlib. Install from
the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/workstation_smoke_test.py
```

Set the import path and run the small test suite:

```bash
export PYTHONPATH="Realistic system size/code"
python -m unittest discover -s "Realistic system size/code/tests" -v
python "Realistic system size/code/rpm_full14_quadrupole_experiment.py" --stage smoke
python "Realistic system size/code/rpm_full14_quadrupole_field_pulse_screen.py" --stage smoke
```

In PowerShell, use `$env:PYTHONPATH = 'Realistic system size/code'` instead
of `export`. The tests and smoke stage do not need saved production data or
access to Kapit Work Station. The `--stage smoke` commands above are quick
numerical check; full experiments are separate, expensive jobs.

See [environment and run provenance](methods/provenance.md) for tested
versions and the source history. Raw trajectory archives, bulk results, and
temporary files are intentionally not checked in. The older thesis and reading
folders remain available in Git history.
