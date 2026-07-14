This repo is for developing quantum mechanical simulations of the radical pair mechanism (RPM) in the hopes of better understanding the engineerability/controllability of said system. 
The inital parameters are based on the Flavin adenine dinucleotide (FAD) and Tryptophan (Trp) radicalm pair found on the crytpochrome-4, responsibe for avian magnetoreception. This being said, the ultimate goal is to have tuneable, scalable quantum mechanical models applicable to a variety of RPMs. 
Some context for the current state of the project: This work is a continuation of my Master's thesis in quantum engineering. For the thesis I developed an 8-qubit representative system (2 electrons + 6 nuclei). The aim of the thesis was to create a fully quantum mechanical model of the system, not aopting many literature-common assumptions and simplifications, in order to study the true quantum dynamics of this many-body quantum system. We studied singlet dynamics and singlet yield of the electron pair, and entanglement dynamics throughout the system.  
The next project goal is to simulate larger system sizes, ideally reaching the ~22 qubit size of the true FAD-Trp system. The thesis work evolved the full density matrix of the mixed system. This approach is not tractable for the scalability of the system size. I expect using sparse hamiltonians and sparse state vectors should be enough without needing more complicated methods but we'll see. 
Since this RPM should be responsible for avian magnetreception, the next immediate goal is to have a full system (all 22 qubits) that is magnetosensitive. We'll use the singlet yield of the RPM as our observable for magneosensitivty. See 'Realistic system size' -> 'literature' -> 'plenio quantum limits' for a description of the singlet yield function. Once we have a good baseline like this, we can further investigate the controllability and engineerability of the system. 
I've attached a folder 'realistic system size' which contains a few helpful files. Firstly, under 'code' it contains an exmaple python script used to run some of the smaller (max 8-qubit) systems used in the thesis. Then we have a 'literature' folder with select research articles doing work on larger (further than toy model) RPM systems. We'll use this work as inspiration for parameter choices and maybe benchmarking our models. A copy of the current master's thesis is attached for context. 
Ultimately, we aim to have a modifiable full quantum model of the Radical pair mechanism. We're starting with the FAD-Trp molecular inspiration since it's the most studied, but hopefully we can adjust simulate other RPM molecules which live in different parameter regimes and have different dynamics/yields/control levers. 

## Danny workstation smoke test

After cloning or pulling this repository on Danny, activate the intended Conda
environment and run this read-only environment check from the repository root:

```bash
conda activate <environment-name>
python scripts/danny_smoke_test.py
```

It reports the active Python interpreter and Conda environment, visible CPU and
memory, repository-disk capacity, and NVIDIA GPU detection. It then checks the
installed NumPy, SciPy, and Matplotlib packages with small numerical and
headless-plotting tests. It is deliberately not a sustained hardware benchmark.

If the smoke test passes, exercise the actual RPM code with:

```bash
python "Realistic system size/code/rpm_8qubit_magnetosensitivity.py" --quick
```

Some details about the system: 1. we'll use closed system dynamics. 2. The hamiltonian will have the Zeeman, hyperfine, exchange, and full (non-secular) dipolar interactions (maybe we'll add the quadrupolar interaction later). 3. We'll treat electrons and all nuclei as spin-1/2 qubits for now. To reiterate, the first goal is to have a magnetosensitive model and we'll go from there. See literature/code/thesis for paramter choices. One unclear parameter choice is the hyperfine coupling tensors for the full system. The 'quantum needle supp' under the 'literature' folder details hyperfine tensor values calculated using Density Functional Theory so we can use that a starting point, which we've already done for the 8-qubit code, and maybe tweak the tensors from there. 

## Getting the current baseline running

The current executable code lives in:

```powershell
Realistic system size\code\rpm_8qubit_magnetosensitivity.py
```

It is a standalone Python/SciPy sparse-state simulation of the 8-qubit baseline:
2 electron spins and 6 spin-1/2 nuclear spins. From the repository root, install
the dependencies with:

```powershell
python -m pip install -r requirements.txt
```

Then run the fast sanity check:

```powershell
python "Realistic system size\code\rpm_8qubit_magnetosensitivity.py" --quick
```

If that passes, run the default theta sweep:

```powershell
cd "Realistic system size\code"
python rpm_8qubit_magnetosensitivity.py
```

The script saves a CSV and PNG under `outputs\8qubit_magnetosensitivity`. The
observable is the recombination-weighted singlet yield
`Phi_S = integral k exp(-k t) P_S(t) dt`, truncated at the configured
`t_max_us`.

## Full FAD-Trp production baseline

The next standalone model is:

```powershell
Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py
```

## Reproducing the 2016 quantum-needle benchmark

The independent reproduction of Fig. 1B from Hiscock et al., *PNAS* 2016 is:

```bash
python "Realistic system size/code/rpm_quantum_needle_reproduction.py" --self-test
python "Realistic system size/code/rpm_quantum_needle_reproduction.py" --quick --workers 1
python "Realistic system size/code/rpm_quantum_needle_reproduction.py" --workers 32
```

It transcribes Tables S1-S2 and implements the radical-factorized correlation
formula in Section S5 of the supplement, including the physical spin-1 nature
of the three selected nitrogen nuclei. The production run recreates the 50-uT
field-angle traces for radical-pair lifetimes of 1, 2, 5, 10, 20, and 100 us.
Results are written under `outputs/quantum_needle_2016`.

### Matrix-free state-vector validation

The sampled state-vector reproduction uses the same paper model and random-phase
nuclear states.  Its overnight production backend exploits the validation
Hamiltonian's exact decomposition into independent 576-dimensional FAD and
384-dimensional TrpH radical blocks, diagonalizes those blocks, and projects
their evolved state-vector factors onto the singlet without forming the full
221,184-component pair state:

```bash
python "Realistic system size/code/rpm_quantum_needle_statevector.py" --stage auto --walltime-hours 12
```

The runner first uses a five-angle pilot to choose global nuclear and
reaction-time sample counts.  It starts the complete 199-angle sweep only when
the stochastic 95% half-width and Lanczos error gates pass.  Every angle/sample
cell is checkpointed under `outputs/quantum_needle_statevector`, so an
interrupted run continues with:

```bash
python "Realistic system size/code/rpm_quantum_needle_statevector.py" --stage auto --resume
```

Pass `--workers 0` (the default) to benchmark 1, 2, 4, 8, 16, and 32 worker
processes and select the smallest count within 5% of the best throughput.  Use
`--stage pilot`, `--stage full`, or `--stage report` to run an individual phase.
The runner stops launching new checkpoint cells at the configured wall-time;
the Danny launch also uses a hard 12-hour process limit.  `--backend factorized`
retains the slower radical-block Krylov diagnostic, while `--backend full`
retains the direct full-state matrix-free propagator for short-time tests and
future Hamiltonians with electron-electron interactions.  Those two diagnostic
backends are not intended for the overnight Fig. 1B production sweep.
The exact reproduction CSV under `outputs/quantum_needle_2016` is used for final
acceptance.  Generate it first with the exact runner above if it is absent.

The shared physical model is in `rpm_quantum_needle_model.py`; the reusable
mixed-spin local-term and adaptive Lanczos implementation is in
`rpm_statevector_engine.py`.  Focused tests run with:

```bash
python -m unittest discover -s "Realistic system size/code/tests" -v
```

It uses the FAD-TrpC hyperfine tensor ordering and geometry from the broadband
upper-bound/action-spectrum work. The current main production baseline is:

- `max_nuclei = 16`
- `hyperfine_mode = full`
- `J = 3 MHz`
- optimistic axial electron-electron dipole enabled by default
- `t_max = 2 us`
- `n_steps = 101`
- `theta = 0, 45, 90, 135, 180 deg`
- one quantum sample unless otherwise specified

For angular sweeps, the same random-phase nuclear sample ensemble is reused at
each theta value so the reported contrast reflects field orientation rather
than sample-to-sample bath noise.

This replaces the earlier 20-nucleus `z-only` hyperfine workflow as the thesis
baseline. The 20-nucleus `z-only` results are still useful for approximate
screening and scaling intuition, but should be labelled as optimistic/screening
only. The side-thread 14-nucleus comparison showed that full anisotropic
hyperfines roughly halved the singlet yield and reduced angular contrast from
about 2.59% to about 0.32%, so the full-anisotropic model is now the more
defensible baseline.

The published ordered list contains 27 hyperfine-coupled nuclei: 15 on FAD and
12 on TrpH. Since exact density-matrix evolution is not tractable at that size,
the script uses state-vector evolution and approximates the initially mixed
nuclear bath by averaging over random-phase nuclear state samples when multiple
samples are requested.

Fast check:

```powershell
python "Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py" --quick
```

Production magnetosensitivity baseline:

```powershell
python "Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py" --output-dir "outputs\baseline_16n_full\magnetosensitivity"
```

The default exchange strength for this full-system script is `J = 3 MHz`.

Use the physical Cry4a FAD-TrpC dipolar tensor instead of the optimistic
best-case axial dipole-axis assumption:

```powershell
python "Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py" --physical-dipole
```

Overhauser comparison on the same 16-nucleus baseline:

```powershell
python "Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py" --model overhauser --samples 400 --output-dir "outputs\baseline_16n_full\overhauser"
```

Entanglement dynamics with and without exchange/dipole:

```powershell
python "Realistic system size\code\rpm_entanglement_dynamics.py" --output-dir "outputs\baseline_16n_full\entanglement"
```

For larger attempts, increase `--max-nuclei`. A 20-nucleus run corresponds to
22 total qubits including the two electrons; with `--hyperfine-mode z-only`,
that path should be treated as an approximate screening calculation rather than
the main baseline. The quantum-needle S2 selection has 22 nuclei, which would be
24 total qubits and should be treated as the next scaling step. The script
automatically uses assembled SciPy sparse matrices for systems at or below the
configured sparse threshold and switches to a matrix-free state-vector operator
beyond it.

## Closed-system pulse control

The first ideal-pulse control screen is:

```powershell
Realistic system size\code\rpm_pulse_control.py
```

It supports no-pulse, Hahn echo, and CPMG-N protocols with ideal instantaneous
global electron pi pulses. The default screening protocols are `none`, `hahn_y`,
`cpmg2_y`, and `cpmg4_y`.

Fast smoke test:

```powershell
python "Realistic system size\code\rpm_pulse_control.py" --max-nuclei 7 --protocols none,hahn_y --theta-start 0 --theta-stop 180 --theta-step 90 --t-max-us 0.2 --n-steps 21 --verify-no-pulse
```

Production pulse-control screen:

```powershell
python "Realistic system size\code\rpm_pulse_control.py" --verify-no-pulse --output-dir "outputs\baseline_16n_full\pulse_control"
```
