This repo is for developing quantum mechanical simulations of the radical pair mechanism (RPM) in the hopes of better understanding the engineerability/controllability of said system. 
The inital parameters are based on the Flavin adenine dinucleotide (FAD) and Tryptophan (Trp) radicalm pair found on the crytpochrome-4, responsibe for avian magnetoreception. This being said, the ultimate goal is to have tuneable, scalable quantum mechanical models applicable to a variety of RPMs. 
Some context for the current state of the project: This work is a continuation of my Master's thesis in quantum engineering. For the thesis I developed an 8-qubit representative system (2 electrons + 6 nuclei). The aim of the thesis was to create a fully quantum mechanical model of the system, not aopting many literature-common assumptions and simplifications, in order to study the true quantum dynamics of this many-body quantum system. We studied singlet dynamics and singlet yield of the electron pair, and entanglement dynamics throughout the system.  
The next project goal is to simulate larger system sizes, ideally reaching the ~22 qubit size of the true FAD-Trp system. The thesis work evolved the full density matrix of the mixed system. This approach is not tractable for the scalability of the system size. I expect using sparse hamiltonians and sparse state vectors should be enough without needing more complicated methods but we'll see. 
Since this RPM should be responsible for avian magnetreception, the next immediate goal is to have a full system (all 22 qubits) that is magnetosensitive. We'll use the singlet yield of the RPM as our observable for magneosensitivty. See 'Realistic system size' -> 'literature' -> 'plenio quantum limits' for a description of the singlet yield function. Once we have a good baseline like this, we can further investigate the controllability and engineerability of the system. 
I've attached a folder 'realistic system size' which contains a few helpful files. Firstly, under 'code' it contains an exmaple python script used to run some of the smaller (max 8-qubit) systems used in the thesis. Then we have a 'literature' folder with select research articles doing work on larger (further than toy model) RPM systems. We'll use this work as inspiration for parameter choices and maybe benchmarking our models. A copy of the current master's thesis is attached for context. 
Ultimately, we aim to have a modifiable full quantum model of the Radical pair mechanism. We're starting with the FAD-Trp molecular inspiration since it's the most studied, but hopefully we can adjust simulate other RPM molecules which live in different parameter regimes and have different dynamics/yields/control levers. 

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

## Entanglement-size scaling study

The scaling study described for the new subproject lives in:

```powershell
Realistic system size\code\rpm_entanglement_scaling.py
```

It starts from the two strongest FAD-coupled nuclei and the two strongest
Trp-coupled nuclei, then adds one nucleus on each radical side per run in the
published approximate hyperfine-strength order. For each truncation it evolves
the closed system with electron Zeeman, full anisotropic hyperfine, exchange
(`J = 3 MHz` by default), electron-electron dipole, and a simple N-14
quadrupolar term for spin-1 nitrogens. It tracks the log negativity for the two
thesis cuts, `e1+n1 | e2+n2` and `e1+e2 | n1+n2`, writes time-series and summary
CSVs, plots both cuts separately for every completed system size, and fits a
power law to the time-averaged log negativity of each cut.

The default initial nuclear state is a fixed separable minimum-Overhauser
product state. Spin-1 nitrogens begin in `m=0` along their quadrupolar
principal axes. Spin-1/2 nuclei begin in fixed `m=+/-1/2` states along their
strongest hyperfine axes, with signs chosen once from the complete FAD and Trp
lists to minimize each radical's initial Overhauser field. The assignment is
kept unchanged as the system grows.

Fast smoke test:

```powershell
python "Realistic system size\code\rpm_entanglement_scaling.py" --quick --output-dir "outputs\entanglement_scaling_quick"
```

Production-style physical-spin run, stopping when the Hilbert-space dimension
exceeds about 22 effective qubits:

```powershell
python "Realistic system size\code\rpm_entanglement_scaling.py" --max-nuclei 14 --max-effective-qubits 22 --stop-after-size-hours 3 --extrapolate-to-nuclei 26 --output-dir "outputs\entanglement_minimum_overhauser_scaling_full"
```

To let the sweep stop cleanly after a single truncation takes more than a few
hours, add for example `--stop-after-size-hours 3`.

To instead use the older one-nucleus-at-a-time global growth order, add
`--growth-mode global`.

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
