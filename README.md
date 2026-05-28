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

## Full FAD-Trp direction

The next standalone model is:

```powershell
Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py
```

It uses the FAD-TrpC hyperfine tensor ordering and geometry from the broadband
upper-bound/action-spectrum work. The published ordered list contains 27
hyperfine-coupled nuclei: 15 on FAD and 12 on TrpH. Since exact density-matrix
evolution is not tractable at that size, the script uses state-vector evolution
and approximates the initially mixed nuclear bath by averaging over random-phase
nuclear state samples.

Fast check:

```powershell
python "Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py" --quick
```

Coarse 11-nucleus literature truncation:

```powershell
python "Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py" --max-nuclei 11 --samples 1 --theta-step 45 --t-max-us 4 --n-steps 121
```

Use the physical Cry4a FAD-TrpC dipolar tensor instead of the optimistic
best-case axial dipole-axis assumption:

```powershell
python "Realistic system size\code\rpm_full_fad_trp_magnetosensitivity.py" --max-nuclei 11 --physical-dipole
```

For larger attempts, increase `--max-nuclei`. The 22-qubit target corresponds to
`--max-nuclei 20`. The script automatically uses assembled SciPy sparse matrices
for smaller systems and switches to a matrix-free state-vector operator beyond
the configured sparse threshold.
