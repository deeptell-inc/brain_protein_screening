# qbscreen

**Radical-pair magnetic field effects and reservoir computing in brain flavoenzymes**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`qbscreen` computes radical-pair-mechanism (RPM) physics for brain flavoenzymes
using an **exact Liouville-space master equation that includes electronic spin
decoherence**. It supports two studies:

1. **Weak-field magnetosensitivity (negative result).** With radical-pair
   parameters fixed by the crystallographic active-site geometry, the
   geomagnetic-field effect on the **singlet yield** of monoamine oxidase A/B and
   D-amino-acid oxidase is **1.7 × 10⁻¹² %**, capped independently by a purely
   kinetic ceiling **(ω_B τ)² ≈ 7.8 × 10⁻⁷ %** that the picosecond lifetime
   imposes on its own. Cryptochrome, in the identical model, gives **+1.0 %**.
   The discriminating variable is the inter-radical separation the protein
   scaffold imposes, not the flavin cofactor the two share. That fast
   recombination and strong exchange suppress weak-field effects *in general* is
   established (Binhi, *Phys.-Usp.* **68**, 1242 (2025)); what is new here is the
   enzyme-specific number, the attribution to the lifetime rather than to
   exchange or decoherence, and a falsifiable experiment at the 26.8 mT
   singlet–triplet crossing.
   → `qbscreen/master_equation.py`, `qbscreen/honest_mfe.py`

2. **Quantum reservoir computing (drafted → `manuscript/reservoir_rev/`).** The same
   radical-pair + nuclear-spin system is a genuine *physical* quantum reservoir:
   with a full readout it reaches a **held-out** information-processing capacity
   **IPC ≈ 6.9** (≤ Dambre bound 12), **~72 % of which is coherence-derived**, and
   it survives ensemble averaging. Framed as the quantitative test of the
   **three-layer quantum-brain hypothesis**: the real cryptochrome pair at the
   geomagnetic field is *intrinsically* capable (**IPC ≈ 5.4**) but *biologically
   non-functional* — with the readout biology can access the capacity falls to
   **≈ 0**, and the memory horizon **MC·τ ≈ 3 µs** is 10³–10⁵× shorter than neural
   timescales. The result is a quantitative, falsifiable **no-go** plus a reusable
   capacity/readout/timescale framework. All capacities are out-of-sample
   (train/test split). Holds only in the cryptochrome-like (separated, long-lived)
   regime.
   → `qbscreen/reservoir.py`, `qbscreen/ensemble.py`, `qbscreen/quantum_vs_classical.py`, `qbscreen/qrc_benchmarks.py`

> **Note.** An earlier framing of this project (a heuristic "Q_RPM" screening
> ranking, a ³¹P nuclear-T₂ "discriminant", and a three-layer "quantum brain"
> model) was superseded after it was found to omit electronic decoherence from
> the spin dynamics, which inflated the predicted effects by ~2 orders of
> magnitude. The legacy screening pipeline (`screen_quantum_sites.py`) is
> retained as a parameter-extraction tool (hyperfine magnitudes, inter-radical
> distances from PDB structures) only; its original conclusions are not used.

The electronic-structure inputs use:

- **GFN2-xTB** semi-empirical quantum chemistry (hyperfine/distance parameters)
- **DFT/TDDFT** (B3LYP/def2-SVP via PySCF) for validation
- **CASSCF(2,2)/NEVPT2** to test whether MAO forms a radical pair at all

## Installation

### From PyPI (when published)

```bash
pip install qbscreen
```

### From source

```bash
git clone https://github.com/deeptell-inc/brain_protein_screening.git
cd brain_protein_screening
pip install -e .
```

### With DFT/CASSCF support

```bash
pip install -e ".[dft]"
```

### External dependencies

- **xtb** (v6.7.1+): [github.com/grimme-lab/xtb](https://github.com/grimme-lab/xtb)
- **Open Babel** (3.1+): [openbabel.org](http://openbabel.org)

```bash
conda install -c conda-forge xtb openbabel
```

## Quick Start

### Screen a single protein

```bash
# By PDB ID (auto-downloads)
qbscreen screen 4I6G -o results_cry

# From local PDB file
qbscreen screen pdb_files/pdb2bxr.ent -o results_maoa

# With ONIOM QM/MM embedding
qbscreen screen 4I6G --embedding oniom -o results_oniom
```

### Run spin dynamics simulation

```bash
qbscreen simulate
```

### DFT validation

```bash
pip install pyscf  # required
qbscreen dft-validate
```

### CASSCF mechanism analysis

```bash
qbscreen casscf
```

## Reproducing Paper Results

**Paper 1 — weak-field MFE (negative result):**

```bash
qbscreen mfe                         # Table 1 + Fig. 1, one command, a few seconds
python -m qbscreen.master_equation   # solver validation (reproduces coherent limit)
```

`qbscreen mfe` writes `simulation_results/honest_mfe_results.json` and
regenerates `manuscript/figures/fig_honest_mfe.pdf`. The ESI robustness analyses
(quantum Zeno, singlet–triplet dephasing, Δg mechanism, B² scaling) run as
`pytest qbscreen/tests/test_master_equation.py -v`.

**Paper 2 — quantum reservoir computing (draft: `manuscript/reservoir_rev/`):**

```bash
python -m qbscreen.reservoir          # memory capacity / IPC, engineered vs in-vivo
python -m qbscreen.ensemble           # pooled vs spatially-resolved IPC
python -m qbscreen.quantum_vs_classical  # is the capacity quantum? (72% coherence)
```

**Parameter inputs (legacy electronic-structure tools):**

```bash
qbscreen dft-validate                 # B3LYP/TDDFT benchmarks
qbscreen casscf                       # CASSCF(2,2)/NEVPT2 mechanism test
```

Verify all claims:

```bash
pytest qbscreen/tests/ -v             # 36 tests, all passing
```

## Key Results

Weak-field (50 µT) magnetic field effect on the time-integrated singlet yield,
computed with the decoherence-included master equation:

| Enzyme | Radical pair | exchange J | lifetime τ | MFE (coherent) | MFE (T₂ᵉ = 1 ns) |
|--------|--------------|-----------|-----------|----------------|------------------|
| MAO-A  | active-site (3.5 Å) | ~750 MHz | ~10 ps | 1.7 × 10⁻¹² % | 1.6 × 10⁻¹² % |
| MAO-B  | active-site         | ~750 MHz | ~10 ps | 1.7 × 10⁻¹² % | 1.6 × 10⁻¹² % |
| DAO    | active-site (~4 Å)  | ~400 MHz | ~10 ps | 5.6 × 10⁻¹³ % | 5.0 × 10⁻¹³ % |
| CRY    | separated (15–20 Å) | < 1 MHz  | ~1 µs  | **+2.0 %**    | **+1.0 %**    |

The neurotransmitter-metabolising flavoenzymes are magnetically inert; only
cryptochrome's purpose-built *separated* radical pair responds. The verdict is
independent of the hyperfine assignment (MAO MFE stays at this level for
couplings 5–200 MHz) and of the coherence time, since the null already holds in
the coherent limit — the picosecond lifetime, not decoherence, is what enforces
it. The active-site values are physical rather than a double-precision floor:
they follow the perturbative B² law to better than 0.3 % from 100 µT to 25 mT.

Reproduce: `qbscreen mfe` &nbsp;|&nbsp; verify: `pytest qbscreen/tests/test_master_equation.py`

## Package Structure

```
qbscreen/
├── __init__.py          # Version, metadata
├── cli.py               # Command-line interface
├── screener.py           # Main screening pipeline
├── spin_dynamics.py      # RPM spin Hamiltonian simulation
├── dft_validation.py     # B3LYP/TDDFT benchmarks
├── casscf_analysis.py    # CASSCF(2,2)/NEVPT2
├── reaction_scan.py      # QM/MM reaction coordinate
├── master_equation.py    # Liouville-space Haberkorn solver (+ decoherence)
├── honest_mfe.py         # Paper 1: decoherence-included MFE table
├── cry_lindblad_mfe.py   # Paper 1: cryptochrome Lindblad reference
├── cry_8spin_mfe.py      # Paper 1: 8-spin cryptochrome convergence check
├── final_numbers.py      # Paper 1: reported-value assembly
├── reanalysis.py         # Paper 1: re-analysis vs earlier parameter sets
├── corrected_injection.py   # Paper 1: spin-injection correction
├── reservoir.py          # Paper 2: QRC protocol, memory capacity / IPC
├── ensemble.py           # Paper 2: pooled vs spatially-resolved capacity
├── quantum_vs_classical.py  # Paper 2: coherence-fraction dephasing sweep
├── qrc_benchmarks.py     # Paper 2: reservoir benchmarks
├── readout_routes.py     # Paper 2: readout channel inventory
└── tests/                # 36 tests, all passing
    ├── test_screening.py
    ├── test_spin_dynamics.py
    ├── test_master_equation.py
    ├── test_reservoir.py
    ├── test_quantumness.py
    └── ENVIRONMENT.txt   # recorded software / hardware environment
```

## Citation

If you use this software, please cite:

```bibtex
@article{wakaura2026magneto,
  title={Active-site geometry, not cofactor chemistry, sets the weak-field
         magnetosensitivity of neurotransmitter-metabolising flavoenzymes},
  author={Wakaura, Hikaru and Tanimae, Taiki},
  journal={Phys. Chem. Chem. Phys.},
  year={2026},
  note={Under review, CP-ART-07-2026-002815}
}

@article{wakaura2026reservoir,
  title={Cryptochrome radical pairs as quantum reservoirs: a microsecond
         memory bound on neural reservoir computing},
  author={Wakaura, Hikaru and Tanimae, Taiki},
  journal={PRX Life},
  year={2026},
  note={Submitted}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [GFN2-xTB](https://github.com/grimme-lab/xtb) by S. Grimme and co-workers
- [PySCF](https://pyscf.org/) for DFT/CASSCF calculations
- [Biopython](https://biopython.org/) for PDB parsing
