# qbscreen

**Quantum-active site screening in brain proteins for radical pair mechanism competence**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`qbscreen` is a computational pipeline for screening phosphorus-containing brain proteins for radical pair mechanism (RPM) suitability under physiological conditions (50 μT, 310 K). It combines:

- **GFN2-xTB** semi-empirical quantum chemistry for electronic structure
- **Spin Hamiltonian analysis** with cofactor-specific ³¹P hyperfine coupling constants
- **Five-channel electron spin relaxation** (Anderson-Weiss, Solomon, g-anisotropy, HFC modulation, SOC)
- **Paramagnetic relaxation enhancement** (PRE) with r⁻⁶ distance scaling
- **Density matrix spin dynamics** for magnetic field effect (MFE) prediction
- **DFT/TDDFT validation** (B3LYP/def2-SVP via PySCF)
- **CASSCF(2,2)/NEVPT2** for SET vs polar mechanism analysis

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

All results in the paper can be reproduced with:

```bash
# 1. Screen all 9 proteins (Table 1)
for pdb in 4I6G 2BXR 1OJA 2DU8 3RBF 2OKK 3L6B 1T0L 1I10; do
  qbscreen screen $pdb -o results_${pdb} --no-hess
done

# 2. Spin dynamics (Fig. 4, Extended Data Fig. 6, 8)
qbscreen simulate

# 3. DFT validation (Extended Data Table 3)
qbscreen dft-validate

# 4. CASSCF analysis (Extended Data Fig. 7)
qbscreen casscf

# 5. Reaction coordinate scan (Extended Data Fig. 7a)
qbscreen reaction-scan
```

Or run the full reproducibility test:

```bash
pytest qbscreen/tests/ -v
```

## Key Results

| Protein | Cofactor | Q_RPM | RP confirmed? | Tier |
|---------|----------|-------|---------------|------|
| CRY     | FAD      | 391*  | Yes           | 1    |
| MAO-A   | FAD      | 457   | Uncertain     | 1    |
| MAO-B   | FAD      | 383   | Uncertain     | 1    |
| DDC     | PLP      | 10    | Weak          | 2    |
| SRR     | PLP+Mn   | ~0    | No            | 3    |

*CRY ranks first when J-aware weighting is applied (Q_eff = 391)

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
├── tests/
│   ├── test_screening.py
│   ├── test_spin_dynamics.py
│   └── test_reproducibility.py
└── data/                 # Reference data for tests
```

## Citation

If you use this software, please cite:

```bibtex
@article{wakaura2026qbscreen,
  title={Systematic screening of brain neurotransmitter enzymes for radical
         pair mechanism competence reveals quantitative constraints on
         magnetic field sensitivity},
  author={Wakaura, Hikaru},
  journal={Phys. Chem. Chem. Phys.},
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
