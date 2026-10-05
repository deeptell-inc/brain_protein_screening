# Skill: Quantum-Active Site Screening in Brain Proteins

## Meta

- **Purpose**: Systematic computational screening of quantum-active sites in brain proteins, assessment of radical pair mechanism (RPM) suitability, decoherence analysis, and manuscript preparation
- **Reproducibility target**: Any LLM agent with shell access, Python, and xtb can execute this pipeline
- **Session origin**: Multi-session iterative development (2026-03)
- **Domain**: Quantum biology, spin chemistry, computational chemistry, neuroscience

---

## Phase 0: Problem Formulation

### Prompt Pattern
```
"Can we screen quantum-active sites in brain proteins for transition dipole moments
using xtb, including nuclear/electron spin analysis?"
```

### Thinking Process
1. **Decompose the question**: What physical quantities define "quantum activity"? → TDM, HFC, SOC, S-T gap, spin relaxation
2. **Tool selection**: GFN2-xTB (semi-empirical QM, fast, handles 100+ atoms) over DFT (too slow for screening)
3. **Scope definition**: Start with one well-studied protein (cryptochrome/FAD), then expand
4. **Output design**: JSON results + human-readable summary + publication-quality figures

### Rule: Start Narrow, Expand Systematically
- Begin with ONE known system (cryptochrome 4I6G, FAD cofactor) to validate the pipeline
- Only after validation, expand to 9+ proteins
- Never screen broadly before confirming the pipeline produces physically reasonable results on a known system

---

## Phase 1: Pipeline Architecture

### Design Decisions and Rationale

```
PDB fetch → Site extraction → H addition → xTB calculations → Scoring → Output
```

| Step | Tool | Rationale |
|---|---|---|
| PDB fetch | Biopython/urllib | Standard, reliable |
| Site extraction | Custom (ProteinSiteExtractor) | No existing tool detects chromophores + pi-stacks + metal centers generically |
| H addition | Open Babel | xTB requires explicit H atoms |
| QM calculation | GFN2-xTB | Fast (seconds per site), reasonable accuracy for trends |
| Environment | ONIOM (gfn2:gfnff) | Point-charge embedding was NON-FUNCTIONAL in xtb 6.7.1 |
| Dynamics | OpenMM (optional) | MD snapshots for conformational sampling |

### Rule: Verify Tool Functionality Before Building On It
- **Critical lesson**: `$embedding` in xTB's xcontrol was documented but NON-FUNCTIONAL
- **Verification method**: Run extreme test case (+10 charge at 2 A) and check if results change
- **Resolution**: Switched to `--oniom gfn2:gfnff` which demonstrably works
- **General principle**: Always verify that a tool actually does what documentation claims before building a pipeline around it

### Implementation Pattern
```python
# Single-file architecture (screen_quantum_sites.py)
# Rationale: One file = easy to transfer, no import issues, grep-able
#
# Class hierarchy:
#   ProteinSiteExtractor  → finds quantum-active sites in PDB
#   ProteinPreparer       → adds H, prepares for QM
#   QMMMPartitioner       → ONIOM inner/outer splitting
#   XTBRunner             → runs xtb, parses output
#   QuantumSiteScreener   → orchestrates pipeline
#   DecoherenceAnalyzer   → post-processing relaxation times
#   NuclearRelaxAnalyzer  → nuclear spin T1/T2
#
# Data flows through a single @dataclass (QuantumSite) with ~80 fields
```

---

## Phase 2: Physical Quantity Calculations

### 2.1 Electronic TDM (Transition Dipole Moment)

**Prompt pattern**: "Calculate electronic transition dipole moments"

**Method**: Unsold approximation + TRK sum rule (upper bound x 0.3 scaling)

```
|mu_if|^2 = (3 * hbar^2 * f_osc) / (2 * m_e * Delta_E)
f_osc = (N_el * e^2) / (6 * pi * epsilon_0 * m_e * c) * 0.3  (scaling)
```

**Rule**: Always state that semi-empirical TDM is an ORDER-OF-MAGNITUDE estimate. Never claim quantitative accuracy. The 0.3 scaling factor is empirical.

### 2.2 Hyperfine Coupling Constants (HFC)

**Initial approach**: Uniform A(31P) = 200 MHz for all proteins

**Critical revision** (Phase 7): Cofactor-specific HFC assignment

| Cofactor | P position | Distance to radical | Assigned A(31P) | Source |
|---|---|---|---|---|
| FAD (FMN-P) | Riboside phosphate | ~5-7 A from N5 | 200 MHz | Weber 1998 ENDOR |
| FAD (AMP-P) | Adenine phosphate | ~12-15 A from N5 | 2 MHz | Distance attenuation |
| PLP | Direct C5-O-P bond | ~3.5 A from ring | 50 MHz | McConnell + through-bond |
| NADPH (NMN-P) | Near nicotinamide | ~6 A | 100 MHz | Interpolation |
| NADPH (AMP-P) | Adenine side | ~14 A | 1 MHz | Distance attenuation |

**Rule: Never Use Uniform Parameters When Structure Provides Differentiation**
- The 200 MHz uniform assumption was questioned late in the project
- It changed T2e values by up to 3.2x and shifted rankings
- But it did NOT change the qualitative Tier 1/Tier 2 separation (because that's PRE-dominated, r^-6)
- **Lesson**: Always check whether "simplifying assumptions" are load-bearing for conclusions

### 2.3 Spin-Orbit Coupling (SOC)

```python
xi_eff = sqrt(sum(xi_atom^2 * n_atom) / N_total)
# Atomic SOC constants: C=29, N=42, O=79, P=230, S=382, Mn=350 cm^-1
```

**Rule**: SOC from atomic constants is a ZEROTH-ORDER estimate. Only useful for excluding metal-containing sites (xi > 200 cm^-1). Never claim precision better than factor-of-2.

### 2.4 Electron Spin Relaxation (T2e)

Five mechanisms, each with different B-field dependence:

```
1/T2e = 1/T2*(HFC)        # Anderson-Weiss, B-independent
       + 1/T2(dipolar)     # Solomon, B-independent
       + 1/T2(g-aniso)     # proportional to omega_0^2, VANISHES at Earth field
       + 1/T2(A-mod)       # HFC secular modulation, B-independent
       + 1/T2(SOC)         # Orbach-like, weak B-dependence
```

**Critical insight**: At Earth's field (50 uT), g-anisotropy relaxation VANISHES because omega_0 is 45000x smaller than X-band. This EXTENDS T2e by 2-7x compared to lab conditions.

### 2.5 Nuclear Spin Relaxation

**Key physics**: At Earth's field, omega_0 * tau_c << 1 (extreme narrowing)
- T1 = T2 (unlike high-field NMR where T2 << T1)
- Cross-relaxation rates INCREASE by ~350,000x vs X-band
- PRE increases by ~5x

**Rule: Always Specify the Magnetic Field for Relaxation Times**
- Lab (X-band, 0.34 T) and brain (Earth field, 50 uT) give QUALITATIVELY different physics
- Rankings change when moving from lab to brain conditions
- IDH jumped from #4 to #2 specifically because g-anisotropy vanished

### 2.6 Coupled Electron-Nuclear Spin System

**4-level system** (S=1/2, I=1/2):

```
|alpha_e, alpha_n>: E = +omega_S/2 + omega_I/2 + A/4
|alpha_e, beta_n>:  E = +omega_S/2 - omega_I/2 - A/4
|beta_e, alpha_n>:  E = -omega_S/2 + omega_I/2 - A/4
|beta_e, beta_n>:   E = -omega_S/2 - omega_I/2 + A/4
```

**Strong coupling regime**: A (200 MHz) >> omega_S (1.4 MHz) at Earth's field
- Mixing angle theta -> pi/4
- "Forbidden" ZQ transition acquires TDM = 1.22 muB (70% of allowed!)
- This is the PHYSICAL BASIS of RPM: flip-flop transitions drive S-T conversion

---

## Phase 3: Multi-Protein Screening

### Prompt Pattern
```
"Which brain proteins with phosphorus have long relaxation times and large TDM?"
```

### Candidate Selection Strategy

1. **Identify cofactor classes containing P**: PLP, FAD/FMN, NAD(P)H, ATP, TPP
2. **Filter for chromophoric cofactors** (need radical pair generation): PLP, FAD, NAD(P)H
3. **Select representative brain enzymes** for each class:
   - PLP: DDC (dopamine synthesis), GAD67 (GABA), SRR (D-serine)
   - FAD: MAO-A (serotonin), MAO-B (dopamine), DAO (D-serine), CRY (circadian)
   - NADPH: IDH (TCA cycle)
   - NADH: LDH (lactate shuttle)

4. **Download PDB structures** and run pipeline on all 9

### Rule: Parallelize Independent Calculations
- Run all 9 proteins simultaneously as background tasks
- Each takes ~2-5 minutes
- Total wall time: ~5 min (not 45 min sequential)

---

## Phase 4: RPM Suitability Scoring

### 7 Necessary Conditions

| # | Condition | Threshold | Physical basis |
|---|---|---|---|
| C1 | RP generation | f_osc > 0.01 | Need photon absorption for electron transfer |
| C2 | 31P present | n_P >= 1 | HFC driving source for S-T mixing |
| C3 | Electron T2e | > 0.1 ns | Minimum spin evolution time |
| C4 | Strong coupling | A/omega_S > 10 | ZQ flip-flop must be allowed |
| C5 | T1 >> T2 | T1e/T2e > 10 | Relaxation hierarchy maintenance |
| C6 | Nuclear T2 | > 1 us | Nuclear spin state preservation during RP |
| C7 | Moderate SOC | 10 < xi < 200 cm^-1 | Allow ISC without destroying T2 |

### Composite Quality Index

```
Q_RPM = f_osc * T2e(ns) * n_P_eff * [T2(31P)/1us] * [100/xi]^2
```

Where `n_P_eff` = number of 31P with A > 20 MHz (NOT total P count).

### Rule: The Ranking Formula Must Reflect Physics
- The initial formula used total n_P, which was incorrect after cofactor-specific HFC
- Always re-derive scoring when assumptions change
- The DOMINANT discriminant (nuclear T2, r^-6 dependence) was robust to all revisions

---

## Phase 5: Spin Dynamics Simulation

### Simulation 1: RPM Singlet Yield

**Hamiltonian**:
```
H = J S1.S2 + sum_i A_i S_e.I_i + g*muB*B*(S1z+S2z) + sum_j omega_j*I_jz
```

**Performance-critical decision**: Use DIAGONALIZATION, not time-stepping

```python
# SLOW (expm per time step):
U = expm(-2j*pi*H*dt)  # O(dim^3) per step, 100 steps x 40 fields = 4000 calls

# FAST (diagonalize once, analytical integration):
E, V = np.linalg.eigh(H)  # O(dim^3) once
dE = E[:, None] - E[None, :]  # O(dim^2)
G = 1.0 / (k + 1j*dE)  # Green's function
phi_S = k * np.real(np.sum(P_S_eig * rho0_eig.T * G))  # O(dim^2)
```

**Rule: Always Consider Analytical Solutions Before Numerical Time-Stepping**
- The diagonalization approach is EXACT and ~100x faster
- 1024-dim (10 spins) with expm would take hours; with diag, minutes
- This is a general principle in quantum dynamics: if H is time-independent, DIAGONALIZE

### Simulation 2: Lindblad Decoherence

```python
# Lindblad master equation:
# drho/dt = -i[H,rho] + sum_k gamma_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
# Use scipy.integrate.solve_ivp with RK45
```

### Simulation 3: Quantum Reservoir Computing

**Key lesson**: Input encoding matters more than reservoir size
- First attempt: tiny B-field modulation → QRC accuracy BELOW random (36%)
- Fix: Encode input as HFC modulation (physically: different substrates have different HFC)
- After fix: QRC accuracy ~85%

**Rule: QRC Input Must Couple Strongly to Reservoir Dynamics**

### Simulation 4: Time Crystal Nuclear Spin Memory

- Floquet driving of 31P nuclear spins
- DTC order parameter: |<Mz(even)> - <Mz(odd)>|/2
- Coherence lifetime vs noise parameter d
- Initial state matters: |++++> (X-polarized) shows oscillation; |0000> (Z-polarized) may not

---

## Phase 6: Three-Layer Quantum Brain Hypothesis

### Layer Definition

```
Layer 1: Nuclear spin quantum memory
  Carrier: 31P nuclear spins in PO4 groups
  Timescale: T2 ~ 3.2 ms (diamagnetic), 160 us (with PRE)
  Energy: peV - feV (Zeeman, dipolar)

Layer 2: Quantum-classical interface (radical pair)
  Carrier: FADH semiquinone + substrate radical
  Mechanism: HFC-driven S-T mixing → singlet yield Phi_S(B)
  Timescale: T2e ~ 1 ns (electron), RP lifetime ~ 1 us
  Energy: neV - ueV (HFC, Zeeman)

Layer 3: Classical electrochemistry
  Carrier: Neurotransmitters (5-HT, DA, GABA)
  Mechanism: Enzyme kinetics, synaptic transmission
  Timescale: ms - s
  Energy: meV - eV (redox, bond, kT)
```

### Information Flow

```
Layer 1 → Layer 2:  HFC (A=200 MHz) couples 31P nuclear spin to electron spin
                     Nuclear spin state modulates S-T mixing rate

Layer 2 → Layer 3:  Singlet yield Phi_S(B) determines reaction branching ratio
                     MFE of ~0.3-4.4% modulates enzyme turnover rate

Layer 3 → Biology:  Changed neurotransmitter concentration → synaptic effect
```

---

## Phase 7: Manuscript Preparation

### Target: Nature (Article format)

**Structure**:
- Title: < 80 characters, no acronyms
- Abstract: Single paragraph, ~150 words, no references
- Main text: ~3000-4000 words, IMRAD-like but without explicit section headers in Nature style
- Methods: After references, detailed but concise
- Figures: 5 main + Extended Data
- References: ~30-40, Nature numbered style

### Iterative Review Process

**Prompt pattern** (repeated 10x):
```
1. Be Nature's most CRITICAL reviewer. List 5 problems, largest first.
2. Fix those problems.
3. Be Nature's most NEGATIVE reviewer. List 5 problems, largest first.
4. Fix those problems.
```

**Critical vs Negative distinction**:
- **Critical reviewer**: Attacks methodology, asks for controls, questions assumptions
- **Negative reviewer**: Attacks significance, novelty, and whether the work should exist at all

### Key Revisions from Review Cycles

| Round | Issue | Resolution |
|---|---|---|
| 1 | "No experimental validation" | Added explicit "predictions" section with testable hypotheses |
| 2 | "Uniform HFC assumption" | Implemented cofactor-specific HFC (Phase 7 revision) |
| 3 | "Overclaiming significance" | Reframed from "proves quantum brain" to "identifies candidates" |
| 4 | "MFE assumes J=0" | Added J-dependence discussion, acknowledged upper-bound nature |
| 5 | "Title too bold" | Changed from "predicts magnetic field sensitivity" to "identifies candidates" |
| 6-10 | Various precision issues | Quantified uncertainties, added caveats throughout |

### Rule: Reviewer Simulation is the Most Valuable Pre-Submission Step
- Each round caught real problems that would have been fatal in peer review
- The "negative reviewer" mode is especially valuable — it attacks the PREMISE, not just details
- 10 rounds is not excessive; each round finds genuinely new issues

### arXiv Preparation

```
arxiv_submission/
├── main.tex          # Combined main + supplementary (single file)
├── 00README.XXX      # arXiv compilation instructions
└── figures/          # PDF only (no PNG duplicates)
```

**Key steps**:
1. Merge main.tex + supplementary.tex body (extract between \begin{document}...\end{document})
2. Merge bibliographies (deduplicate by bibitem key)
3. Add supplementary counter resets (\setcounter, \renewcommand)
4. Remove \linenumbers (not needed for arXiv)
5. Verify compilation from scratch in clean directory
6. tar.gz < 10 MB

---

## Phase 8: Lessons and Anti-Patterns

### DO

1. **Validate on known system first** before screening broadly
2. **Compute actual distances from PDB** instead of assuming uniform values
3. **Specify magnetic field** for every relaxation time quoted
4. **Use diagonalization** for time-independent Hamiltonians
5. **Parallelize independent calculations** (background tasks)
6. **Question uniform assumptions** — they may be load-bearing
7. **Simulate reviewer attacks** before submission
8. **Track all parameters in a dataclass** with explicit units in field names
9. **Auto-scale display units** (ps/ns/us/ms) based on magnitude

### DON'T

1. **Don't trust tool documentation** without verification (xtb $embedding was broken)
2. **Don't use expm in a loop** when diagonalization gives exact analytical results
3. **Don't confuse lab and brain conditions** — g-anisotropy vanishes at Earth field
4. **Don't use total atom count** when effective count matters (n_P_eff vs n_P)
5. **Don't claim precision beyond method capability** — xTB TDM is order-of-magnitude
6. **Don't write "we prove" in a computational study** — write "our calculations predict"
7. **Don't batch task completions** — mark each done immediately
8. **Don't overfit the scoring formula** — the dominant physics (r^-6 for nuclear T2) is robust

---

## Appendix: Prompt Templates

### A. Research Question Decomposition
```
User: "Can we calculate X for Y?"
Agent thinking:
  1. What physical quantities define X?
  2. What computational tools can estimate them?
  3. What are the accuracy limitations?
  4. What is the minimal test case to validate?
  5. What is the full scope after validation?
```

### B. Debugging Failed Calculations
```
User: "The results look wrong"
Agent thinking:
  1. What is the expected order of magnitude?
  2. Run extreme test case (10x parameter) — does it change results?
  3. If no change: tool is not actually using the parameter (BUG)
  4. If wrong direction: check sign conventions and units
  5. If overflow/underflow: reformulate (e.g., energy gap law overflow fix)
```

### C. Literature-Grounded Parameter Assignment
```
When assigning physical parameters:
  1. Search for EXPERIMENTAL values (ENDOR, EPR, NMR)
  2. If unavailable, use STRUCTURE-BASED estimation (distances, bonds)
  3. If structure unavailable, use ELEMENT-BASED defaults
  4. ALWAYS state the source and estimated uncertainty
  5. ALWAYS check if conclusions change when parameter varies by 2-5x
```

### D. Manuscript Improvement Loop
```
For i in range(10):
  # Critical reviewer (methodology)
  problems = reviewer_critical(manuscript)
  manuscript = fix(manuscript, problems[:5])

  # Negative reviewer (significance)
  problems = reviewer_negative(manuscript)
  manuscript = fix(manuscript, problems[:5])

  # Key: Critical and Negative attack DIFFERENT aspects
  # Critical: "Your method is wrong because..."
  # Negative: "Even if correct, this doesn't matter because..."
```

### E. Figure Generation
```
For each figure:
  1. Define the SINGLE message the figure must convey
  2. Choose the simplest plot type that conveys it
  3. Use ENGLISH ONLY for publication figures (no CJK — causes tofu)
  4. Generate PDF (vector) for LaTeX, PNG for preview
  5. Verify by reading the generated image back
  6. If text is garbled: switch to ASCII-safe font (DejaVu Sans)
```

---

## Appendix: Numerical Validation Checkpoints

| Quantity | Expected range | Source |
|---|---|---|
| FAD electronic TDM | 5-15 D | CIS/TD-DFT literature |
| 31P HFC (FAD) | 100-300 MHz | Weber 1998 ENDOR |
| Electron T2e (organic radical, 310K) | 0.1-10 ns | EPR literature |
| Nuclear T2 (31P, protein) | 10-100 ms (diamagnetic) | NMR literature |
| MFE (radical pair, ~mT) | 1-30% | Steiner & Ulrich 1989 |
| SOC (organic, no metals) | 20-100 cm^-1 | Marian 2012 |
| SOC (Mn2+) | 300-500 cm^-1 | Atomic spectroscopy |
| tau_c (50 kDa protein, 310K) | 5-20 ns | NMR relaxation |

**Rule**: If any computed value falls outside these ranges by >10x, investigate before proceeding.
