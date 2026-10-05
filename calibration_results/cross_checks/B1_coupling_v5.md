# B1 — v5 coupling scan, t(r) (2026-09-29)

Source: ct_coupling_scan_v5.json (112 geometries; B3LYP fragment projection,
GFN2-optimised amines). Geometries with a fragment–fragment contact < 2.0 Å
are excluded as steric clashes.

| pair                | valid r (N5–N)  | |t| max per r (meV)                      | β for |t|² (max / median) |
|---------------------|-----------------|------------------------------------------|---------------------------|
| methylamine, face   | 3.0–6.0 Å (35)  | 312, 106, 38, 31, 15, 4.2, 1.1           | 3.5 / 4.2 Å⁻¹             |
| methylamine, edge   | 4.5–6.0 Å (11)  | 171, 81, 35, 13                          | 3.5 / 4.5 Å⁻¹             |
| benzylamine, face   | 3.5–6.0 Å (22)  | 52, 7.9, 35, 21, 8.3, 2.0                | 1.9 / 2.9 Å⁻¹ (noisy)     |
| benzylamine, edge   | only r = 6.0 (1)| 0.44                                     | —                         |

Findings
- At van der Waals contact |t| = 50–300 meV. With λ = 0.5–1.4 eV the back
  transfer is then adiabatic (k_NA/k_TST ≫ ADIABATIC_LIMIT), so golden-rule
  rates and the Fay reactive J are outside validity at contact; they apply
  only beyond ≈ 5 Å (|t| ≲ 15 meV).
- The competence bound in B3 does not use t, so the thermodynamic conclusion
  is unaffected.
- The benzylamine edge placement clashes (contact 0.8–1.8 Å) at every r below
  6 Å: `place_amine(approach="edge")` puts the phenyl ring into the flavin.
  Benzylamine-edge is dropped rather than patched; the face approach covers
  benzylamine.
