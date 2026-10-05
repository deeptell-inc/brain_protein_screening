#!/usr/bin/env python3
"""Cryptochrome FAD•−/TrpH•+ in the same model as the enzyme map.

Same Hamiltonian, same reaction scheme and same observable (product_yield):
singlet back transfer k_S and spin-independent forward reaction k_P, both
1/µs, product yield Φ_P (the signalling state). The pair is photo-formed, so
no competence constraint applies.

  r = 1.90 nm (FAD–terminal Trp), D = −77.9/r³ MHz, axial along the
  interradical axis; J_EH = J0 exp(−β r), β = 14 nm⁻¹, J0 scanned continuously
  over 8e12–8e14 µT with both signs, J_gap = −2 J_EH (Efimova & Hore,
  Biophys. J. 94, 1565 (2008)).
  FAD•− N5, N10: 523, 189 µT (Lee et al., J. R. Soc. Interface 11, 20131063
  (2014), main text), I = 1.
  TrpH•+: 3-methylindole•+ Fermi contact (hyperfine_dft, B3LYP/def2-TZVP),
  the two largest β-type protons and N1 (I = 1).
Hyperfine couplings are isotropic, so the response depends only on the angle
θ between B and the interradical axis; the powder average is over cos θ.
"""

import json
from pathlib import Path

import numpy as np

from qbscreen.product_yield import build_system, hamiltonian, axial_tensor, yields_hilbert, yields_liouville

MUT_TO_MHZ = 0.028025
R_NM = 1.90
BETA = 14.0
D_EH = -77.9 / R_NM ** 3
B = 50e-6
OUT = Path("calibration_results") / "cry_control.json"


def trp_nuclei():
    d = json.loads(Path("calibration_results/lambda/hfc_def2-tzvp.json").read_text())["methylindole_q+1"]
    H = sorted((x for x in d if x["element"] == "H"), key=lambda x: -abs(x["a_MHz"]))[:2]
    N = [x for x in d if x["element"] == "N"]
    return [(1, 0.5, h["a_MHz"]) for h in H] + [(1, 1.0, n["a_MHz"]) for n in N]


FAD = [(0, 1.0, 523 * MUT_TO_MHZ), (0, 1.0, 189 * MUT_TO_MHZ)]


def mfe_theta(sysd, J_gap, theta, k_S=1.0, k_P=1.0, T2e_ns=None):
    T = axial_tensor(D_EH)
    b = np.array([np.sin(theta), 0.0, np.cos(theta)])
    solve = (lambda H: yields_hilbert(sysd, H, k_S, k_P)[0]) if T2e_ns is None else \
            (lambda H: yields_liouville(sysd, H, k_S, k_P, T2e_ns)[0])
    p0 = solve(hamiltonian(sysd, 0.0, b, J_gap, T))
    return (solve(hamiltonian(sysd, B, b, J_gap, T)) / p0 - 1) * 100


def scan(nuclei, J0s, n_theta=8, T2e_ns=None):
    sysd = build_system(nuclei)
    x, w = np.polynomial.legendre.leggauss(n_theta)      # cos θ on [0, 1]
    cth, w = (x + 1) / 2, w / 2
    rows = []
    for J0 in J0s:
        J_eh = J0 * MUT_TO_MHZ * np.exp(-BETA * R_NM)
        for sign in (+1, -1):
            J = -2 * sign * J_eh
            m = np.array([mfe_theta(sysd, J, np.arccos(c), T2e_ns=T2e_ns) for c in cth])
            rows.append(dict(J0_uT=float(J0), J_EH_MHz=float(sign * J_eh), J_gap_MHz=float(J),
                             mfe_powder=float(np.sum(w * m)),
                             mfe_perp=float(mfe_theta(sysd, J, np.pi / 2, T2e_ns=T2e_ns)),
                             mfe_par=float(mfe_theta(sysd, J, 0.0, T2e_ns=T2e_ns))))
            print(f"J0={J0:.2e} sign={sign:+d} J_gap={J:+8.2f} MHz  powder {rows[-1]['mfe_powder']:+.3f} %  "
                  f"perp {rows[-1]['mfe_perp']:+.3f} %", flush=True)
    return rows


def main():
    J0s = np.logspace(np.log10(8e12), np.log10(8e14), 21)
    full = FAD + trp_nuclei()
    res = dict(r_nm=R_NM, D_MHz=D_EH, nuclei=full, k_S=1.0, k_P=1.0,
               coherent=scan(full, J0s),
               # relaxation check on a reduced pair (FAD N5, N10 + largest Trp proton)
               reduced_coherent=scan(FAD + full[2:3], J0s[::5]),
               reduced_T2e_1us=scan(FAD + full[2:3], J0s[::5], T2e_ns=1000.0))
    OUT.write_text(json.dumps(res, indent=2))
    for k in ("coherent", "reduced_coherent", "reduced_T2e_1us"):
        p = [abs(r["mfe_powder"]) for r in res[k]]
        print(f"{k}: |MFE| powder {min(p):.3f}–{max(p):.3f} %")


if __name__ == "__main__":
    main()
