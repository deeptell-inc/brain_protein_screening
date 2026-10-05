#!/usr/bin/env python3
"""Free energy Δ of the flavin–amine radical pair above the closed-shell state.

Δ = [E(Fl•−) + E(amine•+)] − [E(Fl) + E(amine)] + E_pair,

each species at its own GFN2-xTB optimum (adiabatic), B3LYP/def2-SVP in a
polarisable continuum of dielectric ε (PCM), and E_pair = −e²/(4πε0 ε_pair r)
the Coulomb stabilisation of the contact ion pair at charge-centre separation r.
No zero-point or thermal corrections: Δ is an electronic free-energy estimate
with an uncertainty of a few tenths of an eV, reported as a range over ε and r.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent / "calibration_results"
HARTREE_EV = 27.211386
COULOMB_EV_A = 14.399645        # e²/(4πε0) in eV·Å


def energy(xyz_path, charge, spin, eps, basis="def2-svp", xc="b3lyp"):
    from pyscf import gto, dft, solvent
    lines = [l for l in Path(xyz_path).read_text().splitlines()[2:] if l.strip()]
    atoms = [(l.split()[0], tuple(float(v) for v in l.split()[1:4])) for l in lines]
    m = gto.Mole(atom=atoms, basis=basis, charge=charge, spin=spin, verbose=0, max_memory=4000).build()
    ks = (dft.UKS(m) if spin else dft.RKS(m)).density_fit()
    ks.xc = xc
    mf = solvent.PCM(ks)
    mf.with_solvent.eps = eps
    mf.conv_tol = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF not converged: {xyz_path} q={charge} eps={eps}")
    return float(mf.e_tot)


def main(eps_list=(4.0, 10.0)):
    L = HERE / "lambda"
    out = {}
    for eps in eps_list:
        E = {}
        for tag, q, s in (("lumiflavin_q+0", 0, 0), ("lumiflavin_q-1", -1, 1),
                          ("methylamine_q+0", 0, 0), ("methylamine_q+1", 1, 1),
                          ("benzylamine_q+0", 0, 0), ("benzylamine_q+1", 1, 1)):
            E[tag] = energy(L / f"opt_{tag}" / "xtbopt.xyz", q, s, eps)
            print(f"eps={eps} {tag}: {E[tag]:.6f} Ha", flush=True)
        fl = (E["lumiflavin_q-1"] - E["lumiflavin_q+0"]) * HARTREE_EV
        row = {"E_Ha": E, "flavin_EA_eV": -fl}
        for am in ("methylamine", "benzylamine"):
            ion = (E[f"{am}_q+1"] - E[f"{am}_q+0"]) * HARTREE_EV
            sep = fl + ion                                # separated ions in the continuum
            row[am] = {"separated_ions_eV": sep,
                       "Delta_eV": {f"r{r:.1f}": sep - COULOMB_EV_A / (eps * r) for r in (4.0, 5.0, 6.0)}}
        out[f"eps{eps:g}"] = row
    (HERE / "driving_force.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        for am in ("methylamine", "benzylamine"):
            d = v[am]["Delta_eV"]
            print(f"{k} {am}: separated {v[am]['separated_ions_eV']:.2f} eV; Δ(r=4–6 Å) = "
                  f"{min(d.values()):.2f}–{max(d.values()):.2f} eV")


SHE_ABS_V = 4.44                 # absolute potential of the aqueous SHE (Jonsson et al. 1996, eq 3)
EXPERIMENT_V = {                 # aqueous, V vs NHE
    "dimethylamine": 1.30,       # E°(R2NH•+/R2NH), Jonsson, Wayner & Lusztyk, JPC 100, 17539 (1996)
    "lumiflavin": -0.313,        # FMN ox/semiquinone, pH 7, Mayhew, EJB 265, 698 (1999)
}


def aqueous_potentials(eps=78.4):
    """One-electron potentials in water with the same protocol, against experiment.

    The flavin couple is written as a reduction potential, the amines as
    E°(amine•+/amine); the per-half-reaction error is DFT − experiment."""
    L = HERE / "lambda"
    out = {}
    for mol, q in (("dimethylamine", +1), ("methylamine", +1), ("benzylamine", +1), ("lumiflavin", -1)):
        e0 = energy(L / f"opt_{mol}_q+0" / "xtbopt.xyz", 0, 0, eps)
        e1 = energy(L / f"opt_{mol}_q{q:+d}" / "xtbopt.xyz", q, 1, eps)
        E = q * (e1 - e0) * HARTREE_EV - SHE_ABS_V
        out[mol] = {"E_dft_V": E, "E_exp_V": EXPERIMENT_V.get(mol)}
        if mol in EXPERIMENT_V:
            out[mol]["error_V"] = E - EXPERIMENT_V[mol]
        print(f"{mol}: DFT {E:+.2f} V" + (f", exp {EXPERIMENT_V[mol]:+.3f} V" if mol in EXPERIMENT_V else ""), flush=True)
    (HERE / "aqueous_potentials.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    aqueous_potentials() if sys.argv[1:] == ["aqueous"] else main()
