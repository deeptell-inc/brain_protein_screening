#!/usr/bin/env python3
"""Inner-sphere reorganisation energy of the flavin–amine back electron transfer.

Back ET:  Fl•− + amine•+  →  Fl + amine.
Nelsen four-point scheme, each redox couple separately:

    λ_i(Fl)    = [E(Fl  @ Fl•− geom) − E(Fl  @ Fl  geom)]
               + [E(Fl•− @ Fl  geom) − E(Fl•− @ Fl•− geom)]      (then halved)
    λ_i(amine) = same for amine / amine•+

The inner-sphere λ of the reaction is the sum of the two halved terms. Geometries
are GFN2-xTB optima (xtb --opt); energies are B3LYP/def2-SVP single points
(unrestricted for the radicals), gas phase. Outer-sphere λ is not computed here.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent / "calibration_results" / "lambda"
HARTREE_EV = 27.211386

SPECIES = {                         # name: (source xyz, [(charge, unpaired), ...])
    "lumiflavin":  ("lumiflavin.xyz",  [(0, 0), (-1, 1)]),
    "methylamine": ("methylamine.xyz", [(0, 0), (+1, 1)]),
    "benzylamine": ("benzylamine.xyz", [(0, 0), (+1, 1)]),
    "alanine":     ("alanine.xyz",     [(0, 0), (+1, 1)]),
}


def xtb_opt(src, chrg, uhf, tag):
    wd = HERE / f"opt_{tag}"
    wd.mkdir(exist_ok=True)
    (wd / "in.xyz").write_text((HERE / src).read_text())
    # xtb 6.7.1 (macOS arm64) aborts on a Fortran format error while printing the
    # ANC optimiser's first cycle; the L-BFGS engine avoids that code path.
    (wd / "xcontrol").write_text("$opt\n   engine=lbfgs\n$end\n")
    run = subprocess.run(["xtb", "in.xyz", "--opt", "tight", "--input", "xcontrol",
                          "--chrg", str(chrg), "--uhf", str(uhf)],
                         cwd=wd, check=True, capture_output=True, text=True)
    (wd / "xtb.out").write_text(run.stdout)
    if "GEOMETRY OPTIMIZATION CONVERGED" not in run.stdout:
        raise RuntimeError(f"xtb optimisation of {tag} did not converge")
    return (wd / "xtbopt.xyz").read_text()


def read_xyz_text(txt):
    lines = [l for l in txt.splitlines()[2:] if l.strip()]
    return [(l.split()[0], tuple(float(v) for v in l.split()[1:4])) for l in lines]


def dft_energy(atoms, chrg, uhf, xc="b3lyp", basis="def2-svp"):
    from pyscf import gto, dft
    m = gto.Mole(atom=atoms, basis=basis, charge=chrg, spin=uhf, verbose=0, max_memory=4000).build()
    mf = (dft.UKS(m) if uhf else dft.RKS(m)).density_fit()
    mf.xc = xc
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF not converged for charge {chrg}, uhf {uhf}")
    return float(mf.e_tot)


def main(stage):
    if stage == "opt":
        for name, (src, states) in SPECIES.items():
            for chrg, uhf in states:
                tag = f"{name}_q{chrg:+d}"
                xtb_opt(src, chrg, uhf, tag)
                print("optimised", tag, flush=True)
        return

    geoms = {f"{n}_q{c:+d}": read_xyz_text((HERE / f"opt_{n}_q{c:+d}" / "xtbopt.xyz").read_text())
             for n, (_, st) in SPECIES.items() for c, _ in st}
    res = {}
    for name, (_, states) in SPECIES.items():
        (c0, u0), (c1, u1) = states
        g0, g1 = geoms[f"{name}_q{c0:+d}"], geoms[f"{name}_q{c1:+d}"]
        E = {("0", "0"): dft_energy(g0, c0, u0), ("0", "1"): dft_energy(g1, c0, u0),
             ("1", "1"): dft_energy(g1, c1, u1), ("1", "0"): dft_energy(g0, c1, u1)}
        lam_half = 0.5 * ((E[("0", "1")] - E[("0", "0")]) + (E[("1", "0")] - E[("1", "1")])) * HARTREE_EV
        res[name] = dict(lambda_half_eV=lam_half,
                         E_Ha={f"state{a}@geom{b}": v for (a, b), v in E.items()})
        print(f"{name:12s} λ_i/2 contribution = {lam_half:.3f} eV", flush=True)
    for amine in ("methylamine", "benzylamine", "alanine"):
        res[f"lambda_i_lumiflavin+{amine}_eV"] = res["lumiflavin"]["lambda_half_eV"] + res[amine]["lambda_half_eV"]
    (HERE / "lambda_inner.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("lambda_i")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "opt")
