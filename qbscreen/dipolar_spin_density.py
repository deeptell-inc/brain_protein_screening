#!/usr/bin/env python3
"""Interradical dipolar coupling from distributed spin, not point dipoles.

At van der Waals contact the point-dipole approximation places all spin on one
point per radical; the flavin spin is delocalised over the isoalloxazine ring.
Following Efimova & Hore (2008, their eq. 13), the dipolar tensor is summed over
the spin distributions, here as Löwdin atomic spin populations ρ_i, ρ_j:

    T_ab = d Σ_ij ρ_i ρ_j (δ_ab − 3 r̂_a r̂_b) / r_ij³ ,   d = 52.04 MHz nm³

Our Hamiltonian's dipolar term 2D[(S1·n)(S2·n) − S1·S2/3] has principal value
4D/3 along n, so D = (3/4) T_n with T_n the principal value of largest
magnitude (for a single point pair this gives D = −77.9/r³ MHz, Efimova's D).
Spin populations: UKS B3LYP/def2-SVP of the vertical radical ions at the contact
geometry of ct_coupling.place_amine.
"""

import json
import sys
from pathlib import Path

import numpy as np

from qbscreen.ct_coupling import place_amine

D_PREF = 52.04        # MHz nm³, μ0 g² μB² / (4π h) for g = 2.0023


def spin_populations(sym, xyz, idx, charge, basis="def2-svp", xc="b3lyp"):
    from pyscf import gto, dft, lo
    m = gto.Mole(atom=[(sym[i], tuple(xyz[i])) for i in idx], basis=basis,
                 charge=charge, spin=1, verbose=0, max_memory=4000).build()
    mf = dft.UKS(m).density_fit()
    mf.xc = xc
    mf.conv_tol = 1e-8
    mf.kernel()
    dm = mf.make_rdm1()
    S = m.intor("int1e_ovlp")
    w, v = np.linalg.eigh(S)
    Sh = v @ np.diag(np.sqrt(w)) @ v.T                       # Löwdin S^½
    Ps = Sh @ (dm[0] - dm[1]) @ Sh
    pops = np.zeros(m.natm)
    for mu, (ia, *_rest) in enumerate(m.ao_labels(fmt=False)):
        pops[ia] += Ps[mu, mu]
    return pops


def dipolar_tensor(xa, pa, xb, pb):
    T = np.zeros((3, 3))
    for ri, pi in zip(xa, pa):
        for rj, pj in zip(xb, pb):
            rv = (rj - ri) / 10.0                            # Å → nm
            r = np.linalg.norm(rv)
            u = rv / r
            T += pi * pj * D_PREF * (np.eye(3) - 3 * np.outer(u, u)) / r ** 3
    return T


def effective_D(T):
    w, v = np.linalg.eigh(T)
    k = int(np.argmax(abs(w)))
    others = np.delete(w, k)
    return dict(D_MHz=float(0.75 * w[k]), E_MHz=float(0.25 * (others[0] - others[1])),
                principal_MHz=[float(x) for x in w], axis=v[:, k].tolist())


def main(r=3.5, amine="methylamine", approach="face", twists=(0, 60, 120), min_contact_A=2.4):
    from qbscreen.ct_coupling import min_contact
    for tw in twists:
        sym, xyz, nf = place_amine(r, tw, 0.0, amine=amine, approach=approach)
        if min_contact(sym, xyz, nf) >= min_contact_A:
            break
    else:
        raise ValueError(f"no clash-free orientation for {amine} {approach} at r={r}")
    pa = spin_populations(sym, xyz, range(nf), -1)
    pb = spin_populations(sym, xyz, range(nf, len(sym)), +1)
    T = dipolar_tensor(xyz[:nf], pa, xyz[nf:], pb)
    ca = (pa[:, None] * xyz[:nf]).sum(0) / pa.sum()
    cb = (pb[:, None] * xyz[nf:]).sum(0) / pb.sum()
    rc = np.linalg.norm(cb - ca) / 10
    res = dict(amine=amine, approach=approach, r_N5N_A=r, twist=tw,
               min_contact_A=float(min_contact(sym, xyz, nf)),
               spin_sum_flavin=float(pa.sum()), spin_sum_amine=float(pb.sum()),
               centroid_distance_nm=float(rc), D_point_centroid_MHz=float(-77.9 / rc ** 3),
               D_point_N5N_MHz=float(-77.9 / (r / 10) ** 3), T_full_MHz=T.tolist(), **effective_D(T))
    out = Path("calibration_results/dipolar_spin_density_v5.json")
    allr = json.loads(out.read_text()) if out.exists() else {}
    allr[f"{amine}_{approach}_{r:.2f}"] = res
    out.write_text(json.dumps(allr, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "T_full_MHz"}, indent=2))


if __name__ == "__main__":
    # a single unit spin pair must reproduce Efimova's point-dipole D
    T = dipolar_tensor(np.zeros((1, 3)), [1.0], np.array([[10.0, 0, 0]]), [1.0])
    assert abs(effective_D(T)["D_MHz"] - (-77.9)) < 0.2, effective_D(T)
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 3.5
    for am in (sys.argv[2:] or ["methylamine", "benzylamine"]):
        main(r, am)
