#!/usr/bin/env python3
"""Electronic coupling H_DA for the flavin–amine charge-transfer radical pair.

The putative MAO radical pair is Fl•− / amine•+. Its back electron transfer moves
an electron from the flavin SOMO (the LUMO of oxidised flavin) into the amine
SOMO (the HOMO of the neutral amine, its N lone pair). The coupling that drives
that transfer is

    t = < LUMO(Fl) | F | HOMO(amine) >

evaluated with the dimer Kohn–Sham Fock matrix in the basis of Löwdin-
orthogonalised fragment orbitals (the fragment-projection method). The same t
fixes the back-transfer rate and the reactive exchange (rp_theory, after Fay,
Lindoy & Manolopoulos 2018).

Geometry: the amine is placed on the flavin si face with its nitrogen on the
ring normal through N5 at distance r, lone pair pointing at N5. This is a
controlled model of the substrate position in the MAO active site, not a
docked structure.

Runs under any Python with PySCF (here: /opt/anaconda3/envs/tensor).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
HARTREE_MEV = 27211.386


def read_xyz(path):
    lines = [l for l in Path(path).read_text().splitlines()[2:] if l.strip()]
    sym = [l.split()[0] for l in lines]
    xyz = np.array([[float(v) for v in l.split()[1:4]] for l in lines])
    return sym, xyz


def _plane_normal(xyz):
    c = xyz - xyz.mean(0)
    return np.linalg.svd(c)[2][2]


def flavin_n5(sym, xyz):
    """N5: the two-coordinate ring N whose carbon neighbours carry no oxygen.

    N1 is also two-coordinate but sits next to C2(=O); N3 carries an H and N10 a
    methyl, so both are three-coordinate.
    """
    def nbrs(i, cut=1.75):
        d = np.linalg.norm(xyz - xyz[i], axis=1)
        return [j for j in np.where((d > 0.1) & (d < cut))[0]]

    for i, s in enumerate(sym):
        if s != "N":
            continue
        nb = nbrs(i)
        if len(nb) != 2 or any(sym[j] != "C" for j in nb):
            continue
        if not any(sym[k] == "O" for j in nb for k in nbrs(j, 1.35)):
            return i
    raise ValueError("N5 not found")


COVALENT_RADIUS = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}


def bonded(sym, xyz, i, scale=1.2):
    """Indices bonded to atom i by the covalent-radius criterion."""
    return [j for j in range(len(sym)) if j != i and
            np.linalg.norm(xyz[j] - xyz[i]) < scale * (COVALENT_RADIUS[sym[i]] + COVALENT_RADIUS[sym[j]])]


def lone_pair_direction(sym, xyz, iN):
    """Opposite of the sum of unit vectors to all atoms bonded to the nitrogen."""
    nb = bonded(sym, xyz, iN)
    if len(nb) != 3:
        raise ValueError(f"amine nitrogen has {len(nb)} bonded atoms, expected 3")
    lp = -np.sum([(xyz[j] - xyz[iN]) / np.linalg.norm(xyz[j] - xyz[iN]) for j in nb], axis=0)
    return lp / np.linalg.norm(lp)


def place_amine(r, twist_deg=0.0, tilt_deg=0.0, face=+1, amine="methylamine", approach="face"):
    """Return (symbols, xyz, n_flavin_atoms) for the dimer at N5···N distance r.

    approach="face": the amine N sits on the flavin ring normal through N5.
    approach="edge": the amine N sits in the flavin plane, on the outward
    bisector at N5 (away from the ring centroid), i.e. an edge-on contact.
    In both cases the nitrogen lone pair (opposite the three bonds) is aimed at
    N5; twist rotates the amine about that axis, tilt tips the axis.
    """
    fs, fx = read_xyz(DATA / "lumiflavin.xyz")
    ms, mx = read_xyz(DATA / f"{amine}.xyz")

    n5 = flavin_n5(fs, fx)
    if approach == "face":
        axis = face * _plane_normal(fx)
    elif approach == "edge":
        out = fx[n5] - fx.mean(0)
        nrm = _plane_normal(fx)
        out -= (out @ nrm) * nrm
        axis = out / np.linalg.norm(out)
    else:
        raise ValueError(approach)

    iN = ms.index("N")
    lp = lone_pair_direction(ms, mx, iN)

    if tilt_deg:
        perp = np.cross(axis, _plane_normal(fx) if approach == "edge" else [1.0, 0.0, 0.0])
        perp /= np.linalg.norm(perp)
        axis = _rot(perp, np.radians(tilt_deg)) @ axis
    R1 = _align(lp, -axis)
    body = (mx - mx[iN]) @ R1.T
    if twist_deg:
        body = body @ _rot(axis, np.radians(twist_deg)).T
    amine_xyz = body + fx[n5] + r * axis

    sym = list(fs) + list(ms)
    xyz = np.vstack([fx, amine_xyz])
    return sym, xyz, len(fs)


def _rot(u, a):
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _align(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def min_contact(sym, xyz, nf):
    d = np.linalg.norm(xyz[:nf, None, :] - xyz[None, nf:, :], axis=2)
    return float(d.min())


def _mol(sym, xyz, idx, basis):
    from pyscf import gto
    m = gto.Mole()
    m.atom = [(sym[i], tuple(xyz[i])) for i in idx]
    m.basis = basis
    m.charge = 0
    m.spin = 0
    m.verbose = 0
    m.max_memory = 2500
    return m.build()


def _ks(m, xc):
    from pyscf import dft
    mf = dft.RKS(m).density_fit()   # RI-J: ~10x faster, negligible effect on t
    mf.xc = xc
    mf.conv_tol = 1e-9
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF not converged")
    return mf


def flavin_fragment(xc="b3lyp", basis="def2-svp"):
    """Oxidised lumiflavin SCF, shared by every scan point (its geometry is fixed)."""
    fs, fx = read_xyz(DATA / "lumiflavin.xyz")
    m = _mol(fs, fx, range(len(fs)), basis)
    mf = _ks(m, xc)
    return dict(nao=m.nao, lumo=mf.mo_coeff[:, m.nelectron // 2].copy(),
                e_lumo_eV=float(mf.mo_energy[m.nelectron // 2] * 27.211386))


def coupling(sym, xyz, nf, frag_fl, xc="b3lyp", basis="def2-svp"):
    """Fragment-projection coupling between LUMO(flavin) and HOMO(amine), meV."""
    ia = list(range(nf))
    ib = list(range(nf, len(sym)))
    mB, mD = _mol(sym, xyz, ib, basis), _mol(sym, xyz, ia + ib, basis)
    fB, fD = _ks(mB, xc), _ks(mD, xc)

    nA = frag_fl["nao"]
    assert nA + mB.nao == mD.nao
    lumo = np.zeros(mD.nao)
    lumo[:nA] = frag_fl["lumo"]
    homo = np.zeros(mD.nao)
    homo[nA:] = fB.mo_coeff[:, mB.nelectron // 2 - 1]

    C = np.column_stack([lumo, homo])
    H = C.T @ fD.get_fock() @ C
    Sm = C.T @ mD.intor("int1e_ovlp") @ C
    w, v = np.linalg.eigh(Sm)
    X = v @ np.diag(w ** -0.5) @ v.T
    He = X @ H @ X
    return dict(
        t_meV=float(abs(He[0, 1]) * HARTREE_MEV),
        e_lumo_Fl_eV=float(He[0, 0] * HARTREE_MEV / 1e3),
        e_homo_am_eV=float(He[1, 1] * HARTREE_MEV / 1e3),
        overlap=float(Sm[0, 1]),
        E_dimer_Ha=float(fD.e_tot),
    )


def _point(args):
    r, tw, tl, frag_fl, xc, amine, approach = args
    sym, xyz, nf = place_amine(r, tw, tl, amine=amine, approach=approach)
    t0 = time.time()
    res = coupling(sym, xyz, nf, frag_fl, xc=xc)
    res.update(r_N5N=r, twist=tw, tilt=tl, xc=xc, amine=amine, approach=approach,
               min_contact=min_contact(sym, xyz, nf),
               seconds=round(time.time() - t0, 1))
    return res


SCAN_R = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
SCAN_ORIENT = {"face": ((0, 0), (60, 0), (120, 0), (0, 15), (0, -15)),
               "edge": ((0, 0), (90, 0), (0, 20))}


def scan(amines=("methylamine", "benzylamine"), approaches=("face", "edge"), rs=SCAN_R,
         xc="b3lyp", out="calibration_results/ct_coupling_scan_v5.json", procs=6):
    from multiprocessing import Pool
    frag_fl = flavin_fragment(xc)
    jobs = [(r, tw, tl, frag_fl, xc, am, ap)
            for am in amines for ap in approaches for r in rs for tw, tl in SCAN_ORIENT[ap]]
    rows = []
    with Pool(procs) as pool:
        for res in pool.imap_unordered(_point, jobs):
            rows.append(res)
            print(f"{res['amine']:11s} {res['approach']:4s} r={res['r_N5N']:4.2f} "
                  f"twist={res['twist']:3d} tilt={res['tilt']:4d}  t={res['t_meV']:8.3f} meV  "
                  f"contact={res['min_contact']:.2f} A  ({res['seconds']} s)", flush=True)
            rows.sort(key=lambda d: (d["amine"], d["approach"], d["r_N5N"], d["twist"], d["tilt"]))
            Path(out).parent.mkdir(exist_ok=True)
            Path(out).write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    scan()
