#!/usr/bin/env python3
"""
cry_lindblad_mfe.py — Cryptochrome FAD*-/Trp*+ radical pair:
magnetic field effect on the singlet yield, computed two ways.

BACKGROUND
----------
An adversarial review panel (2026-05-14) established three things about the
earlier treatment of this system in this repository:

 1. The reported MFE = -0.55% "at 50 uT" is the 35.714 uT grid point of a
    stored curve whose generating script is absent from the repository.
 2. No code here implements the 8-spin cryptochrome model the manuscripts
    describe (14N carried as I=1, Hilbert dimension 384).
 3. The dissipator was physically wrong.  1/T2* = sqrt(sum_i A_i^2 I(I+1)/3)
    is the *static inhomogeneous* hyperfine linewidth.  When the nuclear
    spins are carried explicitly in H -- as they must be in a radical-pair
    calculation -- that quantity IS the coherent S-T mixing drive.  Feeding
    it back in as an irreversible relaxation rate double-counts the same
    hyperfine coupling in both H and the dissipator.

A first attempt at repair (qbscreen/cry_8spin_mfe.py) introduced a *different*
error: it damped coherences in the eigenbasis of H.  Eigenbasis dephasing
conserves eigenstate populations forever, so it is a quantum-Zeno-type model,
not spin relaxation, and it makes the MFE *grow* as T2 shortens.  That is an
artefact of the model, not physics.

This module fixes that by solving the Lindblad equation in Liouville space
with lab-frame electron Sz dephasing, which is the physically correct
phenomenological channel.

METHOD
------
Singlet-born radical pair, spin-independent (Haberkorn) recombination k:

    drho/dt = -i[H, rho] - k rho + sum_j Lambda (Sjz rho Sjz - rho/4)

Because the recombination is spin independent it factors out exactly, so

    Phi_S = k * Tr[ P_S (k*Id - L0)^{-1} rho0 ],     rho0 = P_S / M

with L0 the Liouvillian of the coherent + dephasing parts.  This is exact for
the model (no Redfield/secular approximation), and is solved by a sparse
linear solve in the d^2-dimensional Liouville space.

    MFE(B) = 100 * [Phi_S(B) - Phi_S(0)] / Phi_S(0)     [percent]

Two independent implementations are cross-validated at Lambda = 0:
  * eigenbasis closed form (Timmel/Hore exponential model), and
  * the Liouville solve above.
They must agree to ~1e-9.  This is the validation the previous code lacked.

All rates are converted to angular frequency (rad/us) consistently.

Deterministic: exact linear algebra, no random numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qbscreen.cry_8spin_mfe import (  # reuse the vetted operator machinery
    GE_MHZ_PER_MT,
    GAMMA_N_MHZ_PER_T,
    Nucleus,
    RPModel,
    build_model,
    cry_nuclei,
    embed,
    spin_matrices,
)

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------
# Liouville-space Lindblad solve
# ----------------------------------------------------------------------
def _electron_sz(model: RPModel) -> list[np.ndarray]:
    _, _, sz = spin_matrices(0.5)
    return [embed(sz, 0, model.dims), embed(sz, 1, model.dims)]


def singlet_yield_lindblad(model: RPModel, B_mT: float, k_MHz: float,
                           Lambda_MHz: float, theta: float, phi: float,
                           ) -> float:
    """Exact Lindblad singlet yield with lab-frame electron Sz dephasing."""
    d = model.dim
    with np.errstate(all="ignore"):          # BLAS sets spurious FPU flags
        H = model.H_static() + model.H_orient(B_mT, theta, phi)
    Hang = TWO_PI * H
    Id = sp.identity(d, dtype=complex, format="csr")
    Hs = sp.csr_matrix(Hang)

    # -i[H, .]  with row-major vec:  vec(AXB) = (A kron B^T) vec(X)
    L = -1j * (sp.kron(Hs, Id, format="csr") - sp.kron(Id, Hs.T, format="csr"))

    if Lambda_MHz > 0.0:
        lam = TWO_PI * Lambda_MHz
        for szop in _electron_sz(model):
            S = sp.csr_matrix(szop)
            L = L + lam * (sp.kron(S, S.T, format="csr")
                           - 0.25 * sp.identity(d * d, dtype=complex,
                                                format="csr"))

    kang = TWO_PI * k_MHz
    A = (kang * sp.identity(d * d, dtype=complex, format="csr") - L).tocsc()

    PS = model.singlet_projector()
    rho0 = (PS / model.M).astype(complex)
    x = spla.spsolve(A, rho0.reshape(-1))
    X = x.reshape(d, d)
    val = kang * np.trace(PS @ X).real
    return float(val)


# ----------------------------------------------------------------------
# Eigenbasis closed form (validation reference, Lambda = 0 only)
# ----------------------------------------------------------------------
def singlet_yield_coherent(model: RPModel, B_mT: float, k_MHz: float,
                           theta: float, phi: float) -> float:
    with np.errstate(all="ignore"):
        H = model.H_static() + model.H_orient(B_mT, theta, phi)
        w, V = np.linalg.eigh(H)
        PSe = V.conj().T @ model.singlet_projector() @ V
    P2 = np.abs(PSe) ** 2
    wr = TWO_PI * w
    dw = wr[:, None] - wr[None, :]
    kk = TWO_PI * k_MHz
    lor = kk * kk / (kk * kk + dw ** 2)
    return float(P2.__mul__(lor).sum().real / model.M)


# ----------------------------------------------------------------------
# Orientation averaging
# ----------------------------------------------------------------------
def orient_avg(fn, model, B_mT, k_MHz, Lambda_MHz, n_theta, n_phi,
               lindblad=True) -> float:
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    thetas = np.arccos(x)
    phis = np.linspace(0.0, TWO_PI, n_phi, endpoint=False)
    tot = wsum = 0.0
    for th, wt in zip(thetas, wx):
        for ph in phis:
            v = (fn(model, B_mT, k_MHz, Lambda_MHz, th, ph) if lindblad
                 else fn(model, B_mT, k_MHz, th, ph))
            tot += wt * v
            wsum += wt
    return tot / wsum


def mfe_at(model, B_list, k_MHz, Lambda_MHz, n_theta=4, n_phi=4,
           lindblad=True):
    fn = singlet_yield_lindblad if lindblad else singlet_yield_coherent
    p0 = orient_avg(fn, model, 0.0, k_MHz, Lambda_MHz, n_theta, n_phi, lindblad)
    out = []
    for B in B_list:
        p = orient_avg(fn, model, float(B), k_MHz, Lambda_MHz,
                       n_theta, n_phi, lindblad)
        out.append(100.0 * (p - p0) / p0)
    return np.array(out), p0


# ----------------------------------------------------------------------
# Reduced model for the Lindblad study (Liouville space must stay tractable)
# ----------------------------------------------------------------------
def build_lindblad_model(A_P31: float = 200.0, J: float = 0.0,
                         n_nuc: int = 3) -> RPModel:
    """2 electrons + n_nuc nuclei.  n_nuc=3 -> d=32, Liouville 1024."""
    return RPModel(nuclei=cry_nuclei(A_P31)[:n_nuc], J=J, r_nm=1.75,
                   include_dipolar=True)


def run(quick: bool = False) -> dict:
    t0 = time.time()
    k_MHz = 1.0                      # tau_RP ~ 1 us
    B_earth = 0.050                  # mT, EXACTLY on grid
    nth, nph = (3, 3) if quick else (4, 4)

    res: dict = {"_meta": {
        "module": "qbscreen/cry_lindblad_mfe.py",
        "method": ("exact Lindblad in Liouville space, lab-frame electron Sz "
                   "dephasing, Haberkorn spin-independent recombination, "
                   "orientation-averaged"),
        "k_MHz": k_MHz,
        "B_earth_mT": B_earth,
        "orientation_grid": {"n_theta_gauss": nth, "n_phi": nph},
        "dephasing_channel": "sum_j Lambda (Sjz rho Sjz - rho/4), j = both electrons",
        "why_not_1_over_T2star": ("1/T2* is the static inhomogeneous HFC width; "
                                  "the HFC is already carried coherently in H, "
                                  "so using it as an irreversible rate double-"
                                  "counts the same interaction."),
    }}

    # ---------- (0) cross-validation of the two solvers at Lambda = 0 ----------
    print("=" * 74)
    print("(0) Solver cross-validation at Lambda = 0 (must agree)")
    print("=" * 74)
    mv = build_lindblad_model(n_nuc=3)
    vals = []
    for B in (0.0, B_earth, 0.29):
        a = orient_avg(singlet_yield_lindblad, mv, B, k_MHz, 0.0, nth, nph, True)
        b = orient_avg(singlet_yield_coherent, mv, B, k_MHz, 0.0, nth, nph, False)
        vals.append({"B_mT": B, "lindblad": a, "eigenbasis": b,
                     "abs_diff": abs(a - b)})
        print(f"  B = {B:6.3f} mT : Lindblad {a:.12f} | eigenbasis {b:.12f} "
              f"| diff {abs(a-b):.2e}")
    res["solver_crossvalidation"] = vals
    worst = max(v["abs_diff"] for v in vals)
    print(f"  worst |diff| = {worst:.2e}  ->  "
          f"{'PASS' if worst < 1e-8 else 'FAIL'}")
    res["_meta"]["crossvalidation_worst_absdiff"] = worst
    res["_meta"]["crossvalidation_pass"] = bool(worst < 1e-8)

    # ---------- (1) THE headline table: MFE vs T2e, physically correct ----------
    print()
    print("=" * 74)
    print("(1) MFE at 50.000 uT vs electron T2  [exact Lindblad, Sz dephasing]")
    print("=" * 74)
    T2_list = [None, 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.54, 1.10, 0.346]
    m = build_lindblad_model(n_nuc=3)
    table = []
    for T2 in T2_list:
        L = 0.0 if T2 is None else 1e3 / T2       # ns -> MHz
        curve, p0 = mfe_at(m, [B_earth, 0.29], k_MHz, L, nth, nph, True)
        row = {"T2e_ns": T2, "Lambda_MHz": L,
               "MFE_50uT_pct": float(curve[0]),
               "MFE_290uT_pct": float(curve[1]),
               "Phi_S_zero": float(p0)}
        table.append(row)
        lab = "coherent" if T2 is None else f"{T2:g} ns"
        print(f"  T2e = {lab:>9s} | Lam = {L:9.2f} MHz | "
              f"MFE(50uT) = {row['MFE_50uT_pct']:+10.6f} % | "
              f"MFE(290uT) = {row['MFE_290uT_pct']:+9.4f} %")
    res["MFE_vs_T2e"] = table

    # ---------- (2) is the suppression the R^2 = (w_S/Gamma)^2 law? ----------
    print()
    print("(2) Suppression law check:  R = w_S/Gamma,  Gamma = 1/T2 + k")
    wS = GE_MHZ_PER_MT * B_earth          # MHz
    ref = [r for r in table if r["T2e_ns"] is None][0]["MFE_50uT_pct"]
    law = []
    for r in table:
        if r["T2e_ns"] is None:
            continue
        G = 1e3 / r["T2e_ns"] + k_MHz
        R = wS / G
        law.append({"T2e_ns": r["T2e_ns"], "R": R, "R2": R * R,
                    "MFE_over_coherent": r["MFE_50uT_pct"] / ref})
        print(f"  T2e={r['T2e_ns']:8g} ns | R={R:9.5f} | R^2={R*R:11.3e} | "
              f"MFE/MFE_coh = {r['MFE_50uT_pct']/ref:11.3e}")
    res["suppression_law"] = {"omega_S_MHz": wS, "rows": law}

    # ---------- (3) nuclear convergence, coherent 8-spin exact ----------
    print()
    print("(3) Nuclear truncation convergence (coherent, exact, 50.000 uT)")
    conv = []
    for n in ([3, 6] if quick else [1, 2, 3, 4, 5, 6]):
        mm = build_model(n_nuc=n)
        c, _ = mfe_at(mm, [B_earth], k_MHz, 0.0, nth, nph, lindblad=False)
        conv.append({"n_nuclei": n, "n_spins": 2 + n, "dim": mm.dim,
                     "MFE_50uT_pct": float(c[0])})
        print(f"  n_nuc={n} ({2+n} spins, dim={mm.dim:4d}) : "
              f"MFE(50uT) = {c[0]:+10.6f} %")
    res["truncation_convergence_coherent"] = conv

    res["_meta"]["runtime_s"] = round(time.time() - t0, 1)
    print()
    print(f"runtime {res['_meta']['runtime_s']} s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run(quick=a.quick)
    out = Path(a.out) if a.out else (
        Path(__file__).resolve().parent.parent / "results_cry_8spin"
        / "cry_lindblad_mfe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
