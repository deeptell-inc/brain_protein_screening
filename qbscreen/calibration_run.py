#!/usr/bin/env python3
"""Every calibrated number in the manuscript, written to calibration_results/*.json.

Inputs and their sources
  contact coupling t(r)          : ct_coupling_fit.json (B3LYP fragment scan)
  inner-sphere λ                 : lambda/lambda_inner.json (4-point, GFN2 geometries)
  measured contact-pair back ET  : Liu et al., PNAS 110, 12966 (2013), photolyase
                                   FAD•−/W382•+: τ = 70 ps, λ ≈ 0.97 eV, ΔG ≈ −1.78 eV
  flavin hyperfine               : Lee et al., J. R. Soc. Interface 11, 20131063 (2014)
                                   N5 523 µT, N10 189 µT
  partner hyperfine              : lambda/hfc_def2-tzvp.json (DFT Fermi contact)
  cryptochrome J0, D, r          : Efimova & Hore, Biophys. J. 94, 1565 (2008)
  electron spin relaxation       : T2e = 1 µs (Kattnig, Solov'yov & Hore, PCCP 18, 12443 (2016))
"""

import json
from pathlib import Path

import numpy as np

from qbscreen.calibrated_mfe import mfe, mfe_general, dipolar_D_MHz
from qbscreen.rp_calibration import (J_superexchange_MHz, k_interpolated, J_tau,
                                     delta_max_for_turnover)

OUT = Path("calibration_results")
B = 50e-6
T2E_NS = 1000.0
MUT_TO_MHZ = 0.028025
FL = [14.66, 5.30]                       # FAD•− N5, N10 (MHz), spin-½ effective
PARTNER = [44.0]                         # overwritten from the DFT table when present


def _partner_from_dft():
    p = OUT / "lambda" / "hfc_def2-tzvp.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    out = {}
    for tag in ("methylamine_q+1", "benzylamine_q+1", "alanine_q+1"):
        big = max(d[tag], key=lambda r: abs(r["a_MHz"]))
        out[tag] = abs(big["a_MHz"])
    return out


def locus(delta, lam, r_nm, partner, ts=(290, 160, 100, 30, 22, 10, 3, 1, 0.3, 0.1, 0.03, 0.01)):
    D = dipolar_D_MHz(r_nm)
    rows = []
    for t in ts:
        J = -J_superexchange_MHz(t, delta)
        k = k_interpolated(t, -delta, lam) * 1e-6
        row = dict(t_meV=t, J_gap_MHz=J, tau_s=1e-6 / k)
        for th in (0, 90):
            row[f"mfe_coh_th{th}"] = mfe(B, np.radians(th), FL, partner, J, D, k)
            row[f"mfe_T2e_th{th}"] = mfe(B, np.radians(th), FL, partner, J, D, k, T2E_NS)
        rows.append(row)
    return dict(delta_eV=delta, lambda_eV=lam, r_nm=r_nm, D_MHz=D,
                J_tau_nonadiabatic=J_tau(delta, lam), rows=rows)


def dipolar_cap(r_nm, partner, tau_us=(1.0, 10.0, 100.0)):
    """Coherent MFE with J = 0 and long lifetimes: what D alone permits."""
    D = dipolar_D_MHz(r_nm)
    return dict(r_nm=r_nm, D_MHz=D, rows=[
        dict(tau_us=tau, **{f"mfe_th{th}": mfe(B, np.radians(th), FL, partner, 0.0, D, 1 / tau)
                            for th in (0, 45, 90)})
        for tau in tau_us])


def competence_grid(ts=(22, 109, 160, 290), lams=(0.7, 1.0, 1.4), kcats=(1, 10, 100)):
    rows = []
    for t in ts:
        for lam in lams:
            for kc in kcats:
                d = delta_max_for_turnover(t, lam, kc)
                k = k_interpolated(t, -d, lam)
                rows.append(dict(t_meV=t, lambda_eV=lam, k_cat=kc, delta_max_eV=d,
                                 tau_s=1 / k, J_gap_MHz=-J_superexchange_MHz(t, d)))
    return rows


def _powder(fn, n=6):
    x, w = np.polynomial.legendre.leggauss(n)            # nodes in cosθ ∈ [−1, 1]
    th = np.arccos(x)
    return float(np.sum(w * np.array([fn(t) for t in th])) / 2)


def cry_control(r_nm=1.90, tau_us=1.0):
    D = dipolar_D_MHz(r_nm)
    models = {"spin_half": [(0, 0.5, FL[0]), (0, 0.5, FL[1]), (1, 0.5, 45.0)],
              "spin_one_N": [(0, 1.0, FL[0]), (0, 1.0, FL[1]), (1, 0.5, 45.0)]}
    rows = []
    for J0 in (8e12, 8e13, 8e14):
        J_eh = J0 * MUT_TO_MHZ * np.exp(-14 * r_nm)
        for sign in (+1, -1):
            J = -2 * sign * J_eh
            for name, nuc in models.items():
                row = dict(model=name, J0_uT=J0, J_EH_MHz=sign * J_eh, J_gap_MHz=J)
                for th in (0, 90):
                    row[f"mfe_T2e_th{th}"] = mfe_general(B, np.radians(th), nuc, J, D, 1 / tau_us, T2E_NS)
                if name == "spin_half":
                    row["mfe_T2e_powder"] = _powder(
                        lambda t: mfe_general(B, t, nuc, J, D, 1 / tau_us, T2E_NS))
                rows.append(row)
    return dict(r_nm=r_nm, D_MHz=D, tau_us=tau_us, T2e_ns=T2E_NS, rows=rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    dft = _partner_from_dft()
    partner = {"MAO": [dft["benzylamine_q+1"]] if dft else PARTNER,
               "DAO": [dft["alanine_q+1"]] if dft else PARTNER}
    lam = json.loads((OUT / "lambda" / "lambda_inner.json").read_text())
    res = dict(
        partner_hfc_MHz=partner, lambda_inner=lam,
        competence=competence_grid(),
        MAO_locus_inverted=locus(1.78, 0.97, 0.35, partner["MAO"]),
        MAO_locus_activationless=locus(1.0, 1.0, 0.35, partner["MAO"]),
        MAO_dipolar_cap=dipolar_cap(0.35, partner["MAO"]),
        DAO_dipolar_cap=dipolar_cap(0.40, partner["DAO"]),
        CRY=cry_control(),
    )
    (OUT / "manuscript_numbers.json").write_text(json.dumps(res, indent=2))
    print("partner hfc (MHz):", partner)
    print("Δ_max range over grid: %.3f–%.3f eV; τ at Δ_max: %.2g–%.2g s" % (
        min(r["delta_max_eV"] for r in res["competence"]), max(r["delta_max_eV"] for r in res["competence"]),
        min(r["tau_s"] for r in res["competence"]), max(r["tau_s"] for r in res["competence"])))
    for k in ("MAO_dipolar_cap", "DAO_dipolar_cap"):
        cap = max(abs(v) for row in res[k]["rows"] for kk, v in row.items() if kk.startswith("mfe"))
        print(f"{k}: max |MFE| = {cap:.2e} %  (D = {res[k]['D_MHz']:.0f} MHz)")
    for k in ("MAO_locus_inverted", "MAO_locus_activationless"):
        top = max(abs(r["mfe_coh_th0"]) for r in res[k]["rows"])
        print(f"{k}: max |MFE_coh| = {top:.2e} %")
    cr = [r for r in res["CRY"]["rows"] if r["model"] == "spin_half"]
    print("CRY powder |MFE|: %.2f–%.2f %%" % (min(abs(r["mfe_T2e_powder"]) for r in cr),
                                           max(abs(r["mfe_T2e_powder"]) for r in cr)))
