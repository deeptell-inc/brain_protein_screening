#!/usr/bin/env python3
"""Field-effect search with the exchange coupling tied to the back-transfer rate.

Default model: anisotropic flavin N5/N10 tensors and the two aminium α-protons
(model_sensitivity._tensors()["aniso"]); dipolar tensor = s × contact tensor with
s ∈ [S_LO, 1], i.e. a stacked pair between the 3.5 Å contact geometry and about
6 Å (point-dipole scaling (3.5/6)³ ≈ 0.2).

Each cell (k_S, k_P) stores F(J) = max over s and field direction of |MFE| on the
J scan of worst_case_map (sinh grid plus zooms around local maxima), together
with the zero-field onward yield at each optimum. The exchange constraint is
applied afterwards, so that the reorganization-energy range can be varied
without searching again: for a pair at free energy Δ the reactive exchange is

    J = r k_S / 2π,  r = ½ erfi[(λ − Δ_S) / 2√(λ k_BT)],  Δ_S = Δ + k_BT ln 4

(Fay, Lindoy & Manolopoulos 2018, eq. 90, in our sign convention; rp_theory),
and only λ for which the golden rule holds, k_S ≤ 0.3 ν_n e^{−E_a/k_BT}, are
admitted. F_c(k_S, k_P; Δ) is the largest stored F(J) with J in the admitted
interval, the interval ends included by linear interpolation of the stored F(J)
(a fixed λ admits a single J); J beyond the scanned range takes the edge value.
interp_check compares interpolated values with direct evaluations.
"""

import json
import sys
from pathlib import Path

import numpy as np

from qbscreen.worst_case_map import Cell, J_grid, _local_maxima, directions
from qbscreen.rp_theory import J_over_k, ADIABATIC_LIMIT, NU_N, KB_EV

R = Path("calibration_results")
S_LO = 0.2
S_GRID = (0.2, 0.35, 0.6, 1.0)
KT = KB_EV * 310.0


def scan_cell(sysd, T_contact, k_S, k_P, s_grid=S_GRID, dirs=None, refine=2):
    """F(J) on the J scan: for every J point the best (s, direction) and its yield."""
    cell = Cell(sysd, T_contact, k_S, k_P)
    dirs = dirs or directions()
    span = max(4000.0, 5 * (k_S + k_P) / (2 * np.pi))
    best = {}

    def visit(J):
        J = float(J)
        if J in best:
            return best[J][0]
        v = max(((*cell.over_dirs(J, s, dirs), s) for s in s_grid), key=lambda t: t[0])
        best[J] = (v[0], v[2], v[1], cell.p0(J, v[2]))
        return v[0]

    Jg = J_grid(span)
    vals = [visit(j) for j in Jg]
    for p in _local_maxima(vals):
        lo, hi = Jg[max(p - 1, 0)], Jg[min(p + 1, len(Jg) - 1)]
        for _ in range(refine):
            Jz = np.linspace(lo, hi, 21)
            vz = [visit(j) for j in Jz]
            k = int(np.argmax(vz))
            lo, hi = Jz[max(k - 1, 0)], Jz[min(k + 1, 20)]
    pts = sorted(best.items())
    return dict(k_S=float(k_S), k_P=float(k_P), J_span=span,
                J=[j for j, _ in pts], F=[v[0] for _, v in pts], s=[v[1] for _, v in pts],
                dir=[list(v[2]) for _, v in pts], Phi_P0=[v[3] for _, v in pts])


def allowed_J(k_S, delta, lams, T=310.0):
    """Interval [J_lo, J_hi] (MHz) of reactive exchange admitted at free energy Δ,
    or None if no λ in `lams` keeps the back transfer non-adiabatic."""
    kT = KB_EV * T
    dS = delta + kT * np.log(4)
    rs = [J_over_k(dS, lam, T) for lam in lams
          if k_S * 1e6 <= ADIABATIC_LIMIT * NU_N * np.exp(-(lam - dS) ** 2 / (4 * lam * kT))]
    if not rs:
        return None
    J = np.array(rs) * k_S / (2 * np.pi)
    return float(J.min()), float(J.max())


def F_constrained(row, delta, lams, T=310.0):
    """(F, Φ_P(0)) at the admitted J of largest stored F, or None if no J is admitted.
    Interval ends are interpolated linearly in the stored F(J); beyond the scan, edge values."""
    iv = allowed_J(row["k_S"], delta, lams, T)
    if iv is None:
        return None
    J, F, P = (np.array(row[k]) for k in ("J", "F", "Phi_P0"))
    lo, hi = np.clip(iv, J[0], J[-1])
    pts = [(float(np.interp(x, J, F)), float(np.interp(x, J, P))) for x in (lo, hi)]
    pts += [(F[i], P[i]) for i in np.nonzero((J > lo) & (J < hi))[0]]
    return max(pts)


LAMBDA_SETS = {"wide": tuple(np.linspace(0.2, 2.0, 91)), "0.5": (0.5,), "1.0": (1.0,), "1.5": (1.5,)}
# limiting rates of flavin reduction at saturating substrate, bovine MAO B, 25 °C:
# benzylamine 760 min⁻¹ (Walker & Edmondson 1994, Table 3); β-phenylethylamine 34 300 min⁻¹
# (Husain, Edmondson & Singer 1982)
K_RED = {"benzylamine": 760 / 60, "phenylethylamine": 34300 / 60}


def analyze(path="calibration_results/constrained_map.json", out="calibration_results/constrained_G.json",
            k_cats=(1.0, 10.0, 100.0, 1000.0), T=310.0, T_red=298.15):
    """G(Δ) on the joint-condition region for each λ set and for J unconstrained, with
    the cell, J and zero-field yield that give it (absolute yield change ΔΦ_P).
    The general curves are at T; the measured MAO rates (25 °C) at T_red.
    G is reported as the running maximum from larger Δ (an envelope); n_raised counts
    the Δ points it raises above the value found at that Δ."""
    from qbscreen.competence_sensitivity import G_joint, non_increasing, threshold
    rows = json.loads(Path(path).read_text())["rows"]
    deltas = np.round(np.arange(0.0, 1.2001, 0.01), 2)

    def best_point(r, delta, lams, t):
        if lams is None:
            k = int(np.argmax(r["F"]))
            return r["F"][k], r["Phi_P0"][k]
        return F_constrained(r, delta, lams, t)

    series = [(f"{kc:g}", kc, T) for kc in k_cats] + [(name, k, T_red) for name, k in K_RED.items()]
    res = {}
    for name, lams in list(LAMBDA_SETS.items()) + [("free", None)]:
        res[name] = {}
        for key, kc, t in series:
            raw = [dict(delta_eV=float(d), G_percent=G_joint(
                       rows, d, kc, T=t, value=lambda r: (best_point(r, d, lams, t) or (None,))[0])) for d in deltas]
            g = non_increasing([dict(x) for x in raw])
            n_raised = int(sum(a["G_percent"] is not None and b["G_percent"] > a["G_percent"] for a, b in zip(raw, g)))
            res[name][key] = dict(G=g, T=t, k_cat=kc, n_raised=n_raised,
                                  thr_0p1=threshold(g, 0.1), thr_0p01=threshold(g, 0.01),
                                  defined=[min((x["delta_eV"] for x in g if x["G_percent"] is not None), default=None),
                                           max((x["delta_eV"] for x in g if x["G_percent"] is not None), default=None)])
    at = {}                                          # pair that sets G at the 0.1 % threshold
    d = res["wide"]["1"]["thr_0p1"]
    if d is not None:
        v, r = G_joint(rows, d, 1.0, T=T, value=lambda r: (best_point(r, d, LAMBDA_SETS["wide"], T) or (None,))[0], arg=True)
        F, phi0 = best_point(r, d, LAMBDA_SETS["wide"], T)
        at["1"] = dict(delta_eV=d, k_S=r["k_S"], k_P=r["k_P"], F_percent=F, Phi_P0=phi0, abs_change=F / 100 * phi0)
    n_invalid = sum(allowed_J(r["k_S"], 0.5, LAMBDA_SETS["wide"], T) is None for r in rows)
    Path(out).write_text(json.dumps(dict(n_cells=len(rows), n_invalid_at_0p5eV=n_invalid, k_red=K_RED, T=T, T_red=T_red,
                                         at_threshold=at, G=res), indent=1))
    return res


def interp_check(n=40, procs=8, out="calibration_results/constrained_interp_check.json", seed=1):
    """Direct evaluation (all s values and directions of the search) at the single J
    admitted by a fixed λ, against the interpolated stored F(J), on cells and Δ that
    the fixed-λ thresholds use (k_cat = 1 s⁻¹, Δ within 0.05 eV of the threshold)."""
    from multiprocessing import Pool
    from qbscreen.competence_sensitivity import k_min_per_us
    rows = json.loads((R / "constrained_map.json").read_text())["rows"]
    G = json.loads((R / "constrained_G.json").read_text())["G"]
    rng = np.random.default_rng(seed)
    cand = []
    for lam in ("0.5", "1.0", "1.5"):
        thr = G[lam]["1"]["thr_0p1"]
        for d in np.round(np.arange(thr - 0.05, thr + 0.051, 0.01), 2):
            km = k_min_per_us(d, 1.0)
            for r in rows:
                if km * (4 / r["k_S"] + 1 / r["k_P"]) <= 10 and allowed_J(r["k_S"], d, LAMBDA_SETS[lam]):
                    cand.append((r["k_S"], r["k_P"], float(d), lam))
    pick = [cand[i] for i in rng.choice(len(cand), size=min(n, len(cand)), replace=False)]
    with Pool(int(procs)) as pool:
        res = pool.map(_direct, [(kS, kP, d, lam, allowed_J(kS, d, LAMBDA_SETS[lam])[0]) for kS, kP, d, lam in pick])
    for (kS, kP, d, lam), x in zip(pick, res):
        r = next(r for r in rows if r["k_S"] == kS and r["k_P"] == kP)
        x.update(k_S=kS, k_P=kP, delta_eV=d, lam=lam, F_interp=F_constrained(r, d, LAMBDA_SETS[lam])[0])
    ratios = [x["F_direct"] / x["F_interp"] for x in res if x["F_interp"] > 0]
    Path(out).write_text(json.dumps(dict(n=len(res), ratio_min=min(ratios), ratio_max=max(ratios), points=res), indent=1))
    return min(ratios), max(ratios)


def interp_shift(path="calibration_results/constrained_interp_check.json"):
    """Largest change of a fixed-λ 0.1 % threshold when every interpolated value is
    multiplied by the largest direct/interpolated ratio of interp_check."""
    from qbscreen.competence_sensitivity import G_joint, non_increasing, threshold
    chk = json.loads(Path(path).read_text())
    G = json.loads((R / "constrained_G.json").read_text())["G"]
    rows = json.loads((R / "constrained_map.json").read_text())["rows"]
    shift = 0.0
    for lam in ("0.5", "1.0", "1.5"):
        for key, v in G[lam].items():
            f = lambda r, d: F_constrained(r, d, LAMBDA_SETS[lam], v["T"])
            g = non_increasing([dict(delta_eV=float(d), G_percent=G_joint(
                    rows, d, v["k_cat"], T=v["T"], value=lambda r: None if f(r, d) is None else chk["ratio_max"] * f(r, d)[0]))
                for d in np.round(np.arange(0.0, 1.2001, 0.01), 2)])
            t = threshold(g, 0.1)
            if t is not None and v["thr_0p1"] is not None:
                shift = max(shift, abs(t - v["thr_0p1"]))
    chk["max_threshold_shift_eV"] = round(shift, 2)
    Path(path).write_text(json.dumps(chk, indent=1))
    return shift


def _direct(args):
    from qbscreen.model_sensitivity import _tensors
    from qbscreen.product_yield import build_system
    kS, kP, d, lam, J = args
    T = np.array(json.loads((R / "worst_case_v6.json").read_text())["T_contact_MHz"])
    cell = Cell(build_system(_tensors()["aniso"]), T, kS, kP)
    return dict(J=J, F_direct=max(cell.over_dirs(J, s, directions())[0] for s in S_GRID))


def s_profile(kS=10 ** 2.5, kP=10 ** 1.5, s_values=(0.0, 0.01, 0.03, 0.1, 0.2, 0.35, 0.6, 1.0, 2.0, 3.0), procs=10,
              out="calibration_results/constrained_s_profile.json"):
    """F at one cell as a function of the dipolar scale s, J unconstrained (same J scan)."""
    from multiprocessing import Pool
    T = json.loads((R / "worst_case_v6.json").read_text())["T_contact_MHz"]
    with Pool(int(procs)) as pool:
        res = pool.map(_s_cell, [(kS, kP, T, s) for s in s_values])
    Path(out).write_text(json.dumps(dict(k_S=kS, k_P=kP, rows=[dict(s=s, F=f) for s, f in zip(s_values, res)]), indent=1))
    return res


def _s_cell(args):
    from qbscreen.model_sensitivity import _tensors
    from qbscreen.product_yield import build_system
    kS, kP, T, s = args
    return max(scan_cell(build_system(_tensors()["aniso"]), np.array(T), kS, kP, s_grid=(s,))["F"])


def _cell(args):
    from qbscreen.model_sensitivity import _tensors
    from qbscreen.product_yield import build_system
    kS, kP, T = args
    return scan_cell(build_system(_tensors()["aniso"]), np.array(T), kS, kP)


def main(procs=6, out="calibration_results/constrained_map.json"):
    from multiprocessing import Pool
    T = json.loads((R / "worst_case_v6.json").read_text())["T_contact_MHz"]
    out = Path(out)
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    done = {(round(np.log10(r["k_S"]), 3), round(np.log10(r["k_P"]), 3)) for r in rows}
    jobs = [(kS, kP, T) for kP in np.logspace(-2, 4, 13) for kS in np.logspace(-2, 5, 15)
            if (round(np.log10(kS), 3), round(np.log10(kP), 3)) not in done]
    jobs.sort(key=lambda j: abs(np.log10(j[1]) - 1.75))           # threshold rows first
    with Pool(int(procs)) as pool:
        for r in pool.imap_unordered(_cell, jobs):
            rows.append(r)
            print(f"k_P={r['k_P']:8.1e} k_S={r['k_S']:8.1e}/us  F={max(r['F']):.3e} %  ({len(r['J'])} J)", flush=True)
            out.write_text(json.dumps(dict(model="aniso", s_range=[S_LO, 1.0], rows=rows)))


if __name__ == "__main__":
    if sys.argv[1:2] == ["analyze"]:
        r = analyze()
        for name, d in r.items():
            print(name, {k: (v["thr_0p1"], v["defined"], v["n_raised"]) for k, v in d.items()})
    elif sys.argv[1:2] == ["interp_check"]:
        print(interp_check(), interp_shift())
    elif sys.argv[1:2] == ["s_profile"]:
        print(s_profile())
    elif sys.argv[1:2] == ["time"]:
        import time
        from qbscreen.model_sensitivity import _tensors
        from qbscreen.product_yield import build_system
        T = np.array(json.loads((R / "worst_case_v6.json").read_text())["T_contact_MHz"])
        t = time.time()
        r = scan_cell(build_system(_tensors()["aniso"]), T, 10 ** 2.5, 10 ** 1.5, dirs=directions()[:5], s_grid=(1.0,))
        print(f"{len(r['J'])} J points, {time.time() - t:.0f} s for 1 s value × 5 directions; F max {max(r['F']):.3g} %")
    else:
        main(*sys.argv[1:])
