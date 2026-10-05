#!/usr/bin/env python3
"""Largest geomagnetic response found at given decay rates (a model-internal search).

For the singlet back-transfer rate k_S and the spin-independent onward rate
k_P, search

    F(k_S, k_P) = max over J, field direction n̂ and dipolar scale s ∈ [0, 3]
                  of |MFE_P(50 µT)|,  T_dip = s · T_contact,

for the fixed nuclei below, with coherent spin dynamics. F is the largest value
FOUND by the search, not a proven supremum: a grid over (s, J, n̂) with
zoom refinement around the three largest local maxima in J for each s, followed
by a Nelder–Mead polish in (J, θ, φ, s) from the four best candidates. The
search, the nuclei and the neglect of relaxation are the stated limits of F.

Field directions cover a full hemisphere (the yield is invariant under B → −B).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qbscreen.product_yield import build_system, hamiltonian, yields_hilbert, yields_t2e

S_GRID = (0.0, 0.3, 1.0, 3.0)
S_MAX = 3.0
B_FIELD = 50e-6
NUCLEI = [(0, 1.0, 14.66), (0, 1.0, 5.30), (1, 0.5, 256.9), (1, 0.5, 196.2)]


def directions(n_polar=4, n_az=8):
    """Pole plus rings at θ = 30°, 60°, 90° with 8 azimuths over the full 2π."""
    out = [(0.0, 0.0)]
    for th in np.linspace(np.pi / (2 * (n_polar - 1)), np.pi / 2, n_polar - 1):
        out += [(th, ph) for ph in np.linspace(0, 2 * np.pi, n_az, endpoint=False)]
    return out


def _unit(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


class Cell:
    """MFE evaluator for one (k_S, k_P) with a cached zero-field yield per (J, s)."""

    def __init__(self, sysd, T_contact, k_S, k_P, T2e_ns=None):
        self.sysd, self.Tc, self.kS, self.kP = sysd, np.asarray(T_contact), k_S, k_P
        self._p0 = {}
        self._y = (lambda H: yields_hilbert(sysd, H, k_S, k_P)[0]) if T2e_ns is None else \
                  (lambda H: yields_t2e(sysd, H, k_S, k_P, T2e_ns)[0])

    def p0(self, J, s):
        key = (float(J), float(s))
        if key not in self._p0:
            self._p0[key] = self._y(hamiltonian(self.sysd, 0.0, np.array([0, 0, 1.0]), J, s * self.Tc))
        return self._p0[key]

    def mfe(self, J, s, th, ph):
        pB = self._y(hamiltonian(self.sysd, B_FIELD, _unit(th, ph), J, s * self.Tc))
        return abs(pB / self.p0(J, s) - 1) * 100

    def over_dirs(self, J, s, dirs):
        v = [self.mfe(J, s, th, ph) for th, ph in dirs]
        i = int(np.argmax(v))
        return v[i], dirs[i]


def J_grid(J_span, n_side=50, J0=40.0):
    """Symmetric grid, fine near zero and reaching ±J_span (sinh spacing)."""
    u = np.linspace(0.0, np.arcsinh(J_span / J0), n_side)
    half = J0 * np.sinh(u)
    return np.unique(np.concatenate([-half[::-1], half]))


def _local_maxima(m, n=3):
    idx = [i for i in range(len(m)) if (i == 0 or m[i] >= m[i - 1]) and (i == len(m) - 1 or m[i] >= m[i + 1])]
    return sorted(idx, key=lambda i: -m[i])[:n]


def scan_J(cell, s, span, dirs, refine=2):
    """Grid in J, then zoom twice around each of the three largest local maxima."""
    J = J_grid(span)
    vals = [cell.over_dirs(j, s, dirs) for j in J]
    cands = [(vals[i][0], s, float(J[i]), vals[i][1]) for i in range(len(J))]
    for p in _local_maxima([v[0] for v in vals]):
        lo, hi = J[max(p - 1, 0)], J[min(p + 1, len(J) - 1)]
        for _ in range(refine):
            Jz = np.linspace(lo, hi, 21)
            vz = [cell.over_dirs(j, s, dirs) for j in Jz]
            k = int(np.argmax([v[0] for v in vz]))
            cands.append((vz[k][0], s, float(Jz[k]), vz[k][1]))
            lo, hi = Jz[max(k - 1, 0)], Jz[min(k + 1, 20)]
    return cands


def polish(cell, cand, maxfev=160):
    """Nelder–Mead in (J, θ, φ, s) from one candidate; s is clipped to [0, S_MAX]."""
    v0, s0, J0, (th0, ph0) = cand
    f = lambda x: -cell.mfe(x[0], float(np.clip(x[3], 0.0, S_MAX)), x[1], x[2])
    x0 = np.array([J0, th0, ph0, s0])
    step = np.array([max(0.5, 0.05 * abs(J0)), 0.15, 0.3, 0.1])
    simplex = np.vstack([x0] + [x0 + np.eye(4)[i] * step[i] for i in range(4)])
    r = minimize(f, x0, method="Nelder-Mead", options=dict(initial_simplex=simplex, maxfev=maxfev, xatol=1e-4, fatol=1e-9))
    x = r.x
    return (-r.fun, float(np.clip(x[3], 0.0, S_MAX)), float(x[0]), (float(x[1]), float(x[2]))) if -r.fun > v0 else cand


def search(sysd, T_contact, k_S, k_P, span=None, T2e_ns=None):
    cell = Cell(sysd, T_contact, k_S, k_P, T2e_ns)
    span = span or max(4000.0, 5 * (k_S + k_P) / (2 * np.pi))
    dirs = directions()
    cands = [c for s in S_GRID for c in scan_J(cell, s, span, dirs)]
    grid_best = max(cands, key=lambda c: c[0])
    top = sorted(cands, key=lambda c: -c[0])
    seeds, seen = [], set()
    for c in top:                                   # four distinct seeds
        key = (c[1], round(c[2], 1))
        if key not in seen:
            seen.add(key)
            seeds.append(c)
        if len(seeds) == 4:
            break
    best = max((polish(cell, c) for c in seeds), key=lambda c: c[0])
    return dict(max_mfe=float(best[0]), s=best[1], J_at_max=best[2], theta=best[3][0], phi=best[3][1],
                grid_max=float(grid_best[0]), at_grid_edge=bool(abs(best[2]) > 0.95 * span), J_span=span)


S_FINE = (0.0,) + tuple(np.logspace(-3, np.log10(S_MAX), 40))


def supplement(sysd, T_contact, row, s_grid=None):
    """Second pass over one searched cell: a fine logarithmic grid in the dipolar
    scale (narrow resonances appear where s·D is comparable with the hyperfine and
    Zeeman terms), at a coarse J grid plus the stored optimum, then a polish. The
    cell keeps the larger of the two values."""
    cell = Cell(sysd, T_contact, row["k_S"], row["k_P"])
    dirs = directions()
    Js = sorted(set(J_grid(row["J_span"], n_side=11).tolist() + [row["J_at_max"]]))
    cands = []
    for s in (S_FINE if s_grid is None else s_grid):
        for J in Js:
            v, d = cell.over_dirs(J, s, dirs)
            cands.append((v, float(s), float(J), d))
    seeds = sorted(cands, key=lambda c: -c[0])[:4]
    best = max((polish(cell, c) for c in seeds), key=lambda c: c[0])
    out = dict(row, supplement_max=float(best[0]))
    if best[0] > row["max_mfe"]:
        out.update(max_mfe=float(best[0]), s=best[1], J_at_max=best[2], theta=best[3][0], phi=best[3][1],
                   source="supplement")
    return out


def _supp(args):
    row, T = args
    return supplement(build_system(NUCLEI), np.array(T), row)


def run_supplement(procs=6, path="calibration_results/worst_case_v6.json"):
    from multiprocessing import Pool
    path = Path(path)
    d = json.loads(path.read_text())
    todo = [r for r in d["rows"] if "supplement_max" not in r]
    with Pool(int(procs)) as pool:
        for r in pool.imap_unordered(_supp, [(r, d["T_contact_MHz"]) for r in todo]):
            d["rows"] = [r if (x["k_S"], x["k_P"]) == (r["k_S"], r["k_P"]) else x for x in d["rows"]]
            print(f"k_P={r['k_P']:8.1e} k_S={r['k_S']:8.1e}/us  F={r['max_mfe']:.3e} % (supplement {r['supplement_max']:.3e}) "
                  f"s={r['s']:.4f}", flush=True)
            path.write_text(json.dumps(dict(d, s_fine=list(S_FINE)), indent=2))


def _conv(args):
    row, T = args
    r = supplement(build_system(NUCLEI), np.array(T), row, s_grid=(0.0,) + tuple(np.logspace(-3, np.log10(S_MAX), 120)))
    return dict(k_S=row["k_S"], k_P=row["k_P"], stored=row["max_mfe"], dense=r["supplement_max"])


def convergence_check(procs=4, path="calibration_results/worst_case_v6.json"):
    """Repeat the supplement with a three-times denser s grid on the threshold-setting
    row maxima and on the cell the supplement raised most."""
    from multiprocessing import Pool
    d = json.loads(Path(path).read_text())
    p1 = {(r["k_S"], r["k_P"]): r["max_mfe"] for r in json.loads(Path(path.replace(".json", "_pass1.json")).read_text())["rows"]}
    rows = d["rows"]
    pick = [max(rows, key=lambda r: r["max_mfe"] / p1[(r["k_S"], r["k_P"])])]
    for kp in (10.0, 10 ** 1.5, 100.0):
        pick.append(max((r for r in rows if np.isclose(r["k_P"], kp)), key=lambda r: r["max_mfe"]))
    with Pool(int(procs)) as pool:
        res = pool.map(_conv, [(r, d["T_contact_MHz"]) for r in pick])
    Path("calibration_results/s_convergence.json").write_text(json.dumps(res, indent=2))
    for r in res:
        print(f"k_S={r['k_S']:.3g} k_P={r['k_P']:.3g}  stored {r['stored']:.4g}  dense-s {r['dense']:.4g}", flush=True)


T2E_ROWS = (10.0, 10 ** 1.5, 100.0, 10 ** 2.5)     # onward rates that set the thresholds of G
T2E_NS = (10.0, 100.0, 1000.0)


def _t2e_cell(args):
    kS, kP, T2, T = args
    sysd = build_system(NUCLEI)
    r = search(sysd, np.array(T), kS, kP, T2e_ns=T2)
    if r["at_grid_edge"]:                           # widen until the maximum is interior
        r = search(sysd, np.array(T), kS, kP, span=16 * abs(r["J_at_max"]), T2e_ns=T2)
    return dict(k_S=float(kS), k_P=float(kP), T2e_ns=T2, **r)


def run_t2e(procs=6, out="calibration_results/worst_case_v6_t2e.json"):
    """First-pass search with single-electron dephasing in the four-nucleus model,
    on the threshold rows and k_S ≥ k_P (the cells G uses there)."""
    from multiprocessing import Pool
    T = json.loads(Path("calibration_results/worst_case_v6.json").read_text())["T_contact_MHz"]
    out = Path(out)
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    done = {(round(np.log10(r["k_S"]), 3), round(np.log10(r["k_P"]), 3), r["T2e_ns"]) for r in rows}
    jobs = [(kS, kP, t2, T) for t2 in T2E_NS for kP in T2E_ROWS for kS in np.logspace(-2, 5, 15)
            if kS >= kP * (1 - 1e-9) and (round(np.log10(kS), 3), round(np.log10(kP), 3), t2) not in done]
    with Pool(int(procs)) as pool:
        for r in pool.imap_unordered(_t2e_cell, jobs):
            rows.append(r)
            print(f"T2e={r['T2e_ns']:7.0f} ns k_P={r['k_P']:8.1e} k_S={r['k_S']:8.1e}/us  F={r['max_mfe']:.3e} %", flush=True)
            out.write_text(json.dumps(dict(nuclei=NUCLEI, T_contact_MHz=T, rows=rows), indent=2))


def _cell(args):
    kS, kP, T = args
    sysd = build_system(NUCLEI)
    r = search(sysd, np.array(T), kS, kP)
    if r["at_grid_edge"]:                           # widen until the maximum is interior
        r = search(sysd, np.array(T), kS, kP, span=16 * abs(r["J_at_max"]))
    return dict(k_S=float(kS), k_P=float(kP), **r)


def main(procs=6, out="calibration_results/worst_case_v6.json"):
    from multiprocessing import Pool
    T = json.loads(Path("calibration_results/dipolar_spin_density_v5.json").read_text())["methylamine_face_3.50"]["T_full_MHz"]
    kS_grid = np.logspace(-2, 5, 15)          # 1/µs
    kP_grid = np.logspace(-2, 4, 13)          # 1/µs
    out = Path(out)
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    done = {(round(np.log10(r["k_S"]), 3), round(np.log10(r["k_P"]), 3)) for r in rows}
    jobs = [(kS, kP, T) for kP in kP_grid for kS in kS_grid
            if (round(np.log10(kS), 3), round(np.log10(kP), 3)) not in done]
    meta = dict(nuclei=NUCLEI, T_contact_MHz=T, s_grid=S_GRID, s_max=S_MAX, n_directions=len(directions()))
    with Pool(int(procs)) as pool:
        for r in pool.imap_unordered(_cell, jobs):
            rows.append(r)
            rows.sort(key=lambda d: (d["k_P"], d["k_S"]))
            print(f"k_P={r['k_P']:8.1e} k_S={r['k_S']:8.1e}/us  F={r['max_mfe']:.3e} % (grid {r['grid_max']:.3e})  "
                  f"s={r['s']:.2f} J={r['J_at_max']:+.1f}{'  [EDGE]' if r['at_grid_edge'] else ''}", flush=True)
            out.write_text(json.dumps(dict(meta, rows=rows), indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        d = directions()
        assert len(d) == 25 and max(ph for _, ph in d) > np.pi        # full azimuth
        sysd = build_system(NUCLEI[:2])
        c = Cell(sysd, np.zeros((3, 3)), 1.0, 1.0)
        base = scan_J(c, 0.0, 500.0, d)
        best = max(base, key=lambda x: x[0])
        assert polish(c, best)[0] >= best[0]                           # polish never loses ground
        print("worst_case_map self-checks passed")
    elif sys.argv[1:2] == ["t2e"]:
        run_t2e(*sys.argv[2:])
    elif sys.argv[1:2] == ["converge"]:
        convergence_check(*sys.argv[2:])
    elif sys.argv[1:2] == ["supplement"]:
        run_supplement(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
