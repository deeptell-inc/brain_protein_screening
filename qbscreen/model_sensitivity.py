#!/usr/bin/env python3
"""How the searched field effect F changes with the spin model, at the rates
that set the thresholds of G.

For each onward rate k_P in THRESHOLD_ROWS, the row-maximum cell of the
four-nucleus map (worst_case_v6.json) is re-searched in richer models:

  aniso       flavin N5/N10 as full tensors (literature isotropic part, Lee et al.
              2014; DFT spin-dipolar part, rotated into the dipolar-tensor frame);
              aminium α-protons isotropic (see _tensors);
  aniso+N     plus the aminium ¹⁴N (axial DFT tensor along the placed lone pair, I = 1);
  aniso+H8    plus the two largest flavin methyl protons (DFT tensors);
  T2e         a reduced pair (flavin N5, aminium Hα1, isotropic) searched with
              single-electron dephasing, T2e = 10 ns – 10 µs, against the same
              reduced pair without relaxation.

The search is seeded with the four-nucleus optimum and a J grid at its field
direction and dipolar scale, then polished (worst_case_map.polish); like F
itself, the result is the largest value found. The ratio X = F_model / F_base
per row feeds competence_sensitivity.G(factor=…) as a stress test: only the
reference row-maximum cell is re-searched, so the factors are not bounds.
"""

import json
import sys
from pathlib import Path

import numpy as np

from qbscreen.product_yield import build_system
from qbscreen.worst_case_map import Cell, J_grid, polish, search, NUCLEI

R = Path("calibration_results")
THRESHOLD_ROWS = (10.0, 10 ** 1.5, 100.0, 10 ** 2.5)
MUT_TO_MHZ = 0.028025
T2E_NS = (10.0, 100.0, 1000.0, 10000.0)


def _complex_frame():
    """Rotation of the relaxed flavin anion onto the flavin of the contact complex
    (the frame of the dipolar tensor), and the placed amine's lone-pair axis."""
    from qbscreen.ct_coupling import read_xyz, place_amine, lone_pair_direction
    _, a = read_xyz(Path(__file__).parent / "data" / "lumiflavin.xyz")
    _, b = read_xyz(R / "lambda" / "opt_lumiflavin_q-1" / "xtbopt.xyz")
    U, _, Vt = np.linalg.svd((b - b.mean(0)).T @ (a - a.mean(0)))
    Rm = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt       # row vectors: x_complex = x_opt Rm
    sym, xyz, nf = place_amine(3.5, 0.0, 0.0)
    iN = next(i for i in range(nf, len(sym)) if sym[i] == "N")
    return Rm, lone_pair_direction(sym, xyz, iN)


def _tensors():
    """Hyperfine tensors in the frame of the contact complex.

    Flavin tensors are rotated from the relaxed anion onto the complex flavin.
    The complex was built with methylamine, so the benzylaminium tensors have no
    orientation there: its α-protons are taken isotropic (anisotropy ≤ 5 % of the
    isotropic part), and its ¹⁴N tensor, nearly axial about the SOMO, is placed
    with its axis along the lone pair of the placed amine."""
    h = json.loads((R / "lambda" / "hfc_tensors_def2-tzvp.json").read_text())
    Rm, lp = _complex_frame()
    rot = lambda A: Rm.T @ np.asarray(A) @ Rm
    fl = sorted((x for x in h["lumiflavin_q-1"] if x["element"] == "N"), key=lambda x: -abs(x["a_MHz"]))
    flH = sorted((x for x in h["lumiflavin_q-1"] if x["element"] == "H"), key=lambda x: -abs(x["a_MHz"]))
    am = h["benzylamine_q+1"]
    aH = sorted((x for x in am if x["element"] == "H"), key=lambda x: -abs(x["a_MHz"]))[:2]
    aN = next(x for x in am if x["element"] == "N")
    flav = lambda x, iso=None: ((x["a_MHz"] if iso is None else iso) * np.eye(3) + rot(x["A_dip_MHz"])).tolist()
    ev = np.linalg.eigvalsh(np.array(aN["A_dip_MHz"]))
    par, perp = ev[np.argmax(abs(ev))], np.mean(np.delete(ev, np.argmax(abs(ev))))
    n = lp / np.linalg.norm(lp)
    aN_t = (aN["a_MHz"] * np.eye(3) + perp * np.eye(3) + (par - perp) * np.outer(n, n)).tolist()
    aniso = [(0, 1.0, flav(fl[0], 523 * MUT_TO_MHZ)), (0, 1.0, flav(fl[1], 189 * MUT_TO_MHZ)),
             (1, 0.5, aH[0]["a_MHz"]), (1, 0.5, aH[1]["a_MHz"])]
    return {"aniso": aniso,
            "aniso+N": aniso + [(1, 1.0, aN_t)],
            "aniso+H8": aniso + [(0, 0.5, flav(flH[0])), (0, 0.5, flav(flH[1]))]}


def _row_max_cells(rows):
    out = []
    for kp in THRESHOLD_ROWS:
        rr = [r for r in rows if np.isclose(r["k_P"], kp)]
        out.append(max(rr, key=lambda r: r["max_mfe"]))
    return out


def research(nuclei, T, base):
    """Seeded search: J grid at the base direction and scale, then polish from the best seeds."""
    cell = Cell(build_system(nuclei), T, base["k_S"], base["k_P"])
    d = (base["theta"], base["phi"])
    cands = [(cell.mfe(base["J_at_max"], base["s"], *d), base["s"], base["J_at_max"], d)]
    for J in J_grid(base["J_span"], n_side=20):
        cands.append((cell.mfe(J, base["s"], *d), base["s"], float(J), d))
    seeds = sorted(cands, key=lambda c: -c[0])[:2]
    return max((polish(cell, c) for c in seeds), key=lambda c: c[0])[0]


def _job(args):
    name, nuclei, T, base = args
    if name.startswith("T2e"):
        red = [(0, 1.0, 523 * MUT_TO_MHZ), (1, 0.5, NUCLEI[2][2])]
        t2 = None if name == "T2e:coherent" else float(name.split(":")[1])
        r = search(build_system(red), np.array(T), base["k_S"], base["k_P"], T2e_ns=t2)
        return name, base["k_P"], r["max_mfe"]
    return name, base["k_P"], research(nuclei, np.array(T), base)


def main(procs=4):
    from multiprocessing import Pool
    m = json.loads((R / "worst_case_v6.json").read_text())
    T = m["T_contact_MHz"]
    cells = _row_max_cells(m["rows"])
    models = _tensors()
    jobs = [(name, nuc, T, c) for c in cells for name, nuc in models.items()]
    jobs += [(f"T2e:{t}", None, T, c) for c in cells for t in ("coherent",) + T2E_NS]
    res = {}
    with Pool(int(procs)) as pool:
        for name, kp, v in pool.imap_unordered(_job, jobs):
            res.setdefault(name, {})[str(kp)] = v
            print(f"{name:12s} k_P={kp:7.3g}  F={v:.4g} %", flush=True)
            (R / "model_sensitivity.json").write_text(json.dumps(dict(
                base={str(c["k_P"]): c for c in cells}, models=res), indent=2))


def _aniso_cell(args):
    from qbscreen.worst_case_map import supplement
    kS, kP, T = args
    sysd = build_system(_tensors()["aniso"])
    r = search(sysd, np.array(T), kS, kP)
    if r["at_grid_edge"]:
        r = search(sysd, np.array(T), kS, kP, span=16 * abs(r["J_at_max"]))
    row = dict(k_S=float(kS), k_P=float(kP), **r)
    return supplement(sysd, np.array(T), row)


def run_aniso_full(procs=6, out="calibration_results/worst_case_v6_aniso.json"):
    """Full search (first pass, edge rule and fine-s pass) of the anisotropic model
    on the threshold rows, k_S ≥ k_P: replaces the local re-search for this model."""
    from multiprocessing import Pool
    T = json.loads((R / "worst_case_v6.json").read_text())["T_contact_MHz"]
    out = Path(out)
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    done = {(round(np.log10(r["k_S"]), 3), round(np.log10(r["k_P"]), 3)) for r in rows}
    jobs = [(kS, kP, T) for kP in THRESHOLD_ROWS for kS in np.logspace(-2, 5, 15)
            if kS >= kP * (1 - 1e-9) and (round(np.log10(kS), 3), round(np.log10(kP), 3)) not in done]
    with Pool(int(procs)) as pool:
        for r in pool.imap_unordered(_aniso_cell, jobs):
            rows.append(r)
            print(f"aniso k_P={r['k_P']:8.1e} k_S={r['k_S']:8.1e}/us  F={r['max_mfe']:.3e} %", flush=True)
            out.write_text(json.dumps(dict(model="aniso", rows=rows), indent=2))


if __name__ == "__main__":
    if sys.argv[1:2] == ["aniso_full"]:
        run_aniso_full(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
