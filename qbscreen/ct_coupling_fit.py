#!/usr/bin/env python3
"""Fit t(r) = t0 exp(−β r/2) to the fragment-coupling scan; report t at contact.

Points whose closest flavin–amine atom contact is below 2.4 Å are steric clashes
of the rigid placement, not physical geometries, and are excluded.
"""

import json
import sys
from pathlib import Path

import numpy as np

from qbscreen.rp_calibration import fit_decay

MIN_CONTACT = 2.4


def analyse(path="calibration_results/ct_coupling_scan_b3lyp.json"):
    rows = [r for r in json.loads(Path(path).read_text()) if r["min_contact"] >= MIN_CONTACT]
    out = {"source": path, "min_contact_cut_A": MIN_CONTACT, "n_points": len(rows), "by_orientation": {}}
    for key in sorted({(r["twist"], r["tilt"]) for r in rows}):
        sub = sorted((r for r in rows if (r["twist"], r["tilt"]) == key), key=lambda r: r["r_N5N"])
        rr = [r["r_N5N"] for r in sub]
        tt = [r["t_meV"] for r in sub]
        t0, beta = fit_decay(rr, tt)
        out["by_orientation"][f"twist{key[0]}_tilt{key[1]}"] = dict(
            r=rr, t_meV=tt, t0_meV=t0, beta_per_A=beta)
    t0, beta = fit_decay([r["r_N5N"] for r in rows], [r["t_meV"] for r in rows])
    out["pooled"] = dict(t0_meV=t0, beta_per_A=beta)
    at35 = [r["t_meV"] for r in rows if abs(r["r_N5N"] - 3.5) < 1e-6]
    out["t_at_3p5A_meV"] = dict(min=min(at35), max=max(at35), geomean=float(np.exp(np.mean(np.log(at35)))),
                               values=at35)
    return out


if __name__ == "__main__":
    res = analyse(*sys.argv[1:])
    Path("calibration_results/ct_coupling_fit.json").write_text(json.dumps(res, indent=2))
    print(f"points used: {res['n_points']} (contact >= {MIN_CONTACT} A)")
    for k, v in res["by_orientation"].items():
        print(f"  {k:16s} beta = {v['beta_per_A']:.2f} /A   t0 = {v['t0_meV']:.3g} meV   n={len(v['r'])}")
    print(f"  pooled           beta = {res['pooled']['beta_per_A']:.2f} /A")
    a = res["t_at_3p5A_meV"]
    print(f"t(3.5 A) = {a['min']:.0f}–{a['max']:.0f} meV, geometric mean {a['geomean']:.0f} meV")
