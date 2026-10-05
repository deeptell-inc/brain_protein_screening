#!/usr/bin/env python3
"""Supporting-information tables for manuscript/magnetobio_jcp, from calibration_results/*.json."""

import json
from pathlib import Path

import numpy as np

R = Path("calibration_results")
OUT = Path("manuscript/magnetobio_jcp/si_tables")


def _j(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def _tab(name, cols, head, rows, size=""):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    (OUT / name).write_text(f"{size}\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")


def g2(x):
    return "0" if x == 0 else (f"{x:.2g}" if abs(x) >= 1e-3 else f"{x:.1e}".replace("e-0", "e-").replace("e-", r"\text{e-}"))


def coupling():
    scan = _j("ct_coupling_scan_v5.json")
    rows = []
    for am, ap in (("methylamine", "face"), ("methylamine", "edge"), ("benzylamine", "face"), ("benzylamine", "edge")):
        for r in sorted({x["r_N5N"] for x in scan}):
            s = [x for x in scan if x["amine"] == am and x["approach"] == ap and x["r_N5N"] == r]
            ok = [abs(x["t_meV"]) for x in s if x["min_contact"] >= 2.0]
            rows.append([am, ap, f"{r:.1f}", f"{len(ok)}/{len(s)}",
                         f"{np.median(ok):.1f}" if ok else "--", f"{max(ok):.1f}" if ok else "--",
                         f"{min(x['min_contact'] for x in s):.2f}"])
    _tab("coupling.tex", "llrrrrr", r"amine & approach & $r_{\rm N5N}$ (\AA) & valid & median $|t|$ (meV) & max $|t|$ (meV) & min contact (\AA)", rows, r"\small")


def hyperfine():
    h = _j("lambda/hfc_def2-tzvp.json")
    names = {"lumiflavin_q-1": r"lumiflavin$^{\bullet-}$", "methylamine_q+1": r"methylaminium$^{\bullet+}$",
             "benzylamine_q+1": r"benzylaminium$^{\bullet+}$", "methylindole_q+1": r"3-methylindole$^{\bullet+}$"}
    rows = []
    for tag, lab in names.items():
        for x in sorted(h[tag], key=lambda x: -abs(x["a_MHz"])):
            if abs(x["a_MHz"]) >= 1.0:
                rows.append([lab, f"{x['element']}{x['index']}", f"{x['a_MHz']:+.1f}", f"{1000 * x['a_mT']:+.0f}"])
                lab = ""
    _tab("hyperfine.tex", "llrr", r"radical & nucleus (atom index) & $a_{\rm iso}$ (MHz) & $a_{\rm iso}$ ($\mu$T)", rows, r"\small")


def dipolar():
    d = _j("dipolar_spin_density_v5.json")["methylamine_face_3.50"]
    rows = [["spin-population tensor", f"{d['D_MHz']:.0f}", f"{d['E_MHz']:.0f}",
             ", ".join(f"{v:.0f}" for v in d["principal_MHz"])],
            ["point dipole at spin centroids", f"{d['D_point_centroid_MHz']:.0f}", "0", "--"],
            ["point dipole at N5 and N", f"{d['D_point_N5N_MHz']:.0f}", "0", "--"]]
    _tab("dipolar.tex", "lrrl", r"model & $D$ (MHz) & $E$ (MHz) & principal values of $\mathsf T$ (MHz)", rows)


def driving_force():
    aq = _j("aqueous_potentials.json")
    lab = {"dimethylamine": r"Me$_2$NH$^{\bullet+}$/Me$_2$NH", "methylamine": r"MeNH$_2^{\bullet+}$/MeNH$_2$",
           "benzylamine": r"BnNH$_2^{\bullet+}$/BnNH$_2$", "lumiflavin": r"Fl/Fl$^{\bullet-}$ (exp.: FMN, pH 7)"}
    rows = [[lab[k], f"{v['E_dft_V']:+.2f}", f"{v['E_exp_V']:+.3f}" if v["E_exp_V"] is not None else "--",
             f"{v['error_V']:+.2f}" if "error_V" in v else "--"] for k, v in aq.items()]
    _tab("potentials.tex", "lrrr", r"couple & DFT (V vs NHE) & experiment & error (V)", rows)
    df = _j("driving_force.json")
    rows = []
    for eps, e in df.items():
        for am in ("methylamine", "benzylamine"):
            rows.append([eps.replace("eps", r"$\varepsilon=") + "$", am, f"{e[am]['separated_ions_eV']:.2f}"] +
                        [f"{e[am]['Delta_eV'][k]:.2f}" for k in ("r4.0", "r5.0", "r6.0")])
    _tab("delta_dft.tex", "llrrrr", r"continuum & amine & separated (eV) & $r=4$ \AA & 5 \AA & 6 \AA", rows)


def worst_case(name, out):
    d = _j(name)
    if not d:
        return
    rows = d["rows"]
    kS = sorted({r["k_S"] for r in rows})
    kP = sorted({r["k_P"] for r in rows})
    val = {(r["k_S"], r["k_P"]): r["max_mfe"] for r in rows}
    body = [[f"{np.log10(p):.1f}"] + [g2(val[(s, p)]) if (s, p) in val else "" for s in kS] for p in kP]
    _tab(out, "r" + "r" * len(kS), r"$\log k_P \backslash \log k_S$ & " + " & ".join(f"{np.log10(s):.1f}" for s in kS),
         body, r"\scriptsize\setlength{\tabcolsep}{2pt}")


def tensors():
    h = _j("lambda/hfc_tensors_def2-tzvp.json")
    rows = []
    for tag, lab in (("lumiflavin_q-1", r"lumiflavin$^{\bullet-}$"), ("benzylamine_q+1", r"benzylaminium$^{\bullet+}$")):
        sel = [x for x in h[tag] if x["element"] == "N"] + \
              sorted((x for x in h[tag] if x["element"] == "H"), key=lambda x: -abs(x["a_MHz"]))[:4]
        for x in sorted(sel, key=lambda x: -abs(x["a_MHz"])):
            ev = np.linalg.eigvalsh(np.array(x["A_dip_MHz"]))
            rows.append([lab, f"{x['element']}{x['index']}", f"{x['a_MHz']:+.1f}", ", ".join(f"{v:+.1f}" for v in ev)])
            lab = ""
    _tab("tensors.tex", "llrl", r"radical & nucleus & $a_{\rm iso}$ (MHz) & principal values of $\mathsf A_{\rm dip}$ (MHz)",
         rows, r"\small")


def sensitivity():
    d = _j("model_sensitivity.json")
    if not d:
        return
    rows = []
    for kp, b in sorted(d["base"].items(), key=lambda kv: float(kv[0])):
        r = [f"{np.log10(float(kp)):.1f}", f"{np.log10(b['k_S']):.1f}", g2(b["max_mfe"])]
        for name in ("aniso", "aniso+N", "aniso+H8", "T2e:coherent", "T2e:10.0", "T2e:100.0", "T2e:1000.0", "T2e:10000.0"):
            v = d["models"].get(name, {}).get(kp)
            r.append(g2(v) if v is not None else "--")
        rows.append(r)
    _tab("sensitivity.tex", "rrrrrrrrrrr",
         r"$\log k_P$ & $\log k_S$ & reference & aniso & +$^{14}$N & +2H & red.\ coh. & 10 ns & 100 ns & 1 $\mu$s & 10 $\mu$s",
         rows, r"\scriptsize\setlength{\tabcolsep}{3pt}")


def relaxation():
    d, m = _j("worst_case_v6_t2e.json"), _j("worst_case_v6.json")
    if not d:
        return
    coh = {(r["k_S"], r["k_P"]): r["max_mfe"] for r in m["rows"]}
    val = {(r["k_S"], r["k_P"], r["T2e_ns"]): r["max_mfe"] for r in d["rows"]}
    t2s = sorted({r["T2e_ns"] for r in d["rows"]})
    rows = []
    for kP in sorted({r["k_P"] for r in d["rows"]}):
        for kS in sorted({r["k_S"] for r in d["rows"] if r["k_P"] == kP}):
            rows.append([f"{np.log10(kP):.1f}", f"{np.log10(kS):.1f}", g2(coh[(kS, kP)])] +
                        [g2(val[(kS, kP, t)]) if (kS, kP, t) in val else "--" for t in t2s])
    _tab("relaxation.tex", "rr" + "r" * (1 + len(t2s)),
         r"$\log k_P$ & $\log k_S$ & coherent & " + " & ".join(f"{t:g} ns" for t in t2s), rows, r"\small")


def cryptochrome():
    c = _j("cry_control.json")
    rows = [[f"{r['J0_uT']:.2e}".replace("e+", r"\text{e}"), f"{r['J_gap_MHz']:+.2f}",
             f"{r['mfe_powder']:+.3f}", f"{r['mfe_perp']:+.3f}", f"{r['mfe_par']:+.3f}"] for r in c["coherent"]]
    _tab("cry.tex", "rrrrr", r"$J_0$ ($\mu$T) & $J$ (MHz) & powder (\%) & $\perp$ (\%) & $\parallel$ (\%)", rows, r"\small")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    coupling(); hyperfine(); tensors(); dipolar(); driving_force(); cryptochrome(); sensitivity(); relaxation()
    worst_case("worst_case_v6.json", "map.tex")
    print("SI tables →", OUT)
