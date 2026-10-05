#!/usr/bin/env python3
"""Supporting-information tables for manuscript/magnetobio_v4, from calibration_results."""

import json
from pathlib import Path

from qbscreen.make_numbers_tex import sci

R = Path("calibration_results")
OUT = Path("manuscript/magnetobio_v4/si_tables")


def _tab(path, cols, header, rows):
    body = ["\\begin{tabular}{" + cols + "}", "\\toprule", header + " \\\\", "\\midrule"]
    body += [" & ".join(r) + " \\\\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}"]
    (OUT / path).write_text("\n".join(body) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scan = json.loads((R / "ct_coupling_scan_b3lyp.json").read_text())
    _tab("scan.tex", "rrrrrl", "$r$ (\\AA) & twist ($^\\circ$) & tilt ($^\\circ$) & $t$ (meV) & closest contact (\\AA) & used",
         [[f"{r['r_N5N']:.2f}", f"{r['twist']}", f"{r['tilt']}", f"{r['t_meV']:.1f}", f"{r['min_contact']:.2f}",
           "yes" if r["min_contact"] >= 2.4 else "no (clash)"] for r in scan])

    lam = json.loads((R / "lambda" / "lambda_inner.json").read_text())
    rows = []
    for name in ("lumiflavin", "methylamine", "benzylamine", "alanine"):
        E = lam[name]["E_Ha"]
        rows.append([name] + [f"{E[k]:.6f}" for k in ("state0@geom0", "state0@geom1", "state1@geom1", "state1@geom0")]
                    + [f"{lam[name]['lambda_half_eV']:.3f}"])
    _tab("lambda.tex", "lrrrrr",
         "species & $E_0(G_0)$ & $E_0(G_1)$ & $E_1(G_1)$ & $E_1(G_0)$ & $\\lambda_i/2$ (eV)", rows)

    hfc = json.loads((R / "lambda" / "hfc_def2-tzvp.json").read_text())
    rows = []
    for tag, lab in (("lumiflavin_q-1", "flavin$^{\\bullet-}$"), ("benzylamine_q+1", "benzylaminium$^{\\bullet+}$"),
                     ("methylamine_q+1", "methylaminium$^{\\bullet+}$"), ("alanine_q+1", "alaninium$^{\\bullet+}$")):
        for r in sorted(hfc[tag], key=lambda d: -abs(d["a_MHz"]))[:6]:
            rows.append([lab, f"{r['element']}{r['index']}", f"{r['a_MHz']:+.1f}", f"{r['a_mT']:+.3f}"])
    _tab("hfc.tex", "llrr", "radical & nucleus & $a_{\\rm iso}$ (MHz) & $a_{\\rm iso}$ (mT)", rows)

    num = json.loads((R / "manuscript_numbers.json").read_text())
    _tab("competence.tex", "rrrrrr",
         "$t$ (meV) & $\\lambda$ (eV) & $k_{\\rm cat}$ (s$^{-1}$) & $\\Delta_{\\max}$ (eV) & $\\tau$ (s) & $|J|$ (MHz)",
         [[f"{r['t_meV']:.0f}", f"{r['lambda_eV']:.1f}", f"{r['k_cat']}", f"{r['delta_max_eV']:.2f}",
           f"${sci(r['tau_s'])}$", f"${sci(abs(r['J_gap_MHz']))}$"] for r in num["competence"]])

    for key, fn in (("MAO_locus_inverted", "locus_inverted.tex"), ("MAO_locus_activationless", "locus_actless.tex")):
        _tab(fn, "rrrrrrr",
             "$t$ (meV) & $|J|$ (MHz) & $\\tau$ (s) & coh.\\ 0$^\\circ$ & coh.\\ 90$^\\circ$ & $T_2^e$ 0$^\\circ$ & $T_2^e$ 90$^\\circ$",
             [[f"{r['t_meV']:g}", f"${sci(abs(r['J_gap_MHz']))}$", f"${sci(r['tau_s'])}$"]
              + [f"${sci(r[k])}$" for k in ("mfe_coh_th0", "mfe_coh_th90", "mfe_T2e_th0", "mfe_T2e_th90")]
              for r in num[key]["rows"]])

    ext = json.loads((R / "extras.json").read_text())
    _tab("partner.tex", "rrrrr", "partner $a$ (MHz) & $\\tau$ ($\\mu$s) & 0$^\\circ$ & 45$^\\circ$ & 90$^\\circ$",
         [[f"{r['partner_MHz']:g}", f"{r['tau_us']:g}"] + [f"${sci(r[k])}$" for k in ("mfe_th0", "mfe_th45", "mfe_th90")]
          for r in ext["partner_sensitivity"]["rows"]])

    dsd = json.loads((R / "dipolar_spin_density.json").read_text())
    _tab("dipolar.tex", "rrrrrr", "$r$ (\\AA) & $D_{\\rm point}$ (MHz) & centroid dist.\\ (nm) & $D_{\\rm spin}$ (MHz) & $E$ (MHz) & principal values (MHz)",
         [[f"{v['r_N5N_A']:.2f}", f"{v['D_point_N5N_MHz']:.0f}", f"{v['centroid_distance_nm']:.3f}", f"{v['D_MHz']:.0f}",
           f"{v['E_MHz']:.0f}", ", ".join(f"{x:.0f}" for x in v["principal_MHz"])] for v in dsd.values()])
    _tab("partner_spinD.tex", "lrrrrr", "enzyme & partner $a$ (MHz) & $\\tau$ ($\\mu$s) & 0$^\\circ$ & 45$^\\circ$ & 90$^\\circ$",
         [[lab, f"{r['partner_MHz']:g}", f"{r['tau_us']:g}"] + [f"${sci(r[k])}$" for k in ("mfe_th0", "mfe_th45", "mfe_th90")]
          for lab, key in (("MAO", "MAO_spinD"), ("DAO", "DAO_spinD")) for r in ext[key]["rows"]])

    _tab("cry.tex", "llrrrr", "model & $|J_0|$ ($\\mu$T) & $J$ (MHz) & 0$^\\circ$ & 90$^\\circ$ & powder",
         [[r["model"].replace("_", " "), f"${sci(r['J0_uT'], 0)}$", f"{r['J_gap_MHz']:+.2f}",
           f"{r['mfe_T2e_th0']:+.3f}", f"{r['mfe_T2e_th90']:+.3f}",
           f"{r['mfe_T2e_powder']:+.3f}" if "mfe_T2e_powder" in r else "--"] for r in num["CRY"]["rows"]])
    print("SI tables written to", OUT)


if __name__ == "__main__":
    main()
