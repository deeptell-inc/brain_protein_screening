#!/usr/bin/env python3
"""Write manuscript/magnetobio_v4/numbers.tex from the calibration JSON files.

Every number quoted in the manuscript prose is a macro defined here, so the text
cannot drift from the computed results. Re-run after any calibration change.
"""

import json
from pathlib import Path

import numpy as np

R = Path("calibration_results")
OUT = Path("manuscript/magnetobio_v4/numbers.tex")


def sci(x, digits=1):
    """LaTeX scientific notation, e.g. 2.1\\times10^{-6}."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(abs(m), digits) >= 10:
        m, e = m / 10, e + 1
    return f"\\ensuremath{{{m:.{digits}f}\\times10^{{{e}}}}}"


def render():
    """Return (numbers.tex text, table_competence.tex text)."""
    fit = json.loads((R / "ct_coupling_fit.json").read_text())
    lam = json.loads((R / "lambda" / "lambda_inner.json").read_text())
    hfc = json.loads((R / "lambda" / "hfc_def2-tzvp.json").read_text())
    num = json.loads((R / "manuscript_numbers.json").read_text())
    ext = json.loads((R / "extras.json").read_text())
    psens = [abs(v) for r in ext["partner_sensitivity"]["rows"] for k, v in r.items() if k.startswith("mfe")]
    sp = {}
    for r in num["CRY"]["rows"]:
        sp.setdefault((r["J0_uT"], r["J_EH_MHz"] > 0), {})[r["model"]] = r["mfe_T2e_th90"]
    spin_ratio = max(max(abs(v["spin_one_N"]), abs(v["spin_half"])) / min(abs(v["spin_one_N"]), abs(v["spin_half"]))
                     for v in sp.values() if min(abs(v["spin_one_N"]), abs(v["spin_half"])) > 1e-3)

    from qbscreen.ct_coupling import read_xyz, flavin_n5
    betas = [v["beta_per_A"] for v in fit["by_orientation"].values()]
    t35 = fit["t_at_3p5A_meV"]
    fl = {r["index"]: r for r in hfc["lumiflavin_q-1"]}
    s, x = read_xyz(R / "lambda" / "opt_lumiflavin_q-1" / "xtbopt.xyz")
    n5 = flavin_n5(s, x)
    n10 = next(i for i, e in enumerate(s) if e == "N" and sum(
        1 for j, f in enumerate(s) if f == "C" and 0.1 < np.linalg.norm(x[j] - x[i]) < 1.75) == 3)
    comp = num["competence"]
    cap_mao = max(abs(v) for row in num["MAO_dipolar_cap"]["rows"] for k, v in row.items() if k.startswith("mfe"))
    cap_dao = max(abs(v) for row in num["DAO_dipolar_cap"]["rows"] for k, v in row.items() if k.startswith("mfe"))
    cry = [r for r in num["CRY"]["rows"]]
    cry90 = [abs(r["mfe_T2e_th90"]) for r in cry]
    cry_pow = [abs(r["mfe_T2e_powder"]) for r in cry if "mfe_T2e_powder" in r]
    tau_c = [r["tau_s"] for r in comp]
    J_c = [abs(r["J_gap_MHz"]) for r in comp]

    macros = {
        # electronic coupling
        "tContactMin": f"{t35['min']:.0f}", "tContactMax": f"{t35['max']:.0f}",
        "tContactGeo": f"{t35['geomean']:.0f}",
        "betaMin": f"{min(betas):.1f}", "betaMax": f"{max(betas):.1f}",
        "nScanPoints": f"{fit['n_points']}",
        # reorganisation
        "lamFlHalf": f"{lam['lumiflavin']['lambda_half_eV']:.2f}",
        "lamMethyl": f"{lam['lambda_i_lumiflavin+methylamine_eV']:.2f}",
        "lamBenzyl": f"{lam['lambda_i_lumiflavin+benzylamine_eV']:.2f}",
        "lamAla": f"{lam['lambda_i_lumiflavin+alanine_eV']:.2f}",
        # hyperfine validation against Lee et al. (2014): N5 523 µT, N10 189 µT
        "aNfiveDFT": f"{fl[n5]['a_mT'] * 1000:.0f}", "aNtenDFT": f"{fl[n10]['a_mT'] * 1000:.0f}",
        "partnerMAO": f"{num['partner_hfc_MHz']['MAO'][0]:.0f}",
        "partnerDAO": f"{num['partner_hfc_MHz']['DAO'][0]:.0f}",
        # catalytic competence
        "DeltaMaxMin": f"{min(r['delta_max_eV'] for r in comp):.2f}",
        "DeltaMaxMax": f"{max(r['delta_max_eV'] for r in comp):.2f}",
        "tauCompMax": sci(max(tau_c)), "tauCompMin": sci(min(tau_c)),
        "JCompMin": sci(min(J_c)), "JCompMax": sci(max(J_c)),
        # J–τ link
        "JtauInverted": f"{num['MAO_locus_inverted']['J_tau_nonadiabatic']:.0f}",
        "JtauActless": f"{num['MAO_locus_activationless']['J_tau_nonadiabatic']:.2f}",
        # dipolar caps
        "DMAO": f"{num['MAO_dipolar_cap']['D_MHz']:.0f}", "DDAO": f"{num['DAO_dipolar_cap']['D_MHz']:.0f}",
        "capMAO": sci(cap_mao), "capDAO": sci(cap_dao),
        # photolyase anchor and partner sensitivity
        "tAnchor": f"{ext['t_anchor_meV']:.0f}", "tAnchorWb": f"{ext['t_anchor_W384_meV']:.0f}",
        "partnerCapMin": sci(min(psens)), "partnerCapMax": sci(max(psens)),
        "spinOneRatio": f"{spin_ratio:.1f}",
        # dipolar coupling summed over DFT spin distributions, and the bound it implies
        "DMAOspin": f"{ext['MAO_spinD']['D_MHz']:.0f}", "DDAOspin": f"{ext['DAO_spinD']['D_MHz']:.0f}",
        "capMAOspin": sci(max(abs(v) for r in ext["MAO_spinD"]["rows"] for k, v in r.items() if k.startswith("mfe"))),
        "capDAOspin": sci(max(abs(v) for r in ext["DAO_spinD"]["rows"] for k, v in r.items() if k.startswith("mfe"))),
        # cryptochrome
        "DCRY": f"{num['CRY']['D_MHz']:.1f}",
        "cryNinetyMin": f"{min(cry90):.2f}", "cryNinetyMax": f"{max(cry90):.2f}",
        "cryPowderMin": f"{min(cry_pow):.2f}", "cryPowderMax": f"{max(cry_pow):.2f}",
    }
    rows = ["\\begin{tabular}{rrrccc}", "\\toprule",
            "$t$ (meV) & $\\lambda$ (eV) & $k_{\\rm cat}$ (s$^{-1}$) & $\\Delta_{\\max}$ (eV) & $\\tau$ (s) & $|J|$ (MHz) \\\\",
            "\\midrule"]
    for r in comp:
        if r["k_cat"] != 10:
            continue
        rows.append(f"{r['t_meV']:.0f} & {r['lambda_eV']:.1f} & {r['k_cat']} & {r['delta_max_eV']:.2f} & "
                    f"${sci(r['tau_s'])}$ & ${sci(abs(r['J_gap_MHz']))}$ \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(rows) + "\n"

    lines = ["% generated by qbscreen/make_numbers_tex.py — do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    return "\n".join(lines) + "\n", table


def main():
    text, table = render()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    OUT.with_name("table_competence.tex").write_text(table)
    print(text)


if __name__ == "__main__":
    main()
