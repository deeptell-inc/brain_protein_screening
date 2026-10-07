#!/usr/bin/env python3
"""Every number quoted in manuscript/magnetobio_jcp/main.tex, as LaTeX macros.

Reads calibration_results/*.json and the measured potentials below; writes
manuscript/magnetobio_jcp/numbers.tex. A macro whose input is missing is set to
\\textbf{??} so an incomplete build is visible.
"""

import json
from pathlib import Path

import numpy as np

from qbscreen.competence_sensitivity import G, non_increasing, threshold

R = Path("calibration_results")
OUT = Path("manuscript/magnetobio_jcp/numbers.tex")
KT = 8.617333262e-5 * 310.0
COULOMB_EV_A = 14.399645
# measured one-electron potentials, V vs NHE
E_AMINE = 1.30            # dialkylamine radical cations, water (Jonsson et al. 1996)
AMINE_ERR = 0.05          # its stated uncertainty (ibid.)
E_MAO = {"A": -0.159, "B": -0.167}   # ox/sq, covalent FAD, no substrate (Sablin & Ramsay 2001)
E_MAO_SQRED = {"A": -0.262, "B": -0.275}   # sq/red, same source
SUB_SHIFT = 0.5           # largest substrate-induced shift of the full-reduction midpoint (ibid.)


def _j(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=2):
    return f"{x:.{d}f}"


def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def pct(x):
    return sci(x, 1) if abs(x) < 1e-3 else f"{x:.2g}"


def medium_correction():
    """Change of Δ when the contact pair moves from water to a medium of effective
    dielectric ε (4 or 10), at r = 3.5 or 6 Å.

    one-sided: Born desolvation of the aminium only, radius from the measured
    solvation-energy difference of dimethylamine (244 kJ/mol, Jonsson et al. 1996,
    p. 17541), plus the stronger Coulomb attraction; the flavin anion is left solvated,
    so this lowers Δ as much as the continuum picture allows.
    two-ion: both ions moved with the same DFT/PCM protocol (driving_force.json at
    ε = 4, 10; aqueous_potentials.json at ε = 78.4), plus the Coulomb change."""
    ddg = 244e3 / 96485.33212                                       # eV
    a_am = COULOMB_EV_A * (1 - 1 / 78.4) / (2 * ddg)                 # Å
    coul = lambda eps, r: -COULOMB_EV_A / r * (1 / eps - 1 / 78.4)
    one = {(eps, r): COULOMB_EV_A / (2 * a_am) * (1 / eps - 1 / 78.4) + coul(eps, r)
           for eps in (4.0, 10.0) for r in (3.5, 6.0)}
    two = {}
    df, aq = _j("driving_force.json"), _j("aqueous_potentials.json")
    if df and aq:
        for am in ("methylamine", "benzylamine"):
            sep_w = aq[am]["E_dft_V"] - aq["lumiflavin"]["E_dft_V"]
            for eps in (4.0, 10.0):
                sep_e = df[f"eps{eps:g}"][am]["separated_ions_eV"]
                for r in (3.5, 6.0):
                    two[(am, eps, r)] = sep_e - sep_w + coul(eps, r)
    return a_am, one, two


def macros():
    m = dict(DeltaMaxLo=f(KT * np.log(1e13 / (4 * 1e3))), DeltaMaxHi=f(KT * np.log(1e14 / 4.0)),
             DeltaMaxPLo=f(KT * np.log(1e13 / 1e3)), DeltaMaxPHi=f(KT * np.log(1e14 / 1.0)), kTln=f(KT * np.log(4), 3),
             sMax="3", EamExp=f(E_AMINE), EflMAOA=f"{1000 * E_MAO['A']:.0f}", EflMAOB=f"{1000 * E_MAO['B']:.0f}",
             SubShift=f(SUB_SHIFT, 1))
    w = {r: COULOMB_EV_A / (78.4 * r) for r in (3.5, 6.0)}          # contact Coulomb term in water
    lo = E_AMINE - max(E_MAO.values()) - w[3.5]
    hi = E_AMINE - min(E_MAO.values()) - w[6.0]
    sub_lo = E_AMINE - (max(E_MAO.values()) + SUB_SHIFT) - w[3.5]
    sub_hi = E_AMINE - (min(E_MAO.values()) + SUB_SHIFT) - w[6.0]
    # equilibrium reading of 'no semiquinone with substrate': E1 ≤ E_m(sub) ≤ E_m(free) + 0.5 V (MAO A)
    em_free_A = (E_MAO["A"] + E_MAO_SQRED["A"]) / 2
    e1_cap = em_free_A + SUB_SHIFT
    m.update(DeltaMAOlo=f(lo), DeltaMAOhi=f(hi), DeltaMAOsub=f(sub_lo), CoulLo=f(w[6.0]), CoulHi=f(w[3.5]),
             AmineErr=f(AMINE_ERR), DeltaMAOsubLo=f(sub_lo - AMINE_ERR), DeltaMAOsubHi=f(sub_hi + AMINE_ERR),
             gridStep=f(KT * np.log(10 ** 0.5), 2), EmFreeA=f"{1000 * em_free_A:.0f}", EoneCap=f(e1_cap),
             EoneShiftCap=f(e1_cap - E_MAO["A"]), EsqredA=f"{1000 * E_MAO_SQRED['A']:.0f}",
             EsqredB=f"{1000 * E_MAO_SQRED['B']:.0f}")
    a_am, one, two = medium_correction()
    m.update(BornRadius=f(a_am), MedOneLo=f(min(one.values())), MedOneHi=f(max(one.values())))
    if two:
        m.update(MedTwoLo=f(min(two.values())), MedTwoHi=f(max(two.values())),
                 DeltaSiteLo=f(sub_lo - AMINE_ERR + min(one.values())),
                 DeltaSiteHi=f(hi + max(two.values())))
    oc = _j("g_offgrid_check.json")
    if oc:
        m.update(offDelta=f(oc["delta_eV"]), offKP=f"{oc['k_P']:.1f}", offMax=f"{oc['max_offgrid_percent']:.4f}",
                 offG=f"{oc['G_percent']:.4f}", offKSlo=f"{oc['points'][0]['k_S']:.0f}",
                 offKShi=f"{oc['points'][-1]['k_S']:.0f}")

    sh = _j("phi_bound_sharpness.json")
    if sh:
        m["sharpRatio"] = f"{max(r['ratio'] for r in sh['rows']):.6f}"

    bt = _j("phi_bound_test.json")
    if bt:
        m.update(nBoundTest=str(bt["n"]), maxBoundRatio=f"{bt['max_ratio']:.4f}")

    wc = _j("worst_case_v6.json")
    if wc:
        rows = wc["rows"]
        env = {kp: max(r["max_mfe"] for r in rows if r["k_P"] == kp) for kp in sorted({r["k_P"] for r in rows})}
        kps = sorted(env)
        m.update(FkPlow=pct(env[kps[0]]), FkPhundred=pct(env[min(kps, key=lambda k: abs(np.log10(k) - 2))]),
                 FkPtop=pct(env[kps[-1]]), FkPthousand=pct(env[min(kps, key=lambda k: abs(np.log10(k) - 3))]), nCells=str(len(rows)), nEdge=str(sum(r["at_grid_edge"] for r in rows)),
                 nSuppRaised=str(sum(r.get("source") == "supplement" for r in rows)))
        p1 = _j("worst_case_v6_pass1.json")
        if p1:
            first = {(r["k_S"], r["k_P"]): r["max_mfe"] for r in p1["rows"]}
            m["polishGain"] = f"{max(100 * (r['max_mfe'] / r['grid_max'] - 1) for r in p1['rows']):.0f}"
            m["suppMaxRatio"] = f"{max(r['max_mfe'] / first[(r['k_S'], r['k_P'])] for r in rows):.1f}"
            rm = lambda rr, kp: max(v for (s_, p_), v in rr if p_ == kp)
            m["rowMaxChange"] = f"{max(100 * (rm([((r['k_S'], r['k_P']), r['max_mfe']) for r in rows], kp) / rm(list(first.items()), kp) - 1) for kp in kps):.1f}"
        conv = _j("s_convergence.json")
        if conv:
            m["convMaxRatio"] = f"{max(c['dense'] / c['stored'] for c in conv):.3f}"
        cs = _j("competence_sensitivity.json")
        if cs and "worst_case_v6.json" in cs:
            g1 = cs["worst_case_v6.json"]["k_cat"]["1.0"]
            g3 = cs["worst_case_v6.json"]["k_cat"]["1000.0"]
            m.update(GdefOne=f(min(x["delta_eV"] for x in g1 if x["G_percent"] is not None)),
                     GdefOneHi=f(max(x["delta_eV"] for x in g1 if x["G_percent"] is not None)),
                     ThrTenthOne=f(threshold(g1, 0.1)), ThrHundredthOne=f(threshold(g1, 0.01)),
                     ThrTenthThousand=f(threshold(g3, 0.1)))
        if cs and "with_relaxation" in cs:
            gr = cs["with_relaxation"]
            m.update(ThrTenthOneRelax=f(threshold(gr["1.0"], 0.1)), ThrHundredthOneRelax=f(threshold(gr["1.0"], 0.01)),
                     ThrTenthThousandRelax=f(threshold(gr["1000.0"], 0.1)))
        an = _j("worst_case_v6_aniso.json")
        if cs and "with_aniso" in cs and an:
            ga = cs["with_aniso"]
            ref = {(r["k_S"], r["k_P"]): r["max_mfe"] for r in rows}
            key = lambda r: min(ref, key=lambda k: abs(np.log(k[0] / r["k_S"])) + abs(np.log(k[1] / r["k_P"])))
            rr = lambda rr_, kp: max(v for (s_, p_), v in rr_ if np.isclose(p_, kp))
            akp = sorted({r["k_P"] for r in an["rows"]})
            arow = [rr([((r["k_S"], r["k_P"]), r["max_mfe"]) for r in an["rows"]], kp) / rr(ref.items(), kp) for kp in akp]
            at = lambda g, d: next(x["G_percent"] for x in g if np.isclose(x["delta_eV"], d))
            if "DeltaSiteLo" in m:
                m.update(GsiteRef=f"{at(cs['worst_case_v6.json']['k_cat']['1.0'], float(m['DeltaSiteLo'])):.2f}",
                         GsiteAniso=f"{at(ga['1.0'], float(m['DeltaSiteLo'])):.2f}")
            m.update(ThrTenthOneAniso=f(threshold(ga["1.0"], 0.1)), ThrTenthThousandAniso=f(threshold(ga["1000.0"], 0.1)),
                     nAnisoCells=str(len(an["rows"])),
                     AnisoCellMax=f"{max(r['max_mfe'] / ref[key(r)] for r in an['rows']):.1f}",
                     AnisoRowMin=f"{min(arow):.2f}", AnisoRowMax=f"{max(arow):.2f}",
                     nAnisoEdge=str(sum(r["at_grid_edge"] for r in an["rows"])))
        t2 = _j("worst_case_v6_t2e.json")
        if t2:
            coh = {(r["k_S"], r["k_P"]): r["max_mfe"] for r in rows}
            ratios = [r["max_mfe"] / coh[(r["k_S"], r["k_P"])] for r in t2["rows"]]
            rowmax = lambda rr, kp: max(r["max_mfe"] for r in rr if np.isclose(r["k_P"], kp))
            rowratio = [max(rowmax([r for r in t2["rows"] if r["T2e_ns"] == t], kp) for t in {r["T2e_ns"] for r in t2["rows"]})
                        / rowmax(rows, kp) for kp in sorted({r["k_P"] for r in t2["rows"]})]
            m.update(nTtwoCells=str(len(t2["rows"])), TtwoCellMax=f"{max(ratios):.2f}", TtwoCellMin=f"{min(ratios):.2f}",
                     TtwoRowMax=f"{max(rowratio):.2f}", TtwoRowMin=f"{min(rowratio):.2f}")
        ms = _j("model_sensitivity.json")
        if ms:
            base = {float(k): v["max_mfe"] for k, v in ms["base"].items()}
            ratio = lambda name: [v / base[float(k)] for k, v in ms["models"].get(name, {}).items()]
            for name, key in (("aniso", "XanisoMax"), ("aniso+N", "XanisoNMax"), ("aniso+H8", "XanisoHMax")):
                if ratio(name):
                    m[key] = f"{max(ratio(name)):.1f}"
            coh = {float(k): v for k, v in ms["models"].get("T2e:coherent", {}).items()}
            t2 = [v / coh[float(k)] for name, d in ms["models"].items() if name.startswith("T2e:") and name != "T2e:coherent"
                  for k, v in d.items() if float(k) in coh]
            if t2:
                m.update(XTtwoMin=f"{min(t2):.2f}", XTtwoMax=f"{max(t2):.2f}")
            # stress test from the nuclear models; relaxation is computed directly (worst_case_v6_t2e.json)
            X = max([1.0] + [x for n in ("aniso", "aniso+N", "aniso+H8") for x in ratio(n)])
            kc = 1.0
            g = non_increasing([dict(delta_eV=d, G_percent=G(rows, d, kc, factor={kp: X for kp in kps}))
                                for d in np.linspace(0, 2, 201)])
            m["ThrTenthOneSens"] = f(threshold(g, 0.1))
            m["Xsens"] = f"{X:.1f}"
            if "ThrTenthOneSens" in m and "DeltaMAOsub" in m:
                m["DeltaToThr"] = f(float(m["DeltaMAOsub"]) - float(m["ThrTenthOneSens"]))

    dsd = _j("dipolar_spin_density_v5.json")
    if dsd:
        d = dsd["methylamine_face_3.50"]
        m.update(Dcontact=f"{d['D_MHz']:.0f}", Econtact=f"{d['E_MHz']:.0f}")
    scan = _j("ct_coupling_scan_v5.json")
    if scan:
        t35 = [abs(r["t_meV"]) for r in scan if r["amine"] == "methylamine" and r["approach"] == "face"
               and r["min_contact"] >= 2.0 and r["r_N5N"] == 3.5]
        m.update(tContactMed=f"{np.median(t35):.0f}", tContactMax=f"{max(t35):.0f}", nScan=str(len(scan)),
                 nScanValid=str(sum(r["min_contact"] >= 2.0 for r in scan)))
    cry = _j("cry_control.json")
    if cry:
        p = [abs(r["mfe_powder"]) for r in cry["coherent"]]
        m.update(cryMax=pct(max(p)), cryDcoup=f(cry["D_MHz"], 1), nCry=str(len({r["J0_uT"] for r in cry["coherent"]})))
    hf = _j("lambda/hfc_def2-tzvp.json")
    if hf:
        big = lambda tag: sorted((abs(x["a_MHz"]) for x in hf[tag]), reverse=True)
        m.update(aBenzOne=f"{big('benzylamine_q+1')[0]:.0f}", aBenzTwo=f"{big('benzylamine_q+1')[1]:.0f}",
                 aTrpOne=f"{big('methylindole_q+1')[0]:.0f}", aTrpTwo=f"{big('methylindole_q+1')[1]:.0f}")
    return m


EXPECTED = ("GsiteRef GsiteAniso GdefOne GdefOneHi sharpRatio ThrTenthOneAniso ThrTenthThousandAniso nAnisoCells AnisoCellMax AnisoRowMin AnisoRowMax nAnisoEdge ThrTenthOne ThrHundredthOne ThrTenthThousand ThrTenthOneSens DeltaToThr XanisoMax XanisoNMax "
            "XanisoHMax XTtwoMin XTtwoMax Xsens FkPlow FkPhundred FkPtop FkPthousand nCells nEdge polishGain nBoundTest "
            "nSuppRaised suppMaxRatio rowMaxChange convMaxRatio ThrTenthOneRelax ThrHundredthOneRelax "
            "ThrTenthThousandRelax nTtwoCells TtwoCellMax TtwoCellMin TtwoRowMax TtwoRowMin "
            "maxBoundRatio").split()


def main():
    m = macros()
    for k in EXPECTED:
        m.setdefault(k, r"\textbf{??}")
    missing = [k for k, v in m.items() if "??" in v]
    if missing:
        print("placeholders:", " ".join(missing))
    OUT.write_text("% generated by qbscreen/make_numbers_v6.py — do not edit\n" +
                   "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items())))
    print(f"{len(m)} macros → {OUT}")


if __name__ == "__main__":
    main()
