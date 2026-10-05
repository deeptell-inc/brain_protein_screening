#!/usr/bin/env python3
"""Figures for manuscript/magnetobio_v4, drawn only from calibration_results/*.json."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("manuscript")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import apply_style, SINGLE_COL, DOUBLE_COL

from qbscreen.rp_calibration import J_tau, EV_TO_MHZ

R = Path("calibration_results")
OUT = Path("manuscript/magnetobio_v4")
C = dict(mao="#c0392b", cry="#1f6fb4", grey="#606060", anchor="#2e7d32")


def fig_coupling():
    fit = json.loads((R / "ct_coupling_fit.json").read_text())
    num = json.loads((R / "manuscript_numbers.json").read_text())
    ext = json.loads((R / "extras.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7))

    # (a) t(r)
    marks = "osD^"
    for (k, v), mk in zip(fit["by_orientation"].items(), marks):
        r = np.array(v["r"]); t = np.array(v["t_meV"])
        ax[0].semilogy(r, t, mk, ms=3.5, color=C["mao"], mfc="none", lw=0)
        rr = np.linspace(2.9, 6.1, 50)
        ax[0].semilogy(rr, v["t0_meV"] * np.exp(-v["beta_per_A"] * rr / 2), "-", lw=0.6, color=C["mao"], alpha=0.6)
    for t in (ext["t_anchor_meV"], ext["t_anchor_W384_meV"]):
        ax[0].axhline(t, color=C["anchor"], ls="--", lw=0.8)
    ax[0].text(5.95, ext["t_anchor_meV"] * 1.15, "photolyase W382", ha="right", fontsize=6.5, color=C["anchor"])
    ax[0].text(5.95, ext["t_anchor_W384_meV"] * 1.15, "photolyase W384", ha="right", fontsize=6.5, color=C["anchor"])
    ax[0].axvline(3.5, color=C["grey"], lw=0.5, ls=":")
    ax[0].set_xlabel(r"N5$\cdots$N distance $r$ (Å)")
    ax[0].set_ylabel(r"coupling $t$ (meV)")
    ax[0].set_title("(a)", loc="left", fontweight="bold")

    # (b) the (τ, |J|) plane
    a = ax[1]
    tau = np.logspace(-14, -4, 200)
    for (d, l, lab) in ((1.78, 0.97, r"$\Delta$=1.78, $\lambda$=0.97 eV"), (1.0, 1.0, r"$\Delta$=$\lambda$=1.0 eV")):
        Jt = J_tau(d, l)                                   # (|J|/ħ)τ, dimensionless
        J_MHz = Jt / tau / (2 * np.pi) / 1e6
        a.loglog(tau, J_MHz, "-", lw=0.9, color=C["grey"], alpha=0.9)
        i = np.argmin(abs(tau - 3e-9))
        a.text(tau[i], J_MHz[i] * 1.6, lab, fontsize=6, color=C["grey"], rotation=-33)
    comp = num["competence"]
    a.loglog([r["tau_s"] for r in comp], [abs(r["J_gap_MHz"]) for r in comp], "o", ms=3,
             color=C["mao"], label="catalytically competent")
    for tau_s, t in ((70e-12, ext["t_anchor_meV"]), (120e-12, ext["t_anchor_W384_meV"])):
        dl = 1.78 if tau_s < 1e-10 else 1.88
        a.loglog(tau_s, 2 * (t * 1e-3) ** 2 / dl * EV_TO_MHZ, "s", ms=4, color=C["anchor"],
                 label="photolyase (measured $\\tau$)" if tau_s < 1e-10 else None)
    cry = [abs(r["J_gap_MHz"]) for r in num["CRY"]["rows"] if r["model"] == "spin_half"]
    a.loglog([1e-6] * len(cry), cry, "^", ms=4, color=C["cry"], label="cryptochrome")
    a.fill_between([1e-7, 1e-4], 1e-3, 30, color=C["cry"], alpha=0.08, lw=0)
    a.text(2e-7, 1e-2, "weak-field\nwindow", fontsize=6.5, color=C["cry"])
    a.set_xlim(1e-14, 1e-4); a.set_ylim(1e-3, 1e9)
    a.set_xlabel(r"lifetime $\tau$ (s)")
    a.set_ylabel(r"$|J|$ (MHz)")
    a.set_title("(b)", loc="left", fontweight="bold")
    a.legend(fontsize=6, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_coupling.pdf")


def fig_mfe():
    num = json.loads((R / "manuscript_numbers.json").read_text())
    ext = json.loads((R / "extras.json").read_text())
    fig, a = plt.subplots(figsize=(SINGLE_COL, 2.4))
    def span(vals):
        v = [abs(x) for x in vals if abs(x) > 0]
        return min(v), max(v)
    def mf(rows):
        return [v for r in rows for k, v in r.items() if k.startswith("mfe")]
    groups = [
        ("MAO\n(3.5 Å)", span(mf(ext["MAO_spinD"]["rows"])), span(mf(num["MAO_dipolar_cap"]["rows"])
                                                                  + mf(ext["partner_sensitivity"]["rows"])), C["mao"]),
        ("DAO\n(4.0 Å)", span(mf(ext["DAO_spinD"]["rows"])), span(mf(num["DAO_dipolar_cap"]["rows"])), C["mao"]),
        ("CRY\n(1.90 nm)", span([r["mfe_T2e_powder"] for r in num["CRY"]["rows"] if "mfe_T2e_powder" in r]
                              + [r["mfe_T2e_th90"] for r in num["CRY"]["rows"]]), None, C["cry"]),
    ]
    for i, (lab, (lo, hi), pd, col) in enumerate(groups):
        if pd:
            a.semilogy([i - 0.12, i - 0.12], pd, "-", lw=4, color=col, solid_capstyle="butt", alpha=0.3)
        a.semilogy([i + (0.08 if pd else 0), i + (0.08 if pd else 0)], [lo, hi], "-", lw=6, color=col,
                   solid_capstyle="butt", alpha=0.85)
    a.set_xticks(range(len(groups)), [g[0] for g in groups])
    a.set_ylabel(r"$|$MFE$|$ at 50 $\mu$T (%)")
    a.set_ylim(1e-9, 1e1)
    a.axhline(0.1, color=C["grey"], ls="--", lw=0.6)
    a.text(1.0, 0.13, "0.1 %", fontsize=6, ha="center", color=C["grey"])
    fig.tight_layout()
    fig.savefig(OUT / "fig_mfe.pdf")


def _le_sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return rf"$\leq {x / 10 ** e:.0f}\times10^{{{e}}}\,\%$"


def fig_toc():
    num = json.loads((R / "manuscript_numbers.json").read_text())
    from matplotlib.patches import Circle
    plt.rcParams["savefig.bbox"] = None
    plt.rcParams["savefig.pad_inches"] = 0.0
    CM = 1 / 2.54
    fig, ax = plt.subplots(figsize=(8.25 * CM, 4.45 * CM))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis("off")
    cap = max(abs(v) for r in num["MAO_dipolar_cap"]["rows"] for k, v in r.items() if k.startswith("mfe"))
    cp = [abs(r["mfe_T2e_powder"]) for r in num["CRY"]["rows"] if "mfe_T2e_powder" in r]
    for xc, sep, col, title, how, val in (
            (25, 6, C["mao"], "Dark enzyme (MAO)", "formed thermally", _le_sci(cap)),
            (75, 30, C["cry"], "Cryptochrome", "formed by light", rf"${min(cp):.1f}$–${max(cp):.1f}\,\%$")):
        ax.add_patch(Circle((xc - sep / 2, 28), 2.6, fc=col, ec="none"))
        ax.add_patch(Circle((xc + sep / 2, 28), 2.6, fc=col, ec="none", alpha=0.55))
        ax.text(xc, 47, title, ha="center", fontsize=8, fontweight="bold")
        ax.text(xc, 38, how, ha="center", fontsize=6.5, color=C["grey"], style="italic")
        ax.text(xc, 13, val, ha="center", fontsize=10, color=col, fontweight="bold")
    ax.plot([50, 50], [6, 44], lw=0.5, color="0.82")
    ax.text(50, 1.5, "geomagnetic-field effect on the singlet yield", ha="center", fontsize=6, color=C["grey"], style="italic")
    fig.savefig(OUT / "fig_toc.pdf", dpi=600)


if __name__ == "__main__":
    apply_style()
    fig_coupling()
    fig_mfe()
    fig_toc()
    print("figures written to", OUT)
