#!/usr/bin/env python3
"""Figures for manuscript/magnetobio_jcp from calibration_results/*.json."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qbscreen.make_numbers_v6 import macros

R = Path("calibration_results")
OUT = Path("manuscript/magnetobio_jcp")
KT = 8.617333262e-5 * 310.0
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig_map():
    rows = json.loads((R / "worst_case_v6.json").read_text())["rows"]
    kS = sorted({r["k_S"] for r in rows})
    kP = sorted({r["k_P"] for r in rows})
    Z = np.full((len(kP), len(kS)), np.nan)
    for r in rows:
        Z[kP.index(r["k_P"]), kS.index(r["k_S"])] = np.log10(max(r["max_mfe"], 1e-14))
    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
    im = ax.pcolormesh(np.log10(kS), np.log10(kP), Z, shading="nearest", cmap="viridis", vmin=-8, vmax=1.5)
    cs = ax.contour(np.log10(kS), np.log10(kP), Z, levels=[-2, -1, 0], colors="w", linewidths=0.6)
    ax.clabel(cs, fmt=lambda v: f"{10 ** v:g} %", fontsize=6)
    ax.set_xlabel(r"$\log_{10}\,k_S$ ($\mu$s$^{-1}$)")
    ax.set_ylabel(r"$\log_{10}\,k_P$ ($\mu$s$^{-1}$)")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\,F$ (%)")
    fig.savefig(OUT / "fig_map.pdf")


def _G(ax, small=False):
    g_all = json.loads((R / "competence_sensitivity.json").read_text())["worst_case_v6.json"]["k_cat"]
    m = macros()
    colours = {"1.0": "C0", "10.0": "C1", "100.0": "C2", "1000.0": "C3"}
    for kc, g in g_all.items():
        if small and kc != "1.0":
            continue
        d = np.array([x["delta_eV"] for x in g])
        G = np.array([np.nan if x["G_percent"] is None else max(x["G_percent"], 1e-12) for x in g])
        ax.semilogy(d, G, "-", color=colours[kc], lw=1,
                    label=None if small else rf"$k_{{\rm cat}}={float(kc):g}$ s$^{{-1}}$")
    ax.axvspan(KT * np.log(1e14 / 4), 2.0, color="0.88", lw=0)
    ax.axvspan(0, float(m["GdefOne"]), facecolor="none", hatch="////", edgecolor="0.6", lw=0)
    ax.axvspan(float(m["GdefOneHi"]), 2.0, facecolor="none", hatch="////", edgecolor="0.6", lw=0)
    # thick: in water; thin: down to the most favourable medium correction
    ax.plot([float(m["DeltaSiteLo"]), float(m["DeltaMAOsubLo"])], [3e-9, 3e-9], color="k", lw=1)
    ax.plot([float(m["DeltaMAOsubLo"]), float(m["DeltaMAOhi"])], [3e-9, 3e-9], color="k", lw=3, solid_capstyle="butt")
    ax.text(float(m["DeltaSiteLo"]), 1.2e-8, "MAO (est.)", fontsize=7)
    ax.axhline(0.1, color="k", lw=0.4, ls=":")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(1e-10, 50)
    ax.set_xlabel(r"$\Delta$ (eV)")
    ax.set_ylabel(r"$G$ (%)")


def fig_G():
    fig, ax = plt.subplots(figsize=(3.3, 2.6), constrained_layout=True)
    _G(ax)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.savefig(OUT / "fig_G.pdf")
    fig, ax = plt.subplots(figsize=(3.25, 1.75), constrained_layout=True)
    _G(ax, small=True)
    fig.savefig(OUT / "fig_toc.pdf")


if __name__ == "__main__":
    if (R / "worst_case_v6.json").exists():
        fig_map()
    if (R / "competence_sensitivity.json").exists():
        fig_G()
