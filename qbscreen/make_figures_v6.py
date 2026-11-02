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
    """F_free of the constrained-model search (J unconstrained, s ∈ [0.2, 1])."""
    rows = json.loads((R / "constrained_map.json").read_text())["rows"]
    kS = sorted({r["k_S"] for r in rows})
    kP = sorted({r["k_P"] for r in rows})
    Z = np.full((len(kP), len(kS)), np.nan)
    for r in rows:
        Z[kP.index(r["k_P"]), kS.index(r["k_S"])] = np.log10(max(max(r["F"]), 1e-14))
    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
    im = ax.pcolormesh(np.log10(kS), np.log10(kP), Z, shading="nearest", cmap="viridis", vmin=-8, vmax=1.5)
    cs = ax.contour(np.log10(kS), np.log10(kP), Z, levels=[-2, -1, 0], colors="w", linewidths=0.6)
    ax.clabel(cs, fmt=lambda v: f"{10 ** v:g} %", fontsize=6)
    ax.set_xlabel(r"$\log_{10}\,k_S$ ($\mu$s$^{-1}$)")
    ax.set_ylabel(r"$\log_{10}\,k_P$ ($\mu$s$^{-1}$)")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\,F_{\rm free}$ (%)")
    fig.savefig(OUT / "fig_map.pdf")


def _G(ax, small=False):
    cg = json.loads((R / "constrained_G.json").read_text())
    g_all = cg["G"]["wide"]
    m = macros()
    curves = [("1", "C0", "-", r"$k_{\rm cat}=1$ s$^{-1}$"), ("10", "C1", "-", r"10 s$^{-1}$"),
              ("100", "C2", "-", r"100 s$^{-1}$"), ("1000", "C3", "-", r"$10^3$ s$^{-1}$"),
              ("benzylamine", "k", "--", r"MAO B, benzylamine (298 K)"),
              ("phenylethylamine", "0.45", "--", r"MAO B, $\beta$-PEA (298 K)")]
    for kc, col, ls, lab in curves:
        if small and kc != "1":
            continue
        g = g_all[kc]["G"]
        d = np.array([x["delta_eV"] for x in g])
        G = np.array([np.nan if x["G_percent"] is None else max(x["G_percent"], 1e-12) for x in g])
        ax.semilogy(d, G, ls, color=col, lw=1, label=None if small else lab)
    ax.axvspan(KT * np.log(1e14 / 5), 2.0, color="0.88", lw=0)
    lo, hi = g_all["1"]["defined"]
    if lo is not None:
        ax.axvspan(0, lo, facecolor="none", hatch="////", edgecolor="0.6", lw=0)
        ax.axvspan(hi, 2.0, facecolor="none", hatch="////", edgecolor="0.6", lw=0)
    # scenarios of Sec. VII: (a) water, (b) substrate-shifted flavin, (c) low-dielectric site, ε = 2–10
    y = 3e-9
    ax.plot([float(m["DeltaCepsTwo"]), float(m["DeltaMAOsubLo"])], [y, y], color="k", lw=1)
    for key, lab in (("Two", "2"), ("Three", "3"), ("Four", "4"), ("Ten", "10")):
        x = float(m[f"DeltaCeps{key}"])
        ax.plot([x, x], [y / 2.5, y * 2.5], color="k", lw=0.6)
        if not small:
            ax.text(x, y / 4, lab, fontsize=5, ha="center", va="top")
    ax.plot([float(m["DeltaMAOsubLo"]), float(m["DeltaMAOsubHi"])], [y, y], color="k", lw=2, solid_capstyle="butt")
    ax.plot([float(m["DeltaMAOlo"]), float(m["DeltaMAOhi"])], [y, y], color="k", lw=4, solid_capstyle="butt")
    ax.text(float(m["DeltaCepsTwo"]), 1.2e-8, r"MAO SET pair (scenarios; ticks: $\varepsilon$)", fontsize=6)
    ax.axhline(0.1, color="k", lw=0.4, ls=":")
    ax.set_xlim(0, 1.6)
    ax.set_ylim(1e-10, 50)
    ax.set_xlabel(r"$\Delta$ (eV)")
    ax.set_ylabel(r"$G$ (%)")


def fig_G():
    fig, ax = plt.subplots(figsize=(3.3, 2.8), constrained_layout=True)
    _G(ax)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    fig.savefig(OUT / "fig_G.pdf")
    fig, ax = plt.subplots(figsize=(3.25, 1.75), constrained_layout=True)
    _G(ax, small=True)
    fig.savefig(OUT / "fig_toc.pdf")


if __name__ == "__main__":
    if (R / "constrained_map.json").exists():
        fig_map()
    if (R / "constrained_G.json").exists():
        fig_G()
