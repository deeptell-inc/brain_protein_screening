#!/usr/bin/env python3
"""Fig 5 v5: 3-layer diagram — no overlap, larger canvas, generous spacing."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D
from pathlib import Path

from figure_style import apply_style
apply_style()
# Override font size for schematic (needs larger text)
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10

OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def spin_arr(ax, x, y, up=True, color="blue", sz=0.22):
    dy = sz if up else -sz
    ax.annotate("", xy=(x, y + dy), xytext=(x, y - dy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))


def main():
    W, H = 30, 32
    fig, ax = plt.subplots(figsize=(W * 0.75, H * 0.75))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ── Title ──
    ax.text(15, 31.3, "Quantum-Classical Information Flow in the Brain",
            ha="center", fontsize=22, fontweight="bold")
    ax.text(15, 30.5, "Molecular carriers, spin states, and information transfer",
            ha="center", fontsize=14, color="#555")

    # ==========================================================
    # LAYER 1   y: 24 -- 30
    # ==========================================================
    L1 = FancyBboxPatch((0.5, 24.0), 28.5, 6.0, boxstyle="round,pad=0.3",
                         fc="#FFF8E1", ec="#E65100", lw=3)
    ax.add_patch(L1)
    ax.text(15, 29.5, "LAYER 1 :  Nuclear Spin Quantum Memory",
            ha="center", fontsize=19, fontweight="bold", color="#E65100")

    # -- FAD --
    ax.text(4.5, 28.5, "FAD ribose-phosphate", fontsize=14, fontweight="bold",
            color="#333", ha="center")

    for i, (px, lab) in enumerate([(3.0, r"$^{31}$P$_1$"), (6.0, r"$^{31}$P$_2$")]):
        c = Circle((px, 27.3), 0.4, fc="#FF6D00", ec="black", lw=2, zorder=5)
        ax.add_patch(c)
        ax.text(px, 27.3, "P", ha="center", va="center", fontsize=13,
                fontweight="bold", color="white", zorder=6)
        spin_arr(ax, px, 27.3, up=(i == 0), color="#1565C0")
        ax.text(px, 26.5, lab, ha="center", fontsize=12, color="#555")

    ax.text(4.5, 25.5,
            "$I = 1/2$,   100 %\n"
            "$A = 200$ MHz\n"
            "$T_2 = 3.2$ ms  (diamag.)\n"
            "$T_2 = 160$  $\\mu$s  (PRE)",
            ha="center", fontsize=12, color="#666", linespacing=1.6)

    # -- PLP --
    ax.text(12, 28.5, "PLP  pyridine-P", fontsize=14, fontweight="bold",
            color="#333", ha="center")

    c = Circle((12, 27.3), 0.4, fc="#FF6D00", ec="black", lw=2, zorder=5)
    ax.add_patch(c)
    ax.text(12, 27.3, "P", ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", zorder=6)
    spin_arr(ax, 12, 27.3, up=True, color="#1565C0")
    ax.text(12, 26.5, r"$^{31}$P  (direct C5 bond)", ha="center", fontsize=12, color="#555")

    ax.text(12, 25.2,
            "$r = 3.5$  \u00c5\n"
            "$T_2 = 2.5$  $\\mu$s  (PRE)\n"
            "$\\times$64  shorter",
            ha="center", fontsize=12, color="#C62828", linespacing=1.6,
            bbox=dict(fc="#FFF8E1", ec="none", pad=1))

    # -- NADPH --
    ax.text(18, 28.5, "NADPH", fontsize=14, fontweight="bold",
            color="#333", ha="center")

    for px in [16.8, 18.0, 19.2]:
        c = Circle((px, 27.3), 0.32, fc="#FF6D00", ec="black", lw=1.5, zorder=5)
        ax.add_patch(c)
        ax.text(px, 27.3, "P", ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=6)

    ax.text(18, 26.5, r"3$\times$  $^{31}$P  ($n_P$ max)", ha="center",
            fontsize=12, color="#555")

    # -- Quantum info box --
    qi = FancyBboxPatch((22, 25.0), 6.5, 4.5, boxstyle="round,pad=0.2",
                         fc="white", ec="#E65100", lw=1.5, ls="--")
    ax.add_patch(qi)
    ax.text(25.25, 29.0, "Quantum information", fontsize=14,
            fontweight="bold", color="#E65100", ha="center")

    ax.text(22.6, 28.0,
            r"$|m_I\!=\!+\frac{1}{2}\rangle$  ,  "
            r"$|m_I\!=\!-\frac{1}{2}\rangle$",
            fontsize=13, color="#1565C0")

    ax.text(22.6, 27.0,
            r"Singlet :  $(|\!\uparrow\downarrow\rangle"
            r" - |\!\downarrow\uparrow\rangle)/\sqrt{2}$",
            fontsize=13, color="#7B1FA2")

    ax.text(22.6, 26.1,
            r"Coherence :  160 $\mu$s  --  3.2 ms",
            fontsize=13, color="#333")

    ax.text(22.6, 25.3,
            r"CIDNP :  $\varepsilon \sim 10^5$",
            fontsize=13, color="#BF360C")

    # ==========================================================
    # INTERFACE 1-2     y: 21.5 -- 23.5
    # ==========================================================
    ax.annotate("", xy=(8, 23.5), xytext=(8, 24.0),
                arrowprops=dict(arrowstyle="<->", color="#E65100", lw=5))

    hb = FancyBboxPatch((2, 21.8), 20, 1.6, boxstyle="round,pad=0.15",
                         fc="#FFF3E0", ec="#E65100", lw=2.5)
    ax.add_patch(hb)
    ax.text(12, 22.9,
            r"HYPERFINE COUPLING       $A = 200$ MHz  $= 0.83$  $\mu$eV",
            ha="center", fontsize=16, fontweight="bold", color="#E65100")
    ax.text(12, 22.1,
            r"$^{31}$P nuclear spin   $\longleftrightarrow$   "
            "electron spin precession       |       quantum channel",
            ha="center", fontsize=11, color="#BF360C")

    ax.text(25.5, 22.5,
            r"$^{31}$P state" + "\ndetermines\n" + "$S$-$T$ rate",
            fontsize=13, color="#E65100", ha="center",
            bbox=dict(fc="lightyellow", alpha=0.9, boxstyle="round,pad=0.4"))

    # ==========================================================
    # LAYER 2   y: 8.5 -- 21.0
    # ==========================================================
    L2 = FancyBboxPatch((0.5, 8.5), 28.5, 12.5, boxstyle="round,pad=0.3",
                         fc="#F3E5F5", ec="#7B1FA2", lw=3)
    ax.add_patch(L2)
    ax.text(15, 20.5, "LAYER 2 :  Radical Pair Quantum-Classical Interface",
            ha="center", fontsize=19, fontweight="bold", color="#7B1FA2")

    # -- FADH box --
    fb = FancyBboxPatch((1.2, 12.5), 8.5, 7.2, boxstyle="round,pad=0.2",
                         fc="#E8EAF6", ec="#3F51B5", lw=2)
    ax.add_patch(fb)
    ax.text(5.45, 19.3, r"FADH$^{\bullet}$  semiquinone", fontsize=15,
            fontweight="bold", color="#1A237E", ha="center")

    # electron 1
    c1 = Circle((3.0, 17.5), 0.35, fc="#C62828", ec="black", lw=2, zorder=5)
    ax.add_patch(c1)
    ax.text(3.0, 17.5, r"$e^-$", ha="center", va="center", fontsize=11,
            fontweight="bold", color="white", zorder=6)
    spin_arr(ax, 3.0, 17.5, up=True, color="#C62828", sz=0.2)

    ax.text(3.0, 16.7, r"$\mathbf{S}_1$", ha="center", fontsize=14, color="#C62828")

    # Isoalloxazine label (well away from electron)
    ax.text(7.0, 17.8, "Isoalloxazine", ha="center", fontsize=12,
            color="#3F51B5", style="italic")
    ax.text(7.0, 17.2, r"$\pi$-system", ha="center", fontsize=12,
            color="#3F51B5", style="italic")

    # coupled P (lower right, away from e-)
    cp = Circle((8.2, 15.5), 0.3, fc="#FF6D00", ec="black", lw=1.5, zorder=5)
    ax.add_patch(cp)
    ax.text(8.2, 15.5, "P", ha="center", va="center", fontsize=11,
            fontweight="bold", color="white", zorder=6)

    # HFC line (e- to P)
    ax.plot([3.35, 7.9], [17.5, 15.5], "r--", lw=2, alpha=0.4)
    ax.text(4.2, 16.8, "HFC  200 MHz", fontsize=11, color="#C62828",
            ha="center",
            bbox=dict(fc="#E8EAF6", ec="none", pad=1))

    # T2, SOC (bottom of box, plenty of space)
    ax.text(5.45, 13.8, r"$T_2^e = 1.1$ ns", fontsize=13,
            ha="center", color="#4A148C")
    ax.text(5.45, 13.0, r"SOC $= 63$ cm$^{-1}$", fontsize=13,
            ha="center", color="#4A148C")

    # -- Substrate box --
    sb = FancyBboxPatch((10.5, 12.5), 7.0, 7.2, boxstyle="round,pad=0.2",
                         fc="#FBE9E7", ec="#BF360C", lw=2)
    ax.add_patch(sb)
    ax.text(14.0, 19.3, "Substrate amine radical", fontsize=15,
            fontweight="bold", color="#BF360C", ha="center")

    # electron 2
    c2 = Circle((12.2, 17.5), 0.35, fc="#C62828", ec="black", lw=2, zorder=5)
    ax.add_patch(c2)
    ax.text(12.2, 17.5, r"$e^-$", ha="center", va="center", fontsize=11,
            fontweight="bold", color="white", zorder=6)
    spin_arr(ax, 12.2, 17.5, up=False, color="#C62828", sz=0.2)

    ax.text(12.2, 16.7, r"$\mathbf{S}_2$", ha="center", fontsize=14, color="#C62828")

    ax.text(15.5, 17.5, "5-HT amine\nnitrogen\nradical",
            ha="center", fontsize=12, color="#BF360C", linespacing=1.4)

    ax.text(14.0, 13.8, r"$^1$H  HFC $= 2.7$ MHz", fontsize=13,
            ha="center", color="#555")
    ax.text(14.0, 13.0, r"$J \approx 0$  (separated)", fontsize=13,
            ha="center", color="#555")

    # RP link
    ax.annotate("", xy=(10.5, 17.5), xytext=(9.7, 17.5),
                arrowprops=dict(arrowstyle="<->", color="#9C27B0", lw=3, ls="--"))
    ax.text(10.1, 18.1, "RP", fontsize=14, color="#9C27B0", ha="center",
            fontweight="bold")

    # -- S-T Mixing box --
    stb = FancyBboxPatch((18.5, 9.5), 10.0, 10.5, boxstyle="round,pad=0.2",
                          fc="white", ec="#7B1FA2", lw=2)
    ax.add_patch(stb)
    ax.text(23.5, 19.5, "$S$-$T$  Mixing", fontsize=18,
            fontweight="bold", color="#7B1FA2", ha="center")
    ax.text(23.5, 18.6, "(quantum process)", fontsize=13,
            color="#7B1FA2", ha="center")

    # Singlet state (generous vertical space)
    ax.text(19.3, 17.0,
            r"$|S\rangle = (|\!\uparrow\downarrow\rangle"
            r" - |\!\downarrow\uparrow\rangle) \;/\; \sqrt{2}$",
            fontsize=16, color="#2E7D32")

    # Triplet state
    ax.text(19.3, 14.8,
            r"$|T_0\rangle = (|\!\uparrow\downarrow\rangle"
            r" + |\!\downarrow\uparrow\rangle) \;/\; \sqrt{2}$",
            fontsize=16, color="#C62828")

    # mixing arrow
    ax.annotate("", xy=(27.5, 15.0), xytext=(27.5, 17.2),
                arrowprops=dict(arrowstyle="<->", color="#FF6D00", lw=4,
                                connectionstyle="arc3,rad=0.5"))
    ax.text(28.2, 16.0, "HFC\ndriven", fontsize=13, color="#FF6D00",
            fontweight="bold", ha="center")

    # Products (well spaced from states above)
    ax.text(19.3, 12.8,
            r"$\Phi_S(B)$   $\rightarrow$   Product A  (singlet)",
            fontsize=14, color="#2E7D32")

    ax.text(19.3, 11.3,
            r"$1 - \Phi_S(B)$   $\rightarrow$   Product B  (triplet)",
            fontsize=14, color="#C62828")

    ax.text(19.3, 10.0,
            "Branching ratio is $B$-field sensitive",
            fontsize=14, color="#7B1FA2", fontweight="bold")

    # ZQ TDM (below Layer 2 boxes, clear space)
    ax.text(5.45, 9.5,
            r"ZQ TDM $= 1.22\;\mu_B$   (70 % of allowed)",
            fontsize=13, ha="center", color="#E65100", fontweight="bold",
            bbox=dict(fc="#FFF3E0", ec="#E65100", boxstyle="round,pad=0.3"))

    # ==========================================================
    # INTERFACE 2-3     y: 5.5 -- 7.8
    # ==========================================================
    ax.annotate("", xy=(8, 7.8), xytext=(8, 8.5),
                arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=5))

    yb = FancyBboxPatch((2, 5.8), 22, 1.8, boxstyle="round,pad=0.15",
                         fc="#EDE7F6", ec="#7B1FA2", lw=2.5)
    ax.add_patch(yb)
    ax.text(13, 7.0,
            r"SPIN-SELECTIVE YIELD       "
            r"$\Phi_S(B) \;\rightarrow\; k_{\mathrm{cat}}$  modulation",
            ha="center", fontsize=16, fontweight="bold", color="#7B1FA2")
    ax.text(13, 6.1,
            "quantum spin state   -->   product ratio   -->   enzyme rate"
            "       |       quantum --> classical",
            ha="center", fontsize=11, color="#4A148C")

    ax.text(27, 6.5,
            "MFE $= -4.4$ %  (1 mT)\n"
            "MFE $= -0.3$ %  (Earth)",
            fontsize=13, color="#7B1FA2", ha="center", fontweight="bold",
            bbox=dict(fc="#EDE7F6", alpha=0.9, boxstyle="round,pad=0.4"))

    # ==========================================================
    # LAYER 3   y: 0.5 -- 5.2
    # ==========================================================
    L3 = FancyBboxPatch((0.5, 0.5), 28.5, 4.7, boxstyle="round,pad=0.3",
                         fc="#E8F5E9", ec="#2E7D32", lw=3)
    ax.add_patch(L3)
    ax.text(15, 4.7, "LAYER 3 :  Classical Electrochemical Processing",
            ha="center", fontsize=19, fontweight="bold", color="#2E7D32")

    nodes = [
        (2.5, "5-HT\n(serotonin)"),
        (7.5, "MAO-A\n$k_{\\mathrm{cat}} \\approx 10$/s"),
        (12.5, "5-HIAA"),
        (17.0, "Synaptic\n[5-HT]"),
        (21.5, "5-HT\nreceptors"),
        (26.0, "Neural\ncircuit"),
    ]
    for i, (x, lab) in enumerate(nodes):
        ax.text(x, 3.3, lab, ha="center", fontsize=13, color="#1B5E20",
                fontweight="bold")
        if i < len(nodes) - 1:
            xn = nodes[i + 1][0]
            ax.annotate("", xy=(xn - 1.2, 3.3), xytext=(x + 1.2, 3.3),
                        arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=3))

    ax.text(12.5, 1.5, "Rate modulated 0.3 % by Earth field",
            fontsize=13, color="#C62828", fontweight="bold", ha="center",
            bbox=dict(fc="#E8F5E9", ec="none", pad=1))

    ax.text(26, 1.3, "Mood\nCircadian rhythm\nCognition",
            fontsize=14, color="#1B5E20", fontweight="bold", ha="center",
            bbox=dict(fc="#C8E6C9", alpha=0.9, boxstyle="round,pad=0.4"))

    ax.text(15, 0.8,
            "Classical :  concentration ,  membrane potential ,  ion current",
            fontsize=10, color="#666", ha="center", style="italic")

    # ==========================================================
    # Energy scale bar (far right, no overlap)
    # ==========================================================
    xE = 29.5
    ax.plot([xE, xE], [1.5, 29.0], "k-", lw=1.5, alpha=0.3)
    for y, lab, col in [
        (28.5, "3.6 peV", "#E65100"),
        (27.5, r"0.83 $\mu$eV", "#FF6D00"),
        (17.0, r"$\sim$neV  ($J$)", "#7B1FA2"),
        (13.0, r"$\sim\mu$eV  (HFC)", "#7B1FA2"),
        (3.5, "26.7 meV  ($k_BT$)", "#2E7D32"),
        (2.0, r"$\sim$eV  (bonds)", "#2E7D32"),
    ]:
        ax.plot([xE - 0.15, xE + 0.15], [y, y], color=col, lw=2.5)
        ax.text(xE + 0.3, y, lab, fontsize=10, color=col, va="center")

    ax.annotate("", xy=(xE, 2.3), xytext=(xE, 28.2),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=2))
    ax.text(xE, 15, "14 orders\nof magnitude", fontsize=11, ha="center",
            color="#333", fontweight="bold", rotation=90)

    # ==========================================================
    # Legend
    # ==========================================================
    le = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF6D00",
               markersize=14, label=r"$^{31}$P nuclear spin  (quantum)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C62828",
               markersize=14, label=r"Electron spin  (quantum)"),
        Line2D([0], [0], color="#E65100", lw=4,
               label="HFC coupling  (quantum channel)"),
        Line2D([0], [0], color="#7B1FA2", lw=4,
               label=r"Spin-selective yield  (Q $\rightarrow$ C)"),
        Line2D([0], [0], color="#2E7D32", lw=4,
               label="Chemical / electrical  (classical)"),
    ]
    ax.legend(handles=le, loc="lower left", fontsize=12,
              title="Information Carriers", title_fontsize=13,
              bbox_to_anchor=(0.01, -0.02), framealpha=0.95, edgecolor="#333")

    plt.savefig(OUT / "fig5_three_layer.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig5_three_layer.png", bbox_inches="tight")
    plt.close()
    print("Fig 5 v5 saved")


if __name__ == "__main__":
    main()
