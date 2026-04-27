#!/usr/bin/env python3
"""3-Layer Quantum Brain Hypothesis — Energy Level Diagrams (English, publication quality)

Generates 5 figures showing molecular information carriers, hyperfine structure,
and energy levels at brain temperature (310 K) for each layer.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path("simulation_results")
OUT_DIR.mkdir(exist_ok=True)

# Use standard fonts only (no CJK)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "figure.dpi": 150,
})

# Physical constants
KB = 8.617e-5
KT_310 = KB * 310
MU_B_EV_T = 5.788e-5
G_E = 2.002
GAMMA_P = 17.235e6
GAMMA_H = 42.577e6
B_EARTH = 50e-6


def fig1_energy_scales():
    """All energy scales across 3 layers in one plot."""
    fig, ax = plt.subplots(figsize=(14, 9))

    scales = [
        ("C-C bond", 4.0, "L3", "#2e7d32"),
        ("S0->S1 (FAD, 450nm)", 2.76, "L3/L2", "#1565c0"),
        ("S-T gap (MAO-A)", 1.30, "L2", "#7b1fa2"),
        ("HOMO-LUMO (MAO-A)", 1.25, "L2", "#7b1fa2"),
        ("5-HT oxidation pot.", 0.8, "L3", "#2e7d32"),
        ("H-bond (protein)", 0.15, "L3", "#2e7d32"),
        ("kT (310 K)", KT_310, "thermal", "#d32f2f"),
        ("HFC 31P (A=200MHz)", 200e6 * 4.136e-15, "L1/L2", "#e65100"),
        ("e- Zeeman @50uT", G_E * MU_B_EV_T * B_EARTH, "L2", "#7b1fa2"),
        ("14N quadrupole (3MHz)", 3e6 * 4.136e-15, "L1", "#ff8f00"),
        ("31P Zeeman @50uT", GAMMA_P * B_EARTH * 4.136e-15, "L1", "#e65100"),
        ("1H Zeeman @50uT", GAMMA_H * B_EARTH * 4.136e-15, "L1", "#ff8f00"),
        ("31P-31P dipolar", 0.01e6 * 4.136e-15, "L1", "#e65100"),
    ]
    scales.sort(key=lambda x: x[1], reverse=True)

    y = np.arange(len(scales))
    energies = [s[1] for s in scales]
    colors = [s[3] for s in scales]

    ax.barh(y, np.log10(energies), color=colors, alpha=0.7,
            edgecolor="black", height=0.6)

    for i, (name, E, layer, _) in enumerate(scales):
        if E >= 1e-3:
            e_str = f"{E*1e3:.1f} meV"
        elif E >= 1e-6:
            e_str = f"{E*1e6:.2f} ueV"
        elif E >= 1e-9:
            e_str = f"{E*1e9:.2f} neV"
        elif E >= 1e-12:
            e_str = f"{E*1e12:.2f} peV"
        else:
            e_str = f"{E*1e15:.1f} feV"
        ax.text(np.log10(E) + 0.15, i, f"{e_str}  [{layer}]",
                va="center", fontsize=8, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in scales], fontsize=9)
    ax.set_xlabel("log10(Energy / eV)", fontsize=12)
    ax.set_title("Three-Layer Quantum Brain Hypothesis\n"
                 "Energy Scale Overview at 310 K", fontsize=13, fontweight="bold")

    ax.axvline(np.log10(KT_310), color="red", ls="--", lw=2, alpha=0.5)
    ax.text(np.log10(KT_310) + 0.1, len(scales) - 0.5,
            f"kT = {KT_310*1e3:.1f} meV", color="red", fontsize=10)

    # Layer shading
    ax.axhspan(9.5, 12.5, alpha=0.06, color="orange")
    ax.axhspan(5.5, 9.5, alpha=0.06, color="purple")
    ax.axhspan(-0.5, 5.5, alpha=0.06, color="green")
    ax.text(-12.5, 11, "Layer 1\nNuclear spin\nquantum memory",
            fontsize=8, color="#e65100", fontweight="bold", ha="center")
    ax.text(-12.5, 7.5, "Layer 2\nRadical pair\ninterface",
            fontsize=8, color="#7b1fa2", fontweight="bold", ha="center")
    ax.text(-12.5, 2.5, "Layer 3\nClassical\nelectrochemistry",
            fontsize=8, color="#2e7d32", fontweight="bold", ha="center")

    ax.set_xlim(-15, 2)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_energy_scales_en.png", dpi=150)
    plt.close()
    print("  fig1 saved")


def fig2_layer1_nuclear():
    """Layer 1: 31P nuclear spin levels at Earth's field."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    # (a) Single 31P
    ax = axes[0]
    nu_P = GAMMA_P * B_EARTH
    E_split = nu_P * 4.136e-15

    ax.plot([0.2, 0.8], [+E_split/2*1e12, +E_split/2*1e12], "b-", lw=3)
    ax.plot([0.2, 0.8], [-E_split/2*1e12, -E_split/2*1e12], "r-", lw=3)
    ax.annotate("", xy=(0.85, E_split/2*1e12), xytext=(0.85, -E_split/2*1e12),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(0.9, 0, f"dE = {nu_P:.0f} Hz\n= {E_split*1e12:.2f} peV",
            fontsize=9, va="center")
    ax.text(0.5, E_split/2*1e12 + 0.3, r"|$m_I$ = +1/2>", ha="center",
            fontsize=10, color="blue")
    ax.text(0.5, -E_split/2*1e12 - 0.3, r"|$m_I$ = -1/2>", ha="center",
            fontsize=10, color="red")
    ax.set_xlim(0, 1.4)
    ax.set_ylabel("Energy (peV)")
    ax.set_title("(a) Single 31P (I=1/2)\n@ Earth field 50 uT\n"
                 "Molecule: PO4 in FAD", fontsize=10)
    ax.set_xticks([])

    # (b) Coupled 31P-31P
    ax = axes[1]
    J_dd = 0.01e6 * 4.136e-15
    E_Z = nu_P * 4.136e-15

    levels_b = [
        (+(E_Z + J_dd/4)*1e15, "|T+> = |up,up>", "blue"),
        (+(J_dd/2)*1e15, "|T0>", "blue"),
        (-(3*J_dd/4)*1e15, "|S> = (|up,dn>-|dn,up>)/sqrt(2)", "red"),
        (-(E_Z - J_dd/4)*1e15, "|T-> = |dn,dn>", "blue"),
    ]
    for E, label, col in levels_b:
        x0 = 0.15 if col == "blue" else 0.55
        x1 = 0.55 if col == "blue" else 0.95
        ax.plot([x0, x1], [E, E], color=col, lw=3)
        ax.text(x1 + 0.02, E, label, fontsize=7, va="center", color=col)

    ax.annotate("", xy=(0.12, (J_dd/2)*1e15), xytext=(0.12, -(3*J_dd/4)*1e15),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(0.0, 0, f"J_dd\n0.01 MHz\n= 41 feV", fontsize=8, color="green",
            va="center", ha="center")

    ax.set_ylabel("Energy (feV)")
    ax.set_title("(b) Coupled 31P-31P (dipolar)\n@ 50 uT\n"
                 "Molecule: FAD phosphate chain", fontsize=10)
    ax.set_xticks([])

    # (c) 14N quadrupole
    ax = axes[2]
    eqQ = 3.0e6
    E_qQ = eqQ * 4.136e-15

    for E, label in [(E_qQ/4, r"|$m_I$=+1>"), (-E_qQ/2, r"|$m_I$=0>"),
                     (E_qQ/4, r"|$m_I$=-1>")]:
        ax.plot([0.2, 0.8], [E*1e9, E*1e9], "b-", lw=3)

    ax.text(0.82, E_qQ/4*1e9, r"|$m_I$=+1> and |$m_I$=-1>", fontsize=9, va="center")
    ax.text(0.82, -E_qQ/2*1e9, r"|$m_I$=0>", fontsize=9, va="center")

    ax.annotate("", xy=(0.15, E_qQ/4*1e9), xytext=(0.15, -E_qQ/2*1e9),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(0.02, 0, f"e2qQ/h\n= {eqQ/1e6:.0f} MHz\n= {E_qQ*1e9:.1f} neV\n"
            f"T1 = 0.73 us", fontsize=8, color="green", va="center")

    ax.set_ylabel("Energy (neV)")
    ax.set_title("(c) 14N quadrupole (I=1)\nIsoalloxazine ring\n"
                 "Molecule: FAD flavin", fontsize=10)
    ax.set_xticks([])

    fig.suptitle("Layer 1: Nuclear Spin Quantum Memory — Carrier Molecules and Energy Levels (310 K)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_layer1_nuclear_en.png", dpi=150)
    plt.close()
    print("  fig2 saved")


def fig3_layer2_radical_pair():
    """Layer 2: Coupled e-31P system and RP energy levels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # (a) Coupled e-31P 4-level system
    ax = axes[0]
    omega_S = G_E * MU_B_EV_T * B_EARTH
    A = 200e6 * 4.136e-15
    omega_I = GAMMA_P * B_EARTH * 4.136e-15
    to_MHz = 1 / (4.136e-15 * 1e6)

    E_aa = (+omega_S/2 + omega_I/2 + A/4) * to_MHz
    E_ab = (+omega_S/2 - omega_I/2 - A/4) * to_MHz
    E_ba = (-omega_S/2 + omega_I/2 - A/4) * to_MHz
    E_bb = (-omega_S/2 - omega_I/2 + A/4) * to_MHz

    for E, label, col, x0, x1 in [
        (E_aa, r"|$\alpha_e\alpha_n$>", "#1565c0", 0.1, 0.45),
        (E_bb, r"|$\beta_e\beta_n$>", "#1565c0", 0.1, 0.45),
        (E_ab, r"|$\alpha_e\beta_n$>", "#c62828", 0.55, 0.9),
        (E_ba, r"|$\beta_e\alpha_n$>", "#c62828", 0.55, 0.9),
    ]:
        ax.plot([x0, x1], [E, E], color=col, lw=3)
        side = x0 - 0.02 if x0 < 0.5 else x1 + 0.02
        ha = "right" if x0 < 0.5 else "left"
        ax.text(side, E, f"{label}\n{E:.1f} MHz", fontsize=7, ha=ha,
                va="center", color=col)

    ax.annotate("", xy=(0.3, E_ba), xytext=(0.3, E_aa),
                arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    ax.text(0.32, (E_aa+E_ba)/2, f"EPR\n{abs(E_aa-E_ba):.1f} MHz\n"
            r"$\mu$=1.73 $\mu_B$", fontsize=7, color="green", va="center")

    ax.annotate("", xy=(0.5, E_ab), xytext=(0.5, E_ba),
                arrowprops=dict(arrowstyle="<->", color="orange", lw=2, ls="--"))
    ax.text(0.52, (E_ab+E_ba)/2, f"ZQ flip-flop\n{abs(E_ab-E_ba):.1f} MHz\n"
            r"$\mu$=1.22 $\mu_B$" + "\n(70% of allowed!)",
            fontsize=7, color="orange", va="center")

    ax.annotate("", xy=(0.95, E_aa), xytext=(0.95, E_ab),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
    ax.text(0.97, (E_aa+E_ab)/2, f"A(31P)\n200 MHz\n0.83 ueV",
            fontsize=7, color="purple", va="center")

    ax.set_ylabel("Energy (MHz)")
    ax.set_title("(a) Coupled e-31P 4-level system\n@ 50 uT (strong coupling: A >> wS)\n"
                 "Molecule: FADH semiquinone radical", fontsize=9)
    ax.set_xticks([])

    # (b) Radical pair S-T levels
    ax = axes[1]
    omega_S_MHz = G_E * MU_B_EV_T * B_EARTH * to_MHz
    J = 0.001

    ax.plot([0.1, 0.5], [omega_S_MHz, omega_S_MHz], "b-", lw=3)
    ax.plot([0.1, 0.5], [-J, -J], "b-", lw=3)
    ax.plot([0.6, 1.0], [+J, +J], "r-", lw=3)
    ax.plot([0.1, 0.5], [-omega_S_MHz, -omega_S_MHz], "b-", lw=3)

    ax.text(0.3, omega_S_MHz+0.08, r"|$T_+$>", ha="center", fontsize=11, color="blue")
    ax.text(0.3, -J+0.08, r"|$T_0$>", ha="center", fontsize=11, color="blue")
    ax.text(0.8, J+0.08, "|S>", ha="center", fontsize=11, color="red")
    ax.text(0.3, -omega_S_MHz-0.15, r"|$T_-$>", ha="center", fontsize=11, color="blue")

    ax.annotate("", xy=(0.55, J), xytext=(0.5, -J),
                arrowprops=dict(arrowstyle="<->", color="orange", lw=2,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(0.53, 0.2, "HFC-driven\n" + r"S$\leftrightarrow$$T_0$ mixing" +
            "\n(A=200 MHz)", fontsize=8, color="orange", ha="center")

    ax.annotate("", xy=(0.05, omega_S_MHz), xytext=(0.05, -omega_S_MHz),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(0.0, 0, f"2g" + r"$\mu_B$B" + f"\n{2*omega_S_MHz:.2f}\nMHz",
            fontsize=7, color="green", va="center", ha="right")

    ax.text(1.05, J, "Product A\n(singlet)", fontsize=8, color="red", va="center")
    ax.text(0.55, -omega_S_MHz-0.25, "Product B\n(triplet)", fontsize=8,
            color="blue", va="center")

    ax.set_ylabel("Energy (MHz)")
    ax.set_ylim(-2, 2)
    ax.set_title("(b) Radical pair S-T levels\n@ 50 uT (J ~ 0)\n"
                 "Molecule: [FADH...Sub] in MAO-A", fontsize=9)
    ax.set_xticks([])

    # (c) Spin relaxation timescales
    ax = axes[2]
    timescales = [
        ("T2e (HFC-mod)\n0.5 ns", 0.5, "#c62828"),
        ("T2* (HFC inhom.)\n1.1 ns", 1.1, "#e65100"),
        ("T2 (g-anisotropy)\ninf @ Earth field", 1e6, "#7b1fa2"),
        ("RP lifetime\n~1 us", 1e3, "#2e7d32"),
        ("T1e (spin-lattice)\n100 us", 1e5, "#1565c0"),
        ("T2(31P) coupled\n160 us", 1.6e5, "#e65100"),
        ("T1(31P)\n4.6 s", 4.6e9, "#ff8f00"),
    ]
    y_pos = np.arange(len(timescales))
    bars = ax.barh(y_pos, [np.log10(t[1]) for t in timescales],
                   color=[t[2] for t in timescales], alpha=0.7,
                   edgecolor="black", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[0] for t in timescales], fontsize=8)
    ax.set_xlabel("log10(time / ns)")
    ax.axvline(3, color="green", ls="--", lw=2, alpha=0.3)
    ax.text(3.1, 6.5, "RP lifetime\n~1 us", fontsize=8, color="green")
    ax.set_title("(c) Spin relaxation timescales\n@ brain (50 uT, 310 K)\n"
                 "Molecule: FADH in MAO-A", fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("Layer 2: Radical Pair Interface — Energy Levels and Timescales (310 K)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_layer2_rp_en.png", dpi=150)
    plt.close()
    print("  fig3 saved")


def fig4_layer3_classical():
    """Layer 3: Classical molecules and integrated energy scales."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    # (a) 5-HT electronic levels
    ax = axes[0]
    levels = [(0, "S0 (ground)", "#2e7d32"),
              (0.8, "Oxidation potential\n(MAO-A substrate)", "#e65100"),
              (3.6, "S1 (275 nm)", "#1565c0"),
              (4.5, "S2 (220 nm)", "#7b1fa2")]
    for E, label, col in levels:
        ax.plot([0.2, 0.8], [E, E], color=col, lw=3)
        ax.text(0.82, E, label, fontsize=9, va="center", color=col)

    ax.annotate("", xy=(0.15, KT_310*1e3), xytext=(0.15, 0),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax.text(0.05, KT_310*1e3/2, f"kT\n{KT_310*1e3:.1f}\nmeV",
            fontsize=8, color="red", ha="center")

    ax.set_ylabel("Energy (eV)")
    ax.set_title("(a) Serotonin (5-HT)\nC10H12N2O (MW 176)\n"
                 "L3 -> L2 entry point", fontsize=10)
    ax.set_xticks([])
    ax.set_ylim(-0.5, 5)

    # (b) FAD electronic levels
    ax = axes[1]
    levels_fad = [(0, "S0", "#2e7d32"),
                  (1.25, "HOMO-LUMO\n(xTB: 1.25 eV)", "#7b1fa2"),
                  (2.76, "S1 (450 nm)\nblue light", "#1565c0"),
                  (2.3, "T1 (after ISC)", "#c62828")]
    for E, label, col in levels_fad:
        ax.plot([0.2, 0.8], [E, E], color=col, lw=3)
        ax.text(0.82, E, label, fontsize=8, va="center", color=col)

    ax.annotate("", xy=(0.5, 2.76), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.text(0.52, 1.4, r"h$\nu$" + "\n450 nm", fontsize=10, color="blue")

    ax.annotate("", xy=(0.7, 2.3), xytext=(0.75, 2.76),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5, ls="--"))
    ax.text(0.77, 2.55, r"ISC ($\xi$=63 cm$^{-1}$)", fontsize=8, color="red")

    ax.text(0.5, 1.9, "RP generation\n[FADH...5-HT]", fontsize=9,
            color="orange", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    ax.set_ylabel("Energy (eV)")
    ax.set_title("(b) FAD (flavin adenine dinucleotide)\nC33H36N9O15P2 (MW 785)\n"
                 "Central molecule of L2", fontsize=10)
    ax.set_xticks([])
    ax.set_ylim(-0.5, 3.5)

    # (c) Integrated energy scale
    ax = axes[2]
    items = [
        ("C-C bond", 4.0, "#2e7d32"),
        ("S0->S1 (FAD)", 2.76, "#1565c0"),
        ("S-T gap", 1.3, "#7b1fa2"),
        ("HOMO-LUMO", 1.25, "#7b1fa2"),
        ("Oxidation", 0.8, "#2e7d32"),
        ("kT (310K)", KT_310, "#d32f2f"),
        ("HFC 31P", 200e6*4.136e-15, "#e65100"),
        ("e- Zeeman", G_E*MU_B_EV_T*B_EARTH, "#7b1fa2"),
        ("31P-31P dip.", 0.01e6*4.136e-15, "#e65100"),
        ("31P Zeeman", GAMMA_P*B_EARTH*4.136e-15, "#e65100"),
    ]
    y = np.arange(len(items))
    ax.barh(y, [np.log10(e[1]) for e in items],
            color=[e[2] for e in items], alpha=0.6, edgecolor="black", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in items], fontsize=8)
    ax.set_xlabel("log10(Energy / eV)")
    ax.axvline(np.log10(KT_310), color="red", ls="--", lw=2, alpha=0.4)

    ax.text(-12.5, 9, "Layer 1", fontsize=9, color="#e65100", fontweight="bold",
            ha="center", bbox=dict(facecolor="lightyellow", alpha=0.8))
    ax.text(-4, 3, "Layer 2", fontsize=9, color="#7b1fa2", fontweight="bold",
            ha="center", bbox=dict(facecolor="lavender", alpha=0.8))
    ax.text(-1, 0.5, "Layer 3", fontsize=9, color="#2e7d32", fontweight="bold",
            ha="center", bbox=dict(facecolor="honeydew", alpha=0.8))

    ax.set_title("(c) Integrated energy hierarchy\n14 orders of magnitude", fontsize=10)
    ax.grid(True, alpha=0.2, axis="x")
    ax.set_xlim(-14, 1)

    fig.suptitle("Layer 3: Classical Electrochemistry and Inter-Layer Energy Connections (310 K)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_layer3_classical_en.png", dpi=150)
    plt.close()
    print("  fig4 saved")


def fig5_molecular_info_flow():
    """3-layer molecular information flow schematic (English, no CJK)."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # === Layer 1 ===
    r1 = mpatches.FancyBboxPatch((0.5, 8.5), 9, 3, boxstyle="round,pad=0.2",
                                  facecolor="#fff3e0", edgecolor="#e65100", lw=2)
    ax.add_patch(r1)
    ax.text(5, 11.2, "Layer 1: Quantum Memory (Nuclear Spins)", fontsize=14,
            ha="center", fontweight="bold", color="#e65100")

    ax.text(1.2, 10.5, "Information carrier:", fontsize=10, fontweight="bold",
            color="#bf360c")
    ax.text(1.2, 10.0, "31P nuclear spin (I = 1/2)\nin PO4 (phosphate group)",
            fontsize=9, color="#333")
    ax.text(1.2, 9.0, "Hyperfine structure:\n"
            "  Zeeman: 862 Hz @50uT (3.6 peV)\n"
            "  Dipolar: 0.01 MHz (41 feV)\n"
            "  T2 = 3.25 ms (diamagnetic)\n"
            "  T2 = 160 us (with PRE, FAD)",
            fontsize=8, color="#555")

    ax.text(5.8, 10.5, "Molecules:", fontsize=10, fontweight="bold", color="#bf360c")
    ax.text(5.8, 10.0, "FAD: C33H36N9O15P2 (MW 785)\n"
            "  2x 31P on ribose-phosphate chain",
            fontsize=9, color="#333")
    ax.text(5.8, 9.0, "PLP: C8H10NO6P (MW 247)\n"
            "  1x 31P directly bonded to C5\n"
            "NADPH: C21H29N7O17P3 (MW 744)\n"
            "  3x 31P (highest n_P)",
            fontsize=8, color="#555")

    # === Layer 2 ===
    r2 = mpatches.FancyBboxPatch((0.5, 4.3), 9, 3.7, boxstyle="round,pad=0.2",
                                  facecolor="#f3e5f5", edgecolor="#7b1fa2", lw=2)
    ax.add_patch(r2)
    ax.text(5, 7.7, "Layer 2: Quantum-Classical Interface (Radical Pair)",
            fontsize=14, ha="center", fontweight="bold", color="#7b1fa2")

    ax.text(1.0, 7.0, "Information carrier:", fontsize=10, fontweight="bold",
            color="#4a148c")
    ax.text(1.0, 6.5, "FADH semiquinone radical\n+ substrate radical (Sub)",
            fontsize=9, color="#333")
    ax.text(1.0, 5.3, "Hyperfine structure:\n"
            "  Coupled e-31P: 4 levels (A=200 MHz)\n"
            "  EPR splitting: 2.80 MHz (@50uT)\n"
            "  Forbidden TDM: 1.22 uB (70% of allowed)\n"
            "  T2e = 1.1 ns, T1e = 100 us",
            fontsize=8, color="#555")

    ax.text(5.5, 7.0, "Conversion mechanism:", fontsize=10, fontweight="bold",
            color="#4a148c")
    ax.text(5.5, 6.5, "S-T0 mixing (HFC-driven)\n-> singlet yield Phi_S(B)",
            fontsize=9, color="#333")
    ax.text(5.5, 5.3, "Magnetic field effect (MFE):\n"
            "  Phi_S(0) = 0.459 -> Phi_S(0.9mT) = 0.439\n"
            "  MFE = -4.4% (MAO-A, Sim 1)\n"
            "  At Earth field @50uT: MFE ~ -0.3%\n"
            "  -> 5-HT degradation rate modulated 0.3%",
            fontsize=8, color="#555")

    # === Layer 3 ===
    r3 = mpatches.FancyBboxPatch((0.5, 0.5), 9, 3.3, boxstyle="round,pad=0.2",
                                  facecolor="#e8f5e9", edgecolor="#2e7d32", lw=2)
    ax.add_patch(r3)
    ax.text(5, 3.5, "Layer 3: Classical Electrochemical Processing",
            fontsize=14, ha="center", fontweight="bold", color="#2e7d32")

    ax.text(1.0, 2.8, "Information carrier:", fontsize=10, fontweight="bold",
            color="#1b5e20")
    ax.text(1.0, 2.3, "Serotonin (5-HT): C10H12N2O\n"
            "Dopamine (DA): C8H11NO2\n"
            "GABA: C4H9NO2", fontsize=9, color="#333")
    ax.text(1.0, 1.2, "Energy scales:\n"
            "  Oxidation potential: 0.8 eV (5-HT)\n"
            "  Synaptic potential: ~100 mV\n"
            "  kT = 26.7 meV",
            fontsize=8, color="#555")

    ax.text(5.5, 2.8, "Processing:", fontsize=10, fontweight="bold", color="#1b5e20")
    ax.text(5.5, 2.3, "MAO-A: 5-HT -> 5-HIAA\n"
            "  kcat ~ 10/s, RPM modulates 0.3%",
            fontsize=9, color="#333")
    ax.text(5.5, 1.2, "Neuroscience output:\n"
            "  5-HT concentration: ~nM change\n"
            "  -> synaptic transmission efficiency\n"
            "  -> mood, emotion, circadian rhythm",
            fontsize=8, color="#555")

    # === Arrows ===
    ax.annotate("", xy=(3, 8.0), xytext=(3, 8.5),
                arrowprops=dict(arrowstyle="<->", color="#e65100", lw=3))
    ax.text(3.3, 8.2, "HFC (A=200 MHz)\n31P nuclear <-> electron spin",
            fontsize=9, color="#e65100", fontweight="bold")

    ax.annotate("", xy=(3, 3.8), xytext=(3, 4.3),
                arrowprops=dict(arrowstyle="->", color="#7b1fa2", lw=3))
    ax.text(3.3, 4.05, "Phi_S(B) -> reaction rate modulation\nQuantum -> Classical conversion",
            fontsize=9, color="#7b1fa2", fontweight="bold")

    ax.text(5, 12.0,
            "Three-Layer Quantum Brain Hypothesis\n"
            "Identification of Molecular Information Carriers and Hyperfine Structure",
            fontsize=14, ha="center", fontweight="bold")

    plt.savefig(OUT_DIR / "fig5_molecular_info_flow_en.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  fig5 saved")


if __name__ == "__main__":
    print("Generating energy level diagrams (English)...")
    fig1_energy_scales()
    fig2_layer1_nuclear()
    fig3_layer2_radical_pair()
    fig4_layer3_classical()
    fig5_molecular_info_flow()
    print("All figures saved -> simulation_results/fig[1-5]_*_en.png")
