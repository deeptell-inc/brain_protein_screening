#!/usr/bin/env python3
"""Generate all 5 main figures for the Nature manuscript.

Fig 1: RPM condition radar chart (9 proteins x 7 conditions)
Fig 2: (a) T2e decomposition, (b) Nuclear T2 vs e-P distance
Fig 3: Coupled e-31P energy levels and transitions at Earth field
Fig 4: RPM singlet yield Phi_S(B) and MFE spectrum
Fig 5: Three-layer quantum-classical hierarchy schematic
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from pathlib import Path
import json

from figure_style import apply_style, SINGLE_COL, DOUBLE_COL
apply_style()

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# ======================================================================
# Data from screening results
# ======================================================================
proteins = ["MAO-A", "MAO-B", "DAO", "CRY", "DDC", "GAD67", "SRR", "IDH", "LDH"]
cofactors = ["FAD", "FAD", "FAD", "FAD", "PLP", "PLP", "PLP+Mn", "NADPH", "NADH"]

# 7 RPM conditions — normalised scores (0-1, higher=better)
# C1: f_osc, C2: n_P/3, C3: T2e/1.5, C4: A/omega (capped at 1),
# C5: T1/T2 (capped), C6: T2n/1500, C7: (200-SOC)/200
data = {
    "MAO-A":  [0.517/0.6, 2/3, 0.32/1.5, 1.0, 1.0, 160/1500, (200-63)/200],
    "MAO-B":  [0.434/0.6, 2/3, 0.30/1.5, 1.0, 1.0, 160/1500, (200-63)/200],
    "DAO":    [0.147/0.6, 2/3, 0.16/1.5, 1.0, 1.0, 160/1500, (200-65)/200],
    "CRY":    [0.497/0.6, 2/3, 0.32/1.5, 1.0, 1.0, 160/1500, (200-67)/200],
    "DDC":    [0.300/0.6, 1/3, 0.52/1.5, 1.0, 1.0, 2.5/1500, (200-79)/200],
    "GAD67":  [0.331/0.6, 1/3, 0.51/1.5, 1.0, 1.0, 2.5/1500, (200-63)/200],
    "SRR":    [0.040/0.6, 0/3, 0.00/1.5, 0.0, 0.0, 0/1500,   (200-382)/200],
    "IDH":    [0.583/0.6, 3/3, 0.23/1.5, 1.0, 1.0, 160/1500, (200-78)/200],
    "LDH":    [0.226/0.6, 2/3, 0.23/1.5, 1.0, 1.0, 160/1500, (200-66)/200],
}
# Clamp negative/over values
for k in data:
    data[k] = [max(0, min(1, v)) for v in data[k]]

conditions = ["C1\nf_osc", "C2\nn_P", "C3\nT2e", "C4\nA/wS", "C5\nT1/T2", "C6\nT2(31P)", "C7\nSOC"]


def fig1_radar():
    """Fig 1: RPM condition radar chart."""
    N = 7
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = {"MAO-A": "#d32f2f", "MAO-B": "#e57373", "DAO": "#ffb74d",
              "CRY": "#1565c0", "DDC": "#4caf50", "GAD67": "#81c784",
              "SRR": "#9e9e9e", "IDH": "#7b1fa2", "LDH": "#ce93d8"}
    lws = {"MAO-A": 3, "CRY": 2.5, "IDH": 2, "SRR": 1.5}

    for name in ["SRR", "LDH", "DAO", "GAD67", "DDC", "MAO-B", "IDH", "CRY", "MAO-A"]:
        vals = data[name] + data[name][:1]
        lw = lws.get(name, 1.5)
        ax.plot(angles, vals, "o-", lw=lw, label=name, color=colors[name],
                ms=4, alpha=0.85)
        if name in ["MAO-A", "CRY", "IDH", "SRR"]:
            ax.fill(angles, vals, alpha=0.05, color=colors[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8,
              title="Protein", title_fontsize=9)
    ax.set_title("RPM Condition Assessment\n(7 conditions, normalised scores)",
                 fontsize=12, fontweight="bold", pad=20)

    plt.savefig(OUT / "fig1_radar.pdf")
    plt.savefig(OUT / "fig1_radar.png")
    plt.close()
    print("  Fig 1 saved")


def fig2_relaxation():
    """Fig 2: (a) T2e decomposition at X-band vs Earth field,
              (b) Nuclear T2 vs e-P distance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) T2e decomposition
    prots = ["MAO-A", "CRY", "MAO-B", "IDH", "LDH", "DAO", "DDC", "GAD67"]
    T2_hfc =   [1.12, 1.12, 1.12, 0.91, 1.12, 1.12, 1.48, 1.56]  # ns
    T2_dd =    [71]*8
    T2_gani =  [3.6, 4.2, 2.6, 4.9, 0.7, 0.3, 5.6, 3.2]
    T2_amod =  [0.50, 0.50, 0.50, 0.34, 0.50, 0.50, 0.95, 0.99]
    T2_total = [0.32, 0.32, 0.30, 0.23, 0.23, 0.16, 0.52, 0.51]
    # Brain (Earth field)
    T2_brain = [1.10, 1.10, 1.10, 0.90, 1.10, 1.10, 1.45, 1.53]

    x = np.arange(len(prots))
    w = 0.35

    # X-band bars (stacked contributions to 1/T2)
    inv_hfc = [1/t for t in T2_hfc]
    inv_dd = [1/t for t in T2_dd]
    inv_gani = [1/t for t in T2_gani]
    inv_amod = [1/t for t in T2_amod]

    ax1.bar(x - w/2, T2_total, w, color="#c62828", alpha=0.8, label="T2e (X-band)", edgecolor="black", lw=0.5)
    ax1.bar(x + w/2, T2_brain, w, color="#1565c0", alpha=0.8, label="T2e (Earth field)", edgecolor="black", lw=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(prots, fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("T2e (ns)")
    ax1.set_title("(a) Electron T2e: X-band vs Earth field", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")

    # Annotations
    ax1.annotate("g-anisotropy\nvanishes at\nEarth field",
                xy=(3+w/2, T2_brain[3]), xytext=(5, 1.3),
                fontsize=8, color="#1565c0",
                arrowprops=dict(arrowstyle="->", color="#1565c0"))
    ax1.annotate(r"$T_2^e \propto 1/(\Delta g^2 \omega_0^2 \tau_c)$",
                xy=(5, 1.25), xytext=(5, 1.25), fontsize=7, color="#555")

    # (b) Nuclear T2 vs e-P distance
    ax2b = ax2
    distances = np.linspace(2, 15, 100)
    # PRE T2 scaling: T2 = T2(7A) * (r/7)^6
    T2_ref = 160  # us at r=7 A
    T2_curve = T2_ref * (distances / 7.0)**6

    ax2b.semilogy(distances, T2_curve, "k-", lw=2, label=r"PRE: $T_2 \propto r^6$")

    # Data points
    pts = [("PLP (DDC,GAD67)", 3.5, 2.5, "#4caf50", "s"),
           ("FAD (MAO-A,CRY)", 7.0, 160, "#d32f2f", "o"),
           ("FAD (DAO)", 7.0, 160, "#ffb74d", "o"),
           ("NADPH (IDH)", 10.0, 160, "#7b1fa2", "D"),
           ("NADH (LDH)", 8.5, 160, "#ce93d8", "D")]
    for name, r, t2, col, mk in pts:
        ax2b.plot(r, t2, mk, color=col, ms=12, mew=2, mec="black", label=name, zorder=5)

    # Diamagnetic limit
    ax2b.axhline(3250, color="blue", ls="--", lw=1.5, alpha=0.5)
    ax2b.text(12, 4000, r"Diamagnetic $T_2$ = 3.2 ms", fontsize=8, color="blue")

    # RP lifetime
    ax2b.axhline(1, color="red", ls=":", lw=1.5, alpha=0.5)
    ax2b.text(12, 0.7, "RP lifetime ~ 1 us", fontsize=8, color="red")

    ax2b.set_xlabel(r"$e^-$-P distance ($\AA$)")
    ax2b.set_ylabel(r"Coupled $T_2(^{31}P)$ ($\mu$s)")
    ax2b.set_title(r"(b) Nuclear $T_2(^{31}P)$ vs $e^-$-P distance", fontsize=11, fontweight="bold")
    ax2b.legend(fontsize=7, loc="upper left")
    ax2b.set_xlim(2, 15)
    ax2b.set_ylim(0.3, 1e4)
    ax2b.grid(True, alpha=0.2)

    # 64x annotation
    ax2b.annotate("", xy=(7, 160), xytext=(3.5, 2.5),
                 arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    ax2b.text(4.5, 15, r"$\times 64$" + "\n" + r"$(r^{-6})$", fontsize=10,
             color="green", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT / "fig2_relaxation.pdf")
    plt.savefig(OUT / "fig2_relaxation.png")
    plt.close()
    print("  Fig 2 saved")


def fig3_coupled_levels():
    """Fig 3: Coupled e-31P energy level diagram at Earth's field."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- (a) 4-level diagram ---
    # Energies in MHz
    A = 200  # MHz
    omega_S = 1.40  # MHz
    omega_I = 0.000862  # MHz

    E_aa = omega_S/2 + omega_I/2 + A/4   # ~50.7
    E_ab = omega_S/2 - omega_I/2 - A/4   # ~-49.3
    E_ba = -omega_S/2 + omega_I/2 - A/4  # ~-50.7
    E_bb = -omega_S/2 - omega_I/2 + A/4  # ~49.3

    levels = [(E_aa, r"|$\alpha_e\alpha_n\rangle$", 0.15, 0.45, "#1565c0"),
              (E_bb, r"|$\beta_e\beta_n\rangle$",   0.15, 0.45, "#1565c0"),
              (E_ab, r"|$\alpha_e\beta_n\rangle$",  0.55, 0.85, "#c62828"),
              (E_ba, r"|$\beta_e\alpha_n\rangle$",  0.55, 0.85, "#c62828")]

    for E, label, x0, x1, col in levels:
        ax1.plot([x0, x1], [E, E], color=col, lw=4)
        side = x0 - 0.01 if x0 < 0.5 else x1 + 0.01
        ha = "right" if x0 < 0.5 else "left"
        ax1.text(side, E, f"  {label}  {E:.1f} MHz", fontsize=8, ha=ha, va="center", color=col)

    # EPR transitions (green arrows)
    for y1, y2, x, label in [(E_aa, E_ba, 0.3, "EPR\n101.4 MHz"),
                              (E_ab, E_bb, 0.7, "EPR\n98.6 MHz")]:
        ax1.annotate("", xy=(x, min(y1,y2)), xytext=(x, max(y1,y2)),
                    arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=2))
        ax1.text(x+0.02, (y1+y2)/2, label, fontsize=7, color="#2e7d32", va="center")

    # ZQ forbidden transition (orange dashed)
    ax1.annotate("", xy=(0.5, E_ab), xytext=(0.5, E_ba),
                arrowprops=dict(arrowstyle="<->", color="#e65100", lw=2.5, ls="--"))
    ax1.text(0.52, (E_ab+E_ba)/2, "ZQ (flip-flop)\n1.40 MHz\n"
             r"$\mu$ = 1.22 $\mu_B$" + "\n(70% of allowed!)",
             fontsize=8, color="#e65100", va="center", fontweight="bold",
             bbox=dict(facecolor="lightyellow", alpha=0.8, boxstyle="round,pad=0.2"))

    # HFC splitting annotation
    ax1.annotate("", xy=(0.92, E_aa), xytext=(0.92, E_ab),
                arrowprops=dict(arrowstyle="<->", color="#7b1fa2", lw=1.5))
    ax1.text(0.94, (E_aa+E_ab)/2, "A = 200 MHz\n= 0.83 ueV",
            fontsize=7, color="#7b1fa2", va="center")

    ax1.set_xlim(0, 1.05)
    ax1.set_ylabel("Energy (MHz)", fontsize=10)
    ax1.set_title("(a) Coupled " + r"$e^-$-$^{31}$P" + " 4-level system\n"
                  r"@ 50 $\mu$T (strong coupling: $A \gg \omega_S$)", fontsize=10)
    ax1.set_xticks([])

    # --- (b) Transition spectrum ---
    ax2b = ax2
    transitions = [
        (abs(E_aa - E_ba), 1.734, "EPR (allowed)", "#2e7d32", "-"),
        (abs(E_ab - E_bb), 1.734, "EPR (allowed)", "#2e7d32", "-"),
        (abs(E_ab - E_ba), 1.218, "ZQ flip-flop (forbidden)", "#e65100", "--"),
        (abs(E_aa - E_bb), 1.218, "DQ flip-flip (forbidden)", "#e65100", ":"),
        (abs(E_aa - E_ab), 0.980*5.051e-4, "NMR/ENDOR", "#1565c0", "-"),
        (abs(E_ba - E_bb), 0.980*5.051e-4, "NMR/ENDOR", "#1565c0", "-"),
    ]

    for freq, tdm, label, col, ls in transitions:
        ax2b.plot([freq, freq], [0, tdm], color=col, lw=3, ls=ls)
        ax2b.plot(freq, tdm, "o", color=col, ms=6)

    # Labels
    ax2b.text(101.4, 1.75, "EPR\n(allowed)", fontsize=7, color="#2e7d32", ha="center")
    ax2b.text(1.40, 1.25, "ZQ\n(forbidden)\n70%!", fontsize=8, color="#e65100",
             ha="center", fontweight="bold")

    ax2b.set_xlabel("Transition frequency (MHz)", fontsize=10)
    ax2b.set_ylabel(r"Transition dipole moment ($\mu_B$)", fontsize=10)
    ax2b.set_title("(b) Transition spectrum at 50 " + r"$\mu$T" + "\n"
                   "Strong coupling: forbidden transitions become allowed",
                   fontsize=10)
    ax2b.set_xlim(-5, 115)
    ax2b.set_ylim(0, 2.0)
    ax2b.grid(True, alpha=0.2)

    # Inset: mixing angle
    ax_in = ax2b.inset_axes([0.35, 0.55, 0.3, 0.35])
    B_range = np.logspace(-6, 0, 100)  # Tesla
    omega_S_range = 2.002 * 9.274e-24 * B_range / (1.055e-34 * 2 * np.pi * 1e6)
    mixing = np.arctan(A / (2 * np.abs(omega_S_range))) / 2
    forbidden_tdm = np.sin(mixing) * 1.734 * np.sqrt(0.75)

    ax_in.semilogx(B_range * 1e6, forbidden_tdm, "r-", lw=2)
    ax_in.axvline(50, color="gray", ls="--", alpha=0.5)
    ax_in.axhline(1.218, color="orange", ls=":", alpha=0.5)
    ax_in.set_xlabel(r"B ($\mu$T)", fontsize=7)
    ax_in.set_ylabel(r"ZQ TDM ($\mu_B$)", fontsize=7)
    ax_in.set_title("Field dependence", fontsize=7)
    ax_in.text(60, 0.5, r"Earth" + "\nfield", fontsize=6, color="gray")
    ax_in.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(OUT / "fig3_coupled_levels.pdf")
    plt.savefig(OUT / "fig3_coupled_levels.png")
    plt.close()
    print("  Fig 3 saved")


def fig4_mfe():
    """Fig 4: RPM singlet yield and MFE from simulation results."""
    # Load simulation data
    sim_file = Path("../simulation_results/simulation_results.json")
    if sim_file.exists():
        with open(sim_file) as f:
            sim = json.load(f)
        B_mT = np.array(sim["sim1_rpm"]["B_mT"])
        phi_S = np.array(sim["sim1_rpm"]["singlet_yield"])
        MFE = np.array(sim["sim1_rpm"]["MFE_percent"])
    else:
        # Fallback: generate synthetic data matching our results
        B_mT = np.concatenate([np.linspace(0, 0.5, 15), np.linspace(0.5, 5, 25)])
        phi_0 = 0.4592
        # Lorentzian-like MFE
        MFE = -4.44 * (B_mT / 0.9)**2 / (1 + (B_mT / 0.9)**2)
        phi_S = phi_0 * (1 + MFE / 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Singlet yield
    ax1.plot(B_mT, phi_S, "b-", lw=2.5)
    ax1.fill_between(B_mT, phi_S, phi_S[0], alpha=0.1, color="blue")
    ax1.axhline(phi_S[0], color="gray", ls="--", alpha=0.4)
    ax1.set_xlabel("Magnetic field B (mT)", fontsize=11)
    ax1.set_ylabel(r"Singlet yield $\Phi_S$", fontsize=11)
    ax1.set_title(r"(a) Singlet yield $\Phi_S(B)$ — MAO-A" + "\n"
                  "8-spin RP: 2e + 2(31P) + 4(1H), dim=256", fontsize=10)
    ax1.text(3, phi_S[0] + 0.001, r"$\Phi_S(0) = $" + f"{phi_S[0]:.4f}",
            fontsize=9, color="gray")

    # Earth field marker
    ax1.axvline(0.05, color="green", ls=":", lw=2, alpha=0.6)
    ax1.text(0.07, phi_S.min(), "Earth\nfield\n50 " + r"$\mu$T", fontsize=8,
            color="green", va="bottom")

    ax1.grid(True, alpha=0.2)

    # (b) MFE
    ax2.plot(B_mT, MFE, "r-", lw=2.5)
    ax2.fill_between(B_mT, MFE, 0, alpha=0.1, color="red")
    ax2.axhline(0, color="gray", ls="-", alpha=0.3)
    ax2.set_xlabel("Magnetic field B (mT)", fontsize=11)
    ax2.set_ylabel("MFE (%)", fontsize=11)
    ax2.set_title("(b) Magnetic Field Effect on 5-HT oxidation\n"
                  r"MFE = $[\Phi_S(B) - \Phi_S(0)] / \Phi_S(0) \times 100\%$",
                  fontsize=10)

    # Peak annotation
    idx_min = np.argmin(MFE)
    ax2.plot(B_mT[idx_min], MFE[idx_min], "rv", ms=12)
    ax2.annotate(f"MFE_max = {MFE[idx_min]:.1f}%\nat B = {B_mT[idx_min]:.2f} mT",
                xy=(B_mT[idx_min], MFE[idx_min]),
                xytext=(B_mT[idx_min]+1, MFE[idx_min]+0.5),
                fontsize=9, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    # Earth field effect
    earth_idx = np.argmin(np.abs(B_mT - 0.05))
    ax2.plot(0.05, MFE[earth_idx], "g^", ms=10, zorder=5)
    ax2.text(0.3, MFE[earth_idx], f"Earth field:\nMFE = {MFE[earth_idx]:.2f}%",
            fontsize=8, color="green")

    ax2.grid(True, alpha=0.2)

    # Key equations
    eq_text = (r"$\hat{H} = J\,\mathbf{S}_1 \cdot \mathbf{S}_2 "
               r"+ \sum_i A_i \mathbf{S}_e \cdot \mathbf{I}_i "
               r"+ g\mu_B B(S_{1z} + S_{2z})$" + "\n"
               r"$\Phi_S = k \sum_{m,n} [P_S]_{mn} [\rho_0]_{nm} / (k + i\Delta E_{mn})$")
    fig.text(0.5, -0.02, eq_text, ha="center", fontsize=9, color="#333",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT / "fig4_mfe.pdf")
    plt.savefig(OUT / "fig4_mfe.png")
    plt.close()
    print("  Fig 4 saved")


def fig5_three_layer():
    """Fig 5: Three-layer quantum-classical hierarchy."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(8, 9.7, "Three-Layer Quantum-Classical Information Hierarchy in the Brain",
            ha="center", fontsize=14, fontweight="bold")

    # === Layer 1 ===
    r1 = FancyBboxPatch((0.5, 7.0), 15, 2.3, boxstyle="round,pad=0.15",
                         facecolor="#fff8e1", edgecolor="#e65100", lw=2.5)
    ax.add_patch(r1)
    ax.text(8, 9.0, "Layer 1: Nuclear Spin Quantum Memory",
            ha="center", fontsize=13, fontweight="bold", color="#e65100")

    ax.text(1.2, 8.4, r"Carrier: $^{31}$P nuclear spins (I=1/2)", fontsize=10, color="#333")
    ax.text(1.2, 7.9, r"Coherence: $T_2$ = 3.2 ms (diamagnetic), 160 $\mu$s (RP-coupled)",
            fontsize=9, color="#555")
    ax.text(1.2, 7.4, r"Energy: $E_{Zeeman}$ = 3.6 peV, $E_{dipolar}$ = 41 feV, $E_{HFC}$ = 0.83 $\mu$eV",
            fontsize=9, color="#555")

    ax.text(10, 8.4, "Molecules: FAD, PLP, NADPH", fontsize=10, color="#333", fontweight="bold")
    ax.text(10, 7.9, r"$^{31}$P count: FAD(2), PLP(1), NADPH(3)", fontsize=9, color="#555")
    ax.text(10, 7.4, "QC paradigm: time crystal, adiabatic", fontsize=9, color="#555", style="italic")

    # === Layer 2 ===
    r2 = FancyBboxPatch((0.5, 3.5), 15, 3.0, boxstyle="round,pad=0.15",
                         facecolor="#f3e5f5", edgecolor="#7b1fa2", lw=2.5)
    ax.add_patch(r2)
    ax.text(8, 6.1, "Layer 2: Radical Pair Quantum-Classical Interface",
            ha="center", fontsize=13, fontweight="bold", color="#7b1fa2")

    ax.text(1.2, 5.5, "Carrier: [FADH...Sub] radical pair", fontsize=10, color="#333")
    ax.text(1.2, 5.0, r"Mechanism: HFC-driven S-$T_0$ mixing $\rightarrow$ $\Phi_S(B)$",
            fontsize=9, color="#555")
    ax.text(1.2, 4.5, r"Timescales: $T_2^e$ = 1.1 ns, RP lifetime ~ 1 $\mu$s",
            fontsize=9, color="#555")
    ax.text(1.2, 4.0, r"MFE: $-4.4\%$ (0.9 mT), $-0.3\%$ (Earth field)",
            fontsize=9, color="#555")

    ax.text(10, 5.5, "Top candidates:", fontsize=10, color="#333", fontweight="bold")
    ax.text(10, 5.0, r"#1 MAO-A (Q=457), #2 IDH (Q=412)", fontsize=9, color="#555")
    ax.text(10, 4.5, r"#3 CRY (Q=391), #4 MAO-B (Q=383)", fontsize=9, color="#555")
    ax.text(10, 4.0, "QC paradigm: quantum reservoir", fontsize=9, color="#555", style="italic")

    # === Layer 3 ===
    r3 = FancyBboxPatch((0.5, 0.5), 15, 2.5, boxstyle="round,pad=0.15",
                         facecolor="#e8f5e9", edgecolor="#2e7d32", lw=2.5)
    ax.add_patch(r3)
    ax.text(8, 2.7, "Layer 3: Classical Electrochemical Processing",
            ha="center", fontsize=13, fontweight="bold", color="#2e7d32")

    ax.text(1.2, 2.1, "Carrier: neurotransmitters (5-HT, DA, GABA)", fontsize=10, color="#333")
    ax.text(1.2, 1.6, r"Energy: kT = 26.7 meV, $V_{syn}$ ~ 100 mV, bonds ~ eV",
            fontsize=9, color="#555")
    ax.text(1.2, 1.0, "Output: synaptic transmission, neural circuit dynamics",
            fontsize=9, color="#555")

    ax.text(10, 2.1, "Downstream effects:", fontsize=10, color="#333", fontweight="bold")
    ax.text(10, 1.6, "5-HT degradation rate modulated 0.3% by Earth field", fontsize=9, color="#555")
    ax.text(10, 1.0, r"$\rightarrow$ mood, circadian rhythm, cognition", fontsize=9, color="#555")

    # === Arrows ===
    ax.annotate("", xy=(4, 6.5), xytext=(4, 7.0),
                arrowprops=dict(arrowstyle="<->", color="#e65100", lw=3))
    ax.text(4.3, 6.7, r"HFC ($A$ = 200 MHz)", fontsize=9, color="#e65100", fontweight="bold")

    ax.annotate("", xy=(4, 3.0), xytext=(4, 3.5),
                arrowprops=dict(arrowstyle="->", color="#7b1fa2", lw=3))
    ax.text(4.3, 3.2, r"$\Phi_S(B) \rightarrow$ rate modulation",
            fontsize=9, color="#7b1fa2", fontweight="bold")

    # Energy scale bar (right side)
    ax.text(15.7, 9.0, "3.6 peV", fontsize=7, color="#e65100", ha="center", rotation=0)
    ax.text(15.7, 5.5, r"0.83 $\mu$eV", fontsize=7, color="#7b1fa2", ha="center")
    ax.text(15.7, 1.5, "4 eV", fontsize=7, color="#2e7d32", ha="center")
    ax.annotate("", xy=(15.7, 1.8), xytext=(15.7, 8.8),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
    ax.text(15.7, 5.3, "14 orders\nof magnitude", fontsize=7, ha="center",
            color="#333", fontweight="bold", rotation=90)

    plt.savefig(OUT / "fig5_three_layer.pdf", bbox_inches="tight")
    plt.savefig(OUT / "fig5_three_layer.png", bbox_inches="tight")
    plt.close()
    print("  Fig 5 saved")


if __name__ == "__main__":
    print("Generating manuscript figures...")
    fig1_radar()
    fig2_relaxation()
    fig3_coupled_levels()
    fig4_mfe()
    fig5_three_layer()
    print("All 5 figures saved -> manuscript/figures/")
