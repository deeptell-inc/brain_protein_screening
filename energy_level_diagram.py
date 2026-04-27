#!/usr/bin/env python3
"""3レイヤー量子脳仮説 — 情報担体分子のエネルギー準位図

各レイヤーの量子/古典情報担体となる分子の超微細構造と
エネルギー準位を脳内温度 (310 K) で図示する。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
from pathlib import Path

OUT_DIR = Path("simulation_results")
OUT_DIR.mkdir(exist_ok=True)

# 物理定数
KB = 8.617e-5       # eV/K
KT_310 = KB * 310   # 26.7 meV
HBAR_EV_S = 6.582e-16
MU_B_EV_T = 5.788e-5
MU_N_EV_T = 3.152e-8
G_E = 2.002
GAMMA_P = 17.235e6  # Hz/T
GAMMA_H = 42.577e6  # Hz/T
B_EARTH = 50e-6     # T

# ═══════════════════════════════════════════════════════════════
# Figure 1: 全体エネルギースケール比較
# ═══════════════════════════════════════════════════════════════

def fig1_energy_scales():
    """3レイヤーのエネルギースケールを1つの図で比較。"""
    fig, ax = plt.subplots(figsize=(16, 10))

    # エネルギースケール (eV)
    scales = [
        # (名前, エネルギー eV, レイヤー, 色, 分子)
        ("kT (310K)", KT_310, "thermal", "#d32f2f", "—"),
        ("S₀→S₁ (FAD)", 2.76, "L3/L2", "#1565c0", "FAD"),
        ("HOMO-LUMO\n(MAO-A/FAD)", 1.25, "L2", "#1565c0", "FAD"),
        ("S-T gap\n(MAO-A)", 1.30, "L2", "#7b1fa2", "FADH•+Sub•"),
        ("C-N bond\n(5-HT)", 3.0, "L3", "#2e7d32", "5-HT"),
        ("H-bond\n(protein)", 0.15, "L3", "#2e7d32", "H₂O"),
        ("HFC ³¹P\n(A=200MHz)", 200e6 * 4.136e-15, "L1/L2", "#e65100", "FADH•-³¹P"),
        ("e⁻ Zeeman\n@50μT", G_E * MU_B_EV_T * B_EARTH, "L2", "#7b1fa2", "FADH•"),
        ("³¹P Zeeman\n@50μT", GAMMA_P * B_EARTH * 4.136e-15, "L1", "#e65100", "³¹PO₄"),
        ("¹H Zeeman\n@50μT", GAMMA_H * B_EARTH * 4.136e-15, "L1", "#ff8f00", "¹H"),
        ("J (exchange)\nestimated", 1e-9, "L2", "#7b1fa2", "RP pair"),
        ("dipolar ³¹P-³¹P\n(0.01 MHz)", 0.01e6 * 4.136e-15, "L1", "#e65100", "³¹P-³¹P"),
    ]

    # ソート
    scales.sort(key=lambda x: x[1], reverse=True)

    y_positions = np.arange(len(scales))
    colors = [s[3] for s in scales]
    energies = [s[1] for s in scales]
    names = [s[0] for s in scales]
    layers = [s[2] for s in scales]
    molecules = [s[4] for s in scales]

    bars = ax.barh(y_positions, np.log10(energies), color=colors, alpha=0.7,
                   edgecolor="black", height=0.6)

    # ラベル
    for i, (name, E, layer, col, mol) in enumerate(scales):
        if E >= 1e-3:
            e_str = f"{E*1e3:.1f} meV"
        elif E >= 1e-6:
            e_str = f"{E*1e6:.2f} μeV"
        elif E >= 1e-9:
            e_str = f"{E*1e9:.2f} neV"
        else:
            e_str = f"{E*1e12:.2f} peV"
        ax.text(np.log10(E) + 0.15, i, f"{e_str}  [{layer}] {mol}",
                va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("log₁₀(Energy / eV)", fontsize=12)
    ax.set_title("3レイヤー量子脳仮説 — エネルギースケール全体図 (310 K)\n"
                 "Layer 1: 核スピン量子メモリ | Layer 2: ラジカルペアインターフェース | "
                 "Layer 3: 古典電気化学", fontsize=11)

    # kTの線
    ax.axvline(np.log10(KT_310), color="red", ls="--", lw=2, alpha=0.5)
    ax.text(np.log10(KT_310) + 0.1, len(scales) - 0.5,
            f"kT = {KT_310*1e3:.1f} meV", color="red", fontsize=10)

    # レイヤー領域
    ax.axhspan(8.5, 11.5, alpha=0.05, color="orange", label="Layer 1")
    ax.axhspan(3.5, 8.5, alpha=0.05, color="purple", label="Layer 2")
    ax.axhspan(-0.5, 3.5, alpha=0.05, color="green", label="Layer 3")

    ax.set_xlim(-13, 2)
    ax.grid(True, alpha=0.2, axis="x")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_energy_scales.png", dpi=150)
    plt.close()
    print("  fig1_energy_scales.png 保存")


# ═══════════════════════════════════════════════════════════════
# Figure 2: Layer 1 — ³¹P核スピンの超微細構造
# ═══════════════════════════════════════════════════════════════

def fig2_layer1_nuclear():
    """³¹P核スピンのZeeman分裂と双極子結合 @地磁気。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # --- (a) 単一³¹P @地磁気 ---
    ax = axes[0]
    B = B_EARTH
    nu_P = GAMMA_P * B  # Hz
    E_split = nu_P * 4.136e-15  # eV

    # エネルギー準位
    y_up = +E_split / 2
    y_down = -E_split / 2

    ax.plot([0.2, 0.8], [y_up * 1e12, y_up * 1e12], "b-", lw=3)
    ax.plot([0.2, 0.8], [y_down * 1e12, y_down * 1e12], "r-", lw=3)
    ax.annotate("", xy=(0.85, y_up * 1e12), xytext=(0.85, y_down * 1e12),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(0.9, 0, f"Δ = {nu_P:.1f} Hz\n= {E_split*1e12:.2f} peV",
            fontsize=9, va="center")
    ax.text(0.5, y_up * 1e12 + 0.3, "|m_I = +½⟩", ha="center", fontsize=10, color="blue")
    ax.text(0.5, y_down * 1e12 - 0.3, "|m_I = −½⟩", ha="center", fontsize=10, color="red")

    # kT比較
    ax.axhline(KT_310 * 1e12, color="red", ls="--", alpha=0.3)
    ax.text(0.05, KT_310 * 1e12, f"kT = {KT_310*1e12:.0e} peV\n(7.4×10⁹倍大きい)",
            fontsize=7, color="red", va="bottom")

    ax.set_xlim(0, 1.3)
    ax.set_ylabel("Energy (peV)", fontsize=11)
    ax.set_title("(a) 単一 ³¹P (I=½)\n@地磁気 50μT\n分子: PO₄ in FAD", fontsize=10)
    ax.set_xticks([])

    # --- (b) 結合2×³¹P (双極子結合) ---
    ax = axes[1]
    J_dd = 0.01e6 * 4.136e-15  # 0.01 MHz → eV

    # 2スピン系: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩
    E_uu = +nu_P * 4.136e-15 + J_dd / 4
    E_ud = +J_dd / 4  # ≈ 0 (Zeeman cancels)
    E_du = -J_dd / 4
    E_dd = -nu_P * 4.136e-15 + J_dd / 4

    # 一重項/三重項分裂
    E_T_plus = E_uu
    E_T_0 = (E_ud + E_du) / 2 + J_dd / 2
    E_T_minus = E_dd
    E_S = (E_ud + E_du) / 2 - 3 * J_dd / 4

    levels = [
        (E_T_plus * 1e15, "|T₊⟩ = |↑↑⟩", "blue", 0.15, 0.55),
        (E_T_0 * 1e15, "|T₀⟩", "blue", 0.15, 0.55),
        (E_S * 1e15, "|S⟩ = (|↑↓⟩−|↓↑⟩)/√2", "red", 0.6, 1.0),
        (E_T_minus * 1e15, "|T₋⟩ = |↓↓⟩", "blue", 0.15, 0.55),
    ]
    levels.sort(key=lambda x: -x[0])

    for i, (E, label, col, x0, x1) in enumerate(levels):
        ax.plot([x0, x1], [E, E], color=col, lw=3)
        side = 1.05 if col == "red" else -0.02
        ha = "left" if col == "red" else "right"
        ax.text(side, E, f"{label}\n{E:.3f} feV", fontsize=8, va="center",
                ha=ha, color=col)

    # S-T分裂
    ax.annotate("", xy=(0.55, E_T_0 * 1e15), xytext=(0.55, E_S * 1e15),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(0.57, (E_T_0 * 1e15 + E_S * 1e15) / 2,
            f"J_dd = {J_dd/4.136e-15*1e-6:.3f} MHz\n核スピン一重項-三重項分裂",
            fontsize=7, color="green", va="center")

    ax.set_ylabel("Energy (feV)", fontsize=11)
    ax.set_title("(b) 結合 ³¹P-³¹P (双極子)\n@地磁気 50μT\n分子: FADリン酸鎖", fontsize=10)
    ax.set_xticks([])

    # --- (c) ¹⁴N 四極子準位 ---
    ax = axes[2]
    # ¹⁴N: I=1, e²qQ/h ≈ 3 MHz
    eqQ = 3.0e6  # Hz
    E_qQ = eqQ * 4.136e-15  # eV

    # 3準位: m_I = +1, 0, -1
    E_p1 = +E_qQ / 4  # 簡略化
    E_0 = -E_qQ / 2
    E_m1 = +E_qQ / 4

    for E, label, y_off in [
        (E_p1, "|m_I = +1⟩", 0.5),
        (E_0, "|m_I = 0⟩", -0.5),
        (E_m1, "|m_I = −1⟩", 0.5),
    ]:
        ax.plot([0.2, 0.8], [E * 1e9, E * 1e9], "b-", lw=3)
        ax.text(0.82, E * 1e9 + y_off * 0.3, label, fontsize=9, va="center")

    ax.annotate("", xy=(0.15, E_p1 * 1e9), xytext=(0.15, E_0 * 1e9),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(0.02, (E_p1 + E_0) / 2 * 1e9,
            f"e²qQ/h\n= {eqQ/1e6:.0f} MHz\n= {E_qQ*1e9:.2f} neV\nT₁ = 0.73 μs",
            fontsize=8, color="green", va="center")

    ax.set_ylabel("Energy (neV)", fontsize=11)
    ax.set_title("(c) ¹⁴N 四極子分裂 (I=1)\nイソアロキサジン環内\n分子: FADフラビン環",
                 fontsize=10)
    ax.set_xticks([])

    fig.suptitle("Layer 1: 核スピン量子メモリの担体分子とエネルギー準位 (310 K)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_layer1_nuclear.png", dpi=150)
    plt.close()
    print("  fig2_layer1_nuclear.png 保存")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Layer 2 — ラジカルペアの超微細構造
# ═══════════════════════════════════════════════════════════════

def fig3_layer2_radical_pair():
    """結合 e⁻-³¹P 4準位系と RPM のエネルギー準位。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # --- (a) 結合 e⁻-³¹P 4準位系 @地磁気 ---
    ax = axes[0]
    omega_S = G_E * MU_B_EV_T * B_EARTH  # eV
    omega_I = GAMMA_P * B_EARTH * 4.136e-15  # eV
    A = 200e6 * 4.136e-15  # 200 MHz → eV

    # 4準位 (MHz単位で表示)
    E_aa = (+omega_S / 2 + omega_I / 2 + A / 4) / (4.136e-15 * 1e6)
    E_ab = (+omega_S / 2 - omega_I / 2 - A / 4) / (4.136e-15 * 1e6)
    E_ba = (-omega_S / 2 + omega_I / 2 - A / 4) / (4.136e-15 * 1e6)
    E_bb = (-omega_S / 2 - omega_I / 2 + A / 4) / (4.136e-15 * 1e6)

    levels = [
        (E_aa, "|α_e α_n⟩", "#1565c0", 0.1, 0.45),
        (E_bb, "|β_e β_n⟩", "#1565c0", 0.1, 0.45),
        (E_ab, "|α_e β_n⟩", "#c62828", 0.55, 0.9),
        (E_ba, "|β_e α_n⟩", "#c62828", 0.55, 0.9),
    ]
    levels.sort(key=lambda x: -x[0])

    for E, label, col, x0, x1 in levels:
        ax.plot([x0, x1], [E, E], color=col, lw=3)
        if x0 < 0.5:
            ax.text(x0 - 0.02, E, f"{label}\n{E:.1f} MHz", fontsize=8,
                    ha="right", va="center", color=col)
        else:
            ax.text(x1 + 0.02, E, f"{label}\n{E:.1f} MHz", fontsize=8,
                    ha="left", va="center", color=col)

    # EPR遷移 (許容)
    ax.annotate("", xy=(0.3, E_ba), xytext=(0.3, E_aa),
                arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    ax.text(0.32, (E_aa + E_ba) / 2, f"EPR₁\n{abs(E_aa-E_ba):.1f} MHz\nμ=1.73 μ_B",
            fontsize=7, color="green", va="center")

    # 禁制遷移 (ZQ)
    ax.annotate("", xy=(0.5, E_ab), xytext=(0.5, E_ba),
                arrowprops=dict(arrowstyle="<->", color="orange", lw=2, ls="--"))
    ax.text(0.52, (E_ab + E_ba) / 2, f"ZQ (flip-flop)\n{abs(E_ab-E_ba):.1f} MHz\n"
            f"μ=1.22 μ_B\n(許容の70%!)",
            fontsize=7, color="orange", va="center")

    # HFC分裂
    ax.annotate("", xy=(0.95, E_aa), xytext=(0.95, E_ab),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))
    ax.text(0.97, (E_aa + E_ab) / 2, f"A(³¹P)\n={200} MHz\n=0.83 μeV",
            fontsize=7, color="purple", va="center")

    ax.set_ylabel("Energy (MHz)", fontsize=11)
    ax.set_title("(a) 結合 e⁻-³¹P 4準位系\n@地磁気 50μT (強結合極限: A≫ω_S)\n"
                 "分子: FADH• セミキノンラジカル", fontsize=9)
    ax.set_xticks([])

    # --- (b) ラジカルペア S-T 準位構造 ---
    ax = axes[1]

    # RP: 2電子スピンの一重項/三重項
    # J ≈ 0 (分離型RP) → S と T₀ がほぼ縮退
    J = 0.001  # MHz (very small exchange)

    # @地磁気: ω_S = 1.4 MHz
    omega_S_MHz = G_E * MU_B_EV_T * B_EARTH / (4.136e-15 * 1e6)

    E_T_plus = +omega_S_MHz
    E_T_0 = -J
    E_S = +J
    E_T_minus = -omega_S_MHz

    # 準位描画
    ax.plot([0.1, 0.5], [E_T_plus, E_T_plus], "b-", lw=3)
    ax.plot([0.1, 0.5], [E_T_0, E_T_0], "b-", lw=3)
    ax.plot([0.6, 1.0], [E_S, E_S], "r-", lw=3)
    ax.plot([0.1, 0.5], [E_T_minus, E_T_minus], "b-", lw=3)

    ax.text(0.3, E_T_plus + 0.08, "|T₊⟩", ha="center", fontsize=10, color="blue")
    ax.text(0.3, E_T_0 + 0.08, "|T₀⟩", ha="center", fontsize=10, color="blue")
    ax.text(0.8, E_S + 0.08, "|S⟩", ha="center", fontsize=10, color="red")
    ax.text(0.3, E_T_minus - 0.12, "|T₋⟩", ha="center", fontsize=10, color="blue")

    # S-T₀ 混合 (HFCによる)
    ax.annotate("", xy=(0.55, E_S), xytext=(0.5, E_T_0),
                arrowprops=dict(arrowstyle="<->", color="orange", lw=2,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(0.52, (E_S + E_T_0) / 2 + 0.15,
            "HFC-driven\nS↔T₀ mixing\n(A=200 MHz)",
            fontsize=8, color="orange", ha="center")

    # Zeeman分裂
    ax.annotate("", xy=(0.05, E_T_plus), xytext=(0.05, E_T_minus),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(0.03, 0, f"2gμ_BB\n={2*omega_S_MHz:.2f}\nMHz",
            fontsize=7, color="green", va="center", ha="right")

    # 反応生成物
    ax.annotate("反応生成物 A\n(一重項)", xy=(1.05, E_S),
                fontsize=8, color="red", va="center")
    ax.annotate("反応生成物 B\n(三重項)", xy=(0.55, E_T_minus - 0.2),
                fontsize=8, color="blue", va="center")

    ax.set_ylabel("Energy (MHz)", fontsize=11)
    ax.set_ylim(-2, 2)
    ax.set_title("(b) ラジカルペア S-T 準位\n@地磁気 50μT (J≈0)\n"
                 "分子: [FADH•...Sub•] in MAO-A", fontsize=9)
    ax.set_xticks([])

    # --- (c) 電子スピン緩和タイムスケール ---
    ax = axes[2]

    timescales = [
        ("T₂ᵉ (HFC-mod)\n0.5 ns", 0.5, "#c62828"),
        ("T₂*(HFC)\n1.1 ns", 1.1, "#e65100"),
        ("T₂(g-ani)\n∞ @地磁気", 1e6, "#7b1fa2"),
        ("T₁ᵉ (spin-lattice)\n100 μs", 1e5, "#1565c0"),
        ("RP寿命\n~1 μs", 1e3, "#2e7d32"),
        ("T₂(³¹P) coupled\n160 μs", 1.6e5, "#e65100"),
        ("T₁(³¹P)\n4.6 s", 4.6e9, "#ff8f00"),
    ]

    y_pos = np.arange(len(timescales))
    t_vals = [t[1] for t in timescales]
    cols = [t[2] for t in timescales]
    names = [t[0] for t in timescales]

    bars = ax.barh(y_pos, np.log10(t_vals), color=cols, alpha=0.7,
                   edgecolor="black", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("log₁₀(時間 / ns)", fontsize=11)

    # RP寿命の線
    ax.axvline(np.log10(1e3), color="green", ls="--", lw=2, alpha=0.3)
    ax.text(np.log10(1e3) + 0.1, len(timescales) - 0.5, "RP寿命\n~1 μs",
            fontsize=8, color="green")

    ax.set_title("(c) スピン緩和タイムスケール\n@脳内 (50μT, 310K)\n"
                 "分子: FADH• in MAO-A", fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("Layer 2: ラジカルペアインターフェースの担体分子とエネルギー準位 (310 K)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_layer2_radical_pair.png", dpi=150)
    plt.close()
    print("  fig3_layer2_radical_pair.png 保存")


# ═══════════════════════════════════════════════════════════════
# Figure 4: Layer 3 — 古典的分子のエネルギー準位
# ═══════════════════════════════════════════════════════════════

def fig4_layer3_classical():
    """神経伝達物質と古典的エネルギースケール。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # --- (a) セロトニン (5-HT) の電子準位 ---
    ax = axes[0]
    # 5-HT: インドール環の電子遷移
    levels_5ht = [
        (0, "S₀ (基底状態)", "#2e7d32"),
        (3.6, "S₁ (π→π*)\n吸収: 275 nm", "#1565c0"),
        (4.5, "S₂\n吸収: 220 nm", "#7b1fa2"),
    ]
    for E, label, col in levels_5ht:
        ax.plot([0.2, 0.8], [E, E], color=col, lw=3)
        ax.text(0.82, E, label, fontsize=9, va="center", color=col)

    # kT矢印
    ax.annotate("", xy=(0.15, KT_310 * 1e3), xytext=(0.15, 0),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax.text(0.02, KT_310 * 1e3 / 2, f"kT\n{KT_310*1e3:.1f}\nmeV",
            fontsize=8, color="red", ha="center")

    # 酸化ポテンシャル
    ax.plot([0.3, 0.7], [0.8, 0.8], "g--", lw=2, alpha=0.5)
    ax.text(0.5, 0.9, "酸化電位 ~0.8 eV\n(MAO-A基質)", fontsize=8,
            ha="center", color="green")

    ax.set_ylabel("Energy (eV)", fontsize=11)
    ax.set_title("(a) セロトニン (5-HT)\n5-ヒドロキシトリプタミン\n"
                 "Layer 3 → Layer 2 への入口", fontsize=9)
    ax.set_xticks([])
    ax.set_ylim(-0.5, 5)

    # --- (b) FAD の電子準位と吸収スペクトル ---
    ax = axes[1]
    levels_fad = [
        (0, "S₀", "#2e7d32"),
        (2.76, "S₁ (π→π*)\n450 nm (青色光)", "#1565c0"),
        (3.3, "T₁ (ISC後)", "#c62828"),
        (1.25, "HOMO-LUMO\n(xTB: 1.25 eV)", "#7b1fa2"),
    ]
    for E, label, col in levels_fad:
        ax.plot([0.2, 0.8], [E, E], color=col, lw=3)
        ax.text(0.82, E, label, fontsize=8, va="center", color=col)

    # 光励起矢印
    ax.annotate("", xy=(0.5, 2.76), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.text(0.52, 1.4, "hν\n450 nm", fontsize=9, color="blue",
            ha="left", fontweight="bold")

    # ISC
    ax.annotate("", xy=(0.7, 3.3), xytext=(0.75, 2.76),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5, ls="--"))
    ax.text(0.77, 3.0, "ISC\nξ=63 cm⁻¹", fontsize=8, color="red")

    # RP生成
    ax.annotate("RP生成\n[FADH•...5-HT•]", xy=(0.5, 2.0),
                fontsize=9, color="orange", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    ax.set_ylabel("Energy (eV)", fontsize=11)
    ax.set_title("(b) FAD (フラビンアデニンジヌクレオチド)\nイソアロキサジン環\n"
                 "Layer 2 の中心分子", fontsize=9)
    ax.set_xticks([])
    ax.set_ylim(-0.5, 4)

    # --- (c) 3レイヤー統合エネルギー図 ---
    ax = axes[2]

    # エネルギースケールのバー
    categories = [
        "C-C bond\n(4.0 eV)",
        "光吸収 S₀→S₁\n(2.76 eV)",
        "S-T gap\n(1.3 eV)",
        "HOMO-LUMO\n(1.25 eV)",
        "酸化電位\n(0.8 eV)",
        "kT (310K)\n(26.7 meV)",
        "HFC ³¹P\n(0.83 μeV)",
        "e⁻ Zeeman @50μT\n(5.8 neV)",
        "³¹P-³¹P dipolar\n(41 feV)",
        "³¹P Zeeman @50μT\n(3.6 peV)",
    ]
    energies = [4.0, 2.76, 1.3, 1.25, 0.8,
                KT_310, 200e6 * 4.136e-15,
                G_E * MU_B_EV_T * B_EARTH,
                0.01e6 * 4.136e-15,
                GAMMA_P * B_EARTH * 4.136e-15]
    layer_colors = ["#2e7d32", "#2e7d32", "#7b1fa2", "#7b1fa2", "#7b1fa2",
                    "#d32f2f", "#e65100", "#7b1fa2", "#e65100", "#e65100"]

    y = np.arange(len(categories))
    ax.barh(y, np.log10(energies), color=layer_colors, alpha=0.6,
            edgecolor="black", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_xlabel("log₁₀(Energy / eV)", fontsize=10)

    # レイヤー注釈
    ax.text(-12.5, 9, "Layer 1\n核スピン", fontsize=9, color="darkorange",
            fontweight="bold", ha="center",
            bbox=dict(facecolor="lightyellow", alpha=0.8))
    ax.text(-4, 3, "Layer 2\nラジカルペア", fontsize=9, color="purple",
            fontweight="bold", ha="center",
            bbox=dict(facecolor="lavender", alpha=0.8))
    ax.text(-1, 0.5, "Layer 3\n古典化学", fontsize=9, color="green",
            fontweight="bold", ha="center",
            bbox=dict(facecolor="honeydew", alpha=0.8))

    # kTの線
    ax.axvline(np.log10(KT_310), color="red", ls="--", lw=2, alpha=0.4)

    ax.set_title("(c) 3レイヤー統合エネルギースケール\n"
                 "14桁にわたるエネルギー階層", fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")
    ax.set_xlim(-14, 1)

    fig.suptitle("Layer 3: 古典電気化学とレイヤー間エネルギー接続 (310 K)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_layer3_classical.png", dpi=150)
    plt.close()
    print("  fig4_layer3_classical.png 保存")


# ═══════════════════════════════════════════════════════════════
# Figure 5: 分子構造と情報フロー
# ═══════════════════════════════════════════════════════════════

def fig5_molecular_info_flow():
    """各レイヤーの担体分子と情報フローの模式図。"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # === Layer 1 (top) ===
    rect1 = mpatches.FancyBboxPatch((0.5, 8.5), 9, 3, boxstyle="round,pad=0.2",
                                     facecolor="#fff3e0", edgecolor="#e65100", lw=2)
    ax.add_patch(rect1)
    ax.text(5, 11.2, "Layer 1: 量子メモリ (核スピン)", fontsize=14,
            ha="center", fontweight="bold", color="#e65100")

    # ³¹P分子情報
    ax.text(1.5, 10.5, "情報担体:", fontsize=10, fontweight="bold", color="#bf360c")
    ax.text(1.5, 10.0, "³¹P 核スピン (I = ½)\nin PO₄³⁻ (リン酸基)",
            fontsize=9, color="#333")
    ax.text(1.5, 9.2, "超微細構造:\n  Zeeman: 862 Hz @50μT (3.6 peV)\n"
            "  双極子: 0.01 MHz (41 feV)\n  T₂ = 3.25 ms (反磁性)",
            fontsize=8, color="#555", family="Hiragino Sans")

    ax.text(5.5, 10.5, "分子:", fontsize=10, fontweight="bold", color="#bf360c")
    ax.text(5.5, 10.0, "FAD: CH₃₃N₁₀O₁₅P₂ (MW 785)\n  リン酸鎖上の ³¹P × 2",
            fontsize=9, color="#333")
    ax.text(5.5, 9.2, "PLP: C₈H₁₀NO₆P (MW 247)\n  ピリジン環C5直結 ³¹P × 1\n"
            "NADPH: C₂₁H₂₉N₇O₁₇P₃ (MW 744)\n  リン酸 × 3",
            fontsize=8, color="#555")

    # === Layer 2 (middle) ===
    rect2 = mpatches.FancyBboxPatch((0.5, 4.5), 9, 3.5, boxstyle="round,pad=0.2",
                                     facecolor="#f3e5f5", edgecolor="#7b1fa2", lw=2)
    ax.add_patch(rect2)
    ax.text(5, 7.7, "Layer 2: 量子-古典インターフェース (ラジカルペア)", fontsize=14,
            ha="center", fontweight="bold", color="#7b1fa2")

    ax.text(1.0, 7.0, "情報担体:", fontsize=10, fontweight="bold", color="#4a148c")
    ax.text(1.0, 6.5, "FADH• セミキノンラジカル\n+ 基質ラジカル (Sub•)",
            fontsize=9, color="#333")
    ax.text(1.0, 5.5, "超微細構造:\n"
            "  結合 e⁻-³¹P: 4準位 (A=200 MHz)\n"
            "  EPR分裂: 2.80 MHz (@50μT)\n"
            "  禁制TDM: 1.22 μ_B (許容の70%)\n"
            "  T₂ᵉ = 1.1 ns, T₁ᵉ = 100 μs",
            fontsize=8, color="#555", family="Hiragino Sans")

    ax.text(5.5, 7.0, "変換機構:", fontsize=10, fontweight="bold", color="#4a148c")
    ax.text(5.5, 6.5, "S-T₀ mixing (HFC駆動)\n→ 一重項収率 Φ_S(B)",
            fontsize=9, color="#333")
    ax.text(5.5, 5.5, "磁場効果 (MFE):\n"
            "  Φ_S(0) = 0.459 → Φ_S(0.9mT) = 0.439\n"
            "  MFE = −4.4% (MAO-A)\n"
            "  地磁気 @50μT: MFE ≈ −0.3%\n"
            "  → 5-HT分解速度 0.3%変調",
            fontsize=8, color="#555", family="Hiragino Sans")

    # === Layer 3 (bottom) ===
    rect3 = mpatches.FancyBboxPatch((0.5, 0.5), 9, 3.5, boxstyle="round,pad=0.2",
                                     facecolor="#e8f5e9", edgecolor="#2e7d32", lw=2)
    ax.add_patch(rect3)
    ax.text(5, 3.7, "Layer 3: 古典的電気化学情報処理", fontsize=14,
            ha="center", fontweight="bold", color="#2e7d32")

    ax.text(1.0, 3.0, "情報担体:", fontsize=10, fontweight="bold", color="#1b5e20")
    ax.text(1.0, 2.5, "セロトニン (5-HT): C₁₀H₁₂N₂O\n"
            "ドーパミン (DA): C₈H₁₁NO₂\n"
            "GABA: C₄H₉NO₂",
            fontsize=9, color="#333")
    ax.text(1.0, 1.2, "エネルギースケール:\n"
            "  酸化電位: 0.8 eV (5-HT)\n"
            "  シナプス電位: ~100 mV\n"
            "  kT = 26.7 meV",
            fontsize=8, color="#555", family="Hiragino Sans")

    ax.text(5.5, 3.0, "情報処理:", fontsize=10, fontweight="bold", color="#1b5e20")
    ax.text(5.5, 2.5, "MAO-A: 5-HT → 5-HIAA\n"
            "  kcat ≈ 10 s⁻¹\n"
            "  RPMにより 0.3% 速度変調",
            fontsize=9, color="#333")
    ax.text(5.5, 1.2, "神経科学的出力:\n"
            "  5-HT濃度: ~nM変化\n"
            "  → シナプス伝達効率変化\n"
            "  → 気分・感情・概日リズム",
            fontsize=8, color="#555", family="Hiragino Sans")

    # === 矢印 (レイヤー間接続) ===
    # Layer 1 → Layer 2
    ax.annotate("", xy=(3, 8.0), xytext=(3, 8.5),
                arrowprops=dict(arrowstyle="<->", color="#e65100", lw=3))
    ax.text(3.2, 8.25, "HFC (A=200 MHz)\n³¹P核スピン ↔ 電子スピン",
            fontsize=8, color="#e65100", fontweight="bold")

    # Layer 2 → Layer 3
    ax.annotate("", xy=(3, 4.0), xytext=(3, 4.5),
                arrowprops=dict(arrowstyle="->", color="#7b1fa2", lw=3))
    ax.text(3.2, 4.25, "Φ_S(B) → 反応速度変調\n量子 → 古典 変換",
            fontsize=8, color="#7b1fa2", fontweight="bold")

    # タイトル
    ax.text(5, 12.0, "3レイヤー量子脳仮説 — 情報担体分子の同定と超微細構造",
            fontsize=15, ha="center", fontweight="bold")

    plt.savefig(OUT_DIR / "fig5_molecular_info_flow.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  fig5_molecular_info_flow.png 保存")


# ═══════════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("エネルギー準位図の生成...")
    fig1_energy_scales()
    fig2_layer1_nuclear()
    fig3_layer2_radical_pair()
    fig4_layer3_classical()
    fig5_molecular_info_flow()
    print("全図完了 → simulation_results/fig[1-5]_*.png")
