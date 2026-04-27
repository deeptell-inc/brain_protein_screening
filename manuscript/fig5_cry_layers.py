#!/usr/bin/env python3
"""Fig 5: CRY-based 3-layer quantum-classical hierarchy (publication quality).

Clean, spacious design with clear information flow arrows.
Grounded entirely in CRY biology with experimental references at each step.
"""
import sys
sys.path.insert(0, '.')
from figure_style import apply_style, DOUBLE_COL
apply_style()
plt_size = 10  # override for schematic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Override some rcParams for this schematic
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
})

fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 1.3))  # ~7.2 x 9.4 inches
ax.set_xlim(0, 100)
ax.set_ylim(0, 130)
ax.axis('off')

# Color palette
C_L1 = '#FFF8E1'    # Layer 1 bg (warm yellow)
C_L1_EDGE = '#E65100'
C_L2 = '#F3E5F5'    # Layer 2 bg (purple)
C_L2_EDGE = '#7B1FA2'
C_L3 = '#E8F5E9'    # Layer 3 bg (green)
C_L3_EDGE = '#2E7D32'
C_ARROW = '#333333'
C_HFC = '#E65100'
C_QUANTUM = '#1565C0'
C_CLASSICAL = '#2E7D32'

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(50, 128, 'Three-Layer Quantum-Classical Information Hierarchy',
        ha='center', fontsize=11, fontweight='bold', color='#222')
ax.text(50, 125, 'Grounded in cryptochrome (CRY) radical pair mechanism',
        ha='center', fontsize=8, color='#333', style='italic')

# ═══════════════════════════════════════════════════════════════
# LAYER 1: Nuclear Spin Quantum Memory
# ═══════════════════════════════════════════════════════════════
y1_top, y1_bot = 118, 92
r1 = FancyBboxPatch((3, y1_bot), 94, y1_top - y1_bot,
                     boxstyle="round,pad=1.5", facecolor=C_L1,
                     edgecolor=C_L1_EDGE, lw=2)
ax.add_patch(r1)

ax.text(50, y1_top - 2.5, 'LAYER 1 :  Nuclear Spin Quantum Memory',
        ha='center', fontsize=10, fontweight='bold', color=C_L1_EDGE)

# Left column: molecule
ax.text(8, 112, 'Molecule', fontsize=8, fontweight='bold', color='#333')
ax.text(8, 109.5, 'FAD in cryptochrome', fontsize=7.5, color='#222')
ax.text(8, 107, r'$^{31}$P on FMN-phosphate', fontsize=7.5, color='#222')
ax.text(8, 104.5, r'$I$ = 1/2,  100% abundance', fontsize=7, color='#444')

# Center: energy level diagram (simplified)
# Draw 2-level system
lx, ly = 38, 110
ax.plot([lx, lx+12], [ly+2, ly+2], color=C_QUANTUM, lw=2.5)
ax.plot([lx, lx+12], [ly-2, ly-2], color='#C62828', lw=2.5)
ax.text(lx+13, ly+2, r'$|m_I = +\frac{1}{2}\rangle$', fontsize=7,
        va='center', color=C_QUANTUM)
ax.text(lx+13, ly-2, r'$|m_I = -\frac{1}{2}\rangle$', fontsize=7,
        va='center', color='#C62828')
ax.annotate('', xy=(lx-2, ly+2), xytext=(lx-2, ly-2),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=1))
ax.text(lx-4, ly, '862 Hz\n3.6 peV', fontsize=6, ha='right', va='center', color='#222')

# Right column: coherence times
ax.text(65, 112, 'Coherence', fontsize=8, fontweight='bold', color='#333')
ax.text(65, 109, r'Diamagnetic:  $T_2$ = 3.2 ms', fontsize=7.5, color='#222')
ax.text(65, 106.5, r'RP-coupled:  $T_2$ = 160 $\mu$s', fontsize=7.5, color='#222')
ax.text(65, 104, r'($r^{-6}$ PRE, $r_{eP}$ = 7 $\AA$)', fontsize=7, color='#555')

# Bottom of Layer 1: quantum information box
qi_box = FancyBboxPatch((15, y1_bot + 1), 70, 6,
                         boxstyle="round,pad=0.5", facecolor='white',
                         edgecolor=C_QUANTUM, lw=1, alpha=0.9)
ax.add_patch(qi_box)
ax.text(50, y1_bot + 5.5, 'Quantum information:', fontsize=7.5,
        ha='center', fontweight='bold', color=C_QUANTUM)
ax.text(50, y1_bot + 2.5,
        r'$^{31}$P spin state encodes $B$-field direction  '
        r'$\rightarrow$  modulates S-T mixing rate via HFC',
        fontsize=7, ha='center', color='#444')

# ═══════════════════════════════════════════════════════════════
# ARROW: Layer 1 → Layer 2 (HFC coupling)
# ═══════════════════════════════════════════════════════════════
arrow_y = (y1_bot + 92) / 2 - 3.5
ax.annotate('', xy=(50, arrow_y - 5), xytext=(50, arrow_y + 5),
            arrowprops=dict(arrowstyle='->', color=C_HFC, lw=3,
                           connectionstyle='arc3'))
ax.text(52, arrow_y, r'HFC:  $A$ = 200 MHz = 0.83 $\mu$eV',
        fontsize=8, fontweight='bold', color=C_HFC, va='center')
ax.text(52, arrow_y - 3, r'$^{31}$P nuclear spin $\leftrightarrow$ electron spin',
        fontsize=7, color=C_HFC, va='center')

# ═══════════════════════════════════════════════════════════════
# LAYER 2: Radical Pair Quantum-Classical Interface
# ═══════════════════════════════════════════════════════════════
y2_top, y2_bot = 80, 48
r2 = FancyBboxPatch((3, y2_bot), 94, y2_top - y2_bot,
                     boxstyle="round,pad=1.5", facecolor=C_L2,
                     edgecolor=C_L2_EDGE, lw=2)
ax.add_patch(r2)

ax.text(50, y2_top - 2.5, 'LAYER 2 :  Radical Pair Quantum-Classical Interface',
        ha='center', fontsize=10, fontweight='bold', color=C_L2_EDGE)

# Left: RP formation
ax.text(8, 74, r'Blue light (450 nm)', fontsize=7.5, fontweight='bold', color='#333')
ax.text(8, 71.5, r'FAD $\rightarrow$ [FAD$^{\bullet-}$ $\cdots$ Trp$^{\bullet+}$]',
        fontsize=8, color='#222')
ax.text(8, 69, r'via Trp triad/tetrad', fontsize=7, color='#444')
ax.text(8, 66.5, r'$r$ = 15-20 $\AA$,  $|J|$ < 1 MHz', fontsize=7, color='#444')
ax.text(8, 64, r'$\tau_{RP}$ = 1-10 $\mu$s', fontsize=7, color='#444')

# Center: S-T mixing diagram
stx = 48
ax.plot([stx-6, stx+6], [72, 72], color=C_QUANTUM, lw=2)  # T0
ax.plot([stx-6, stx+6], [68, 68], color='#C62828', lw=2)   # S
ax.text(stx+7, 72, r'$|T_0\rangle$', fontsize=7, color=C_QUANTUM, va='center')
ax.text(stx+7, 68, r'$|S\rangle$', fontsize=7, color='#C62828', va='center')

# Wavy arrow for S-T mixing
for i in range(8):
    x0 = stx - 5 + i * 1.5
    y0 = 70 + 0.8 * np.sin(i * 1.2)
    ax.plot([x0, x0 + 1], [y0, y0 + 0.8 * np.sin((i+1) * 1.2)],
            color=C_HFC, lw=1.2, alpha=0.7)
ax.text(stx, 74.5, 'HFC-driven\nmixing', fontsize=6, ha='center', color=C_HFC)

# Right: MFE output
ax.text(68, 74, 'Magnetic field effect', fontsize=7.5, fontweight='bold', color='#333')
ax.text(68, 71.5, r'MFE = $-$5.6% @ 0.29 mT', fontsize=7.5, color='#222')
ax.text(68, 69, r'MFE = $-$0.55% @ Earth field', fontsize=7.5, color='#222')
ax.text(68, 66.5, r'$\Phi_S(B)$ = singlet yield', fontsize=7, color='#444')

# Bottom of Layer 2: transduction box
tr_box = FancyBboxPatch((10, y2_bot + 1), 80, 8,
                         boxstyle="round,pad=0.5", facecolor='white',
                         edgecolor=C_L2_EDGE, lw=1, alpha=0.9)
ax.add_patch(tr_box)
ax.text(50, y2_bot + 7, 'Quantum-to-classical transduction:', fontsize=7.5,
        ha='center', fontweight='bold', color=C_L2_EDGE)
ax.text(50, y2_bot + 4,
        'Nuclear spin state  '
        r'$\longrightarrow$  S-T mixing rate  '
        r'$\rightarrow$  $\Phi_S(B)$  $\rightarrow$  '
        'product branching ratio',
        fontsize=7, ha='center', color='#444')

# ═══════════════════════════════════════════════════════════════
# ARROW: Layer 2 → Layer 3
# ═══════════════════════════════════════════════════════════════
arrow_y2 = (y2_bot + 48) / 2 - 3.5
ax.annotate('', xy=(50, arrow_y2 - 5), xytext=(50, arrow_y2 + 5),
            arrowprops=dict(arrowstyle='->', color=C_L2_EDGE, lw=3))
ax.text(52, arrow_y2, r'$\Delta\Phi_S$ = 0.55%  (Earth field)',
        fontsize=8, fontweight='bold', color=C_L2_EDGE, va='center')
ax.text(52, arrow_y2 - 3, 'No homeostatic buffering\n(conformational, not metabolite)',
        fontsize=7, color=C_L2_EDGE, va='center')

# ═══════════════════════════════════════════════════════════════
# LAYER 3: Classical Circadian Signalling
# ═══════════════════════════════════════════════════════════════
y3_top, y3_bot = 36, 8
r3 = FancyBboxPatch((3, y3_bot), 94, y3_top - y3_bot,
                     boxstyle="round,pad=1.5", facecolor=C_L3,
                     edgecolor=C_L3_EDGE, lw=2)
ax.add_patch(r3)

ax.text(50, y3_top - 2.5, 'LAYER 3 :  Classical Circadian Signalling',
        ha='center', fontsize=10, fontweight='bold', color=C_L3_EDGE)

# Signalling cascade (horizontal flow)
steps = [
    (10, r'CRY*', 'Photoactivated'),
    (28, r'PER', 'Binding'),
    (46, 'CLOCK\n/BMAL1', 'Repression'),
    (64, 'Gene\nexpr.', 'Oscillation'),
    (82, 'Period\nshift', '2.7-8.2 min'),
]

for i, (x, label, sub) in enumerate(steps):
    box = FancyBboxPatch((x - 5, 18), 10, 10,
                          boxstyle="round,pad=0.3",
                          facecolor='white', edgecolor=C_L3_EDGE, lw=1)
    ax.add_patch(box)
    ax.text(x, 24.5, label, ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='#333')
    ax.text(x, 16, sub, ha='center', fontsize=6, color='#444')

    # Arrows between boxes
    if i < len(steps) - 1:
        next_x = steps[i + 1][0]
        ax.annotate('', xy=(next_x - 6, 23), xytext=(x + 6, 23),
                    arrowprops=dict(arrowstyle='->', color=C_L3_EDGE, lw=1.5))

# Bottom annotation
ax.text(50, 11, 'Circadian rhythm  '
        r'$\rightarrow$  mood, sleep, seasonal behaviour',
        ha='center', fontsize=7.5, color='#222', style='italic')

# ═══════════════════════════════════════════════════════════════
# Right margin: Energy scale bar
# ═══════════════════════════════════════════════════════════════
ex = 98
ax.annotate('', xy=(ex, y3_bot + 5), xytext=(ex, y1_top - 5),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=1))
ax.text(ex + 1, (y1_top + y1_bot) / 2 + 10, '3.6 peV', fontsize=6,
        color=C_L1_EDGE, rotation=90, va='center', ha='left')
ax.text(ex + 1, (y2_top + y2_bot) / 2 + 5, r'0.83 $\mu$eV', fontsize=6,
        color=C_L2_EDGE, rotation=90, va='center', ha='left')
ax.text(ex + 1, (y3_top + y3_bot) / 2 + 2, '~eV', fontsize=6,
        color=C_L3_EDGE, rotation=90, va='center', ha='left')
ax.text(ex + 1, 65, '14 orders\nof magnitude', fontsize=6,
        color='#333', rotation=90, va='center', ha='left')

# ═══════════════════════════════════════════════════════════════
# Left margin: experimental evidence markers
# ═══════════════════════════════════════════════════════════════
ev_x = 1
ax.text(ev_x, (y1_top + y1_bot) / 2 + 5, 'NMR', fontsize=6,
        color=C_L1_EDGE, rotation=90, va='center', ha='center', fontweight='bold')
ax.text(ev_x, (y2_top + y2_bot) / 2 + 5, 'Xu 2021\nMaeda 2012', fontsize=5,
        color=C_L2_EDGE, rotation=90, va='center', ha='center', fontweight='bold')
ax.text(ev_x, (y3_top + y3_bot) / 2 + 2, 'Partch 2014', fontsize=5,
        color=C_L3_EDGE, rotation=90, va='center', ha='center', fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# Bottom: key advantages
# ═══════════════════════════════════════════════════════════════
ax.text(50, 4, 'Key:  RP confirmed ($|J|$ < 1 MHz)  |  '
        'No homeostatic buffering  |  '
        'Circadian amplification (limit-cycle)',
        ha='center', fontsize=7, color='#333',
        bbox=dict(facecolor='white', edgecolor='#999', boxstyle='round,pad=0.3'))

ax.text(50, 1, 'CRY is the only currently defensible RPM host in the human brain',
        ha='center', fontsize=7, fontweight='bold', color='#333', style='italic')

plt.savefig(OUT / 'fig5_three_layer.pdf', bbox_inches='tight')
plt.savefig(OUT / 'fig5_three_layer.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig 5 saved")
