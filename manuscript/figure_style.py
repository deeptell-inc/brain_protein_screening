"""Unified figure style for all manuscript figures.

Usage:
    from figure_style import apply_style
    apply_style()
    # then create figures as usual
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def apply_style():
    """Apply Nature-style formatting to all matplotlib figures."""
    plt.rcParams.update({
        # Font
        'font.family': 'Arial',
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.title_fontsize': 8,
        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        # Lines
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        # Axes
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        # Grid
        'grid.linewidth': 0.3,
        'grid.alpha': 0.3,
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })

# Nature single column = 89 mm, double = 183 mm
# 1 inch = 25.4 mm
SINGLE_COL = 89 / 25.4   # 3.50 inches
DOUBLE_COL = 183 / 25.4  # 7.20 inches
FULL_PAGE_H = 247 / 25.4  # 9.72 inches
