#!/usr/bin/env python3
"""PCCP table-of-contents graphic: exactly 8 cm x 4 cm, 600 dpi.

Message: the same flavin cofactor gives a dead enzyme and a working
magnetoreceptor; the inter-radical separation the scaffold imposes is what decides.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from figure_style import apply_style

apply_style()
# PCCP requires a hard 8 cm x 4 cm canvas; the shared style's 'tight' bbox
# would pad past it, so disable cropping for this figure only.
plt.rcParams['savefig.bbox'] = None
plt.rcParams['savefig.pad_inches'] = 0.0

CM = 1 / 2.54
fig, ax = plt.subplots(figsize=(8 * CM, 4 * CM))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

RED, BLUE, GREY = "#c0392b", "#1f6fb4", "#606060"
Y = 28.0          # radical-pair centre line
R = 2.6           # radical radius


def pair(xc, sep, colour, title, r_txt, tau_txt, mfe_txt):
    """Draw one radical pair centred on xc with its separation and verdict."""
    x0, x1 = xc - sep / 2, xc + sep / 2
    ax.add_patch(Circle((x0, Y), R, fc=colour, ec="none"))
    ax.add_patch(Circle((x1, Y), R, fc=colour, ec="none", alpha=0.55))
    # separation marker: an arrow only when the radicals are far enough apart
    # for one to be legible, otherwise a plain tick pair.
    if sep > 3 * R:
        ax.annotate("", xy=(x1 - R - 0.6, Y), xytext=(x0 + R + 0.6, Y),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color=GREY))
    else:
        for x in (x0 - R, x1 + R):
            ax.plot([x, x], [Y - 4.2, Y + 4.2], lw=0.6, color=GREY)
        ax.annotate("", xy=(x1 + R, Y + 3.4), xytext=(x0 - R, Y + 3.4),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, color=GREY))
    ax.text(xc, Y + 6.4, r_txt, ha="center", va="bottom", fontsize=7, color=GREY)
    ax.text(xc, Y - 6.4, tau_txt, ha="center", va="top", fontsize=7, color=GREY)
    ax.text(xc, 44.5, title, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(xc, 11.0, mfe_txt, ha="center", va="center", fontsize=11,
            fontweight="bold", color=colour)


pair(25, 6,  RED,  "MAO / DAO active site", "3.5 Å",   r"$\tau \approx 10$ ps",
     r"$10^{-12}\,\%$")
pair(75, 30, BLUE, "Cryptochrome",          "15–20 Å", r"$\tau \approx 1\ \mu$s",
     r"$+1.0\,\%$")

ax.plot([50, 50], [7, 40], lw=0.5, color="0.82")
ax.text(50, 0.8, "geomagnetic-field effect on the singlet yield, 50 μT",
        ha="center", va="bottom", fontsize=6.5, color=GREY, style="italic")

fig.savefig("toc_entry.pdf", dpi=600)
fig.savefig("toc_entry.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("wrote toc_entry.pdf / toc_entry.tif")
