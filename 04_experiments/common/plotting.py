"""Paper-style matplotlib defaults.

Call apply_paper_style() once at the top of any plotting script.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelpad": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "figure.constrained_layout.use": False,
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
    })


# Colour-blind safe palette (Wong 2011)
WONG = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "black":  "#000000",
}


def save_both(fig, basename: str, out_dir) -> tuple:
    """Save figure as both PDF (vector) and PNG (raster) at 300 dpi."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{basename}.pdf"
    png = out_dir / f"{basename}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png
