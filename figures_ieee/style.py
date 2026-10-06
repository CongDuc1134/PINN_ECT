# -*- coding: utf-8 -*-
"""
IEEE Publication Plot Styling & Figure Management Module.

Standard rules for IEEE Transactions figures:
1. No internal plot titles (caption belongs in LaTeX manuscript).
2. Legends placed outside axes or neatly positioned without occluding data.
3. Typography: Times New Roman (or serif fallback), consistent font sizes.
4. Publication-grade palette with high contrast and distinct markers/linestyles.
5. Dual export: High-resolution PNG (300 DPI) and vector PDF for LaTeX.
6. Atomic file writing to prevent corrupt files upon unexpected termination.
"""

import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# IEEE TRANSACTIONS COLOR PALETTES
# =============================================================================
COLOR_TRAIN = "#1f77b4"      # IEEE Royal Blue (Train)
COLOR_VAL = "#d62728"        # IEEE Crimson Red (Val)
COLOR_PINN = "#2ca02c"       # Forest Green (Physics/PINN)
COLOR_BASELINE = "#7f7f7f"   # Slate Gray (Baseline)

# Shape classification colors (5 distinct classes)
SHAPE_COLORS = [
    "#1f77b4",  # Class 0: Blue
    "#ff7f0e",  # Class 1: Orange
    "#2ca02c",  # Class 2: Green
    "#d62728",  # Class 3: Red
    "#9467bd",  # Class 4: Purple
]

# Regression dimension colors
COLOR_W = "#1f77b4"  # Width (W): Blue
COLOR_L = "#ff7f0e"  # Length (L): Orange
COLOR_D = "#2ca02c"  # Depth (D): Green


def setup_ieee_style():
    """Configure matplotlib rcParams to strictly match IEEE Transactions standards."""
    plt.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,

        # Axes & labels (NO TITLE by default in IEEE style)
        "axes.titlesize": 0,
        "axes.titleweight": "normal",
        "axes.labelsize": 10.0,
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # Ticks
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Legends (Clean, border frame, readable)
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#888888",
        "legend.fancybox": False,
        "legend.borderaxespad": 0.5,

        # Figures & Savefig
        "figure.titlesize": 0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def save_ieee_figure(fig, output_path, dpi=300, close_fig=True):
    """
    Save figure as both vector PDF and 300 DPI PNG via atomic file writes.
    
    Args:
        fig: matplotlib Figure object
        output_path: Full file path (with or without extension)
        dpi: Resolution for PNG export (default 300)
        close_fig: Whether to close the figure after saving
    """
    base_no_ext = os.path.splitext(output_path)[0]
    out_dir = os.path.dirname(os.path.abspath(base_no_ext))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    png_path = f"{base_no_ext}.png"
    pdf_path = f"{base_no_ext}.pdf"

    # Atomic write for PNG
    tmp_png = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=out_dir, prefix=".tmp_fig_", suffix=".png") as tf:
            tmp_png = tf.name
        fig.savefig(tmp_png, dpi=dpi, bbox_inches="tight")
        os.replace(tmp_png, png_path)
    finally:
        if tmp_png and os.path.exists(tmp_png):
            try:
                os.remove(tmp_png)
            except Exception:
                pass

    # Atomic write for PDF
    tmp_pdf = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=out_dir, prefix=".tmp_fig_", suffix=".pdf") as tf:
            tmp_pdf = tf.name
        fig.savefig(tmp_pdf, format="pdf", bbox_inches="tight")
        os.replace(tmp_pdf, pdf_path)
    except Exception as e:
        # Some environments might lack full PDF font embedder; don't crash
        pass
    finally:
        if tmp_pdf and os.path.exists(tmp_pdf):
            try:
                os.remove(tmp_pdf)
            except Exception:
                pass

    if close_fig:
        plt.close(fig)
