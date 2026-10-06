# -*- coding: utf-8 -*-
"""
IEEE Publication Visualization Package for PINN ECT Project.

Provides unified, publication-grade figures conforming to IEEE Transactions standards:
1. No internal plot titles (all captions belong in LaTeX manuscript).
2. Legends placed outside axes or neatly positioned without occluding data.
3. Times New Roman (or serif fallback) typography with consistent sizing.
4. Publication color palettes with high contrast.
5. Dual export: High-resolution PNG (300 DPI) and vector PDF for LaTeX.
"""

from .style import (
    setup_ieee_style,
    save_ieee_figure,
    COLOR_TRAIN,
    COLOR_VAL,
    COLOR_PINN,
    COLOR_BASELINE,
    SHAPE_COLORS,
    COLOR_W,
    COLOR_L,
    COLOR_D,
)

from .training_curves import (
    plot_fig1_training_curves,
    plot_loss_components_breakdown,
)

from .confusion_matrix import (
    plot_fig2_confusion_matrix,
    plot_per_class_accuracy,
)

from .regression_scatter import (
    plot_fig3_regression_scatter,
    save_regression_plots,
)

from .latent_space import (
    plot_fig4_tsne_latent_space,
    plot_pca_latent_space,
)

from .per_shape_metrics import (
    plot_per_shape_comparison,
)

from .physics_diagnostics import (
    plot_physics_comparison,
    visualize_physics_comparison,
)

from .comprehensive_metrics import (
    generate_comprehensive_metrics_visualizations,
)

__all__ = [
    "setup_ieee_style",
    "save_ieee_figure",
    "COLOR_TRAIN",
    "COLOR_VAL",
    "COLOR_PINN",
    "COLOR_BASELINE",
    "SHAPE_COLORS",
    "COLOR_W",
    "COLOR_L",
    "COLOR_D",
    "plot_fig1_training_curves",
    "plot_loss_components_breakdown",
    "plot_fig2_confusion_matrix",
    "plot_per_class_accuracy",
    "plot_fig3_regression_scatter",
    "save_regression_plots",
    "plot_fig4_tsne_latent_space",
    "plot_pca_latent_space",
    "plot_per_shape_comparison",
    "plot_physics_comparison",
    "visualize_physics_comparison",
    "generate_comprehensive_metrics_visualizations",
]
