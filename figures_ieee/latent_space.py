# -*- coding: utf-8 -*-
"""
IEEE Publication Standard: Latent Space & Manifold Clustering (t-SNE) Visualizations.

Key Rules:
- No internal figure/subplot titles.
- Dual panel: (a) Shape Class Clusters, (b) Continuous Depth (D) Gradient.
- Legends placed strictly outside plot bounding box.
- Dual export: Vector PDF & 300 DPI PNG.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .style import setup_ieee_style, save_ieee_figure, SHAPE_COLORS


def plot_fig4_tsne_latent_space(
    model,
    test_loader,
    y_shape_test,
    y_test_denorm,
    unique_shapes,
    output_dir,
    filename_prefix="fig4_tsne_latent_space",
    device="cpu",
    max_samples=1000,
):
    """
    Plot IEEE standard Fig 4: Dual-Panel t-SNE Latent Feature Space.
    Panel (a): Clustered by Defect Geometry Class
    Panel (b): Continuous Manifold colored by Defect Depth D (mm)
    """
    setup_ieee_style()
    model.eval()

    latent_features = []
    collected_shapes = []
    collected_depths = []
    sample_count = 0

    with torch.no_grad():
        for batch_data in test_loader:
            batch_X = batch_data[0].to(device)
            # Extract features from backbone
            if hasattr(model, 'backbone'):
                feat = model.backbone(batch_X)
                feat = feat.reshape(feat.size(0), -1)
            elif hasattr(model, 'feature_extractor'):
                feat = model.feature_extractor(batch_X)
            else:
                # Fallback: flatten input if no explicit backbone
                feat = batch_X.reshape(batch_X.size(0), -1)

            latent_features.append(feat.cpu().numpy())
            sample_count += batch_X.size(0)
            if sample_count >= max_samples:
                break

    if not latent_features:
        return

    features_all = np.concatenate(latent_features, axis=0)[:max_samples]
    shapes_all = np.asarray(y_shape_test)[:len(features_all)]
    depths_all = np.asarray(y_test_denorm)[:len(features_all), 2]  # Depth is index 2

    # Compute t-SNE
    perplexity = min(30, max(5, len(features_all) // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    embedded = tsne.fit_transform(features_all)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    # --- Panel (a): Categorical Clusters by Shape ---
    ax_shape = axes[0]
    markers = ['o', 's', '^', 'D', 'v']
    for idx, shape_name in enumerate(unique_shapes):
        mask = (shapes_all == idx)
        if np.any(mask):
            color = SHAPE_COLORS[idx % len(SHAPE_COLORS)]
            marker = markers[idx % len(markers)]
            ax_shape.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=color, label=shape_name, marker=marker,
                alpha=0.75, s=18, edgecolors='none'
            )

    ax_shape.set_xlabel('t-SNE Dimension 1')
    ax_shape.set_ylabel('t-SNE Dimension 2')
    # Legend outside on top
    ax_shape.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fontsize=7.5)

    # --- Panel (b): Continuous Manifold by Depth D ---
    ax_depth = axes[1]
    sc = ax_depth.scatter(
        embedded[:, 0], embedded[:, 1],
        c=depths_all, cmap='plasma', alpha=0.8,
        s=18, edgecolors='none'
    )
    cbar = fig.colorbar(sc, ax=ax_depth, fraction=0.046, pad=0.04)
    cbar.set_label('Depth, $D$ (mm)', rotation=270, labelpad=12)

    ax_depth.set_xlabel('t-SNE Dimension 1')
    ax_depth.set_ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename_prefix)
    save_ieee_figure(fig, save_path)


def plot_pca_latent_space(features, labels, class_names, output_prefix="pca"):
    """Plot 2D PCA latent clustering without internal titles, legend outside."""
    setup_ieee_style()
    if len(features) == 0:
        return
    from sklearn.decomposition import PCA
    pca2 = PCA(n_components=2)
    f2d = pca2.fit_transform(features)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i, cls in enumerate(class_names):
        idx = (labels == i)
        if np.any(idx):
            color = SHAPE_COLORS[i % len(SHAPE_COLORS)]
            ax.scatter(f2d[idx, 0], f2d[idx, 1], label=cls, color=color, alpha=0.7, s=18, edgecolors='none')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fontsize=8)
    plt.tight_layout()
    save_ieee_figure(fig, f"{output_prefix}_2d")

