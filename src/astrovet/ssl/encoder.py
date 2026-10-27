"""
Unsupervised PCA representation prototype for light-curve views.

Scientific status & limitations (Crucial safeguard):
---------------------------------------------------
This module implements an UNSUPERVISED PCA / SVD PROTOTYPE representation, NOT a fully trained
deep self-supervised learning (SSL) model. It fits a principal component subspace over
unlabeled training light-curve views to extract fixed-dimensional linear embeddings (d=16).

A complete self-supervised representation learning pipeline (e.g. contrastive view alignment
via SimCLR/BYOL or masked autoencoding with the augmentations in `augmentations.py`) is planned
for future development. All representations produced by this module should be interpreted as
unsupervised PCA features.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class LightCurvePCAPrototypeEncoder:
    """
    Unsupervised PCA representation prototype mapping phase-folded light curves (N=200)
    to a compact linear projection vector (d=16).
    """

    def __init__(self, embedding_dim: int = 16, seed: int = 42) -> None:
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.is_fitted = False
        self.model_type = "unsupervised_pca_prototype"
        self.projection_weights: Optional[np.ndarray] = None

    def fit(self, training_views: np.ndarray) -> LightCurvePCAPrototypeEncoder:
        """
        Fit PCA projection basis on unlabeled training light-curve views.
        training_views: shape (M, 200).
        """
        M, N = training_views.shape
        cleaned_views = np.empty_like(training_views)
        for i in range(M):
            row = training_views[i]
            nans = np.isnan(row)
            if np.all(nans):
                cleaned_views[i] = 1.0
            elif np.any(nans):
                x = np.arange(N)
                cleaned_views[i] = np.interp(x, x[~nans], row[~nans])
            else:
                cleaned_views[i] = row

        # Center around out-of-transit baseline 1.0
        diff = cleaned_views - 1.0
        U, S, Vt = np.linalg.svd(diff, full_matrices=False)
        self.projection_weights = Vt[:self.embedding_dim].T
        self.is_fitted = True
        return self

    def encode(self, view: np.ndarray) -> np.ndarray:
        """
        Project a single phase-folded view (200,) into embedding vector (16,).
        """
        N = len(view)
        if self.projection_weights is None or not self.is_fitted:
            # Deterministic harmonic basis initialization if not yet fitted
            x = np.linspace(-1, 1, N)
            filters = []
            for k in range(1, self.embedding_dim + 1):
                f = np.cos(k * np.pi * x) if k % 2 == 1 else np.sin(k * np.pi * x)
                filters.append(f / np.linalg.norm(f))
            self.projection_weights = np.column_stack(filters)
            self.is_fitted = True

        v = view.copy()
        nans = np.isnan(v)
        if np.all(nans):
            return np.zeros(self.embedding_dim, dtype=np.float32)
        elif np.any(nans):
            x = np.arange(N)
            v = np.interp(x, x[~nans], v[~nans])

        diff = v - 1.0
        embedding = np.dot(diff, self.projection_weights)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def encode_batch(self, views: np.ndarray) -> np.ndarray:
        """Encode batch of views (M, N) -> (M, embedding_dim)."""
        return np.array([self.encode(row) for row in views], dtype=np.float32)


# Backward-compatible alias
LightCurveViewEncoder = LightCurvePCAPrototypeEncoder
