"""
Self-supervised neural encoder for transit light-curve representation learning.

Scientific principles:
---------------------
1. Self-supervised objective:
   - Masked Cadence Autoencoding (MAE): reconstructs randomly dropped cadences
     (simulating instrument dropouts and data gaps without corrupting orbital period).
   - View Consistency: minimizes latent embedding divergence between two differently
     augmented views of the same phase-folded transit signal.
2. Architecture:
   - 2-layer MLP encoder with LeakyReLU activations and L2 feature normalization.
   - 2-layer MLP decoder for pretraining view reconstruction.
   - Vectorized forward, backward, and Adam optimization for fast CPU execution.
3. Provenance & Limitations:
   - Intended for CPU training and fast smoke-test verification on representative batches.
   - Large-scale pretraining on the full 60+ sector TESS archive requires GPU compute clusters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from astrovet.ssl.augmentations import random_cadence_mask, bounded_photometric_jitter


class SelfSupervisedTransitEncoder:
    """
    Self-supervised representation learner for phase-folded light curves (N=200).
    Optimizes a joint masked reconstruction and view-consistency objective.
    """

    def __init__(
        self,
        input_dim: int = 200,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
        view_loss_weight: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.view_loss_weight = view_loss_weight
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.is_fitted = False
        self.model_type = "self_supervised_neural_encoder"

        # Initialize network weights (He / Kaiming normal)
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, embedding_dim))
        self.b2 = np.zeros(embedding_dim)

        # Decoder weights
        self.W3 = self.rng.normal(0.0, np.sqrt(2.0 / embedding_dim), (embedding_dim, hidden_dim))
        self.b3 = np.zeros(hidden_dim)
        self.W4 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, input_dim))
        self.b4 = np.zeros(input_dim)

        # Adam optimizer states
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._step = 0

    def _params(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
            "W4": self.W4, "b4": self.b4,
        }

    @staticmethod
    def _leaky_relu(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def _leaky_relu_grad(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)

    def _clean_input(self, views: np.ndarray) -> np.ndarray:
        """Impute gaps and put each folded curve on a robust, unitless scale.

        FITS products may contain flux near one, ppm values, or unnormalized
        instrumental units. Per-view median/MAD normalization makes the SSL
        objective insensitive to those unit choices while retaining transit
        morphology. The original flux remains untouched in the physics branch.
        """
        arr = np.asarray(views, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected views with shape (N, {self.input_dim}); got {arr.shape}")
        cleaned = np.empty_like(arr, dtype=np.float64)
        for i in range(len(arr)):
            row = arr[i]
            missing = ~np.isfinite(row)
            nans = missing
            if np.all(nans):
                cleaned[i] = 0.0
            elif np.any(nans):
                idx = np.arange(len(row))
                cleaned[i] = np.interp(idx, idx[~nans], row[~nans])
            else:
                cleaned[i] = row
            finite = cleaned[i]
            center = float(np.median(finite))
            mad = float(np.median(np.abs(finite - center)))
            # A floor protects nearly flat curves from numerical blow-up.
            scale = max(1.4826 * mad, 1e-8 * max(1.0, abs(center)))
            cleaned[i] = np.clip((finite - center) / scale, -8.0, 8.0)
        return cleaned

    def encode(self, views: np.ndarray) -> np.ndarray:
        """
        Extract compact representation embeddings (B, embedding_dim).
        """
        X = self._clean_input(views)
        h1 = self._leaky_relu(X @ self.W1 + self.b1)
        z = h1 @ self.W2 + self.b2
        # L2 normalize embeddings for stable downstream similarity / fusion
        norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        return z / norms

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruct phase-folded flux view from latent embedding (B, input_dim).
        """
        h3 = self._leaky_relu(z @ self.W3 + self.b3)
        return h3 @ self.W4 + self.b4

    def train_epoch(
        self,
        unlabeled_views: np.ndarray,
        lr: float = 0.005,
        batch_size: int = 16,
    ) -> Dict[str, float]:
        """
        Execute one training epoch with joint masked autoencoding and view consistency.
        """
        X_clean = self._clean_input(unlabeled_views)
        N = len(X_clean)
        indices = np.arange(N)
        self.rng.shuffle(indices)

        total_loss = 0.0
        total_recon = 0.0
        total_view = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = indices[start:start + batch_size]
            B = len(batch_idx)
            if B == 0:
                continue

            X_target = X_clean[batch_idx]

            # Generate View 1: Masked with cadence dropouts
            mask_1 = np.ones((B, self.input_dim), dtype=bool)
            X_aug1 = np.empty_like(X_target)
            for i in range(B):
                masked = random_cadence_mask(X_target[i], mask_fraction=0.15, rng=self.rng)
                mask_1[i] = np.isnan(masked)
                # Inputs are robust-centered/scaled, so zero is the neutral
                # normalized baseline at masked cadences.
                X_aug1[i] = np.where(mask_1[i], 0.0, masked)
            X_aug1 = bounded_photometric_jitter(X_aug1, noise_sigma=0.02, rng=self.rng)

            # Generate View 2: Different photometric jitter + slight mask
            X_aug2 = bounded_photometric_jitter(X_target, noise_sigma=0.04, rng=self.rng)

            # --- Forward Pass View 1 ---
            h1_1 = X_aug1 @ self.W1 + self.b1
            a1_1 = self._leaky_relu(h1_1)
            z1 = a1_1 @ self.W2 + self.b2

            h3_1 = z1 @ self.W3 + self.b3
            a3_1 = self._leaky_relu(h3_1)
            x_recon = a3_1 @ self.W4 + self.b4

            # --- Forward Pass View 2 ---
            h1_2 = X_aug2 @ self.W1 + self.b1
            a1_2 = self._leaky_relu(h1_2)
            z2 = a1_2 @ self.W2 + self.b2

            # --- Compute Losses ---
            # 1. Masked reconstruction loss (focusing on masked coordinates)
            recon_diff = x_recon - X_target
            mask_weights = np.where(mask_1, 2.0, 1.0)
            recon_loss = np.mean(mask_weights * (recon_diff ** 2))

            # 2. View consistency loss between z1 and z2
            view_diff = z1 - z2
            view_loss = np.mean(view_diff ** 2)

            batch_loss = recon_loss + self.view_loss_weight * view_loss

            # --- Backward Pass (Gradients via Chain Rule) ---
            d_x_recon = (2.0 / (B * self.input_dim)) * (mask_weights * recon_diff)
            d_W4 = a3_1.T @ d_x_recon
            d_b4 = np.sum(d_x_recon, axis=0)

            d_a3_1 = d_x_recon @ self.W4.T
            d_h3_1 = d_a3_1 * self._leaky_relu_grad(h3_1)
            d_W3 = z1.T @ d_h3_1
            d_b3 = np.sum(d_h3_1, axis=0)

            d_z1 = d_h3_1 @ self.W3.T + (2.0 * self.view_loss_weight / (B * self.embedding_dim)) * view_diff
            d_W2 = a1_1.T @ d_z1
            d_b2 = np.sum(d_z1, axis=0)

            d_a1_1 = d_z1 @ self.W2.T
            d_h1_1 = d_a1_1 * self._leaky_relu_grad(h1_1)
            d_W1 = X_aug1.T @ d_h1_1
            d_b1 = np.sum(d_h1_1, axis=0)

            # View 2 branch gradients for encoder
            d_z2 = (-2.0 * self.view_loss_weight / (B * self.embedding_dim)) * view_diff
            d_W2 += a1_2.T @ d_z2
            d_b2 += np.sum(d_z2, axis=0)

            d_a1_2 = d_z2 @ self.W2.T
            d_h1_2 = d_a1_2 * self._leaky_relu_grad(h1_2)
            d_W1 += X_aug2.T @ d_h1_2
            d_b1 += np.sum(d_h1_2, axis=0)

            grads = {
                "W1": d_W1, "b1": d_b1,
                "W2": d_W2, "b2": d_b2,
                "W3": d_W3, "b3": d_b3,
                "W4": d_W4, "b4": d_b4,
            }

            # Bound the combined gradient before Adam so unusual/outlier curves
            # cannot send the hand-written NumPy optimizer into a divergent run.
            grad_norm = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
            if not np.isfinite(grad_norm):
                raise FloatingPointError("Non-finite SSL gradient; input data or model state is invalid")
            if grad_norm > 5.0:
                scale = 5.0 / grad_norm
                grads = {name: grad * scale for name, grad in grads.items()}

            # --- Adam Optimizer Update ---
            self._step += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            params = self._params()

            for k in params:
                g = grads[k]
                self._m[k] = beta1 * self._m[k] + (1.0 - beta1) * g
                self._v[k] = beta2 * self._v[k] + (1.0 - beta2) * (g ** 2)
                m_hat = self._m[k] / (1.0 - beta1 ** self._step)
                v_hat = self._v[k] / (1.0 - beta2 ** self._step)
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if not all(np.all(np.isfinite(param)) for param in params.values()):
                raise FloatingPointError("SSL optimizer produced non-finite weights")

            total_loss += batch_loss
            total_recon += recon_loss
            total_view += view_loss
            n_batches += 1

        return {
            "loss": float(total_loss / max(1, n_batches)),
            "recon_loss": float(total_recon / max(1, n_batches)),
            "view_loss": float(total_view / max(1, n_batches)),
        }

    def fit(
        self,
        unlabeled_views: np.ndarray,
        epochs: int = 5,
        lr: float = 0.005,
        batch_size: int = 16,
    ) -> Dict[str, List[float]]:
        """
        Train self-supervised representation learner over unlabeled phase-folded light curves.
        """
        history: Dict[str, List[float]] = {"loss": [], "recon_loss": [], "view_loss": []}
        for ep in range(epochs):
            metrics = self.train_epoch(unlabeled_views, lr=lr, batch_size=batch_size)
            history["loss"].append(metrics["loss"])
            history["recon_loss"].append(metrics["recon_loss"])
            history["view_loss"].append(metrics["view_loss"])

        self.is_fitted = True
        return history

    def save_checkpoint(
        self,
        output_path: str | Path,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save encoder parameters and metadata to compressed numpy archive (.npz)."""
        import json
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        prov = provenance or {}
        data_source = prov.get("data_source", "REAL_TESS_FITS")
        np.savez_compressed(
            out,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            W4=self.W4, b4=self.b4,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            is_fitted=np.array([self.is_fitted]),
            data_source=np.array([str(data_source)]),
            provenance_json=np.array([json.dumps(prov)]),
        )
        return out

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str | Path) -> SelfSupervisedTransitEncoder:
        """Load a trained encoder instance from a saved .npz checkpoint."""
        p = Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")
        data = np.load(p)
        input_dim = int(data.get("input_dim", 200))
        hidden_dim = int(data.get("hidden_dim", 64))
        embedding_dim = int(data.get("embedding_dim", 16))

        instance = cls(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        instance.W1 = data["W1"]
        instance.b1 = data["b1"]
        instance.W2 = data["W2"]
        instance.b2 = data["b2"]
        instance.W3 = data["W3"]
        instance.b3 = data["b3"]
        instance.W4 = data["W4"]
        instance.b4 = data["b4"]
        instance.is_fitted = bool(data["is_fitted"][0]) if "is_fitted" in data else True
        prov_dict = {}
        if "provenance_json" in data:
            try:
                import json
                prov_dict = json.loads(str(data["provenance_json"][0]))
            except Exception:
                pass
        instance.provenance = prov_dict
        instance.data_source = str(data["data_source"][0]) if "data_source" in data else prov_dict.get("data_source", "UNKNOWN")
        return instance
