"""
stagbdt/loss.py
===============
Asymmetric least squares (expectile) loss function:
  L(y, yhat) = w * (y - yhat)^2
  where w = alpha if y >= yhat else (1 - alpha).

Gradient and Hessian (with respect to yhat):
  g = -2 * w * (y - yhat)
  h = 2 * w

Configurable alpha in (0, 1). alpha=0.5 corresponds to symmetric L2 squared loss.
Note: This is asymmetric least squares (expectile regression), NOT pinball loss.
"""

import numpy as np


class AsymmetricLeastSquaresLoss:
    """
    Asymmetric least squares (expectile) loss for Newton boosting.
    """

    def __init__(self, alpha: float = 0.5):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)

    def loss(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """Compute mean asymmetric squared loss."""
        diff = y - yhat
        w = np.where(diff >= 0.0, self.alpha, 1.0 - self.alpha)
        return float(np.mean(w * (diff ** 2)))

    def gradient_and_hessian(self, y: np.ndarray, yhat: np.ndarray):
        """
        Compute gradient and hessian with respect to yhat.
        Returns:
            g: np.ndarray of shape (N,)
            h: np.ndarray of shape (N,)
        """
        diff = y - yhat
        w = np.where(diff >= 0.0, self.alpha, 1.0 - self.alpha)
        g = -2.0 * w * diff
        h = 2.0 * w
        return g, h

    def initial_prediction(self, y: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> float:
        """
        Find constant scalar c minimizing mean expectile loss on y
        using iterative reweighted least squares (IRLS).
        """
        c = float(np.mean(y))
        for _ in range(max_iter):
            w = np.where(y >= c, self.alpha, 1.0 - self.alpha)
            sum_w = np.sum(w)
            if sum_w <= 0.0:
                break
            c_next = float(np.sum(w * y) / sum_w)
            if abs(c_next - c) < tol:
                c = c_next
                break
            c = c_next
        return c
