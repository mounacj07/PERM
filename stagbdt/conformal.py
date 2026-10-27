"""
stagbdt/conformal.py
====================
Split-conformal calibration for upper bounds and early-exit gating.
Pure numpy implementation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


class ConformalCalibrator:
    """
    Split-conformal upper bound predictor:
      y_upper = yhat + q_margin
    where q_margin is the conformal quantile of nonconformity score s = y - yhat
    computed on calibration runs.
    """

    def __init__(self, coverage_level: float = 0.90):
        if not (0.0 < coverage_level < 1.0):
            raise ValueError(f"coverage_level must be in (0, 1), got {coverage_level}")
        self.coverage_level = float(coverage_level)
        self.q_margin_: Optional[float] = None

    def calibrate(self, y_cal: np.ndarray, yhat_cal: np.ndarray) -> float:
        """
        Compute conformal margin q from calibration predictions.
        s = y - yhat (one-sided upper bound score)
        """
        s = y_cal - yhat_cal
        n = len(s)
        if n == 0:
            self.q_margin_ = 0.0
            return 0.0
        # Finite-sample conformal quantile: ceil((n+1) * coverage_level) / n
        q_idx = int(np.ceil((n + 1) * self.coverage_level))
        q_idx = min(n - 1, max(0, q_idx - 1))
        sorted_s = np.sort(s)
        self.q_margin_ = float(sorted_s[q_idx])
        return self.q_margin_

    def predict_upper(self, yhat: np.ndarray) -> np.ndarray:
        if self.q_margin_ is None:
            raise RuntimeError("Calibrator must be calibrated before predict_upper")
        return yhat + self.q_margin_

    def evaluate_coverage(self, y_test: np.ndarray, yhat_test: np.ndarray) -> float:
        y_upper = self.predict_upper(yhat_test)
        covered = (y_test <= y_upper)
        return float(np.mean(covered))


class EarlyExitCalibrator:
    """
    Early-exit gating calibrator:
      Exit early after K0 trees if:
        (yhat_K0 + margin < threshold) AND (abs(last_5_trees_change) < eps)
      where margin = conformal quantile of positive (y - yhat_K0) on calibration runs.
    """

    def __init__(self, coverage_level: float = 0.95):
        self.coverage_level = float(coverage_level)
        self.margin_: Optional[float] = None

    def calibrate(self, y_cal: np.ndarray, yhat_K0_cal: np.ndarray) -> float:
        residuals = y_cal - yhat_K0_cal
        pos_residuals = residuals[residuals > 0.0]
        if len(pos_residuals) == 0:
            self.margin_ = 0.0
            return 0.0
        n = len(pos_residuals)
        q_idx = int(np.ceil((n + 1) * self.coverage_level))
        q_idx = min(n - 1, max(0, q_idx - 1))
        sorted_pos = np.sort(pos_residuals)
        self.margin_ = float(sorted_pos[q_idx])
        return self.margin_
