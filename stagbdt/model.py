"""
stagbdt/model.py
================
Pure NumPy implementation of ST-AGBDT (Self-Throttling Asymmetric Gradient Boosted Decision Trees).
Features:
  - Newton boosting with histogram splits (64 bins)
  - Depth <= 5, shrinkage, L2 regularization, min child hessian, fixed seeds
  - Asymmetric least squares (expectile) loss: g = -2*w*(y-yhat), h = 2*w
  - Flat perfect-binary array compiled trees (child = 2i+1+(x>thr))
  - Fast single-sample paths: recursive, old flat, and stacked 2-D level-walk
  - Early-exit gating after K0 trees with absolute change verification
  - Split-conformal upper bounds
  - JSON serialization / deserialization with lean schema option (zero external ML dependencies)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from stagbdt.loss import AsymmetricLeastSquaresLoss
from stagbdt.tree import (
    FlatTree,
    Tree,
    build_tree,
    compute_feature_bin_edges,
    digitize_features,
)


class STAGBDT:
    """
    Self-Throttling Asymmetric Gradient Boosted Decision Tree (ST-AGBDT) regressor.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        alpha: float = 0.85,
        l2_reg: float = 1.0,
        min_child_hessian: float = 1e-3,
        n_bins: int = 64,
        random_state: int = 42,
    ):
        if max_depth > 5:
            raise ValueError(f"max_depth must be <= 5 per specification, got {max_depth}")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)
        self.min_child_hessian = float(min_child_hessian)
        self.n_bins = int(n_bins)
        self.random_state = int(random_state)

        self.loss_func = AsymmetricLeastSquaresLoss(alpha=self.alpha)
        self.base_pred_: float = 0.0
        self.bin_edges_: List[np.ndarray] = []
        self.trees_: List[Tree] = []
        self.flat_trees_: List[FlatTree] = []
        self.feature_names_: Optional[List[str]] = None

        # Stacked 2D array representation
        self.stacked_features: Optional[np.ndarray] = None
        self.stacked_thresholds: Optional[np.ndarray] = None
        self.stacked_leaves: Optional[np.ndarray] = None
        self._leaf_offset: int = (1 << self.max_depth) - 1
        self._k_range: Optional[np.ndarray] = None

    def _compile_stacked(self) -> None:
        """Compile all flat trees into 2-D stacked arrays for fast vectorized level traversal."""
        if not self.flat_trees_:
            return
        K = len(self.flat_trees_)
        self.stacked_features = np.array([ft.feature_indices for ft in self.flat_trees_], dtype=np.int32)
        self.stacked_thresholds = np.array([ft.thresholds for ft in self.flat_trees_], dtype=np.float64)
        self.stacked_leaves = np.array([ft.leaf_values for ft in self.flat_trees_], dtype=np.float64)
        self._leaf_offset = (1 << self.max_depth) - 1
        self._k_range = np.arange(K, dtype=np.int32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> STAGBDT:
        """Fit ST-AGBDT ensemble using Newton boosting."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N, D = X.shape

        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"f{i}" for i in range(D)]

        # 1. Base prediction minimizing initial loss
        self.base_pred_ = self.loss_func.initial_prediction(y)

        # 2. Compute 64-bin edges per feature
        self.bin_edges_ = compute_feature_bin_edges(X, n_bins=self.n_bins)
        X_binned = digitize_features(X, self.bin_edges_)

        # 3. Newton boosting iterations
        self.trees_ = []
        self.flat_trees_ = []
        yhat = np.full(N, self.base_pred_, dtype=np.float64)

        for _ in range(self.n_estimators):
            g, h = self.loss_func.gradient_and_hessian(y, yhat)
            rec_tree, flat_tree = build_tree(
                X=X,
                X_binned=X_binned,
                g=g,
                h=h,
                bin_edges=self.bin_edges_,
                max_depth=self.max_depth,
                l2_reg=self.l2_reg,
                min_child_hessian=self.min_child_hessian,
                shrinkage=self.learning_rate,
            )
            self.trees_.append(rec_tree)
            self.flat_trees_.append(flat_tree)
            yhat += flat_tree.predict_batch(X)

        self._compile_stacked()
        return self

    def predict(self, X: np.ndarray, use_flat: bool = True) -> np.ndarray:
        """Batch prediction."""
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        out = np.full(N, self.base_pred_, dtype=np.float64)
        if use_flat:
            for ft in self.flat_trees_:
                out += ft.predict_batch(X)
        else:
            for t in self.trees_:
                out += t.predict_batch(X)
        return out

    def predict_one(self, x: np.ndarray, use_flat: bool = True) -> float:
        """Single-sample prediction path."""
        x = np.asarray(x, dtype=np.float64)
        val = self.base_pred_
        if use_flat:
            for ft in self.flat_trees_:
                val += ft.predict_one(x)
        else:
            for t in self.trees_:
                val += t.predict_one(x)
        return float(val)

    def predict_one_stacked(self, x: np.ndarray) -> float:
        """
        Stacked single-sample path:
        Walks depth levels across all K trees simultaneously in 2-D numpy arrays.
        """
        if self.stacked_features is None:
            self._compile_stacked()
        x = np.asarray(x, dtype=np.float64)
        K = len(self.flat_trees_)
        nodes = np.zeros(K, dtype=np.int32)
        for _ in range(self.max_depth):
            feats = self.stacked_features[self._k_range, nodes]
            thrs = self.stacked_thresholds[self._k_range, nodes]
            cond = (x[feats] > thrs).astype(np.int32)
            nodes = 2 * nodes + 1 + cond
        return float(self.base_pred_ + np.sum(self.stacked_leaves[self._k_range, nodes - self._leaf_offset]))

    def predict_staged(self, X: np.ndarray) -> List[np.ndarray]:
        """Return cumulative predictions after each tree."""
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        cur = np.full(N, self.base_pred_, dtype=np.float64)
        staged = []
        for ft in self.flat_trees_:
            cur = cur + ft.predict_batch(X)
            staged.append(cur.copy())
        return staged

    def predict_with_early_exit(
        self,
        x: np.ndarray,
        K0: int,
        threshold: float,
        margin: float,
        eps: float = 1.0,
    ) -> Tuple[float, int, bool]:
        """
        Single-sample early exit evaluation.
        Exit after K0 trees if:
          (yhat_K0 + margin < threshold) AND (sum of ABSOLUTE changes of last 5 trees < eps)
        Returns:
          (final_prediction, trees_evaluated, exited_early)
        """
        x = np.asarray(x, dtype=np.float64)
        val = self.base_pred_
        K0 = min(K0, len(self.flat_trees_))
        last_5_vals: List[float] = []

        # Evaluate up to K0
        for i in range(K0):
            step = self.flat_trees_[i].predict_one(x)
            val += step
            last_5_vals.append(step)
            if len(last_5_vals) > 5:
                last_5_vals.pop(0)

        # Check early exit condition: sum of ABSOLUTE changes of last 5 trees
        last_5_abs_change = sum(abs(v) for v in last_5_vals)
        if (val + margin < threshold) and (last_5_abs_change < eps):
            return float(val), K0, True

        # Otherwise continue to full ensemble
        for i in range(K0, len(self.flat_trees_)):
            val += self.flat_trees_[i].predict_one(x)

        return float(val), len(self.flat_trees_), False

    def predict_batch_early_exit(
        self,
        X: np.ndarray,
        K0: int,
        threshold: float,
        margin: float,
        eps: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch prediction with early exit.
        Returns:
          (predictions, trees_evaluated, exited_mask)
        """
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        preds = np.empty(N, dtype=np.float64)
        trees_used = np.empty(N, dtype=np.int32)
        exited = np.empty(N, dtype=bool)

        for i in range(N):
            p, k, ex = self.predict_with_early_exit(X[i], K0, threshold, margin, eps)
            preds[i] = p
            trees_used[i] = k
            exited[i] = ex

        return preds, trees_used, exited

    def to_dict(self, include_recursive: bool = True) -> Dict[str, Any]:
        """Serialize model to JSON-compatible dictionary."""
        d = {
            "model_type": "STAGBDT",
            "full_name": "Self-Throttling Asymmetric Gradient Boosted Decision Trees",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "alpha": self.alpha,
            "l2_reg": self.l2_reg,
            "min_child_hessian": self.min_child_hessian,
            "n_bins": self.n_bins,
            "random_state": self.random_state,
            "base_pred_": self.base_pred_,
            "feature_names_": self.feature_names_,
            "bin_edges_": [edges.tolist() for edges in self.bin_edges_],
            "flat_trees_": [ft.to_dict() for ft in self.flat_trees_],
        }
        if include_recursive and self.trees_:
            d["trees_"] = [t.to_dict() for t in self.trees_]
        return d

    def save_json(self, filepath: str, include_recursive: bool = True) -> None:
        """Save model to JSON."""
        d = self.to_dict(include_recursive=include_recursive)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(d, f)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> STAGBDT:
        """Restore model from dictionary."""
        model = cls(
            n_estimators=d["n_estimators"],
            learning_rate=d["learning_rate"],
            max_depth=d["max_depth"],
            alpha=d["alpha"],
            l2_reg=d["l2_reg"],
            min_child_hessian=d["min_child_hessian"],
            n_bins=d["n_bins"],
            random_state=d["random_state"],
        )
        model.base_pred_ = float(d["base_pred_"])
        model.feature_names_ = d.get("feature_names_")
        model.bin_edges_ = [np.array(edges, dtype=np.float64) for edges in d["bin_edges_"]]
        model.flat_trees_ = [FlatTree.from_dict(ft_d) for ft_d in d["flat_trees_"]]
        if "trees_" in d:
            model.trees_ = [Tree.from_dict(t_d) for t_d in d["trees_"]]
        else:
            model.trees_ = []
        model._compile_stacked()
        return model

    @classmethod
    def load_json(cls, filepath: str) -> STAGBDT:
        """Load model from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
