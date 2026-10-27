"""
stagbdt/tree.py
===============
Histogram-based Newton decision tree with depth <= 5, L2 regularization,
min child hessian, and compilation to flat perfect-binary arrays:
  child = 2*i + 1 + int(x[thr_feat] > thr)

Includes:
  - Tree: recursive representation with predict_one and predict_batch
  - FlatTree: compiled flat perfect-binary array with vectorized batch and sub-microsecond single-sample path
  - build_tree: histogram-split builder (64 bins)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Tree:
    """Recursive decision tree node."""

    def __init__(
        self,
        is_leaf: bool = False,
        leaf_value: float = 0.0,
        feature_idx: int = 0,
        threshold: float = 0.0,
        left: Optional[Tree] = None,
        right: Optional[Tree] = None,
        depth: int = 0,
    ):
        self.is_leaf = is_leaf
        self.leaf_value = float(leaf_value)
        self.feature_idx = int(feature_idx)
        self.threshold = float(threshold)
        self.left = left
        self.right = right
        self.depth = depth

    def predict_one(self, x: np.ndarray) -> float:
        if self.is_leaf:
            return self.leaf_value
        if x[self.feature_idx] > self.threshold:
            return self.right.predict_one(x)
        return self.left.predict_one(x)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if self.is_leaf:
            return np.full(X.shape[0], self.leaf_value, dtype=np.float64)
        mask = X[:, self.feature_idx] > self.threshold
        out = np.empty(X.shape[0], dtype=np.float64)
        if np.any(~mask):
            out[~mask] = self.left.predict_batch(X[~mask])
        if np.any(mask):
            out[mask] = self.right.predict_batch(X[mask])
        return out

    def to_dict(self) -> Dict[str, Any]:
        if self.is_leaf:
            return {
                "is_leaf": True,
                "leaf_value": self.leaf_value,
                "depth": self.depth,
            }
        return {
            "is_leaf": False,
            "feature_idx": self.feature_idx,
            "threshold": self.threshold,
            "depth": self.depth,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Tree:
        if d.get("is_leaf", False):
            return cls(
                is_leaf=True,
                leaf_value=d["leaf_value"],
                depth=d.get("depth", 0),
            )
        return cls(
            is_leaf=False,
            feature_idx=d["feature_idx"],
            threshold=d["threshold"],
            depth=d.get("depth", 0),
            left=cls.from_dict(d["left"]),
            right=cls.from_dict(d["right"]),
        )


class FlatTree:
    """
    Flat perfect-binary tree array representation.
    Internal nodes: 0 .. 2^depth - 2
    Leaf nodes:     2^depth - 1 .. 2^(depth+1) - 2
    Traversal:
      child = 2 * i + 1 + int(x[feature_indices[i]] > thresholds[i])
    """

    def __init__(
        self,
        depth: int,
        feature_indices: np.ndarray,
        thresholds: np.ndarray,
        leaf_values: np.ndarray,
    ):
        self.depth = int(depth)
        self.feature_indices = np.asarray(feature_indices, dtype=np.int32)
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.leaf_values = np.asarray(leaf_values, dtype=np.float64)
        self._leaf_offset = (1 << self.depth) - 1

    def predict_one(self, x: np.ndarray) -> float:
        """Fast single-sample prediction path."""
        node = 0
        for _ in range(self.depth):
            f = self.feature_indices[node]
            t = self.thresholds[node]
            node = 2 * node + 1 + int(x[f] > t)
        return float(self.leaf_values[node - self._leaf_offset])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized batch prediction path."""
        N = X.shape[0]
        node = np.zeros(N, dtype=np.int32)
        arange_N = np.arange(N)
        for _ in range(self.depth):
            f = self.feature_indices[node]
            t = self.thresholds[node]
            cond = (X[arange_N, f] > t).astype(np.int32)
            node = 2 * node + 1 + cond
        return self.leaf_values[node - self._leaf_offset]

    @classmethod
    def from_tree(cls, tree: Tree, depth: int) -> FlatTree:
        """Compile a recursive Tree of depth <= D into a perfect binary FlatTree of depth D."""
        n_internal = (1 << depth) - 1
        n_leaves = 1 << depth
        feat_arr = np.zeros(n_internal, dtype=np.int32)
        thr_arr = np.zeros(n_internal, dtype=np.float64)
        leaf_arr = np.zeros(n_leaves, dtype=np.float64)

        def _traverse(node: Tree, curr_idx: int, curr_depth: int):
            if curr_depth == depth:
                leaf_idx = curr_idx - n_internal
                leaf_arr[leaf_idx] = node.leaf_value
                return
            if node.is_leaf:
                # Pad out leaf down to target depth
                feat_arr[curr_idx] = 0
                thr_arr[curr_idx] = 0.0
                _traverse(node, 2 * curr_idx + 1, curr_depth + 1)
                _traverse(node, 2 * curr_idx + 2, curr_depth + 1)
            else:
                feat_arr[curr_idx] = node.feature_idx
                thr_arr[curr_idx] = node.threshold
                _traverse(node.left, 2 * curr_idx + 1, curr_depth + 1)
                _traverse(node.right, 2 * curr_idx + 2, curr_depth + 1)

        _traverse(tree, 0, 0)
        return cls(depth, feat_arr, thr_arr, leaf_arr)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "feature_indices": self.feature_indices.tolist(),
            "thresholds": self.thresholds.tolist(),
            "leaf_values": self.leaf_values.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FlatTree:
        return cls(
            depth=d["depth"],
            feature_indices=np.array(d["feature_indices"], dtype=np.int32),
            thresholds=np.array(d["thresholds"], dtype=np.float64),
            leaf_values=np.array(d["leaf_values"], dtype=np.float64),
        )


def compute_feature_bin_edges(X: np.ndarray, n_bins: int = 64) -> List[np.ndarray]:
    """Compute up to (n_bins - 1) threshold split points per feature."""
    D = X.shape[1]
    bin_edges: List[np.ndarray] = []
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    for j in range(D):
        col = X[:, j]
        unique_vals = np.unique(col)
        if len(unique_vals) <= 1:
            edges = np.array([], dtype=np.float64)
        elif len(unique_vals) <= n_bins:
            # Midpoints between unique sorted values
            edges = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        else:
            q_edges = np.percentile(col, quantiles)
            edges = np.unique(q_edges)
        bin_edges.append(edges)
    return bin_edges


def digitize_features(X: np.ndarray, bin_edges: List[np.ndarray]) -> np.ndarray:
    """Digitize X into integer bin indices in [0, n_bins-1]."""
    N, D = X.shape
    X_binned = np.empty((N, D), dtype=np.int16)
    for j in range(D):
        edges = bin_edges[j]
        if len(edges) == 0:
            X_binned[:, j] = 0
        else:
            X_binned[:, j] = np.searchsorted(edges, X[:, j], side="right")
    return X_binned


def build_tree(
    X: np.ndarray,
    X_binned: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    bin_edges: List[np.ndarray],
    max_depth: int = 5,
    l2_reg: float = 1.0,
    min_child_hessian: float = 1e-3,
    shrinkage: float = 0.1,
) -> Tuple[Tree, FlatTree]:
    """
    Build a Newton tree using histogram splits, apply shrinkage to leaf values,
    and return both (Tree, FlatTree).
    """
    N, D = X.shape

    def _make_leaf(indices: np.ndarray, depth: int) -> Tree:
        G = float(np.sum(g[indices])) if len(indices) > 0 else 0.0
        H = float(np.sum(h[indices])) if len(indices) > 0 else 0.0
        # Optimal Newton step: - G / (H + lambda), scaled by shrinkage
        weight = - (G / (H + l2_reg)) * shrinkage
        return Tree(is_leaf=True, leaf_value=weight, depth=depth)

    def _split_node(indices: np.ndarray, depth: int) -> Tree:
        G = float(np.sum(g[indices]))
        H = float(np.sum(h[indices]))

        if depth >= max_depth or len(indices) <= 1 or H < 2.0 * min_child_hessian:
            return _make_leaf(indices, depth)

        parent_score = (G * G) / (H + l2_reg)

        best_gain = -1.0
        best_feat = -1
        best_thr = 0.0
        best_left_idx = None
        best_right_idx = None

        for j in range(D):
            edges = bin_edges[j]
            if len(edges) == 0:
                continue
            n_bins = len(edges) + 1
            bins_j = X_binned[indices, j]
            G_hist = np.bincount(bins_j, weights=g[indices], minlength=n_bins)
            H_hist = np.bincount(bins_j, weights=h[indices], minlength=n_bins)

            G_left = np.cumsum(G_hist[:-1])
            H_left = np.cumsum(H_hist[:-1])
            G_right = G - G_left
            H_right = H - H_left

            valid = (H_left >= min_child_hessian) & (H_right >= min_child_hessian)
            if not np.any(valid):
                continue

            gain = 0.5 * (
                (G_left[valid] ** 2) / (H_left[valid] + l2_reg)
                + (G_right[valid] ** 2) / (H_right[valid] + l2_reg)
                - parent_score
            )
            max_idx = np.argmax(gain)
            if gain[max_idx] > best_gain:
                valid_split_indices = np.where(valid)[0]
                split_bin = valid_split_indices[max_idx]
                best_gain = float(gain[max_idx])
                best_feat = j
                best_thr = float(edges[split_bin])

        if best_gain <= 0.0 or best_feat < 0:
            return _make_leaf(indices, depth)

        col = X[indices, best_feat]
        left_mask = col <= best_thr
        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]

        if len(left_idx) == 0 or len(right_idx) == 0:
            return _make_leaf(indices, depth)

        left_child = _split_node(left_idx, depth + 1)
        right_child = _split_node(right_idx, depth + 1)

        return Tree(
            is_leaf=False,
            feature_idx=best_feat,
            threshold=best_thr,
            left=left_child,
            right=right_child,
            depth=depth,
        )

    all_indices = np.arange(N)
    rec_tree = _split_node(all_indices, 0)
    flat_tree = FlatTree.from_tree(rec_tree, depth=max_depth)
    return rec_tree, flat_tree
