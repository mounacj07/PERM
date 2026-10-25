import json
import os
import math
import numpy as np


class CustomNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, gain=0.0, n_samples=0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain
        self.n_samples = n_samples

    def is_leaf(self):
        return self.value is not None

    def to_dict(self):
        if self.is_leaf():
            return {"value": float(self.value), "n_samples": int(self.n_samples)}
        return {
            "feature_idx": int(self.feature_idx),
            "threshold": float(self.threshold),
            "gain": float(self.gain),
            "n_samples": int(self.n_samples),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        if "value" in d:
            return cls(value=d["value"], n_samples=d.get("n_samples", 0))
        return cls(
            feature_idx=d["feature_idx"],
            threshold=d["threshold"],
            gain=d.get("gain", 0.0),
            n_samples=d.get("n_samples", 0),
            left=cls.from_dict(d["left"]),
            right=cls.from_dict(d["right"]),
        )


class CustomDecisionTreeRegressor:
    def __init__(self, max_depth=6, min_samples_split=4, min_samples_leaf=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None

    def fit(self, X, y, n_features_total=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        if n_features_total is None:
            n_features_total = n_features

        self.feature_importances_ = np.zeros(n_features_total, dtype=np.float64)

        if self.max_features is None:
            n_sub_features = n_features
        elif isinstance(self.max_features, float):
            n_sub_features = max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, str) and self.max_features == "sqrt":
            n_sub_features = max(1, int(np.sqrt(n_features)))
        else:
            n_sub_features = min(n_features, int(self.max_features))

        self.root = self._build_tree(X, y, depth=0, n_sub_features=n_sub_features)

        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def _build_tree(self, X, y, depth, n_sub_features):
        n_samples, n_features = X.shape
        leaf_val = float(np.mean(y))

        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.std(y) < 1e-7):
            return CustomNode(value=leaf_val, n_samples=n_samples)

        if n_sub_features < n_features:
            feat_indices = np.random.choice(n_features, size=n_sub_features, replace=False)
        else:
            feat_indices = np.arange(n_features)

        node_sum = float(np.sum(y))
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        cum_counts = np.arange(1, n_samples)

        for feat_idx in feat_indices:
            col = X[:, feat_idx]
            sort_idx = np.argsort(col)
            col_sorted = col[sort_idx]
            y_sorted = y[sort_idx]

            # Vectorized candidate split locations where values differ
            split_mask = col_sorted[:-1] != col_sorted[1:]
            valid_mask = (cum_counts >= self.min_samples_leaf) & ((n_samples - cum_counts) >= self.min_samples_leaf) & split_mask

            if not np.any(valid_mask):
                continue

            cum_sums = np.cumsum(y_sorted)[:-1]

            left_counts = cum_counts[valid_mask]
            right_counts = n_samples - left_counts
            left_sums = cum_sums[valid_mask]
            right_sums = node_sum - left_sums

            gains = (left_sums**2 / left_counts) + (right_sums**2 / right_counts) - (node_sum**2 / n_samples)
            
            if len(gains) == 0:
                continue

            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]

            if max_gain > best_gain:
                best_gain = max_gain
                best_feat = feat_idx
                valid_split_indices = np.where(valid_mask)[0]
                split_i = valid_split_indices[max_gain_idx]
                best_thresh = float((col_sorted[split_i] + col_sorted[split_i + 1]) / 2.0)

        if best_feat is None or best_gain <= 1e-7:
            return CustomNode(value=leaf_val, n_samples=n_samples)

        self.feature_importances_[best_feat] += best_gain

        left_mask = X[:, best_feat] <= best_thresh
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1, n_sub_features)
        right_node = self._build_tree(X[~left_mask], y[~left_mask], depth + 1, n_sub_features)

        return CustomNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_node,
            right=right_node,
            gain=best_gain,
            n_samples=n_samples
        )

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        preds = np.zeros(len(X), dtype=np.float64)
        for i in range(len(X)):
            preds[i] = self._predict_row(self.root, X[i])
        return preds

    def _predict_row(self, node, row):
        if node.is_leaf():
            return node.value
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)

    def to_dict(self):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "root": self.root.to_dict() if self.root else None,
            "feature_importances_": self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
        }

    @classmethod
    def from_dict(cls, d):
        tree = cls(
            max_depth=d["max_depth"],
            min_samples_split=d["min_samples_split"],
            min_samples_leaf=d["min_samples_leaf"],
        )
        tree.root = CustomNode.from_dict(d["root"])
        if d.get("feature_importances_"):
            tree.feature_importances_ = np.array(d["feature_importances_"], dtype=np.float64)
        return tree


class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=150, learning_rate=0.08, max_depth=5, min_samples_split=4,
                 min_samples_leaf=2, subsample=0.85, max_features=0.85, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.init_value_ = 0.0
        self.estimators_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        self.init_value_ = float(np.mean(y))
        y_pred = np.full(n_samples, self.init_value_, dtype=np.float64)

        self.estimators_ = []
        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)

        for m in range(self.n_estimators):
            residuals = y - y_pred

            if self.subsample < 1.0:
                sub_size = max(1, int(self.subsample * n_samples))
                indices = np.random.choice(n_samples, size=sub_size, replace=False)
                X_train_m = X[indices]
                res_train_m = residuals[indices]
            else:
                X_train_m = X
                res_train_m = residuals

            tree = CustomDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_train_m, res_train_m, n_features_total=n_features)
            
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.estimators_.append(tree)

            if tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_

        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        y_pred = np.full(len(X), self.init_value_, dtype=np.float64)
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def to_dict(self):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": float(self.learning_rate),
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "subsample": float(self.subsample),
            "max_features": float(self.max_features) if isinstance(self.max_features, float) else self.max_features,
            "init_value_": float(self.init_value_),
            "feature_importances_": self.feature_importances_.tolist() if self.feature_importances_ is not None else [],
            "estimators_": [tree.to_dict() for tree in self.estimators_],
        }

    @classmethod
    def from_dict(cls, d):
        model = cls(
            n_estimators=d["n_estimators"],
            learning_rate=d["learning_rate"],
            max_depth=d["max_depth"],
            min_samples_split=d["min_samples_split"],
            min_samples_leaf=d["min_samples_leaf"],
            subsample=d["subsample"],
            max_features=d["max_features"],
        )
        model.init_value_ = d["init_value_"]
        model.feature_importances_ = np.array(d["feature_importances_"], dtype=np.float64) if d.get("feature_importances_") else None
        model.estimators_ = [CustomDecisionTreeRegressor.from_dict(t) for t in d["estimators_"]]
        return model

    def save_model(self, file_path):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_model(cls, file_path):
        with open(file_path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


class CustomNeuralNetworkRegressor:
    def __init__(self, hidden1=64, hidden2=32, lr=0.003, epochs=100, batch_size=64, l2_reg=0.001, random_state=42):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.random_state = random_state
        
        self.mean_ = None
        self.std_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self.weights_ = {}

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.mean_) / self.std_

        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y)) + 1e-8
        y_norm = (y - self.y_mean_) / self.y_std_

        n_samples, n_features = X_norm.shape

        W1 = np.random.randn(n_features, self.hidden1) * np.sqrt(2.0 / n_features)
        b1 = np.zeros((1, self.hidden1))
        W2 = np.random.randn(self.hidden1, self.hidden2) * np.sqrt(2.0 / self.hidden1)
        b2 = np.zeros((1, self.hidden2))
        W3 = np.random.randn(self.hidden2, 1) * np.sqrt(2.0 / self.hidden2)
        b3 = np.zeros((1, 1))

        params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        for epoch in range(self.epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X_norm[perm]
            y_shuffled = y_norm[perm]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                curr_bs = len(X_batch)

                z1 = X_batch @ params["W1"] + params["b1"]
                a1 = np.maximum(0, z1)
                z2 = a1 @ params["W2"] + params["b2"]
                a2 = np.maximum(0, z2)
                z3 = a2 @ params["W3"] + params["b3"]
                y_pred = z3

                dz3 = (y_pred - y_batch) / curr_bs
                dW3 = a2.T @ dz3 + self.l2_reg * params["W3"]
                db3 = np.sum(dz3, axis=0, keepdims=True)

                da2 = dz3 @ params["W3"].T
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T @ dz2 + self.l2_reg * params["W2"]
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = dz2 @ params["W2"].T
                dz1 = da1 * (z1 > 0)
                dW1 = X_batch.T @ dz1 + self.l2_reg * params["W1"]
                db1 = np.sum(dz1, axis=0, keepdims=True)

                grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

                t += 1
                for key in params:
                    m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
                    v[key] = beta2 * v[key] + (1 - beta2) * (grads[key]**2)
                    m_hat = m[key] / (1 - beta1**t)
                    v_hat = v[key] / (1 - beta2**t)
                    params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        self.weights_ = params
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.mean_) / self.std_
        a1 = np.maximum(0, X_norm @ self.weights_["W1"] + self.weights_["b1"])
        a2 = np.maximum(0, a1 @ self.weights_["W2"] + self.weights_["b2"])
        preds_norm = a2 @ self.weights_["W3"] + self.weights_["b3"]
        preds = preds_norm.ravel() * self.y_std_ + self.y_mean_
        return preds

    def to_dict(self):
        return {
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "mean_": self.mean_.tolist(),
            "std_": self.std_.tolist(),
            "y_mean_": float(self.y_mean_),
            "y_std_": float(self.y_std_),
            "weights_": {k: v.tolist() for k, v in self.weights_.items()}
        }

    @classmethod
    def from_dict(cls, d):
        nn = cls(hidden1=d["hidden1"], hidden2=d["hidden2"])
        nn.mean_ = np.array(d["mean_"], dtype=np.float64)
        nn.std_ = np.array(d["std_"], dtype=np.float64)
        nn.y_mean_ = d.get("y_mean_", 0.0)
        nn.y_std_ = d.get("y_std_", 1.0)
        nn.weights_ = {k: np.array(v, dtype=np.float64) for k, v in d["weights_"].items()}
        return nn


class CustomLogTransformedTargetRegressor:
    def __init__(self, base_regressor):
        self.base_regressor = base_regressor

    def fit(self, X, y):
        y_log = np.log1p(np.maximum(0, y))
        self.base_regressor.fit(X, y_log)
        return self

    def predict(self, X):
        y_log_pred = self.base_regressor.predict(X)
        y_log_clipped = np.clip(y_log_pred, 0, 10.0)
        return np.expm1(y_log_clipped)

    def save_model(self, file_path):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_model(cls, file_path):
        with open(file_path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @property
    def feature_importances_(self):
        return getattr(self.base_regressor, "feature_importances_", None)

    def to_dict(self):
        return {
            "type": "CustomLogTransformedTargetRegressor",
            "base_regressor": self.base_regressor.to_dict()
        }

    @classmethod
    def from_dict(cls, d):
        base = CustomGradientBoostingRegressor.from_dict(d["base_regressor"])
        return cls(base)


class CustomEnsembleRegressor:
    def __init__(self, gbdt_weight=0.90, nn_weight=0.10, gbdt_kwargs=None, nn_kwargs=None):
        self.gbdt_weight = gbdt_weight
        self.nn_weight = nn_weight
        self.gbdt_kwargs = gbdt_kwargs or {}
        self.nn_kwargs = nn_kwargs or {}
        
        self.gbdt_ = CustomGradientBoostingRegressor(**self.gbdt_kwargs)
        self.nn_ = CustomNeuralNetworkRegressor(**self.nn_kwargs)
        self.feature_importances_ = None

    def fit(self, X, y):
        self.gbdt_.fit(X, y)
        self.nn_.fit(X, y)
        self.feature_importances_ = self.gbdt_.feature_importances_
        return self

    def predict(self, X):
        gbdt_pred = self.gbdt_.predict(X)
        nn_pred = self.nn_.predict(X)
        return self.gbdt_weight * gbdt_pred + self.nn_weight * nn_pred

    def to_dict(self):
        return {
            "type": "CustomEnsembleRegressor",
            "gbdt_weight": float(self.gbdt_weight),
            "nn_weight": float(self.nn_weight),
            "gbdt_": self.gbdt_.to_dict(),
            "nn_": self.nn_.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        ens = cls(gbdt_weight=d["gbdt_weight"], nn_weight=d["nn_weight"])
        ens.gbdt_ = CustomGradientBoostingRegressor.from_dict(d["gbdt_"])
        ens.nn_ = CustomNeuralNetworkRegressor.from_dict(d["nn_"])
        ens.feature_importances_ = ens.gbdt_.feature_importances_
        return ens

    def save_model(self, file_path):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_model(cls, file_path):
        with open(file_path, "r") as f:
            d = json.load(f)
        if d.get("type") == "CustomEnsembleRegressor":
            return cls.from_dict(d)
        elif d.get("type") == "CustomLogTransformedTargetRegressor":
            return CustomLogTransformedTargetRegressor.from_dict(d)
        else:
            return CustomGradientBoostingRegressor.from_dict(d)
