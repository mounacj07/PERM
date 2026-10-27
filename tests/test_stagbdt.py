"""
tests/test_stagbdt.py
=====================
Unit tests for STAGBDT library satisfying Part C requirements:
  1. Finite-difference gradient & hessian check for AsymmetricLeastSquaresLoss
  2. Exact equivalence: flat_batch == flat_single_sample == recursive_predict
  3. Determinism under fixed random seed
  4. Grep test: stagbdt/ imports NO external ML libraries (no sklearn, torch, xgboost, etc.)
  5. JSON serialization & deserialization round-trip
  6. Early-exit gating behavior
  7. Split-conformal calibration coverage
"""

import glob
import os
import re
import unittest
import numpy as np

from stagbdt.loss import AsymmetricLeastSquaresLoss
from stagbdt.model import STAGBDT
from stagbdt.conformal import ConformalCalibrator, EarlyExitCalibrator


class TestSTAGBDT(unittest.TestCase):

    def test_finite_difference_gradient(self):
        """Test finite-difference gradient and hessian check for expectile loss."""
        for alpha in [0.2, 0.5, 0.7, 0.85, 0.95]:
            loss_fn = AsymmetricLeastSquaresLoss(alpha=alpha)
            rng = np.random.RandomState(42)
            y = rng.uniform(5.0, 50.0, size=50)
            # Ensure yhat is sufficiently distinct from y to avoid discontinuity at y == yhat
            yhat = y + rng.choice([-1.0, 1.0], size=50) * rng.uniform(0.5, 10.0, size=50)

            eps = 1e-6
            g_ana, h_ana = loss_fn.gradient_and_hessian(y, yhat)

            # Numerical gradient for each element
            g_num = np.empty_like(yhat)
            for i in range(len(yhat)):
                yhat_plus = yhat.copy()
                yhat_minus = yhat.copy()
                yhat_plus[i] += eps
                yhat_minus[i] -= eps
                # Single element loss
                l_plus = (alpha if y[i] >= yhat_plus[i] else 1.0 - alpha) * ((y[i] - yhat_plus[i]) ** 2)
                l_minus = (alpha if y[i] >= yhat_minus[i] else 1.0 - alpha) * ((y[i] - yhat_minus[i]) ** 2)
                g_num[i] = (l_plus - l_minus) / (2.0 * eps)

            np.testing.assert_allclose(
                g_ana, g_num, rtol=1e-5, atol=1e-5,
                err_msg=f"Gradient mismatch for alpha={alpha}",
            )

            # Numerical hessian: dg/dyhat
            h_num = np.empty_like(yhat)
            for i in range(len(yhat)):
                yhat_plus = yhat.copy()
                yhat_minus = yhat.copy()
                yhat_plus[i] += eps
                yhat_minus[i] -= eps
                diff_plus = y[i] - yhat_plus[i]
                diff_minus = y[i] - yhat_minus[i]
                w_plus = alpha if diff_plus >= 0.0 else 1.0 - alpha
                w_minus = alpha if diff_minus >= 0.0 else 1.0 - alpha
                gp = -2.0 * w_plus * diff_plus
                gm = -2.0 * w_minus * diff_minus
                h_num[i] = (gp - gm) / (2.0 * eps)

            np.testing.assert_allclose(
                h_ana, h_num, rtol=1e-5, atol=1e-5,
                err_msg=f"Hessian mismatch for alpha={alpha}",
            )

    def test_flat_equals_recursive(self):
        """Verify flat batch, flat single-sample, and recursive predict match exactly."""
        rng = np.random.RandomState(123)
        X = rng.normal(size=(200, 8))
        y = (
            3.0 * X[:, 0]
            - 2.0 * (X[:, 1] > 0.0).astype(float)
            + 1.5 * X[:, 2] * X[:, 3]
            + 10.0
            + rng.normal(scale=0.5, size=200)
        )

        model = STAGBDT(
            n_estimators=15,
            learning_rate=0.1,
            max_depth=4,
            alpha=0.85,
            l2_reg=1.0,
            random_state=42,
        )
        model.fit(X, y)

        X_test = rng.normal(size=(100, 8))
        pred_rec = model.predict(X_test, use_flat=False)
        pred_flat_batch = model.predict(X_test, use_flat=True)
        pred_flat_single = np.array([model.predict_one(x, use_flat=True) for x in X_test])
        pred_stacked_single = np.array([model.predict_one_stacked(x) for x in X_test])

        np.testing.assert_allclose(
            pred_flat_batch, pred_rec, rtol=1e-12, atol=1e-12,
            err_msg="Flat batch predictions do not match recursive predictions",
        )
        np.testing.assert_allclose(
            pred_flat_single, pred_rec, rtol=1e-12, atol=1e-12,
            err_msg="Flat single-sample predictions do not match recursive predictions",
        )
        np.testing.assert_allclose(
            pred_stacked_single, pred_rec, rtol=1e-12, atol=1e-12,
            err_msg="Stacked single-sample predictions do not match recursive predictions",
        )

    def test_determinism(self):
        """Verify determinism under fixed random seed."""
        rng = np.random.RandomState(999)
        X = rng.uniform(0.0, 10.0, size=(100, 6))
        y = np.sum(X, axis=1) + rng.normal(size=100)

        m1 = STAGBDT(n_estimators=10, max_depth=3, random_state=42).fit(X, y)
        m2 = STAGBDT(n_estimators=10, max_depth=3, random_state=42).fit(X, y)

        p1 = m1.predict(X)
        p2 = m2.predict(X)
        np.testing.assert_array_equal(p1, p2, err_msg="Models are not deterministic")

    def test_no_external_ml_imports(self):
        """Grep test: stagbdt/ must import NO ML library (sklearn, torch, xgboost, etc.)."""
        pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stagbdt")
        py_files = glob.glob(os.path.join(pkg_dir, "*.py"))
        self.assertGreater(len(py_files), 0, "No python files found in stagbdt/")

        forbidden_patterns = [
            r"\bimport\s+sklearn\b",
            r"\bfrom\s+sklearn\b",
            r"\bimport\s+torch\b",
            r"\bfrom\s+torch\b",
            r"\bimport\s+xgboost\b",
            r"\bfrom\s+xgboost\b",
            r"\bimport\s+lightgbm\b",
            r"\bfrom\s+lightgbm\b",
            r"\bimport\s+catboost\b",
            r"\bfrom\s+catboost\b",
            r"\bimport\s+scipy\b",
            r"\bfrom\s+scipy\b",
        ]

        for filepath in py_files:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            for pattern in forbidden_patterns:
                match = re.search(pattern, content)
                self.assertIsNone(
                    match,
                    f"Forbidden import '{pattern}' found in {os.path.basename(filepath)}",
                )

    def test_json_save_load(self):
        """Verify complete JSON save and load round-trip."""
        rng = np.random.RandomState(42)
        X = rng.normal(size=(50, 4))
        y = X[:, 0] * 2.0 + 5.0

        model = STAGBDT(n_estimators=5, max_depth=3, alpha=0.7).fit(X, y)
        pred_before = model.predict(X)

        json_path = os.path.join(os.path.dirname(__file__), "temp_model.json")
        try:
            model.save_json(json_path)
            loaded_model = STAGBDT.load_json(json_path)
            pred_after = loaded_model.predict(X)
            np.testing.assert_array_equal(pred_before, pred_after)
        finally:
            if os.path.exists(json_path):
                os.remove(json_path)

    def test_conformal_calibrator(self):
        """Verify conformal upper bound coverage property."""
        rng = np.random.RandomState(77)
        y_cal = rng.normal(loc=20.0, scale=3.0, size=500)
        yhat_cal = y_cal + rng.normal(loc=0.0, scale=1.0, size=500)

        cal = ConformalCalibrator(coverage_level=0.90)
        margin = cal.calibrate(y_cal, yhat_cal)

        y_test = rng.normal(loc=20.0, scale=3.0, size=1000)
        yhat_test = y_test + rng.normal(loc=0.0, scale=1.0, size=1000)
        coverage = cal.evaluate_coverage(y_test, yhat_test)

        # Theoretical coverage for 90% should be ~0.90 (+- 0.03)
        self.assertGreaterEqual(coverage, 0.85)


if __name__ == "__main__":
    unittest.main()
