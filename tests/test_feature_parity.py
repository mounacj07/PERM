"""
tests/test_feature_parity.py
=============================
Unit test verifying exact mathematical and implementation parity between:
  1. Offline batch sequence extraction (used in training)
  2. Online streaming extraction (used in replay_demo.py and inference_service.py)
"""

import os
import unittest
import numpy as np

from stagbdt.features import (
    FEATURE_NAMES,
    extract_features_single,
    extract_features_sequence,
    StreamingArrivalFeatureExtractor,
)


class TestFeatureParity(unittest.TestCase):

    def test_feature_count_and_names(self):
        """Verify 25 arrival-only features with no forbidden tokens."""
        self.assertEqual(len(FEATURE_NAMES), 25)
        for fn in FEATURE_NAMES:
            for forbidden in ["cpu", "mem", "lat", "queue", "flight", "elapsed", "burst", "run_id", "timestamp"]:
                self.assertNotIn(forbidden, fn.lower())

    def test_online_streaming_equals_offline_sequence(self):
        """Stream a recorded run second by second and verify online features equal offline features."""
        # Use a recorded run from data/runs_locust or synthetic realistic trace
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "runs_locust", "random_mix_seed20.csv")
        if os.path.exists(csv_path):
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                arr = np.array([float(r["arrivals_count"]) for r in reader], dtype=np.float64)
        else:
            rng = np.random.RandomState(42)
            arr = rng.poisson(lam=20.0, size=120).astype(np.float64)

        T = len(arr)
        valid_t = np.arange(10, T)

        # 1. Offline batch extraction
        X_offline, names_offline = extract_features_sequence(arr, valid_t=valid_t)
        self.assertEqual(names_offline, FEATURE_NAMES)

        # 2. Online streaming extraction
        extractor = StreamingArrivalFeatureExtractor(max_buffer_len=150)
        X_online = []
        for t in range(T):
            feats_t = extractor.push(arr[t])
            if t in valid_t:
                X_online.append(feats_t)

        X_online = np.vstack(X_online)

        # Assert exact floating point equality
        np.testing.assert_allclose(
            X_online,
            X_offline,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Online streaming features do NOT match offline sequence features!",
        )


if __name__ == "__main__":
    unittest.main()
