"""
stagbdt/features.py
===================
Canonical arrival-only feature extraction module for ST-AGBDT.
Used by:
  - Offline model training and evaluation (train_final_model.py, eval_forecaster.py)
  - Online replay demo (replay_demo.py)
  - Production inference service (/forecast/demand in inference_service.py)

Guarantees 100% bitwise parity between offline and online feature extraction.
Pure NumPy and standard library (zero external ML dependencies).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

# Standard window durations in seconds
WINDOWS = [5, 10, 30, 60]

def get_feature_names() -> List[str]:
    """Return sorted list of the 25 arrival-only feature names."""
    names = ["arr_curr"]
    for w in WINDOWS:
        names.extend([
            f"arr_mean_{w}",
            f"arr_max_{w}",
            f"arr_std_{w}",
            f"arr_slope_{w}",
            f"arr_accel_{w}",
            f"arr_ewma_{w}",
        ])
    return sorted(names)


FEATURE_NAMES = get_feature_names()


def extract_features_single(history_arr: np.ndarray) -> np.ndarray:
    """
    Extract the 25 arrival-only features for the current time step t (the last item in history_arr).
    
    Parameters
    ----------
    history_arr : np.ndarray
        1D array of float arrival counts from t=0 up to current t.
        Must contain at least 1 element.

    Returns
    -------
    np.ndarray
        1D array of shape (25,) sorted by FEATURE_NAMES.
    """
    arr = np.asarray(history_arr, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("history_arr must have at least 1 element")

    curr_val = arr[-1]
    feats: Dict[str, float] = {"arr_curr": float(curr_val)}

    n = len(arr)
    for w in WINDOWS:
        win = arr[max(0, n - w):]
        win_len = len(win)

        mean_val = float(np.mean(win))
        max_val = float(np.max(win))
        std_val = float(np.std(win)) if win_len > 1 else 0.0
        slope_val = float((win[-1] - win[0]) / max(1.0, float(win_len - 1)))

        half_w = max(1, win_len // 2)
        s_recent = float((win[-1] - win[-half_w]) / float(half_w))
        s_older = float((win[-half_w] - win[0]) / float(half_w))
        accel_val = float((s_recent - s_older) / float(half_w))

        # EWMA over window max(0, n - 2*w) to n
        ewma_win = arr[max(0, n - 2 * w):]
        alpha_ewma = 2.0 / (w + 1.0)
        ewma = float(ewma_win[0])
        for v in ewma_win[1:]:
            ewma = alpha_ewma * float(v) + (1.0 - alpha_ewma) * ewma

        feats[f"arr_mean_{w}"] = mean_val
        feats[f"arr_max_{w}"] = max_val
        feats[f"arr_std_{w}"] = std_val
        feats[f"arr_slope_{w}"] = slope_val
        feats[f"arr_accel_{w}"] = accel_val
        feats[f"arr_ewma_{w}"] = ewma

    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract_features_sequence(
    arr: np.ndarray,
    valid_t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract arrival-only features across a full sequence of arrivals.
    
    Parameters
    ----------
    arr : np.ndarray
        Full 1D array of arrivals.
    valid_t : Optional[np.ndarray]
        Indices at which to extract features. If None, extracts for all t >= 10.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (X, feature_names) where X has shape (len(valid_t), 25).
    """
    arr = np.asarray(arr, dtype=np.float64)
    T = len(arr)
    if valid_t is None:
        valid_t = np.arange(10, T)

    N = len(valid_t)
    X = np.empty((N, len(FEATURE_NAMES)), dtype=np.float64)
    for idx, t in enumerate(valid_t):
        X[idx] = extract_features_single(arr[: t + 1])

    return X, FEATURE_NAMES


class StreamingArrivalFeatureExtractor:
    """
    Online stateful feature extractor for second-by-second streaming.
    Maintains a rolling buffer and uses extract_features_single for exact parity.
    """

    def __init__(self, max_buffer_len: int = 150):
        self.buffer: List[float] = []
        self.max_buffer_len = max_buffer_len

    def reset(self) -> None:
        self.buffer.clear()

    def push(self, arrival_count: float) -> np.ndarray:
        """Add arrival count for the current second and return current feature vector."""
        self.buffer.append(float(arrival_count))
        if len(self.buffer) > self.max_buffer_len:
            self.buffer.pop(0)
        return extract_features_single(np.asarray(self.buffer, dtype=np.float64))

    def get_history(self) -> np.ndarray:
        return np.asarray(self.buffer, dtype=np.float64)
