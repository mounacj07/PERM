"""
PERM - Shared Feature Extraction
Single canonical source for sliding-window feature extraction.
Imported by both feature_engineering.py (offline) and inference_service.py (live).

Do NOT duplicate this logic elsewhere.
"""

import numpy as np
import pandas as pd


def window_features(window_df: pd.DataFrame) -> dict:
    """
    Summarize one sliding window of 1-second telemetry into a fixed-size feature vector.

    Parameters
    ----------
    window_df : pd.DataFrame
        Telemetry rows with columns:
            timestamp, cpu_percent, mem_percent,
            avg_latency_ms, p95_latency_ms, request_count

    Returns
    -------
    dict
        Feature name → float.  All keys are always present (never NaN).
    """
    if window_df is None or len(window_df) == 0:
        return _zero_features()

    cpu = window_df["cpu_percent"].to_numpy(dtype=float)
    mem = window_df["mem_percent"].to_numpy(dtype=float)
    lat = window_df["avg_latency_ms"].to_numpy(dtype=float)
    p95 = window_df["p95_latency_ms"].to_numpy(dtype=float)
    req = window_df["request_count"].to_numpy(dtype=float)

    n_w = len(window_df)

    # ── Time span ──────────────────────────────────────────────────────────
    ts_col = window_df["timestamp"]
    try:
        ts_parsed = pd.to_datetime(ts_col, utc=True)
        span_sec = max((ts_parsed.iloc[-1] - ts_parsed.iloc[0]).total_seconds(), 1.0)
    except Exception:
        span_sec = max(float(n_w - 1), 1.0)

    # ── Simple slopes (first → last per second) ────────────────────────────
    cpu_slope = float((cpu[-1] - cpu[0]) / span_sec)
    lat_slope = float((lat[-1] - lat[0]) / span_sec)
    mem_slope = float((mem[-1] - mem[0]) / span_sec)
    req_slope = float((req[-1] - req[0]) / span_sec)
    p95_slope = float((p95[-1] - p95[0]) / span_sec)

    # ── Sub-window dynamics (first/second half, recent 1/3) ────────────────
    half = n_w // 2
    r10 = max(1, n_w // 3)
    r5  = max(1, min(5, n_w // 4))   # last ~5 samples (or fewer)
    half_span = max(span_sec / 2.0, 0.5)

    cpu_accel = float((cpu[half:].mean() - cpu[:half].mean()) / half_span) if half > 0 else 0.0
    lat_accel = float((lat[half:].mean() - lat[:half].mean()) / half_span) if half > 0 else 0.0
    req_accel = float((req[half:].mean() - req[:half].mean()) / half_span) if half > 0 else 0.0

    cpu_rec10 = float(cpu[-r10:].mean())
    lat_rec10 = float(lat[-r10:].mean())
    req_rec10 = float(req[-r10:].mean())

    # Very recent pressure (last ~5 samples) — more responsive to sharp spikes
    cpu_rec5  = float(cpu[-r5:].mean())
    lat_rec5  = float(lat[-r5:].mean())
    req_rec5  = float(req[-r5:].mean())

    cpu_mean = float(cpu.mean())
    lat_mean = float(lat.mean())
    req_mean = float(req.mean())

    # ── EWMA (exponentially weighted, alpha=0.3) ───────────────────────────
    def _ewma(arr, alpha=0.3):
        result = arr[0]
        for x in arr[1:]:
            result = alpha * x + (1 - alpha) * result
        return float(result)

    cpu_ewma = _ewma(cpu)
    lat_ewma = _ewma(lat)
    req_ewma = _ewma(req)

    # ── Regime / saturation flags ──────────────────────────────────────────
    # Continuous saturation: 0=idle, 1=warning zone, 2=critical zone
    cpu_sat = float(np.clip((cpu_rec5 - 50.0) / 35.0, 0.0, 1.0))   # 50→85% maps to 0→1
    lat_sat = float(np.clip((lat_rec5 - 80.0) / 220.0, 0.0, 1.0))  # 80→300ms maps to 0→1

    # Surge flag: is the recent window rising sharply?
    cpu_surge = float(1.0 if cpu_slope > 2.0 else 0.0)
    lat_surge = float(1.0 if lat_slope > 5.0 else 0.0)

    # ── Interactions ───────────────────────────────────────────────────────
    req_x_cpu        = float(req_mean * cpu_mean)
    req_x_lat        = float(req_mean * lat_mean)
    cpu_per_req      = float(cpu_mean / (req_mean + 1e-5))
    lat_p95_ratio    = float(p95.mean() / (lat_mean + 1e-5))
    cpu_lat_product  = float(cpu_mean * lat_mean)
    req_ramp_rate    = float(req_accel / (req_mean + 1e-5))

    # Pressure score: combined load × saturation signal
    pressure_score   = float(cpu_ewma * lat_ewma / 1000.0)

    return {
        # ── Core CPU ──────────────────────────────────────────────────────
        "cpu_mean":      cpu_mean,
        "cpu_std":       float(cpu.std()),
        "cpu_max":       float(cpu.max()),
        "cpu_min":       float(cpu.min()),
        "cpu_last":      float(cpu[-1]),
        "cpu_slope":     cpu_slope,
        "cpu_range":     float(cpu.max() - cpu.min()),
        "cpu_accel":     cpu_accel,
        "cpu_rec10":     cpu_rec10,
        "cpu_rec5":      cpu_rec5,
        "cpu_ewma":      cpu_ewma,
        "cpu_momentum":  float(cpu_rec10 - cpu_mean),
        "cpu_sat":       cpu_sat,
        "cpu_surge":     cpu_surge,

        # ── Core Memory ───────────────────────────────────────────────────
        "mem_mean":      float(mem.mean()),
        "mem_std":       float(mem.std()),
        "mem_max":       float(mem.max()),
        "mem_last":      float(mem[-1]),
        "mem_slope":     mem_slope,

        # ── Core Latency ──────────────────────────────────────────────────
        "lat_mean":      lat_mean,
        "lat_std":       float(lat.std()),
        "lat_max":       float(lat.max()),
        "lat_min":       float(lat.min()),
        "lat_last":      float(lat[-1]),
        "lat_slope":     lat_slope,
        "lat_range":     float(lat.max() - lat.min()),
        "lat_accel":     lat_accel,
        "lat_rec10":     lat_rec10,
        "lat_rec5":      lat_rec5,
        "lat_ewma":      lat_ewma,
        "lat_momentum":  float(lat_rec10 - lat_mean),
        "lat_sat":       lat_sat,
        "lat_surge":     lat_surge,

        # ── Core P95 ──────────────────────────────────────────────────────
        "p95_mean":      float(p95.mean()),
        "p95_max":       float(p95.max()),
        "p95_slope":     p95_slope,
        "lat_p95_ratio": lat_p95_ratio,

        # ── Core Request ──────────────────────────────────────────────────
        "req_mean":      req_mean,
        "req_max":       float(req.max()),
        "req_min":       float(req.min()),
        "req_sum":       float(req.sum()),
        "req_last":      float(req[-1]),
        "req_slope":     req_slope,
        "req_accel":     req_accel,
        "req_rec10":     req_rec10,
        "req_rec5":      req_rec5,
        "req_ewma":      req_ewma,
        "req_momentum":  float(req_rec10 - req_mean),

        # ── Interactions & Regime ─────────────────────────────────────────
        "req_x_cpu":       req_x_cpu,
        "req_x_lat":       req_x_lat,
        "cpu_per_req":     cpu_per_req,
        "cpu_lat_product": cpu_lat_product,
        "req_ramp_rate":   req_ramp_rate,
        "pressure_score":  pressure_score,
        "window_rows":     float(n_w),
    }


def _zero_features() -> dict:
    """Return a zeroed feature dict for empty windows (fallback only)."""
    keys = [
        "cpu_mean", "cpu_std", "cpu_max", "cpu_min", "cpu_last",
        "cpu_slope", "cpu_range", "cpu_accel", "cpu_rec10", "cpu_rec5",
        "cpu_ewma", "cpu_momentum", "cpu_sat", "cpu_surge",
        "mem_mean", "mem_std", "mem_max", "mem_last", "mem_slope",
        "lat_mean", "lat_std", "lat_max", "lat_min", "lat_last",
        "lat_slope", "lat_range", "lat_accel", "lat_rec10", "lat_rec5",
        "lat_ewma", "lat_momentum", "lat_sat", "lat_surge",
        "p95_mean", "p95_max", "p95_slope", "lat_p95_ratio",
        "req_mean", "req_max", "req_min", "req_sum", "req_last",
        "req_slope", "req_accel", "req_rec10", "req_rec5", "req_ewma", "req_momentum",
        "req_x_cpu", "req_x_lat", "cpu_per_req",
        "cpu_lat_product", "req_ramp_rate", "pressure_score", "window_rows",
    ]
    return {k: 0.0 for k in keys}
