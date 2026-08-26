"""
PERM - High-Fidelity Edge Telemetry & Supervised Training Dataset Generator

Generates comprehensive multi-cycle edge telemetry that closely matches
the actual locust burst traffic profile (up to 90 users, load 7-10),
including auto-scaling scenarios where workers ramp up mid-burst.
"""

import random
import pandas as pd
import numpy as np
import features as feat_lib


def _latency(req_count, workers, cpu_pct, base_noise_std=3.0):
    """Realistic queueing-theory latency with non-linear CPU saturation."""
    queue_factor = 1.0
    if cpu_pct > 65.0:
        queue_factor += ((cpu_pct - 65.0) / 12.0) ** 1.7
    base = (8.0 + (req_count * 0.5 / max(1, workers))) * queue_factor
    return float(max(4.0, base + random.gauss(0, base_noise_std)))


def _cpu(req_count, load_val, workers, sys_base=12.0):
    """Edge CPU utilization calibrated to match telemetry_collector formula."""
    demand = (req_count * load_val * 0.30) / max(1, workers)
    return float(np.clip(sys_base + demand + random.gauss(0, 1.5), 8.0, 97.0))


def generate_multi_cycle_telemetry(num_cycles=14):
    """
    Generate multi-cycle telemetry covering:
      1. Standard profiles (steady → ramp → burst → cooldown)
      2. Locust-matched burst profiles (up to 90 users, load 7-10)
      3. Auto-scaling scenarios (workers scale up during burst)
      4. Gradual and step-function ramps
    """
    rows = []
    start_ts = pd.Timestamp("2026-08-20 00:00:00", tz="UTC")
    current_sec = 0

    # Profile library — varied to cover edge cases the model must learn
    profile_library = [
        # ── Standard profiles (long phases — horizon stays within-phase) ───
        {
            "name": "standard",
            "stages": [
                {"phase": "steady",   "dur": 90, "u_s": 4,  "u_e": 8,  "load": (1.0, 2.0), "w": 2},
                {"phase": "ramp",     "dur": 80, "u_s": 8,  "u_e": 40, "load": (2.5, 4.5), "w": 2},
                {"phase": "burst",    "dur": 60, "u_s": 40, "u_e": 70, "load": (5.0, 8.0), "w": 2},
                {"phase": "cooldown", "dur": 70, "u_s": 70, "u_e": 5,  "load": (1.0, 2.0), "w": 2},
            ],
            "cycles": 10
        },
        # ── Locust-matched burst profile (high intensity, but LONG phases) ─
        {
            "name": "locust_intense",
            "stages": [
                {"phase": "steady",   "dur": 70, "u_s": 6,  "u_e": 8,  "load": (2.0, 3.0), "w": 2},
                {"phase": "ramp",     "dur": 65, "u_s": 8,  "u_e": 45, "load": (4.0, 7.0), "w": 2},
                {"phase": "burst",    "dur": 60, "u_s": 45, "u_e": 90, "load": (7.0, 10.0), "w": 2},
                {"phase": "cooldown", "dur": 60, "u_s": 90, "u_e": 5,  "load": (1.0, 2.0), "w": 2},
            ],
            "cycles": 10
        },
        # ── Locust with auto-scaling (workers ramp up mid-burst) ──────────
        {
            "name": "locust_with_scaling",
            "stages": [
                {"phase": "steady",      "dur": 70, "u_s": 6,  "u_e": 8,  "load": (2.0, 3.0), "w": 2},
                {"phase": "ramp",        "dur": 60, "u_s": 8,  "u_e": 45, "load": (4.0, 7.0), "w": 2},
                {"phase": "burst_pre",   "dur": 30, "u_s": 45, "u_e": 90, "load": (7.0, 10.0),"w": 2},
                {"phase": "burst_scaled","dur": 55, "u_s": 90, "u_e": 90, "load": (7.0, 10.0),"w": 4},
                {"phase": "cooldown",    "dur": 60, "u_s": 90, "u_e": 5,  "load": (1.0, 2.0), "w": 4},
                {"phase": "scale_down",  "dur": 55, "u_s": 5,  "u_e": 5,  "load": (1.0, 2.0), "w": 2},
            ],
            "cycles": 8
        },
        # ── Step-function surge (instant spike with no ramp) ─────────────
        {
            "name": "step_surge",
            "stages": [
                {"phase": "steady",   "dur": 70, "u_s": 5,  "u_e": 5,  "load": (1.5, 2.5), "w": 2},
                {"phase": "burst",    "dur": 65, "u_s": 80, "u_e": 90, "load": (8.0, 10.0),"w": 2},
                {"phase": "cooldown", "dur": 65, "u_s": 90, "u_e": 4,  "load": (1.0, 1.5), "w": 2},
            ],
            "cycles": 8
        },
        # ── Sustained high load (workers=4) ──────────────────────────────
        {
            "name": "sustained_load",
            "stages": [
                {"phase": "steady",   "dur": 60, "u_s": 10, "u_e": 15, "load": (2.0, 3.5), "w": 2},
                {"phase": "burst",    "dur": 90, "u_s": 60, "u_e": 80, "load": (6.0, 9.0), "w": 4},
                {"phase": "cooldown", "dur": 60, "u_s": 80, "u_e": 8,  "load": (1.0, 2.0), "w": 4},
            ],
            "cycles": 8
        },
    ]

    for profile in profile_library:
        for _ in range(profile["cycles"]):
            for stage in profile["stages"]:
                phase  = stage["phase"]
                dur    = stage["dur"]
                u_s    = stage["u_s"]
                u_e    = stage["u_e"]
                load_r = stage["load"]
                workers = stage["w"]

                for s in range(dur):
                    # Use sigmoid interpolation for smoother transitions
                    t = s / max(1.0, dur - 1)
                    # Ease-in-out for natural traffic shape
                    t_smooth = t * t * (3 - 2 * t)
                    users = u_s + (u_e - u_s) * t_smooth
                    users = max(1.0, users + random.gauss(0, max(0.5, users * 0.04)))

                    load_val  = random.uniform(load_r[0], load_r[1])
                    req_count = int(round(users * random.uniform(0.9, 1.1)))

                    cpu_pct     = _cpu(req_count, load_val, workers)
                    avg_latency = _latency(req_count, workers, cpu_pct)
                    p95_latency = float(max(avg_latency * 1.25, avg_latency + random.uniform(5.0, 35.0)))
                    mem_pct     = float(np.clip(55.0 + req_count * 0.18 + random.gauss(0, 0.4), 48.0, 88.0))

                    ts = start_ts + pd.Timedelta(seconds=current_sec)
                    rows.append({
                        "timestamp":      ts,
                        "cpu_percent":    round(cpu_pct, 2),
                        "mem_percent":    round(mem_pct, 2),
                        "avg_latency_ms": round(avg_latency, 2),
                        "p95_latency_ms": round(p95_latency, 2),
                        "request_count":  req_count,
                        "active_workers": workers,
                        "phase":          phase,
                    })
                    current_sec += 1

    return pd.DataFrame(rows)


def build_labeled_features(telemetry_df, window_sec=10, stride_sec=3, horizon_sec=30):
    """
    Build sliding-window feature rows with mean-target labels.
    Uses mean of next 15s (not point-in-time) for more robust supervision.

    Skips samples where the transition is inherently unpredictable:
    e.g. steady/cooldown feature window targeting a burst window.
    The model cannot predict a burst 30s from now from steady features alone.
    """
    ts_array = telemetry_df["timestamp"].values
    ts_ns = ts_array.astype("datetime64[ns]").astype(np.int64)
    start_time = pd.Timestamp(ts_array[0])
    end_time   = pd.Timestamp(ts_array[-1])

    # Phase transitions we CANNOT learn from features → exclude from training
    UNPREDICTABLE = {
        ("steady",   "burst"),
        ("steady",   "burst_pre"),
        ("steady",   "burst_scaled"),
        ("cooldown", "burst"),
        ("cooldown", "burst_pre"),
        ("scale_down", "burst"),
    }

    rows = []
    skipped = 0
    window_end = start_time + pd.Timedelta(seconds=window_sec)

    while window_end + pd.Timedelta(seconds=horizon_sec + 15) <= end_time:
        window_start = window_end - pd.Timedelta(seconds=window_sec)
        ws_ns = window_start.value
        we_ns = window_end.value

        i_start = np.searchsorted(ts_ns, ws_ns, side="left")
        i_end   = np.searchsorted(ts_ns, we_ns, side="right")

        window_df = telemetry_df.iloc[i_start:i_end]
        if len(window_df) < 8:
            window_end += pd.Timedelta(seconds=stride_sec)
            continue

        current_phase = window_df["phase"].iloc[-1]

        # Target window: mean of 15s starting at t+horizon
        t_start_ns = (window_end + pd.Timedelta(seconds=horizon_sec)).value
        t_end_ns   = (window_end + pd.Timedelta(seconds=horizon_sec + 15)).value
        t_i_s = np.searchsorted(ts_ns, t_start_ns)
        t_i_e = np.searchsorted(ts_ns, t_end_ns)

        if t_i_s >= len(ts_ns) or t_i_e <= t_i_s:
            window_end += pd.Timedelta(seconds=stride_sec)
            continue

        target_window = telemetry_df.iloc[t_i_s:t_i_e]
        # Use the MAJORITY phase in the target window
        target_phase = target_window["phase"].mode()[0] if len(target_window) > 0 else "unknown"

        # Skip inherently unpredictable transitions
        if (current_phase, target_phase) in UNPREDICTABLE:
            skipped += 1
            window_end += pd.Timedelta(seconds=stride_sec)
            continue

        features = feat_lib.window_features(window_df)
        features["window_end"] = str(window_end)
        features["phase"]      = current_phase
        features["target_cpu_percent"] = float(target_window["cpu_percent"].mean())
        features["target_latency_ms"]  = float(target_window["avg_latency_ms"].mean())
        rows.append(features)

        window_end += pd.Timedelta(seconds=stride_sec)

    print(f"  -> Skipped {skipped} unpredictable phase-transition windows.")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("[*] Generating high-fidelity edge training data (locust-matched profiles)...")
    telem = generate_multi_cycle_telemetry()
    print(f"  -> Generated {len(telem)} 1-second telemetry rows.")
    print(f"  -> CPU range: {telem.cpu_percent.min():.1f} - {telem.cpu_percent.max():.1f}%")
    print(f"  -> Latency range: {telem.avg_latency_ms.min():.1f} - {telem.avg_latency_ms.max():.1f}ms")
    print(f"  -> High-CPU rows (>70%): {(telem.cpu_percent > 70).sum()}")
    print(f"  -> High-latency rows (>150ms): {(telem.avg_latency_ms > 150).sum()}")

    features_df = build_labeled_features(telem)
    print(f"  -> Built {len(features_df)} feature vectors.")

    # Shuffle before splitting to avoid temporal leakage between profiles
    features_df = features_df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(features_df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_df = features_df.iloc[:n_train]
    val_df   = features_df.iloc[n_train:n_train + n_val]
    test_df  = features_df.iloc[n_train + n_val:]

    train_df.to_csv("features_train.csv", index=False)
    val_df.to_csv("features_val.csv",   index=False)
    test_df.to_csv("features_test.csv", index=False)
    print(f"  -> Wrote train({len(train_df)}), val({len(val_df)}), test({len(test_df)}).")
