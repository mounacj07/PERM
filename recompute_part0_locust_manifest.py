"""
recompute_part0_locust_manifest.py
==================================
Updates data/runs_locust/manifest.json for Step 3b-L Part 0:
1. Tags with profile_definition_version = "v1.2".
2. Renames rate_bias_pct to delivery_ratio = arrived / scheduled.
3. Computes real fidelity check:
   - Pearson dispersion: mean((x_t - lambda_t)^2 / lambda_t)
   - Max cumulative deviation: max(|cumsum(x_t) - cumsum(lambda_t)|)
   Replaces detrended-Fano metric.
"""

import os
import json
import csv
import numpy as np
import pandas as pd
import burst_test

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
MANIFEST_PATH = os.path.join(RUNS_DIR, "manifest.json")


def get_nominal_lambda_curve(profile: str, seed: int) -> np.ndarray:
    rng = burst_test.random.Random(seed)
    if profile == "step":
        stages = [(30.0, 4.0, 4.0), (60.0, 40.0, 40.0), (30.0, 4.0, 4.0)]
    elif profile == "ramp":
        stages = [(30.0, 4.0, 4.0), (30.0, 4.0, 40.0), (45.0, 40.0, 40.0), (30.0, 4.0, 4.0)]
    elif profile == "spike":
        stages = [(30.0, 4.0, 4.0), (15.0, 40.0, 40.0), (30.0, 4.0, 4.0)]
    elif profile == "double_burst":
        stages = [(20.0, 4.0, 4.0), (30.0, 40.0, 40.0), (30.0, 4.0, 4.0), (30.0, 40.0, 40.0), (20.0, 4.0, 4.0)]
    elif profile == "sawtooth":
        stages = [(20.0, 4.0, 4.0), (20.0, 4.0, 40.0), (20.0, 40.0, 4.0), (20.0, 40.0, 40.0), (20.0, 40.0, 40.0), (20.0, 4.0, 4.0)]
    elif profile == "random_walk":
        stages = [(20.0, 4.0, 4.0)]
        rem_dur = 100.0
        cur_rate = 4.0
        while rem_dur > 0:
            dur = min(rem_dur, round(rng.uniform(10.0, 20.0), 1))
            next_rate = round(rng.uniform(3.0, 45.0), 1)
            stages.append((dur, cur_rate, next_rate))
            cur_rate = next_rate
            rem_dur -= dur
    elif profile == "random_mix":
        stages = []
        initial_calm_rate = round(rng.uniform(3.0, 6.0), 1)
        initial_calm_dur = round(rng.uniform(10.0, 40.0), 1)
        stages.append((initial_calm_dur, initial_calm_rate, initial_calm_rate))
        current_rate = initial_calm_rate
        n_segments = rng.randint(2, 4)
        for seg_idx in range(n_segments):
            used_dur = sum(d for d, _, _ in stages)
            if used_dur >= 130.0:
                break
            max_avail = max(10.0, 150.0 - used_dur)
            peak_rate = round(rng.uniform(12.0, 52.0), 1)
            seg_type = rng.choice(["step", "ramp"])
            rise_dur = round(rng.uniform(5.0, min(30.0, max(5.0, max_avail * 0.3))), 1)
            hold_dur = round(rng.uniform(10.0, min(30.0, max(5.0, max_avail * 0.4))), 1)
            return_dur = round(rng.uniform(5.0, min(15.0, max(5.0, max_avail * 0.3))), 1)
            calm_rate = round(rng.uniform(3.0, 6.0), 1)
            if seg_type == "step":
                stages.append((rise_dur, peak_rate, peak_rate))
            else:
                stages.append((rise_dur, current_rate, peak_rate))
            stages.append((hold_dur, peak_rate, peak_rate))
            stages.append((return_dur, peak_rate, calm_rate))
            current_rate = calm_rate
        total_d = sum(d for d, _, _ in stages)
        if total_d > 150.0:
            excess = total_d - 150.0
            last_d, r1, r2 = stages[-1]
            if last_d > excess:
                stages[-1] = (round(last_d - excess, 1), r1, r2)
            else:
                stages.pop()
    else:
        raise ValueError(f"Unknown profile {profile}")

    total_dur = sum(d for d, _, _ in stages)
    T = int(np.round(total_dur))
    lambdas = np.zeros(T, dtype=np.float64)
    t_start = 0.0
    for dur, r1, r2 in stages:
        t_end = t_start + dur
        s_idx = int(np.floor(t_start))
        e_idx = min(T, int(np.ceil(t_end)))
        for sec in range(s_idx, e_idx):
            ov_start = max(float(sec), t_start)
            ov_end = min(float(sec + 1), t_end)
            if ov_end > ov_start:
                if dur > 0:
                    u_mid = (ov_start + ov_end) / 2.0
                    rate_mid = r1 + (u_mid - t_start) / dur * (r2 - r1)
                else:
                    rate_mid = r1
                lambdas[sec] += rate_mid * (ov_end - ov_start)
        t_start = t_end
    return lambdas


def update_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["profile_definition_version"] = "v1.2"
    runs = data.get("runs", [])

    for r in runs:
        prof = r["profile"]
        s = r["seed"]
        dur = float(r["duration_sec"])
        csv_file = os.path.join(RUNS_DIR, r.get("csv_file", f"{r['run_id']}.csv"))

        df = pd.read_csv(csv_file)
        zero_idx = df.index[df["elapsed_sec"] == 0.0].tolist()
        start_idx = zero_idx[0] if zero_idx else 0
        T = int(np.round(dur))
        sub_df = df.iloc[start_idx : start_idx + T]

        x_t = sub_df["arrivals_count"].values.astype(np.float64)
        lam_t = get_nominal_lambda_curve(prof, s)
        min_len = min(len(x_t), len(lam_t))
        x_t = x_t[:min_len]
        lam_t = lam_t[:min_len]

        # Pearson dispersion: mean((x_t - lam_t)^2 / lam_t)
        pearson_disp = float(np.mean(((x_t - lam_t) ** 2) / np.maximum(0.1, lam_t)))
        # Max cumulative deviation: max(|cumsum(x_t) - cumsum(lam_t)|)
        max_cum_dev = float(np.max(np.abs(np.cumsum(x_t) - np.cumsum(lam_t))))

        sched = r.get("scheduled_count", int(np.round(np.sum(lam_t))))
        arr = r.get("total_arrivals", int(np.sum(x_t)))
        deliv_ratio = float(arr / sched) if sched > 0 else 1.0

        r["delivery_ratio"] = round(deliv_ratio, 4)
        r["pearson_dispersion"] = round(pearson_disp, 3)
        r["max_cumulative_deviation"] = round(max_cum_dev, 2)

        # Drop old detrended_fano_var_mean if desired, or keep as legacy
        if "detrended_fano_var_mean" in r:
            r["legacy_detrended_fano"] = r.pop("detrended_fano_var_mean")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Successfully updated {len(runs)} runs in {MANIFEST_PATH}")


if __name__ == "__main__":
    update_manifest()
