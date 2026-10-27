# PERM Warm-Up & Forecaster Scaling Diagnostic Handoff

**Date**: October 28, 2026  
**Status**: Complete. Pure diagnostic measurement — zero existing files modified, zero models retuned or retrained.  
**Diagnostic Directory**: `diag/`  
**Repository Scan**: 506/506 files clean (0 hits on local paths, usernames, or secrets).

---

## Executive Summary & Answer to Question

**Question**: *In earlier predictive runs, workers rose from 2 toward 8 at t ~ 10s (when the 10s warm-up ended) on both step and ramp profiles, with pool utilization at only 6-12%, about 20s before the burst. Is the forecaster responding to the traffic, or to warm-up/short-history artifacts, or is it hedging toward high forecasts whenever it sees calm traffic?*

**Diagnostic Conclusion**: **The forecaster is hedging toward high forecasts whenever it sees calm traffic.**
The timing ($t \sim 10\text{s}$) is an artifact of the controller's warm-up gate, not model sensitivity:
1. **Controller Gate Suppression ($t = 0..9\text{s}$)**: In `demo_controller.py` (lines 382–385), `compute_predictive_target` hardcodes:
   ```python
   if len(arrivals_list) < 10:
       log.info("predictive warming up")
       return 0, None
   ```
   No forecast queries are dispatched until $t = 10\text{s}$.
2. **Persistent Over-Prediction on Calm Traffic**: When fed pure, steady 4 req/s Poisson traffic for 120s (with no burst ever occurring), the offline ST-AGBDT $h=20$ model continuously outputs raw forecasts between **35.3 req/s and 52.6 req/s** (upper bound **40.4 to 57.7 req/s**), demanding **8 workers** at every second from $t=3\text{s}$ through $t=120\text{s}$.
3. **Burst Invariance in Live Runs**: When burst onset was shifted back by 30 seconds (`step_late` and `ramp_late` starting at $t=60\text{s}$ instead of $t=30\text{s}$), the first scale-up time did **not** move. It occurred at $t = 9\text{s} - 10\text{s}$ in all runs.
4. **Root Cause in Training Set & Loss Function**:
   - **No Steady-State Data**: The training dataset (`data/runs_locust`) contains zero pure steady-state runs; every run is a burst or dynamic workload.
   - **71.9% Burst Probability**: Among all calm windows (arrivals $< 8$ req/s for 10s) in the training set, **71.93%** were immediately followed by a burst.
   - **85th Expectile Loss ($\alpha = 0.85$)**: The model was trained with asymmetric least squares penalizing under-prediction $5.67\times$ more than over-prediction. The 85th percentile of the target on calm windows is **48.0 req/s**. The model mathematically learned that observing calm traffic means an extreme burst is imminent.

---

## PART 1 — Offline Feature Check

### Execution
Script: `diag/diag_warmup.py`  
Output Data: `diag/offline_warmup_results.csv` (118 per-second rows for $t = 3..120$)

### Measured Values
- **Input Traffic**: Constant $\lambda = 4.0$ req/s Poisson arrivals, 120s duration, seed 44.
  - `numpy.random.default_rng(44)`: mean = 3.83 req/s, min = 0.0, max = 8.0 req/s.
  - `numpy.random.RandomState(44)`: mean = 4.23 req/s, min = 0.0, max = 12.0 req/s.
- **First $t$ where `recommended_workers >= 3`**: **$t = 3$** (the earliest possible evaluation point).
- **Maximum `recommended_workers` over the run**: **8 workers** (the maximum possible pool capacity).
- **Forecast Ranges over 120 seconds of flat 4 req/s load**:
  - Raw forecast range: **35.29 req/s to 50.75 req/s** (default_rng) / **36.50 to 52.63 req/s** (RandomState).
  - Conformal margin: **+5.0763 req/s**.
  - Upper bound range: **40.37 req/s to 55.82 req/s** (default_rng) / **41.57 to 57.71 req/s** (RandomState).
  - Effective capacity per worker: $7.89 \times 0.70 = \mathbf{5.523\text{ req/s}}$.
  - Threshold for 3 workers: $\text{upper\_bound} > 11.046$ req/s (or $\text{raw} > 5.97$ req/s).
  - Threshold for 8 workers: $\text{upper\_bound} > 38.66$ req/s (or $\text{raw} > 33.58$ req/s).
  - Result: Every single evaluated point ($t = 3..120$) has upper bound $\ge 40.37$ req/s, so `recommended_workers` is continuously clamped to **8 workers** for the entire 120 seconds.

### Short-History Handling in `extract_features_single`
From `stagbdt/features.py` (lines 62–90):
```python
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
```

**Quoted Lines & Analysis**:
1. **Truncated Slice (No Padding)**: When $n < w$, `win = arr[max(0, n - w):] = arr[0:n]`. There is **no zero padding, edge padding, or NaN**. The code simply computes the mean, max, and std over whatever slice exists (e.g. for window $w=60$ at $t=5$, `win` is 5 elements long).
2. **Dynamic Denominator Normalization**: The slope is divided by `max(1.0, float(win_len - 1))` and acceleration by `float(half_w)`. These scale with available length rather than full window duration $w$.
3. **EWMA Window Truncation**: When $n < 2w$, `ewma_win` starts from index 0 (`ewma = float(ewma_win[0])`), running with weight $\alpha = \frac{2}{w+1}$.

---

## PART 2 — Live Control Runs (Locust Headless, Seed 44)

### Execution Setup
- **Collector**: `python telemetry_collector.py --port 5000 --workers 2`
- **Inference Service**: `python inference_service.py --port 5001` (with `PERM_FORECASTER=stagbdt`)
- **Controller**: `python demo_controller.py --mode predictive --approval auto --initial-workers 2 --collector-url http://127.0.0.1:5000 --inference-url http://127.0.0.1:5001`
- **Profiles Defined in `diag/diag_profiles.py`**:
  - `flat_4`: 4 req/s Poisson for 120s (no burst).
  - `step_late`: 4 req/s for 60s, then 40 req/s for 60s (burst onset at $t = 60\text{s}$).
  - `ramp_late`: 4 req/s for 60s, ramp 4 $\rightarrow$ 40 req/s over 30s, hold 30s (ramp onset at $t = 60\text{s}$).
  - `step`: standard 30s calm, 60s burst at 40 req/s, 30s calm (burst onset at $t = 30\text{s}$).
  - `ramp`: standard 30s calm, 30s ramp 4 $\rightarrow$ 40 req/s, 45s burst, 30s calm (ramp onset at $t = 30\text{s}$).
- **Locust Wrapper `diag/locustfile_diag.py`**: Imports `PermOpenLoopUser` from `locustfile_perm.py` and delegates schedule generation to `diag_profiles.generate_schedule`. `locustfile_perm.py` and `burst_test.py` were left 100% untouched.

### Empirical Decision Table (Measured Data Only)

| Profile | Scaled > 2? | 1st Scale-Up ($t$) | Burst/Ramp Onset ($t$) | Scaled Before Burst? | Workers Timeline ($w @ t$) |
|---|:---:|:---:|:---:|:---:|---|
| **`flat_4`** | **YES** | **9.03s** (to 3w) | *None (never)* | **YES** | `2w@0.0s -> 3w@9.0s -> 4w@11.0s -> 5w@15.0s -> 7w@17.0s -> 8w@19.0s` |
| **`step_late`** | **YES** | **10.04s** (to 3w) | 60.0s | **YES** (+50s before) | `2w@0.0s -> 3w@10.0s -> 4w@14.0s -> 3w@15.0s -> 4w@16.0s -> 5w@19.1s -> 7w@28.1s -> 8w@33.1s` |
| **`ramp_late`** | **YES** | **9.02s** (to 4w) | 60.0s | **YES** (+51s before) | `2w@0.0s -> 4w@9.0s -> 3w@10.0s -> 4w@12.0s -> 3w@13.0s -> 4w@14.0s -> 5w@17.0s -> 6w@18.0s -> 7w@20.0s -> 8w@22.0s` |
| **`step`** | **YES** | **9.02s** (to 6w) | 30.0s | **YES** (+21s before) | `2w@0.0s -> 6w@9.0s -> 3w@10.0s -> 5w@21.0s -> 7w@32.0s -> 8w@38.0s` |
| **`ramp`** | **YES** | **10.02s** (to 3w) | 30.0s | **YES** (+20s before) | `2w@0.0s -> 3w@10.0s -> 5w@18.0s -> 7w@28.0s -> 8w@33.0s` |

### Direct Answers to Decision Table Questions
1. **`flat_4`: Does it scale above 2 workers? When?**
   - **YES**. It scales above 2 workers at **$t = 9.03\text{s}$** (scaling to 3 workers), reaches 4 workers at $t = 11.0\text{s}$, 5 workers at $t = 15.0\text{s}$, 7 workers at $t = 17.0\text{s}$, and **8 workers at $t = 19.0\text{s}$**.
   - Any scale-up on flat load indicates either an artifact or hedging; here, the forecaster predicted 35–50 req/s on a flat 4 req/s load.
2. **`step` vs `step_late`: Does the first scale-up time move with the burst, or stay at ~10s?**
   - **It stays at ~10s**.
   - In `step` (burst at 30s), first scale-up is at **$t = 9.02\text{s}$**.
   - In `step_late` (burst delayed to 60s), first scale-up is at **$t = 10.04\text{s}$**.
   - Delaying the burst by 30 seconds did **not** delay the initial scale-up time.
3. **`ramp` vs `ramp_late`: Does the first scale-up time move with the burst, or stay at ~10s?**
   - **It stays at ~10s**.
   - In `ramp` (ramp at 30s), first scale-up is at **$t = 10.02\text{s}$**.
   - In `ramp_late` (ramp delayed to 60s), first scale-up is at **$t = 9.02\text{s}$**.
   - Shifting the ramp onset by 30 seconds did **not** move the scale-up time.

---

## PART 3 — Training-Set Audit (READ-ONLY)

### Script & Dataset
Script: `diag/audit_training_set.py`  
Dataset Directory: `data/runs_locust/` (24 training runs defined by `train_and_eval_final.py` lines 87–144)  
Excluded Runs: `sawtooth_seed42..44`, `step_seed42`.  
Calibration Runs: `random_mix_seed15..19`, `step_seed44`, `ramp_seed44`.  
Held-out Test Runs: `random_mix_seed20..29`, `spike_seed44`, `double_burst_seed44`, `random_walk_seed44`.  
Training Runs: 24 runs (`double_burst_seed42/43`, `ramp_seed42/43`, `random_mix_seed0..14`, `random_walk_seed42/43`, `spike_seed42/43`, `step_seed43`).

### Target Definition in Training Code
From `train_and_eval_final.py` (lines 166–174):
```python
valid_t = np.arange(10, T - h)
X, fnames = extract_features_sequence(arr, valid_t=valid_t)

N = len(valid_t)
y_target = np.empty(N, dtype=np.float64)
for idx, t in enumerate(valid_t):
    fut_3s = smoothed_arr[t + 1 : t + h + 1]
    y_target[idx] = np.max(fut_3s)
```
*Note*: The training target for horizon $h=20$ is **not** the point value at $t+20$; it is the **maximum 3-second smoothed arrival in the entire forward 20-second window** $[t+1, t+20]$.

### Raw Audit Metrics

#### a) Calm Training Windows Target Distribution
A calm window is defined as any time $t \in [10, T-20]$ where arrivals for the preceding 10 seconds ($t-9$ to $t$) are $< 2\times\text{baseline}$ ($< 8.0$ req/s).
- Total calm windows across all 24 training runs: **367 windows**.
- **Forward-Max Target ($\max(\text{smoothed}[t+1..t+20])$)**:
  - Windows with target $> 8.0$ req/s (burst): **264 / 367 = 71.93%**.
- **Point Target ($\text{arrivals}[t+20]$)**:
  - Windows with point arrival $> 8.0$ req/s (burst): **259 / 367 = 70.57%**.

#### b) Target Statistics for Calm Windows
- **All calm windows in the training set**:
  - Mean target ($\max_{1..20}$): **26.30 req/s**
  - **85th percentile target**: **48.00 req/s**
  - Median (50th percentile): **26.33 req/s**
  - Min / Max: 4.67 req/s / 60.00 req/s
  - Point target at $t+20$: mean = **23.49 req/s**, 85th percentile = **43.00 req/s**
- **Calm windows from steady profiles only**:
  - **Count = 0**.
  - **Zero steady/flat runs exist in `data/runs_locust`**. Every run in the dataset is a synthetic burst scenario.

#### c) Feature Dependencies on Time & History Length
1. **Elapsed Time**: **None**. Features are strictly arrival-derived (`arr_curr` and window statistics for $w \in [5, 10, 30, 60]$).
2. **Padding**: **None**. As analyzed in Part 1, truncated slices are used (`arr[max(0, n - w):]`), normalized by available window sample counts.
3. **Cycle Position**: **None**.

#### d) Burst Onset Offsets Across All 24 Training Runs
Burst onset is the first second where smoothed arrivals cross $\ge 8.0$ req/s:

| Run ID | Burst Onset Offset ($t$) | Profile Type |
|---|:---:|---|
| `double_burst_seed42` | 26.0s | `double_burst` |
| `double_burst_seed43` | 25.0s | `double_burst` |
| `ramp_seed42` | 40.0s | `ramp` |
| `ramp_seed43` | 38.0s | `ramp` |
| `spike_seed42` | 35.0s | `spike` |
| `spike_seed43` | 35.0s | `spike` |
| `step_seed43` | 36.0s | `step` |
| `random_walk_seed42` | 51.0s | `random_walk` |
| `random_walk_seed43` | 27.0s | `random_walk` |
| `random_mix_seed0` | 32.0s | `random_mix` |
| `random_mix_seed1` | 48.0s | `random_mix` |
| `random_mix_seed2` | 29.0s | `random_mix` |
| `random_mix_seed3` | 34.0s | `random_mix` |
| `random_mix_seed4` | 18.0s | `random_mix` |
| `random_mix_seed5` | 38.0s | `random_mix` |
| `random_mix_seed6` | 41.0s | `random_mix` |
| `random_mix_seed7` | 20.0s | `random_mix` |
| `random_mix_seed8` | 45.0s | `random_mix` |
| `random_mix_seed9` | 26.0s | `random_mix` |
| `random_mix_seed10` | 35.0s | `random_mix` |
| `random_mix_seed11` | 31.0s | `random_mix` |
| `random_mix_seed12` | 37.0s | `random_mix` |
| `random_mix_seed13` | 37.0s | `random_mix` |
| `random_mix_seed14` | 36.0s | `random_mix` |

**Burst Onset Analysis**:
- In standard parameterized profiles (`step`, `ramp`, `spike`, `double_burst`), the initial calm stage is almost always fixed to 20–30 seconds.
- In `random_mix`, initial calm duration is sampled uniformly between 10.0s and 40.0s (`initial_calm_dur = round(rng.uniform(10.0, 40.0), 1)` in `burst_test.py` line 126).
- Because horizon $h=20$ looks 20 seconds ahead, any calm window at $t=10\text{s}$ will have forward horizon $[11, 30\text{s}]$, which overlaps directly with the onset of the first burst in nearly every profile!

---

## Deliverables & Trace Files Reference

All newly created diagnostic code and output traces reside in `diag/`:
1. `diag/diag_warmup.py`: Part 1 offline feature and short-history test script.
2. `diag/offline_warmup_results.csv`: Per-second evaluation for $t = 3..120$ on constant 4 req/s load.
3. `diag/diag_profiles.py`: Diagnostic profile generator for `flat_4`, `step_late`, and `ramp_late`.
4. `diag/locustfile_diag.py`: Non-invasive Locust wrapper reusing `PermOpenLoopUser`.
5. `diag/run_live_diag.py`: Automated harness running all 5 live testbed scenarios.
6. `diag/generate_summary_table.py`: Parser compiling the decision table from raw traces.
7. `diag/audit_training_set.py`: Part 3 dataset audit script.
8. `diag/decision_table_full.json`: Machine-readable decision table metrics.
9. **Raw Trace CSVs (per-second measurements)**:
   - `diag/controller_trace_flat_4.csv`
   - `diag/controller_trace_step_late.csv`
   - `diag/controller_trace_ramp_late.csv`
   - `diag/controller_trace_step.csv`
   - `diag/controller_trace_ramp.csv`
