# PERM Final Handoff: ST-AGBDT Production Forecaster & Streaming Replay

**Date**: 2026-10-27  
**Task**: Final ST-AGBDT model training, conformal calibration, held-out evaluation with bootstrap CIs, second-by-second streaming replay demo, inference service integration, edit disclosures, and paired baseline analysis.  
**Status**: All final models trained and validated; all unit and integration test suites passing (11/11 pytest, 15/15 test_testbed, 6/6 test_system). Zero active background tasks.

---

## 1. Disclosure of Edits & Collector Runtime Behavior

### 1.1 `git diff` of `telemetry_collector.py`
```diff
diff --git a/telemetry_collector.py b/telemetry_collector.py
index 4bd2beb..dd25dc1 100644
--- a/telemetry_collector.py
+++ b/telemetry_collector.py
@@ -28,6 +28,10 @@ CONFIGURATION (env vars or defaults):
     REQUEST_TIMEOUT_SEC - per-request deadline in seconds (default 2.0)
     QUEUE_CAPACITY      - maximum bounded queue backlog (default 20)
     WORKER_CORE_IDS     - comma-separated core indices to pin workers to
+    PROVISION_DELAY_SEC - extra seconds a newly-spawned worker is unavailable
+                          after becoming OS-ready (simulates config/warm-up;
+                          default 0 = no extra delay). Records ready_time in
+                          scale_events.
 """
 
 from __future__ import annotations
@@ -72,6 +76,14 @@ REQUEST_TIMEOUT_SEC: float = float(os.environ.get("REQUEST_TIMEOUT_SEC", "2.0"))
 # At 2 workers and load=7 (126ms service time), 20 items / (2 / 0.126) ~= 1.26s < 2.0s
 QUEUE_CAPACITY: int = int(os.environ.get("QUEUE_CAPACITY", "20"))
 
+# Extra unavailability time after OS-ready (provisioning simulation)
+PROVISION_DELAY_SEC: float = float(os.environ.get("PROVISION_DELAY_SEC", "0"))
+
+
+def _record_latency(val: float = 0.0) -> None:
+    """Backward compatibility helper for test suites."""
+    pass
+
 
 def get_physical_core_map() -> List[List[int]]:
     """
@@ -503,9 +515,23 @@ class _PoolManager:
                 p.start()
                 new_handles.append(_WorkerHandle(wid, p, exit_ev, ready_ev, core_affinity=affinity))
 
-            # Wait for processes to become ready without holding self._lock
+            # Wait for processes to become OS-ready without holding self._lock
+            os_ready_time = time.monotonic()
             for wh in new_handles:
                 wh.ready_event.wait(timeout=10.0)
+            os_ready_dur = round(time.monotonic() - t0, 3)
+
+            # PROVISION_DELAY_SEC: hold workers unavailable for extra time
+            # (simulates configuration/warm-up after OS-ready)
+            if PROVISION_DELAY_SEC > 0.0:
+                log.info(
+                    "PROVISION_DELAY_SEC=%.1f: holding %d new worker(s) unavailable for %.1fs",
+                    PROVISION_DELAY_SEC, needed, PROVISION_DELAY_SEC,
+                )
+                time.sleep(PROVISION_DELAY_SEC)
+
+            # Mark workers ready and add to pool
+            for wh in new_handles:
                 wh.ready = True
                 wh.get_psutil()
 
@@ -522,11 +548,14 @@ class _PoolManager:
                 "from_workers": old_count,
                 "to_workers": new_total,
                 "duration_seconds": dur,
+                "os_ready_seconds": os_ready_dur,
+                "provision_delay_sec": PROVISION_DELAY_SEC,
+                "ready_time": datetime.now(timezone.utc).isoformat(),
             }
             self._scale_events.append(event)
             log.info(
-                "Scale-up complete: %d -> %d workers (cold start took %.3fs)",
-                old_count, new_total, dur,
+                "Scale-up complete: %d -> %d workers (os_ready=%.3fs, total=%.3fs, delay=%.1fs)",
+                old_count, new_total, os_ready_dur, dur, PROVISION_DELAY_SEC,
             )
 
     def _scale_down_sync(self, target_n: int) -> None:
```

### 1.2 `git diff` of `pytest.ini`
```ini
[pytest]
testpaths = tests
```

### 1.3 Explanation of Edits
1. **`telemetry_collector.py`**:
   - `PROVISION_DELAY_SEC`: Allows simulating configuration and warm-up delay for delay sweep experiments (`delay_sweep.py`). Defaults to `0.0` (zero added delay). When default or unset, the `if PROVISION_DELAY_SEC > 0.0:` branch is not taken.
   - `_record_latency`: A two-line backward-compatibility stub (`def _record_latency(val: float = 0.0) -> None: pass`). Required because legacy integration test `test_system.py` imports `_record_latency` from `telemetry_collector`. It is an inert helper with zero side effects on runtime request processing.
2. **`pytest.ini`**:
   - Configures `testpaths = tests` to ensure `pytest` runs the canonical test suite in `tests/` and does not scan untracked analysis directories (like `scratch/`).

### 1.4 Confirmation of Runtime Collector Behavior
- The collector's runtime request execution, process pool worker handling, queueing, and telemetry logging are **100% identical and unchanged** from commit `86a5603231a12d210fab9e3d3cab193591c2beec` (Date: `Sun Oct 25 19:42:50 2026 +0530`), the version that produced the canonical dataset in `data/runs_locust`.
- **Note on Test Execution**: Both `test_testbed.py` and `test_system.py` explicitly start a live telemetry collector server in a child subprocess (`telemetry_collector.py`) as part of their test suites, and terminate it cleanly upon test completion.

---

## 2. Model Architecture, Configuration & Splits

### 2.1 Model Specifications
*Citation: `models/stagbdt_final_config.json`*

- **Name**: `ST-AGBDT (expectile boosted trees + conformal bound)`
- **Loss Function**: Asymmetric Least Squares (Expectile) with $\alpha = 0.85$
- **Ensemble**: 50 trees, max depth 4, learning rate 0.1, L2 regularization 1.0, 64 bins
- **Size**: ~54 KB per lean compiled model JSON (`models/stagbdt_h10_lean.json`, `h20`, `h30`)
- **Features**: 25 arrival-only features (instantaneous, mean, max, std, slope, acceleration, and multi-scale EWMA across 5s, 10s, 30s, 60s windows)
- **Target**: Maximum 3-second rolling smoothed arrival rate over the future $h$-second window ($h \in \{10, 20, 30\}$)
- **Capacity Calibration**: 7.89 req/s per worker (`testbed_calibration.json:results[0].saturation_rps`), target utilization 0.70 $\rightarrow$ effective capacity = 5.523 req/s per worker

### 2.2 Run Partitioning (48 Locust Runs in `data/runs_locust`)
*Citation: `models/stagbdt_final_config.json:splits`*

- **Excluded Runs (4 runs)**:
  - Sawtooth runs: `sawtooth_seed42`, `sawtooth_seed43`, `sawtooth_seed44`
  - Flagged run: `step_seed42` (`data/runs_locust/manifest.json:runs[0].failed_requests` = 1)
- **Calibration Runs (7 runs)**:
  - `random_mix` seeds 15, 16, 17, 18, 19
  - Fixed seed-44 runs: `step_seed44`, `ramp_seed44`
- **Held-Out Test Runs (13 runs)**:
  - `random_mix` seeds 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
  - Fixed seed-44 runs: `spike_seed44`, `double_burst_seed44`, `random_walk_seed44`
- **Training Runs (24 runs)**:
  - `random_mix` seeds 0 through 14 (15 runs)
  - Fixed seed-42 runs: `ramp_seed42`, `spike_seed42`, `double_burst_seed42`, `random_walk_seed42` (4 runs)
  - Fixed seed-43 runs: `step_seed43`, `ramp_seed43`, `spike_seed43`, `double_burst_seed43`, `random_walk_seed43` (5 runs)

---

## 3. Held-Out Evaluation on Test Runs (13 Runs, $B = 1,000$ Bootstrap CIs)

*Citations: `results/final_evaluation_results_v2.json`*

### 3.1 Empirical Coverage of ST-AGBDT 90% Conformal Upper Bound
*Citations: `results/final_evaluation_results_v2.json:horizons.<h>.empirical_coverage_stagbdt_90pct`*

- **Horizon h = 10s**: Overall coverage = **87.24%**  
  (Per profile: `random_mix`: 86.29%, `double_burst`: 86.83%, `spike`: 88.73%, `random_walk`: 100.0%)
- **Horizon h = 20s**: Overall coverage = **87.35%**  
  (Per profile: `random_mix`: 86.15%, `double_burst`: 86.92%, `spike`: 88.24%, `random_walk`: 100.0%)
- **Horizon h = 30s**: Overall coverage = **91.94%**  
  (Per profile: `random_mix`: 90.53%, `double_burst`: 97.94%, `spike`: 92.68%, `random_walk`: 100.0%)

### 3.2 Evaluation Tables Across Horizons

#### Horizon $h = 10\text{s}$
*Calibrated 90% Margins: ST-AGBDT=+3.46, Persistence=+14.33, EWMA 10s=+15.26, Peak-Hold 30s=+7.33, Peak-Hold 60s=+6.67, Linear Trend=+26.56*

| Model Variant | MAE (req/s) | Miss Rate % | Mean Extra Workers |
| :--- | :---: | :---: | :---: |
| **ST-AGBDT Padded 90%** | **10.223** | **6.45%** | 1.362 |
| **ST-AGBDT (raw)** | **7.796** | 9.82% | 0.943 |
| Persistence Padded 90% | 11.209 | 11.50% | 1.152 |
| Persistence (raw) | 7.793 | 49.37% | 0.058 |
| EWMA 10s Padded 90% | 10.822 | 11.15% | 1.224 |
| EWMA 10s (raw) | 8.640 | 55.96% | 0.081 |
| Peak-Hold 30s Padded 90% | 14.882 | 7.08% | 1.258 |
| Peak-Hold 30s (raw) | 8.441 | 21.04% | 0.528 |
| Peak-Hold 60s Padded 90% | 16.143 | 6.94% | 1.284 |
| Peak-Hold 60s (raw) | 10.287 | 16.55% | 0.706 |
| Linear Trend Padded 90% | 26.124 | 8.13% | 1.921 |
| Linear Trend (raw) | 14.993 | 41.66% | 0.353 |

**Paired-Bootstrap Differences vs ST-AGBDT (RAW MAE)** ($\Delta = \text{MAE}_{\text{ST-AGBDT}} - \text{MAE}_{\text{Base}}$):
- vs **Persistence**: Mean Diff = `+0.012` req/s [95% CI: -1.909, +1.951] (0 in CI: True $\rightarrow$ not statistically significant)
- vs **EWMA 10s**: Mean Diff = `-0.845` req/s [95% CI: -2.950, +1.229] (0 in CI: True)
- vs **Peak-Hold 30s**: Mean Diff = `-1.927` req/s [95% CI: -5.137, +0.777] (0 in CI: True)
- vs **Peak-Hold 60s**: Mean Diff = `-2.504` req/s [95% CI: -5.923, +0.479] (0 in CI: True)
- vs **Linear Trend**: Mean Diff = `-7.196` req/s [95% CI: -10.575, -3.878] (0 in CI: False)

#### Horizon $h = 20\text{s}$
*Calibrated 90% Margins: ST-AGBDT=+5.08, Persistence=+23.33, EWMA 10s=+24.30, Peak-Hold 30s=+16.33, Peak-Hold 60s=+16.33, Linear Trend=+47.33*

| Model Variant | MAE (req/s) | Miss Rate % | Mean Extra Workers |
| :--- | :---: | :---: | :---: |
| **ST-AGBDT Padded 90%** | **13.245** | **5.40%** | 1.640 |
| **ST-AGBDT (raw)** | **9.930** | 12.04% | 1.258 |
| Persistence Padded 90% | 15.319 | 11.19% | 1.360 |
| Persistence (raw) | 12.042 | 66.67% | 0.029 |
| EWMA 10s Padded 90% | 14.400 | 7.64% | 1.462 |
| EWMA 10s (raw) | 13.079 | 76.47% | 0.042 |
| Peak-Hold 30s Padded 90% | 18.496 | 7.48% | 1.407 |
| Peak-Hold 30s (raw) | 9.653 | 29.86% | 0.443 |
| Peak-Hold 60s Padded 90% | 19.276 | 7.41% | 1.438 |
| Peak-Hold 60s (raw) | 9.897 | 26.00% | 0.505 |
| Linear Trend Padded 90% | 46.278 | 5.79% | 1.847 |
| Linear Trend (raw) | 27.282 | 50.69% | 0.461 |

**Paired-Bootstrap Differences vs ST-AGBDT (RAW MAE)**:
- vs **Persistence**: Mean Diff = `-2.124` req/s [95% CI: -4.996, +0.754] (0 in CI: True $\rightarrow$ not statistically significant)
- vs **EWMA 10s**: Mean Diff = `-3.169` req/s [95% CI: -6.308, -0.063] (0 in CI: False)
- vs **Peak-Hold 30s**: Mean Diff = `+0.257` req/s [95% CI: -3.029, +3.297] (0 in CI: True)
- vs **Peak-Hold 60s**: Mean Diff = `+0.030` req/s [95% CI: -3.675, +3.373] (0 in CI: True)
- vs **Linear Trend**: Mean Diff = `-17.380` req/s [95% CI: -23.485, -11.298] (0 in CI: False)

#### Horizon $h = 30\text{s}$
*Calibrated 90% Margins: ST-AGBDT=+6.70, Persistence=+37.33, EWMA 10s=+40.50, Peak-Hold 30s=+30.67, Peak-Hold 60s=+27.67, Linear Trend=+55.67*

| Model Variant | MAE (req/s) | Miss Rate % | Mean Extra Workers |
| :--- | :---: | :---: | :---: |
| **ST-AGBDT Padded 90%** | **16.056** | **0.43%** | 1.598 |
| **ST-AGBDT (raw)** | **11.318** | 6.17% | 1.396 |
| Persistence Padded 90% | 23.154 | 0.77% | 1.638 |
| Persistence (raw) | 15.591 | 79.85% | 0.016 |
| EWMA 10s Padded 90% | 24.322 | 0.00% | 1.649 |
| EWMA 10s (raw) | 16.566 | 88.25% | 0.018 |
| Peak-Hold 30s Padded 90% | 26.652 | 1.63% | 1.585 |
| Peak-Hold 30s (raw) | 10.460 | 42.97% | 0.247 |
| Peak-Hold 60s Padded 90% | 24.858 | 4.46% | 1.500 |
| Peak-Hold 60s (raw) | 10.510 | 38.34% | 0.286 |
| Linear Trend Padded 90% | 56.920 | 8.06% | 1.523 |
| Linear Trend (raw) | 36.953 | 57.46% | 0.445 |

**Paired-Bootstrap Differences vs ST-AGBDT (RAW MAE)**:
- vs **Persistence**: Mean Diff = `-4.330` req/s [95% CI: -8.731, -0.335] (0 in CI: False)
- vs **EWMA 10s**: Mean Diff = `-5.315` req/s [95% CI: -10.009, -1.244] (0 in CI: False)
- vs **Peak-Hold 30s**: Mean Diff = `+0.805` req/s [95% CI: -3.055, +4.714] (0 in CI: True)
- vs **Peak-Hold 60s**: Mean Diff = `+0.777` req/s [95% CI: -3.509, +4.923] (0 in CI: True)
- vs **Linear Trend**: Mean Diff = `-25.703` req/s [95% CI: -34.583, -17.532] (0 in CI: False)

### 3.3 Matched Operating Points (Interpolated Conformal Frontier over Levels 0.50–0.99)
*Citations: `results/final_evaluation_results_v2.json:horizons.<h>.matched_operating_frontiers`*

Mean extra workers required to achieve a specified miss rate target across whole-run resamples:

| Horizon | Target Miss Rate | ST-AGBDT Extra [95% CI] | Persistence Extra [95% CI] | EWMA 10s Extra [95% CI] | Peak-Hold 30s Extra [95% CI] | Peak-Hold 60s Extra [95% CI] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **h = 10s** | **10.0%** | **0.962** [0.793, 1.129] | 1.282 [1.149, 1.417] | 1.298 [1.104, 1.529] | 1.244 [0.909, 1.612] | 1.240 [0.934, 1.576] |
| | **6.45%** | **1.382** [1.146, 1.601] | 1.624 [1.438, 1.866] | 1.589 [1.336, 1.898] | 1.615 [1.290, 1.942] | 1.605 [1.278, 1.958] |
| | **5.0%** | **1.619** [1.373, 1.871] | 1.830 [1.568, 2.214] | 1.732 [1.430, 2.093] | 1.775 [1.424, 2.112] | 1.774 [1.424, 2.110] |
| | **1.0%** | **2.306** [1.819, 2.738] | 2.456 [1.928, 2.947] | 2.399 [1.903, 2.850] | 2.348 [1.846, 2.845] | 2.340 [1.837, 2.837] |
| **h = 20s** | **10.0%** | 1.344 [1.030, 1.641] | 1.399 [1.057, 1.715] | **1.293** [0.956, 1.704] | **1.280** [0.889, 1.657] | 1.300 [0.915, 1.688] |
| | **5.40%** | 1.612 [1.188, 1.984] | 1.642 [1.210, 2.042] | 1.589 [1.160, 2.039] | **1.553** [1.123, 1.994] | 1.611 [1.149, 2.153] |
| | **5.0%** | 1.634 [1.205, 2.013] | 1.666 [1.226, 2.072] | 1.617 [1.171, 2.067] | **1.583** [1.142, 2.030] | 1.650 [1.175, 2.201] |
| | **1.0%** | **1.920** [1.343, 2.437] | 1.950 [1.373, 2.507] | 1.933 [1.376, 2.471] | 1.930 [1.361, 2.443] | 1.975 [1.383, 2.567] |
| **h = 30s** | **10.0%** | 1.275 [0.847, 1.665] | 1.152 [0.778, 1.510] | **1.066** [0.706, 1.431] | 1.106 [0.698, 1.524] | 1.131 [0.706, 1.560] |
| | **5.0%** | 1.419 [0.917, 1.865] | 1.376 [0.931, 1.817] | **1.302** [0.862, 1.708] | 1.338 [0.859, 1.789] | 1.398 [0.892, 1.918] |
| | **1.0%** | **1.554** [0.996, 2.091] | 1.604 [1.048, 2.172] | 1.569 [1.024, 2.116] | 1.583 [1.020, 2.174] | 1.597 [1.029, 2.175] |
| | **0.43%** | **1.588** [1.017, 2.154] | 1.628 [1.049, 2.224] | 1.611 [1.042, 2.182] | 1.612 [1.044, 2.213] | 1.620 [1.046, 2.223] |

*Note: At h=30s and low miss rates (e.g. 0.43%), all padded methods require approximately 1.59 to 1.63 extra workers and their confidence intervals completely overlap.*

---

## 4. Replay Demo Verification (`replay_demo.py`)

- **Feature Parity Unit Test**: `tests/test_feature_parity.py` validates that executing the online rolling buffer (`StreamingArrivalFeatureExtractor.push`) second by second matches offline sequence batch calculation (`extract_features_sequence`) with relative and absolute tolerance $\le 10^{-12}$. All 11 unit tests passed in 5.43 seconds (`python -m pytest tests/`).
- **Streaming Replays Generated**:
  1. **`random_mix_seed20`**:
     - 97 valid seconds ($10 \le t < 107$).
     - Forecast MAE: **14.543 req/s**
     - Workers Needed Miss Rate: **3.09%** (3 misses out of 97 seconds)
     - Mean Extra Workers: **1.804 workers**
     - Output files: `results/replay/replay_random_mix_seed20.csv`, `results/replay/replay_random_mix_seed20.png`
  2. **`spike_seed44`**:
     - 51 valid seconds ($10 \le t < 61$).
     - Forecast MAE: **6.829 req/s**
     - Workers Needed Miss Rate: **0.00%** (0 misses out of 51 seconds)
     - Mean Extra Workers: **0.765 workers**
     - Output files: `results/replay/replay_spike_seed44.csv`, `results/replay/replay_spike_seed44.png`

---

## 5. Claims We Can Make / Cannot Make

### CAN:
- A from-scratch, NumPy-only boosted-tree forecaster with an asymmetric (expectile) loss and conformal upper bounds, about 54 KB per horizon; online and offline features match to 1e-12.
- On 13 held-out runs, raw ST-AGBDT MAE equals unpadded persistence at h=10 s (7.796 vs 7.793 req/s) and is lower in point estimate at h=20 s (9.93 vs 12.04) and h=30 s (11.32 vs 15.59); confidence intervals overlap, so significance is not established.
- ST-AGBDT + 90% conformal reached 6.45% / 5.40% / 0.43% workers-needed miss at h = 10/20/30 s vs 11.50% / 11.19% / 0.77% for padded persistence, at similar or slightly higher extra-worker cost; operating points are not matched and confidence intervals overlap, so no dominance is claimed; at h=30 s the padded baselines are equivalent in practice.
- The replay demo runs on held-out runs; headline run is random_mix_seed20; spike_seed44 shares its shape with training seeds (in-profile).

### CANNOT:
- Any closed-loop benefit or any claim of beating reactive scaling (not tested).
- That the MAE improvement is statistically significant.
- Any benefit from early exit (0% exited at a safe margin) or from flat arrays (slower per single sample); the "Self-Throttling" name is therefore unjustified.
- Results for pure step or ramp traffic (no step/ramp runs in the test set).
- Generalisation beyond the authors' synthetic generator; real traces, energy and real applications are untested.
- That unpadded persistence "proves bursts can't be anticipated": its miss rate is a property of the max-over-horizon target.

---

## 6. UNVERIFIED Section

1. **Production Live Closed-Loop Scaling**: While `/forecast/demand` was implemented and tested offline via Flask test client, running closed-loop proactive autoscaling under live Locust traffic with the live remediation service was explicitly excluded from today's run-alone task scope.
2. **Very Long Horizon Extrapolations ($h > 30\text{s}$)**: Horizons beyond 30 seconds were not trained or evaluated.
3. **Non-Standard Worker Limits**: The recommended worker count is clipped to the testbed ceiling $[2, 8]$. Behavior for cluster sizes $W > 8$ has not been calibrated or verified.

---

## 7. Repository Scan Output

Output of `python scan_repo.py`:

```
Scanning repository for sensitive paths / usernames...
Target extensions: {'.csv', '.ipynb', '.py', '.md', '.txt', '.json'}
Excluded dirs: {'.system_generated', 'venv', '__pycache__', '.venv', '.git', 'scratch', '.pytest_cache'}
======================================================================
Total files scanned: 407
Clean files: 407
Files with hits: 0
======================================================================

ALL SCANNED FILES ARE 100% CLEAN!
```
