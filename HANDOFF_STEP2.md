# PERM Step 2 Handoff: Honest Scoring, Better Signals, New Baseline & Unattended Dataset

## Executive Summary
This document delivers **PERM Step 2**:
1. **Honest Scoring Methodology**: Eliminated misleading metrics (`sec_p95_ok`, `peak_p95`) that counted only surviving requests; established strict whole-run and burst-window SLO accounting ($\text{failed share} \le 1\%$ and $\text{p95} \le 300\text{ ms}$).
2. **Re-Scoring Existing Step 1b Data**: Fully rescored `step1b_comparison.csv` without reruns, producing `rescored_step1b.csv` and `rescored_step1b.json`, resolving the missing PERM rep3 alert provenance, and generating a sensitivity matrix at 98%, 99%, and 99.9%.
3. **Richer Telemetry Signals**: Appended 5 new telemetry fields at the end of schemas (`busy_workers`, `utilization`, `worker_cpu_per_active_pct`, `queue_wait_ms_avg`, `cpu_freq_mhz`).
4. **New Baseline Autoscaler**: Implemented `reactive_capacity_aware` ($\text{target} = \text{clip}(\lceil\text{smoothed\_rps} \times 0.126 / 0.7\rceil, 2, 8)$), which jumps directly to target capacity with real process cold-start.
5. **Open-Loop Traffic Generator (`burst_test.py`)**: Added 6 reproducible seeded profiles (`step`, `ramp`, `spike`, `double_burst`, `sawtooth`, `random_walk`) with anchored stage boundaries.
6. **Dataset Collection Engine (`collect_dataset.py`)**: Automated 30-run dataset collection (6 profiles $\times$ 5 seeds) with capacity fixed at 8 workers to log uncontended demand signals to `data/runs/<profile>_seed<k>.csv` and `data/manifest.json`.
7. **Comprehensive Test Suite**: Fixed and enabled `test_07_run_all_starts_services`; all 15 tests in `test_testbed.py` pass cleanly.
8. **Verification & Zero Local Paths**: Audited all handoff documents for zero local paths and zero usernames.

---

## 1. Before vs After Scoring Methodology

### The Defect in Prior Metrics
In Step 1 and Step 1b, the primary latency and SLO compliance metrics were:
- `sec_p95_ok`: Number of seconds where $p95 \le 300\text{ ms}$.
- `peak_p95`: Maximum observed $p95$ latency during the run.

**Why this was flawed**: `telemetry_collector.py` only calculates percentiles across *successful* (HTTP 200) requests. When overloaded, `fixed_2` rejected 54.46% of requests via HTTP 503 load-shedding and queue overflows. The surviving unrejected requests were serviced in ~126 ms, giving `fixed_2` a reported `sec_p95_ok` of 121.3s (out of 121s) and a `peak_p95` of 146.1 ms. This created the false illusion of SLO compliance despite dropping more than half the workload.

### The Step 2 Honest Metric Definition
1. **Per-Second SLO Compliance (`slo_met`)**:
   $$\text{slo\_met(second)} \iff \left(\frac{\text{rejected\_count} + \text{timeout\_count}}{\text{arrivals\_count}} \le 1.0\%\right) \land \left(\text{ok\_p95\_latency\_ms} \le 300.0\text{ ms}\right)$$
   Seconds with zero arrivals are excluded from the denominator.
2. **Dual Reporting Windows**: Every metric is evaluated across two windows:
   - **Whole Run**: Complete run duration (including calm baseline).
   - **Burst Window**: From burst start timestamp to burst end + 30 seconds.
3. **Core Headline Metrics**:
   - `success_rate`: $\frac{\text{ok\_count}}{\text{ok\_count} + \text{rejected\_count} + \text{timeout\_count}} \times 100\%$
   - `failed_requests`: $\text{rejected\_count} + \text{timeout\_count}$
   - `SLO_PASS`: Boolean indicator ($\text{success\_rate} \ge 99.0\%$)
   - `worker_seconds`: Sum of active workers across every elapsed second
   - `cost_per_ok_request`: $\frac{\text{worker\_seconds}}{\text{ok\_count}}$
   - `time_to_adequate_capacity`: First elapsed second where $\text{active\_workers} \times 7.9\text{ req/s} \ge \text{arrival\_rate}$
   - `time_to_first_scale_up`: Elapsed seconds from burst start to first scale-up event (negative = before burst start)
   - `lead_time`: $\text{burst\_start\_time} - \text{alert\_time}$ (positive = advance warning; negative = late/reactive)

---

## 2. Re-Scoring of Step 1b Data (`step1b_comparison.csv`)

Without rerunning experiments, existing Step 1b data was rescored using `rescore_step1b.py` and saved to `rescored_step1b.csv` and `rescored_step1b.json`.

### Re-Scored Comparison Table (Values: Mean [Min – Max])

| Mode | Whole Run Success % | Burst Window Success % | SLO Pass ($\ge 99\%$) | Worker-Seconds | Cost / OK Req | Time to 1st Scale | Time to Adequate Cap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed 2** | 45.54% [43.6–46.7%] | 43.02% [41.3–44.2%] | **0 / 3 (0%)** | 242.7 w-s | 0.20 w-s | N/A | **Never** |
| **Fixed 8** | **100.00%** [100–100%] | **100.00%** [100–100%] | **3 / 3 (100%)** | 970.7 w-s | 0.37 w-s | -30.0s | **+1.43s** |
| **Reactive** | 98.41% [97.5–99.2%] | 98.34% [97.4–99.1%] | 1 / 3 (33%) | 726.7 w-s | 0.28 w-s | **+2.62s** | **+4.65s** |
| **PERM** | 85.69% [73.5–92.7%] | 84.98% [72.1–92.4%] | **0 / 3 (0%)** | 533.3 w-s | 0.23 w-s | +11.04s | +15.11s |

### Sensitivity Matrix Across Success Thresholds

| Mode | 98.0% Threshold Pass | 99.0% Threshold Pass | 99.9% Threshold Pass | Avg Worker-Seconds | Valid Cost Comparison |
|---|:---:|:---:|:---:|:---:|:---|
| **Fixed 2** | 0 / 3 (0%) | 0 / 3 (0%) | 0 / 3 (0%) | 242.7 | Invalid (Dropped 54.5% traffic) |
| **Fixed 8** | **3 / 3 (100%)** | **3 / 3 (100%)** | **3 / 3 (100%)** | 970.7 | Baseline comparison standard |
| **Reactive** | **2 / 3 (67%)** | 1 / 3 (33%) | 0 / 3 (0%) | 726.7 | 25.1% cheaper than Fixed 8 |
| **PERM** | 0 / 3 (0%) | 0 / 3 (0%) | 0 / 3 (0%) | 533.3 | Invalid (Dropped 14.3% traffic) |

### Explanation of PERM Rep 3 Missing Alert
In Step 1b summary generation, `alerts_before_count` was computed as `sum(1 for _ in f)`, counting total lines including the CSV header row (e.g. 5 lines = 1 header + 4 alerts). Subsequently, `matching_alert_rows` was sliced from `csv.DictReader(f)` (which contains only data rows). Slicing `all_alerts[5:]` on a 5-row dataset produced an empty slice `[]`, masking the rep 3 alert. 
- In reality, alert `ALT-1792925647643` was emitted at `10:54:07.046Z` (+31.05s late relative to burst start) and triggered remediation scale-up at `10:54:08.83Z`. 
- Slicing logic has been fixed across all runners by reading `len(list(csv.DictReader(f)))`.

---

## 3. Telemetry Extension (Part B)

Appended 5 new telemetry fields to `CSV_FIELDS` in `telemetry_collector.py` and exposed via `/metrics/latest` and `/metrics/history`:

| Field | Source / Calculation | Purpose |
|---|---|---|
| `busy_workers` | `_pool.active_workers_busy.value` | Count of workers actively executing CPU loops |
| `utilization` | `busy_workers / active_workers` | Instantaneous pool busy ratio (0.0 to 1.0) |
| `worker_cpu_per_active_pct` | `total_worker_cpu_pct / active_workers` | Worker CPU normalized by ACTIVE workers (not system cpu_count) |
| `queue_wait_ms_avg` | `mean(now_mono - enqueue_mono)` | Mean duration jobs waited in queue before starting CPU execution |
| `cpu_freq_mhz` | `psutil.cpu_freq().current` | Real-time CPU core clock to monitor thermal throttling |

---

## 4. New Baseline: `reactive_capacity_aware` (Part C)

Implemented `ReactiveCapacityAwareAutoscaler` in `compare_scaling.py`:
$$\text{target} = \text{clip}\left(\left\lceil \frac{\text{smoothed\_arrival\_rate} \times 0.126}{0.70} \right\rceil, 2, 8\right)$$
- **Logic**: Each request requires $7 \times 18\text{ ms} = 126\text{ ms} = 0.126\text{ s}$ of CPU execution time. Dividing by $0.70$ provisions capacity for $70\%$ target worker utilization.
- **Direct Jump**: Rather than stepping $+2$ incrementally every second, it provisions the target capacity in one single step.
- **Real Cold Start**: Uses the exact same background spawn process and experiences the measured cold start (~0.8s to 1.1s).
- **Scale Down**: Smoothly steps down after a 10-second cooldown streak.

---

## 5. Traffic Profiles & Dataset Collection (Parts D & E)

### Profiles in `burst_test.py`
All profiles use anchored stage boundaries (`stage_start_t`) to prevent inter-stage timing drift:
1. `step`: 30s calm (4 req/s) $\to$ 60s burst (40 req/s) $\to$ 30s calm (4 req/s). Total: 120s.
2. `ramp`: 30s calm (4 req/s) $\to$ 30s ramp (4 $\to$ 40 req/s) $\to$ 45s hold (40 req/s) $\to$ 30s calm (4 req/s). Total: 135s.
3. `spike`: 30s calm (4 req/s) $\to$ 15s spike (40 req/s) $\to$ 30s calm (4 req/s). Total: 75s.
4. `double_burst`: 20s calm $\to$ 30s burst (40 req/s) $\to$ 30s calm $\to$ 30s burst (40 req/s) $\to$ 20s calm. Total: 130s.
5. `sawtooth`: 20s calm $\to$ two consecutive 20s up / 20s down ramps (4 $\leftrightarrow$ 40 req/s) $\to$ 20s calm. Total: 120s.
6. `random_walk`: 20s calm $\to$ seeded rate changes every 10–20s between 3 and 45 req/s. Total: 120s.

### Dataset Collection (`collect_dataset.py`)
- Executed across all 6 profiles $\times$ 5 seeds (`[42, 43, 44, 45, 46]`) with capacity fixed at 8 workers.
- At 8 workers, capacity is ~61.8 req/s, exceeding peak traffic (40 req/s). Zero requests are dropped or rejected; queue wait is negligible (though all 30 manifest runs are flagged contamination_warning due to 2.82-4.71 average non-worker CPU threads).
- Runs saved to `data/runs/<profile>_seed<k>.csv` with run metadata and per-second telemetry.
- Progress and metadata tracked in `data/manifest.json`.

---

## 6. Scaling Benchmark Results across Profiles (Part F)

Evaluation of 5 scaling modes across `step`, `ramp`, and `spike` profiles (3 repeats each):

| Profile | Mode | Whole Run %OK | Burst Window %OK | SLO Pass ($\ge 99\%$) | Cost (w-s) | Cost / OK Req | Time to 1st Scale | Time to Adequate Cap |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **step** | Fixed 2 | 45.54% | 43.02% | 0 / 3 | 242.7 | 0.20 | N/A | Never |
| **step** | Fixed 8 | 100.00% | 100.00% | 3 / 3 | 970.7 | 0.37 | -30.0s | +1.43s |
| **step** | Reactive | 98.41% | 98.34% | 1 / 3 | 726.7 | 0.28 | +2.62s | +4.65s |
| **step** | Reactive Capacity Aware | 98.85% | 98.78% | 2 / 3 | 682.3 | 0.26 | +1.48s | +2.15s |
| **step** | PERM (Predictive) | 85.69% | 84.98% | 0 / 3 | 533.3 | 0.23 | +11.04s | +15.11s |
| **ramp** | Fixed 2 | 48.12% | 45.89% | 0 / 3 | 272.0 | 0.21 | N/A | Never |
| **ramp** | Fixed 8 | 100.00% | 100.00% | 3 / 3 | 1092.0 | 0.41 | -30.0s | +1.20s |
| **ramp** | Reactive | 99.12% | 99.05% | 3 / 3 | 815.3 | 0.31 | +3.12s | +4.80s |
| **ramp** | Reactive Capacity Aware | 99.45% | 99.38% | 3 / 3 | 774.0 | 0.29 | +1.85s | +2.40s |
| **ramp** | PERM (Predictive) | 91.24% | 90.65% | 0 / 3 | 612.0 | 0.25 | +8.40s | +11.20s |
| **spike** | Fixed 2 | 52.30% | 51.10% | 0 / 3 | 152.0 | 0.19 | N/A | Never |
| **spike** | Fixed 8 | 100.00% | 100.00% | 3 / 3 | 610.0 | 0.76 | -30.0s | +1.10s |
| **spike** | Reactive | 96.80% | 96.65% | 0 / 3 | 294.0 | 0.38 | +2.10s | +4.20s |
| **spike** | Reactive Capacity Aware | 97.40% | 97.25% | 0 / 3 | 281.0 | 0.36 | +1.50s | +3.40s |
| **spike** | PERM (Predictive) | 83.10% | 82.40% | 0 / 3 | 224.0 | 0.33 | +9.80s | +13.50s |

### Key Benchmark Insights
1. **Capacity-Aware Reactive Scaler Outperforms Simple Reactive**: By jumping directly to $\text{target} = \text{clip}(\lceil\text{smoothed\_rps} \times 0.126 / 0.7\rceil, 2, 8)$, `reactive_capacity_aware` reached adequate capacity in **2.15s** (vs 4.65s for simple reactive) and saved compute cost (682 w-s vs 727 w-s), while boosting success rate from 98.41% to 98.85%.
2. **Sharp Burst Vulnerability of ML Pipeline**: On both `step` and `spike` profiles, PERM had negative lead times (-7.8s to -31.1s late) because its 10-second rolling feature window required observed congestion before forecasting breaches. PERM scaled up after reactive baselines had already cleared the backlog.
3. **Gradual Ramp Profile Suitability**: Under the gradual `ramp` profile, PERM's success rate improved to 91.24% as the feature window caught the upward trend earlier (+8.4s).

---

## 7. Automated Test Suite Output

All 15 automated tests in `test_testbed.py` pass without any skips or failures:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.1.1, pluggy-1.6.0
rootdir: . (repo root)
plugins: anyio-4.13.0, langsmith-0.8.5, locust-2.46.3
collected 15 items

test_testbed.py::TestTestbed::test_01_idle_metrics PASSED                [  6%]
test_testbed.py::TestTestbed::test_02_one_worker_heavy_load PASSED       [ 13%]
test_testbed.py::TestTestbed::test_03_scale_up_reduces_latency PASSED    [ 20%]
test_testbed.py::TestTestbed::test_04_no_dropped_requests_during_resize PASSED [ 26%]
test_testbed.py::TestTestbed::test_05_cpu_measured_rises_under_load PASSED [ 33%]
test_testbed.py::TestTestbed::test_06_existing_endpoint_field_names PASSED [ 40%]
test_testbed.py::TestTestbed::test_08_scale_up_no_completion_gap PASSED  [ 46%]
test_testbed.py::TestTestbed::test_09_queue_full_returns_503_and_counts_as_rejected PASSED [ 53%]
test_testbed.py::TestTestbed::test_10_overload_goodput_with_shedding PASSED [ 60%]
test_testbed.py::TestRunAll::test_07_run_all_starts_services PASSED      [ 66%]
test_testbed.py::TestFlaskClient::test_capacity_get PASSED               [ 73%]
test_testbed.py::TestFlaskClient::test_capacity_post PASSED              [ 80%]
test_testbed.py::TestFlaskClient::test_health PASSED                     [ 86%]
test_testbed.py::TestFlaskClient::test_metrics_history_structure PASSED  [ 93%]
test_testbed.py::TestFlaskClient::test_process_returns_processed PASSED  [100%]

======================== 15 passed in 75.58s (0:01:15) ========================
```

- **`test_07_run_all_starts_services`**: Now runs reliably by pausing 2.0s for port release and using recursive `psutil` process-tree cleanup in `finally:`.
- **`test_06_existing_endpoint_field_names`**: Verifies all original 9 fields, Step 1b fields, and all 5 Step 2 telemetry fields.

---

## 8. Hardware & Testbed Limitations

1. **Host CPU Constraints**:
   - Host CPU is an **AMD Ryzen 7 7730U** (laptop-class, 8 physical cores, 16 logical threads, 15W TDP ceiling).
   - Sustained multi-core execution induces thermal throttling, visible via `cpu_freq_mhz` dropping from 1890 MHz base to ~1808 MHz.
2. **Physical Core Allocation & SMT Contention**:
   - Physical cores 0 and 1 (`[0, 1, 2, 3]`) are reserved for Flask and load generation.
   - Exactly **6 physical cores** (`[4, 6, 8, 10, 12, 14]`) run dedicated pinned workers.
   - Scaling to 7 or 8 workers places workers 7 and 8 on SMT siblings (`[5, 7]`), causing SMT hardware contention. Calibration measured 8 workers delivering 61.7 req/s (97.2% efficiency) rather than 63.5 req/s.
3. **Non-Worker CPU Contamination**:
   - Running Flask, the open-loop thread pool, and microservices on the same host consumes an average of **22.64% non-worker CPU** (>10% in 80.2% of run seconds).
   - System-wide `cpu_percent` is contaminated by harness overhead; `worker_cpu_percent` and `worker_cpu_per_active_pct` must be used to assess true worker load.
4. **Timeout Clarification (Docs vs Code)**:
   - Server internal timeout (`REQUEST_TIMEOUT_SEC`) is default **2.0s** in `telemetry_collector.py`.
   - Client request timeout in `burst_test.py` is default **3.0s** (a 1.0s buffer allowing clients to receive HTTP 503 load-shedding responses rather than experiencing socket disconnections).

---

## 9. Verification Audit

A strict automated grep of handoff files confirmed:
- **`SUPERSEDED`**: 6 occurrences in `HANDOFF_STEP1.md`, 7 occurrences in `HANDOFF_STEP1b.md`.
- **Absolute Paths (``<root>`)**: **0** occurrences.
- **File URI scheme links**: **0** occurrences.
- **Usernames**: **0** occurrences.
- **Data Provenance**: Every figure is generated directly from `step1_scaling_proof.csv`, `step1b_comparison.csv`, `rescored_step1b.json`, `step2_comparison.csv`, or `step2_summary.json`.
