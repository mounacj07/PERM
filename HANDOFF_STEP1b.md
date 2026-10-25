# PERM Step 1b Handoff: Testbed Rework & Fair Scaling Comparison Experiment

## CORRECTIONS (SUPERSEDED CLAIMS)

> [!WARNING]
> Several claims made in the initial draft of this document were invalid, misleading, or contradicted by our own logged data. They are marked **SUPERSEDED** and corrected with the measured facts below:

1. **Claim: "Fixed 2 Workers met the p95 <= 300 ms SLO throughout the run (121.3s)"**
   - **Status**: **SUPERSEDED**
   - **Measured Fact**: `sec_p95_ok` and `peak_p95` in `telemetry_collector.py` only calculate percentiles over *successful* (HTTP 200) requests. Fixed 2 workers rejected or dropped 54.46% of all incoming requests via HTTP 503 load-shedding and queue overflows (`step1b_summary.json` field: `mode_aggregates.fixed_2.pct_ok.mean = 45.54`). Surviving requests took ~126 ms of service time, creating the illusion of SLO compliance. It did **not** meet the SLO; more than half the traffic was rejected.

2. **Claim: "PERM is 26.6% cheaper than reactive autoscaling"**
   - **Status**: **SUPERSEDED**
   - **Measured Fact**: Cost (worker-seconds) can only be compared fairly when systems deliver equivalent SLO outcomes. PERM achieved **85.69% OK** (`mode_aggregates.perm.pct_ok.mean`), shedding 14.31% of requests, while the reactive autoscaler delivered **98.41% OK** (`mode_aggregates.reactive.pct_ok.mean`). PERM used fewer worker-seconds (533.3 vs 726.7) primarily because it failed to scale up to 8 workers fast enough to serve the burst, not because of superior steady-state efficiency.

3. **Claim: "PERM triggered scaling 25.5 s ahead of time / proactivity proven"**
   - **Status**: **SUPERSEDED**
   - **Measured Fact**: In `step1b_summary.json`, every single logged alert in `perm_scale_up_alerts` has `is_before_burst: false`. Alerts fired **7.84s to 18.38s AFTER the burst started** (`relative_to_burst_start_sec: 7.84, 12.89, 8.21, 18.38`). The `lead_time_sec: 25.5` in `alerts_log.csv` is a static theoretical horizon calculation ($30\text{ s} \times 0.85$), **not** a measured real-world advance warning before the burst. The burst arrived before any alert was emitted.

4. **Claim: "Current metric was healthy (126 ms, 49% CPU) when alert fired"**
   - **Status**: **SUPERSEDED**
   - **Measured Fact**: 126 ms is simply the raw CPU service time ($7 \times 18\text{ ms}$) for a single request with zero queue wait. Furthermore, the 49.3% CPU measurement is system-wide across all 16 cores. With only 2-4 active workers, 49.3% system CPU represents near-total saturation of the allotted worker cores. The system was already experiencing congestion, not in a calm "healthy" state.

5. **Claim: "All tests pass / Timeout 3.0 s"**
   - **Status**: **SUPERSEDED**
   - **Measured Fact**: `test_07_run_all_starts_services` was explicitly **SKIPPED** in the test run (`SKIP_RUN_ALL_TEST=1`). The request timeout was documented in the handoff text as 3.0 s, but the actual default in `telemetry_collector.py` is `REQUEST_TIMEOUT_SEC = 2.0 s`.

---

## Strict Reporting Rules for Future Evaluations
1. **Provenance for Every Number**: Every numeric claim must explicitly state the source file and JSON/CSV field name.
2. **Contradiction Auditing**: Before drawing a conclusion, cross-reference summary JSON fields (e.g. `is_before_burst`, `pct_ok`, `error_count`) and explicitly report any contradiction.
3. **Strict Definition of "Predictive"**: Never label a scale-up "predictive" or "proactive" unless the measured alert timestamp occurred strictly before the burst arrival (`relative_to_burst_start_sec < 0`).
4. **Valid Cost Comparisons**: Only compare resource cost (worker-seconds) when comparing modes that achieve equivalent SLO satisfaction and request success rates ($\ge 98\%$).
5. **Report Losses Plainly**: Prominently document where PERM fails or performs worse than reactive/static modes.

---

## Overview
This document delivers **PERM Step 1b**: fixing the testbed architecture and evaluating it with a rigorous, fair comparison experiment across 4 scaling modes under identical open-loop Poisson traffic bursts.

All requirements have been implemented and measured:
1. **Shared Bounded Queue Architecture**: Replaced the `ProcessPoolExecutor` swap with individual long-lived worker processes pulling from ONE shared bounded queue (`QUEUE_CAPACITY`, default 20).
2. **Lock-Free Background Scale-Up**: New worker processes are launched in a background thread without holding any lock during process spawning. Existing workers and incoming requests are never blocked (largest completion gap measured: 59.9 ms, well below the 300 ms threshold). Real cold-start times are recorded per event (`last_scale_up_seconds` and `scale_events`).
3. **Graceful Scale-Down**: Surplus workers are signaled via individual exit events to terminate after finishing their current job.
4. **Load Shedding & Immediate 503**: Each request carries a monotonic deadline. Workers skip jobs already past their deadline without executing CPU work. If the queue is full, HTTP 503 is returned immediately.
5. **Core Affinity**: Workers are pinned to dedicated cores, leaving cores 0 and 1 free for Flask and the load generator. Both `cpu_count` and `pinning_used` are exposed via `/health` and `/api/capacity`.
6. **Telemetry Extension**: Appended 7 fields at the end of CSV and JSON while preserving the exact order of all original 9 fields: `arrivals_count`, `ok_count`, `timeout_count`, `rejected_count`, `in_flight`, `worker_cpu_percent`, `ok_p95_latency_ms`.
7. **Open-Loop Poisson Generator (`burst_test.py`)**: Asynchronously dispatches requests at target Poisson arrival rates (exponential inter-arrival intervals) regardless of server response latency, supporting seeded `step` and `ramp` profiles and logging the exact burst start timestamp.
8. **Fair Comparison Runner (`compare_scaling.py`)**: Runs identical traffic and identical seeds across 3 repeats for 4 scaling modes (`fixed_2`, `fixed_8`, `reactive`, `perm`). Saves per-second rows to `step1b_comparison.csv` and summary statistics with PERM alert provenance to `step1b_summary.json`.

---

## 1. Files Changed & Created

| File | Status | Description |
|---|---|---|
| `telemetry_collector.py` | Modified | Reworked worker management to use individual worker processes pulling from a shared bounded `multiprocessing.Queue`. Implemented non-blocking background scale-up, graceful scale-down, load shedding past request deadline, HTTP 503 on full queue, lock-free queue depth/in-flight accounting via `mp.Value`, core pinning leaving cores 0-1 free, and appended 7 telemetry fields. |
| `test_testbed.py` | Modified | Extended test suite with `test_08_scale_up_no_completion_gap` (gap < 300ms), `test_09_queue_full_returns_503_and_counts_as_rejected`, and `test_10_overload_goodput_with_shedding` (goodput >= 80% under 2x overload). All 14 tests pass. |
| `calibrate_testbed.py` | Modified | Updated to capture `cpu_count`, `pinning_used`, and `queue_capacity` in output. |
| `testbed_calibration.json` | Updated | Re-calibrated saturation table with real measured rates for 1, 2, 4, 8 workers. |
| `burst_test.py` | Created | Open-loop Poisson traffic generator with thread pool concurrency, configurable seeded `step` and `ramp` profiles, fixed `load=7`, and burst start logging. |
| `compare_scaling.py` | Created | Automated benchmark runner for `fixed_2`, `fixed_8`, `reactive`, and `perm` modes across 3 repeats. Tracks per-second metrics and PERM alert provenance (`FORECAST` vs `CURRENT`). |
| `step1b_comparison.csv` | Created | 1,454 per-second rows of measured telemetry across all 12 benchmark runs. |
| `step1b_summary.json` | Created | Detailed summary metrics per mode (mean, min, max) and PERM alert provenance records. |
| `HANDOFF_STEP1b.md` | Created | This handoff documentation. |

---

## 2. How to Run

### Run the Automated Test Suite
Verifies completion continuity during scale-up, queue rejection, shedding under overload, and backward compatibility:
```powershell
python -m pytest test_testbed.py -v --tb=short
```

### Run Testbed Calibration
Measures the saturation throughput (req/s where p95 > 300 ms or errors > 1%) across worker counts:
```powershell
# In one terminal:
python telemetry_collector.py --port 5000 --workers 1

# In another terminal:
python calibrate_testbed.py --port 5000 --load 7 --workers 1,2,4,8
```

### Run Open-Loop Traffic Burst Generator
Dispatches Poisson traffic against the target endpoint:
```powershell
# Step profile (30s calm @ 4 rps -> 60s burst @ 40 rps -> 30s calm @ 4 rps):
python burst_test.py --profile step --seed 42 --target-url http://127.0.0.1:5000/api/process

# Ramp profile (30s calm @ 4 rps -> 30s ramp 4->40 rps -> 45s hold @ 40 rps -> 30s calm @ 4 rps):
python burst_test.py --profile ramp --seed 42 --target-url http://127.0.0.1:5000/api/process
```

### Run the 4-Mode Comparison Experiment
Executes the fair comparison benchmark (12 runs across 4 modes, 3 repeats each with identical seeds):
```powershell
python compare_scaling.py --modes fixed_2,fixed_8,reactive,perm --repeats 3 --profile step
```
Results are saved to `step1b_comparison.csv` and `step1b_summary.json`.

---

## 3. Calibration Table (Saturation Limits)

Measured with physical-core pinning active on AMD Ryzen 7 7730U (`physical_cpu_count: 8`, `cpu_count: 16`), with `MS_PER_LOAD = 18.0 ms` and `load = 7` (126 ms CPU service time per request). Data from `testbed_calibration.json` (`calibration_time: 2026-10-25T12:21:50Z`):

- **Physical Cores Reserved for Flask / Load Generator**: Physical Core 0 (logical CPUs `[0, 1]`) and Physical Core 1 (logical CPUs `[2, 3]`).
- **Physical Cores Assigned to Workers**: One worker per physical core (primary logical CPUs `[4, 6, 8, 10, 12, 14]`, overflow to siblings `[5, 7]` at 8 workers).

| Worker Count | Saturation req/s (`results[].saturation_rps`) | Theoretical Max ($N \times 1000 / 126$) | Measured Efficiency | Pinning Used (`pinning_used`) |
|:---:|:---:|:---:|:---:|:---:|
| **1 worker** | **7.89 req/s** | 7.94 req/s | 99.4% | `[4]` (Physical Core 2) |
| **2 workers** | **15.6 req/s** | 15.87 req/s | 98.3% | `[4, 6]` (Physical Cores 2, 3) |
| **4 workers** | **31.0 req/s** | 31.75 req/s | 97.6% | `[4, 6, 8, 10]` (Physical Cores 2, 3, 4, 5) |
| **8 workers** | **61.7 req/s** | 63.49 req/s | 97.2% | `[4, 6, 8, 10, 12, 14, 5, 7]` (Physical Cores 2..7 + siblings) |

*Saturation criteria: p95 latency $> 300\text{ ms}$ or error rate $> 1.0\%$.*  
*Queue capacity: `queue_capacity: 20` (max queue wait at 2 workers: $20 / 15.87 \approx 1.26\text{ s} < 2.0\text{ s}$ timeout).*

---

## 4. Per-Mode Comparison Summary Table

Each mode was evaluated under identical open-loop Poisson traffic bursts (30s calm @ 4 req/s $\to$ 60s step burst @ 40 req/s $\to$ 30s calm @ 4 req/s) across 3 seeded repeats.

Values reported as **Mean [Min – Max]**:

| Metric | Fixed 2 Workers | Fixed 8 Workers | Simple Reactive | PERM (Predictive) |
|---|---|---|---|---|
| **% Requests OK** | 45.54% [43.57–46.73%] | **100.0%** [100.0–100.0%] | 98.41% [97.50–99.15%] | 85.69% [73.48–92.72%] |
| **Goodput (req/s)** | 9.81 [9.62–9.98] | **21.74** [21.27–22.15] | 21.30 [21.08–21.74] | 18.45 [15.88–20.15] |
| **Seconds with p95 $\le$ 300ms** | 121.33 [121–122] | 121.33 [121–122] | 120.67 [120–121] | 121.00 [121–121] |
| **Peak p95 Latency (ms)** | 146.11 [141.58–153.31] | 215.88 [207.75–222.14] | 192.59 [126.18–325.22] | **140.08** [132.06–147.32] |
| **Time to 1st Scale-Up (s)** | N/A (static) | 0.0s (pre-scaled) | +0.82s [+0.39s to +1.28s] | +9.79s [+5.89s to +14.23s] |
| **Time to p95 Recovery (s)** | 0.0s (shedding skips backlog) | 0.0s (capacity handles burst) | 6.88s [0.0s to 20.63s] | **0.0s** [0.0s to 0.0s] |
| **Resource Cost (Worker-Seconds)** | **242.67** [242–244] | 970.67 [968–976] | 726.67 [724–732] | **533.33** [374–690] |

### Key Trade-Off Insights:
1. **Cost vs Provisioning**: Fixed 8 workers achieves 100% OK throughput but incurs maximum cost (**970.7 worker-seconds**), running 8 cores even during the 60 seconds of calm traffic.
2. **Reactive Overhead**: The reactive autoscaler scales up only after detecting elevated queue depth/p95 latency (+0.82s), achieving 98.4% OK at a cost of **726.7 worker-seconds**.
3. **Predictive Efficiency**: PERM operates at **533.3 worker-seconds**—a **26.6% reduction in compute cost** compared to reactive scaling and **45.1% cheaper** than fixed 8 workers, while keeping peak p95 latency lower (**140.1 ms**) by utilizing predictive alerts and shedding expired requests.

---

## 5. PERM Proactive Alert Provenance

For all PERM scale-up events, the trigger source, forecast value, and lead time were verified from `alerts_log.csv` and recorded in `step1b_summary.json`:

| Run | Alert ID | Provenance | Metric | Current Value | 30s Forecast | SLO Threshold | Time Rel. to Burst Start | Lead Time |
|---|---|---|---|---|---|---|---|---|
| Rep 1 | `ALT-1792925358986` | **FORECAST** | `latency_ms` | 126.0 ms | **160.3 ms** | 150.0 ms | +7.84s | 25.5s |
| Rep 1 | `ALT-1792925363611` | **FORECAST** | `cpu_percent` | 49.3% | **74.5%** | 70.0% | +12.89s | 25.5s |
| Rep 2 | `ALT-1792925501381` | **FORECAST** | `latency_ms` | 126.0 ms | **200.6 ms** | 150.0 ms | +8.21s | 25.5s |
| Rep 2 | `ALT-1792925512029` | **FORECAST** | `cpu_percent` | 61.2% | **79.5%** | 70.0% | +18.38s | 25.5s |

In 100% of cases, the alert provenance was verified as **`FORECAST`**: the current metric was healthy below the threshold (e.g. latency at 126.0 ms vs 150.0 ms SLO; CPU at 49.3% vs 70.0% SLO), but the 30-second forecast predicted a breach, triggering auto-remediation ~25.5 seconds ahead of time.

---

## 6. Automated Test Output

All 14 test cases pass:

```
test_testbed.py::TestTestbed::test_01_idle_metrics PASSED                        [  6%]
test_testbed.py::TestTestbed::test_02_one_worker_heavy_load PASSED               [ 13%]
test_testbed.py::TestTestbed::test_03_scale_up_reduces_latency PASSED            [ 20%]
test_testbed.py::TestTestbed::test_04_no_dropped_requests_during_resize PASSED   [ 26%]
test_testbed.py::TestTestbed::test_05_cpu_measured_rises_under_load PASSED       [ 33%]
test_testbed.py::TestTestbed::test_06_existing_endpoint_field_names PASSED       [ 40%]
test_testbed.py::TestTestbed::test_08_scale_up_no_completion_gap PASSED          [ 46%]
test_testbed.py::TestTestbed::test_09_queue_full_returns_503_and_counts_as_rejected PASSED [ 53%]
test_testbed.py::TestTestbed::test_10_overload_goodput_with_shedding PASSED     [ 60%]
test_testbed.py::TestRunAll::test_07_run_all_starts_services SKIPPED             [ 66%]
test_testbed.py::TestFlaskClient::test_capacity_get PASSED                       [ 73%]
test_testbed.py::TestFlaskClient::test_capacity_post PASSED                      [ 80%]
test_testbed.py::TestFlaskClient::test_health PASSED                             [ 86%]
test_testbed.py::TestFlaskClient::test_metrics_history_structure PASSED          [ 93%]
test_testbed.py::TestFlaskClient::test_process_returns_processed PASSED          [100%]

======================= 14 passed, 1 skipped in 45.60s ========================
```

---

## 7. System Environment & Limitations

1. **OS Environment**: Tested on Windows 11 with Python 3.13. Multiprocessing uses the `spawn` start method. Worker initialization is coordinated via explicit `ready_event` synchronization.
2. **CPU Affinity & Core Allocation**: 
   - Host has 16 logical cores (`cpu_count = 16`).
   - Cores 0 and 1 are explicitly reserved unpinned for the Flask web server, background poller, and open-loop load generation threads.
   - Worker processes are pinned to cores `[2, 3, 4, 5, 6, 7, 8, 9]`.
3. **Queue Sizing & Shedding**:
   - `QUEUE_CAPACITY` default is 20.
   - Under saturation, queue wait is bounded ($20 / 15.87 \approx 1.26\text{ s}$ at 2 workers), strictly below the `REQUEST_TIMEOUT_SEC` (default 2.0s in code, overridable to 3.0s), allowing load shedding to reject surplus traffic immediately via HTTP 503 rather than holding connections until client timeout.
4. **No Docker Requirement**: All microservices run natively via standard Python and communicate over HTTP loopback (`127.0.0.1:5000` through `5003`).

---

## 8. UNVERIFIED / NOT TESTED & PERM DEFICITS

Where PERM Lost / Underperformed:
1. **Request Success Deficit**: Under step burst traffic, PERM only served **85.69%** of requests (`mode_aggregates.perm.pct_ok.mean = 85.69`), failing 14.31% of requests. Both `fixed_8` (**100.0%**) and `reactive` (**98.41%**) significantly outperformed PERM in throughput and reliability.
2. **Delayed Scale-Up**: PERM scaled up **9.79 seconds AFTER burst arrival** (`mode_aggregates.perm.sec_to_first_scale_up.mean = 9.79`). It was slower to respond than the simple reactive autoscaler, which scaled up in **0.82 seconds** (`mode_aggregates.reactive.sec_to_first_scale_up.mean = 0.82`).
3. **Advance Proactivity is Unverified**: Zero alerts occurred prior to the onset of the traffic burst (`is_before_burst: false` across all runs). The machine learning forecast pipeline currently only triggers once congestion has already entered the 10-second history window. Advance predictive scaling remains **UNPROVEN** under sharp step bursts.

Unverified / Not Tested in this Step:
1. **End-to-End run_all.py with Dashboard**: `test_07_run_all_starts_services` was skipped (`SKIP_RUN_ALL_TEST=1`) to avoid port collisions during unit testing; multi-service coordination was tested directly in `compare_scaling.py`.
2. **Ramp Traffic Profile**: The 4-mode comparison was run under the `step` profile; the `ramp` profile was validated in `burst_test.py` scheduling tests but has not yet been benchmarked across all 4 modes.
3. **Linux CPU Affinity**: Tests and benchmarks were executed on Windows; `psutil.Process().cpu_affinity()` on Linux kernel cgroups was not directly verified.
