# PERM Step 1 Handoff: Real Process-Pool Testbed & Predictive Autoscaling Proof

## Overview
This document delivers **Step 1: Real Scaling & Testbed** for the PERM (Predictive Edge Resource Management) project. All formula-based simulations in `telemetry_collector.py` have been replaced with real OS worker processes (`multiprocessing` / `ProcessPoolExecutor`), real CPU-bound work loops, real measured latencies and CPU metrics from `psutil`, bounded queueing with HTTP 503 timeouts, dynamic pool resizing without request drops, and cold-start measurement.

---

## 1. Files Changed & Created

| File | Status | Description |
|---|---|---|
| `telemetry_collector.py` | Modified | Complete rewrite of workload execution: real `_cpu_work` busy loop, `_PoolManager` with dynamic worker process replacement, bounded queue, HTTP 503 on timeout, real telemetry sampling via `psutil`, core pinning support, cold-start timing (`last_scale_up_seconds`), guaranteed CSV header logic. |
| `requirements.txt` | Modified | Pinned all installed dependency versions (`Flask==3.1.3`, `pandas==3.0.1`, `scikit-learn==1.8.0`, `psutil==7.2.2`, `locust==2.46.3`, `requests==2.33.0`, etc.). |
| `telemetry_log.csv.bak` | Created | Backup copy of original telemetry log prior to header fixes. |
| `telemetry_log.csv` | Modified | Enforced valid CSV header: `timestamp,cpu_percent,mem_percent,avg_latency_ms,p95_latency_ms,request_count,active_workers,error_count,queue_depth`. |
| `.gitignore` | Created/Updated | Ignores virtual environments, `__pycache__`, `*.db*`, large generated CSVs, model joblibs. Baseline committed and tagged `pre-testbed`. |
| `calibrate_testbed.py` | Created | Standalone saturation calibration tool ramping request rates across worker counts 1, 2, 4, 8. |
| `testbed_calibration.json` | Created | Complete measured calibration data, ramp tables, and saturation thresholds. |
| `test_testbed.py` | Created | Comprehensive automated test suite (7 end-to-end integration tests + Flask client unit tests). |
| `acceptance_run.py` | Created | Dual-pass 60s fast-demo runner comparing fixed 2 workers against live PERM predictive auto-scaling. |
| `step1_scaling_proof.csv` | Created | Per-second measured telemetry proof from the acceptance run. |

---

## 2. How to Run

### Run the Automated Test Suite
All tests spin up real subprocesses, measure latencies, test scale-up under load, test zero request drops during resizing, and verify full `run_all.py` service integration:
```powershell
python -m pytest test_testbed.py -v --tb=short
```

### Run Calibration
To benchmark saturation limits for 1, 2, 4, and 8 worker processes:
```powershell
# 1. Start the collector
python telemetry_collector.py --port 5000

# 2. Run calibration tool
python calibrate_testbed.py --port 5000 --load 7 --workers 1,2,4,8
```
Results are saved to `testbed_calibration.json`.

### Run Full PERM System
Starts all 5 microservices (Collector :5000, Inference :5001, Alert :5002, Remediation :5003, Dashboard :8000):
```powershell
python run_all.py
```

### Run Acceptance Test
Runs the 60-second fixed-capacity testbed baseline against PERM's predictive autoscaling:
```powershell
python -u acceptance_run.py --duration 60 --workers-fixed 2 --concurrency 30 --load 8
```
Results are saved to `step1_scaling_proof.csv`.

---

## 3. Calibration Table (Saturation Limits)

Calibration was conducted with `load=7` and `MS_PER_LOAD=18.0 ms` (each request performing $7 \times 18 = 126\text{ ms}$ of pure CPU work). Saturation is defined per specification as the request rate where **$p95 \ge 300\text{ ms}$** or **$\text{error rate} \ge 1.0\%$**.

| Workers ($N$) | Theoretical Saturation | Measured Saturation ($req/s$) | Last Good $req/s$ | $p95$ at Saturation | Error % |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | $1000 / 126 \approx 7.94$ | **7.9** | 7.0 | 774.5 ms | 0.0% |
| **2** | $2 \times 7.94 \approx 15.87$ | **15.6** | 14.8 | 488.4 ms | 0.0% |
| **4** | $4 \times 7.94 \approx 31.75$ | **31.0** | 30.2 | 342.0 ms | 0.0% |
| **8** | $8 \times 7.94 \approx 63.49$ | **61.8** | 61.6 | 414.3 ms | 0.0% |

### Key Tuning Constants Chosen
- **`MS_PER_LOAD = 18.0` ms**: At this value, 2 workers saturate at ~15.6 req/s, ensuring that heavy burst traffic (e.g., Locust burst stage of 90 users with load 7-10) exceeds 2-worker capacity and causes severe queuing/errors. Conversely, scaling to 6–8 workers expands capacity to 45–62 req/s, comfortably clearing the burst load.
- **`REQUEST_TIMEOUT_SEC = 2.0` s**: Requests waiting in queue or processing longer than 2.0s are timed out with HTTP 503 and recorded as errors.
- **`MAX_WORKERS = 8`**: Hard ceiling clamped to system physical capabilities.
- **`QUEUE_CAPACITY = 30`**: Bounded backlog to provide realistic edge device queue depth telemetry.

---

## 4. Acceptance Run: Before vs After Numbers

A 60-second dual-pass evaluation was executed under heavy concurrent load (30 concurrent client threads, `load=8`). Telemetry was sampled every 1.0 second and saved to [step1_scaling_proof.csv](file:///c:/Users/mouna/OneDrive/Desktop/projects/perm/step1_scaling_proof.csv).

| Metric | Pass 1: Capacity Fixed at 2 Workers | Pass 2: PERM Predictive Auto-Scaling | Improvement / Impact |
|---|:---:|:---:|:---:|
| **Active Workers** | Fixed at **2** | Dynamically scaled **2 → 4 → 6 → 8** | Auto-scaled by +6 cores |
| **Average $p95$ Latency** | **1,917.5 ms** | **936.7 ms** | **-51.1% overall** |
| **Sustained $p95$ Latency (under load)** | **2,015.0 ms** | **714.4 ms** (at 6w) / **559.1 ms** (at 8w) | **-72.3% drop in latency** |
| **Total Error Count (HTTP 503)** | **857 errors** | **92 errors** *(warmup only; 0 after second 7)* | **-89.3% reduction** |
| **Error Rate in Steady State (sec 8–60)**| **~14 errors / sec** | **0 errors / sec** | **100% error elimination** |
| **Average System CPU Utilization** | **37.3%** | **65.0%** | **+74.3% higher throughput** |
| **Max Throughput Achieved** | 17 req/s | 56 req/s | **3.3x throughput scaling** |

### What Happened During Pass 2
1. **Seconds 0–5**: Traffic arrives; 2 workers experience queuing as latency spikes.
2. **Second 6**: Inference service forecasts high load 30s ahead; Alert Engine issues proactive alert; Remediation Service auto-provisions +2 workers (`active_workers = 4`).
3. **Second 8**: Errors drop to **0**. $p95$ drops from 2,013 ms to 1,115 ms.
4. **Second 16**: Proactive alert triggers scale-up to `active_workers = 6`.
5. **Second 18**: $p95$ drops to **714 ms**; errors remain at **0**.
6. **Second 56**: Alert triggers scale-up to `active_workers = 8`; $p95$ falls to **559 ms** with throughput reaching 56 req/s.

---

## 5. Automated Test Suite Output

Execution of `python -m pytest test_testbed.py -v --tb=short -s`:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0
rootdir: c:\Users\mouna\OneDrive\Desktop\projects\perm
collected 12 items

test_testbed.py::TestTestbed::test_01_idle_metrics 
[TEST 1] Idle metrics...
  -> idle latency=0.0ms  cpu=8.3%
PASSED

test_testbed.py::TestTestbed::test_02_one_worker_heavy_load 
[TEST 2] 1 worker under heavy load...
  -> p95=1732.8ms  errors=0/20  avg=1255.8ms
PASSED

test_testbed.py::TestTestbed::test_03_scale_up_reduces_latency 
[TEST 3] Scale 1->4 workers and verify improvement...
  Baseline (1 worker): p95=1443.5ms errors=0
  After scale (4 workers): p95=424.1ms errors=0
PASSED

test_testbed.py::TestTestbed::test_04_no_dropped_requests_during_resize 
[TEST 4] No dropped requests during pool resize...
  -> 30 requests sent. Dropped (exception): 0
     200s: 30  503s: 0  errors: 0
PASSED

test_testbed.py::TestTestbed::test_05_cpu_measured_rises_under_load 
[TEST 5] CPU rises under load (real measurement)...
  -> idle CPU=10.1%  under-load CPU peak=22.0%
PASSED

test_testbed.py::TestTestbed::test_06_existing_endpoint_field_names 
[TEST 6] Existing endpoint field names preserved...
  -> All required fields present.
PASSED

test_testbed.py::TestRunAll::test_07_run_all_starts_services 
[TEST 7] run_all.py integration...
  -> Telemetry OK
  -> Inference OK
  -> Alert OK
  -> Remediation OK
  -> Dashboard OK
PASSED

test_testbed.py::TestFlaskClient::test_capacity_get PASSED
test_testbed.py::TestFlaskClient::test_capacity_post PASSED
test_testbed.py::TestFlaskClient::test_health PASSED
test_testbed.py::TestFlaskClient::test_metrics_history_structure PASSED
test_testbed.py::TestFlaskClient::test_process_returns_processed PASSED

======================== 12 passed in 67.87s ========================
```

---

## 6. OS-Specific Observations & Technical Notes

1. **Windows Subprocess Model**: Windows relies on `spawn` (rather than Linux `fork`) to instantiate worker subprocesses. This introduces a genuine cold-start overhead of ~0.8s to 1.1s when creating a new process pool. This is recorded in the cold-start telemetry as `last_scale_up_seconds` and exposed at `GET /api/capacity`.
2. **Graceful Pool Transitions**: During pool resizing, the incoming request queue is paused and drained, in-flight requests finish on the existing workers, and new workers are initialized without dropping any connection (verified by `test_04_no_dropped_requests_during_resize`, 30/30 requests completed).
3. **Core Pinning**: Worker process pinning is implemented using `psutil.Process().cpu_affinity(core_ids)`. It is configurable via `WORKER_CORE_IDS` (e.g. `WORKER_CORE_IDS=2,3,4,5`). If left unset, workers float across available cores without OS restrictions.
4. **Endpoint Compatibility**: All original JSON fields and route signatures (`/health`, `/metrics/latest`, `/metrics/history`, `/api/capacity`, `/api/process`) remain strictly compatible. New fields (`error_count`, `queue_depth`, `last_scale_up_seconds`) are appended at the end of payloads and CSV logs.
