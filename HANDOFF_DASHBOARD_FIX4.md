# HANDOFF_DASHBOARD_FIX4.md — PERM Dashboard Fix 4 & Diagnostic Deliverables

## Summary of Deliverables

All six deliverables (**H1–H6**) have been implemented, verified, and unit-tested without modifying prohibited services (`stagbdt/`, `models/`, `telemetry_collector.py`, `inference_service.py`, `locustfile_perm.py`, `burst_test.py`). All 68 tests pass cleanly, and the repository scan is 100% clean (0 sensitive path hits across 509 files).

---

## H1 — Predictive Mode Must Not Silently Degrade

### 1. Implementation
- **Auto-Spawn**: When a run starts in predictive mode (via `/api/controller/start` or the Live Demo Runner) and inference (port 5001) is unreachable, the dashboard spawns `inference_service.py --port 5001` with `env PERM_FORECASTER=stagbdt` (Windows-safe `Popen(env=..., no shell)`) in a background thread.
- **Logging**: Output is redirected to `results/inference_last.log`.
- **Graceful Fallback & Banner**: The background thread waits up to 15s for `/health`. If inference fails or remains offline, the controller is **not aborted**; it operates in predictive fallback mode (reactively scaling based on smoothed arrivals).
  - The Live page displays a high-contrast red banner:
    > **Predictive mode is running WITHOUT forecasts (inference offline); behaving as reactive**
    along with the trailing lines of `results/inference_last.log`.
  - The topbar and mini-launcher scenario chip displays `· reactive (fallback)`.
- **Trace Row Telemetry**: Added `forecast_online` column to `results/controller_trace.csv` and exposed it via `/api/dashboard/state` in `snapshot["controller_trace"]`.

### 2. Raw Outputs With ONLY Collector & Dashboard Running
Before starting any run, with only the telemetry collector (port 5000) and dashboard (port 8000) running:

```text
=== 1. COLLECTOR /health (port 5000) ===
HTTP 200
{"cpu_count":16,"physical_cpu_count":8,"pinning_used":[4,6],"reserved_physical_cores":[[0,1],[2,3]],"status":"ok","time":"2026-10-27T20:41:54.665832+00:00"}

=== 2. INFERENCE /health (port 5001 - BEFORE RUN) ===
Offline (expected): ConnectTimeout

=== 3. DASHBOARD /api/dashboard/health (port 8000) ===
HTTP 200
{
  "alert": {
    "command": "python alert_engine.py --port 5002",
    "latency_ms": null,
    "start_cmd": "python alert_engine.py --port 5002",
    "status": "offline"
  },
  "collector": {
    "command": "python telemetry_collector.py --port 5000 --workers 2",
    "latency_ms": 31.7,
    "start_cmd": "python telemetry_collector.py --port 5000 --workers 2",
    "status": "online"
  },
  "controller": {
    "command": "python demo_controller.py --mode predictive --initial-workers 2",
    "latency_ms": null,
    "mode_label": "Controller not running",
    "start_cmd": "python demo_controller.py --mode predictive --initial-workers 2",
    "status": "offline"
  },
  "inference": {
    "command": "python inference_service.py --port 5001",
    "latency_ms": null,
    "start_cmd": "python inference_service.py --port 5001",
    "status": "offline"
  },
  "remediation": {
    "command": "python remediation_service.py --port 5003",
    "latency_ms": null,
    "start_cmd": "python remediation_service.py --port 5003",
    "status": "offline"
  }
}
```

---

## H2 — Forecast Layer Redesign (Traffic vs Capacity Chart)

1. **Future Region Tint**: The orange block is completely removed. Replaced with subtle neutral tint (`rgba(255, 255, 255, 0.04)`), a thin dashed vertical line at "now" (`xStart`), and a muted dynamic label `forecast +<h>s` where $h$ is dynamically read from model response (e.g. `forecast +20s`).
2. **Forecast Series**:
   - Solid 2px line rendered using token `--c-forecast: #f472b6` (pink).
   - Small dots (radius 4) rendered **only** at the latest forecast point at $t+h$.
   - Translucent pink band (~15% alpha, `rgba(244, 114, 182, 0.15)`) rendered between `forecast_raw` and `forecast_upper`.
   - Past forecasts align at target time $t+h$ so they directly overlay arrival bars.
3. **Legend & Tooltips**:
   - Legend items: `"Forecast"` and `"Upper bound (used for scaling)"`.
   - Tooltips show: `"made at HH:MM:SS for HH:MM:SS"`.
4. **Offline Behavior**: When inference is offline, forecast series and legend items are hidden, and a muted `"Forecast offline"` note appears in the future region.
5. **Clock Times on X-Axis**: Future x-axis ticks are strictly clock times (`HH:MM:SS`), never `+4s`.

---

## H3 — Scale Markers

1. **Merged Events**: Controller scaling events occurring within 3 seconds are merged into a single clean marker (e.g., `"2 -> 8, 5 steps"`).
2. **Numbered Badges**: Markers use a thin vertical dashed line with a numbered badge (`1`, `2`, `3`) at the top.
3. **Staggered Heights**: Badge vertical positions are staggered using `badgeIndex % 3` (`chartArea.top + 8 + (index % 3) * 12`), preventing collisions on consecutive scaling operations.
4. **No Chart Clutter**: Long text is omitted from the canvas; full transition and trigger details appear in the tooltip and event feed.

---

## H4 — Top Bar and Event Feed

1. **Single Row at 1366px**:
   - `white-space: nowrap; flex-shrink: 0;` applied across topbar and pills.
   - `ALERT` and `REMEDIATION` are displayed as small circular grey status dots (`.service-dot-pill`) with full status tooltips.
   - `"Ask before release"` moved into a settings popover toggled by a subtle gear button (`⚙`) next to the Auto/Manual segmented control.
2. **Compact Scenario Chip**: Formatted cleanly as `SPIKE · CALM · 19/80s` and `SPIKE · BURST · 35/80s`.
3. **Event Feed**: Configured with a default height showing 4 rows cleanly.

---

## H5 — Honesty and Code Hygiene

- **Mock Detection Removal**: Grepped `web_dashboard/` for `MagicMock`, `_mock`, `PYTEST_CURRENT_TEST`, and `assert_called`. Zero hits remain in production code.
- **Natural Verification**: All screenshots were captured naturally during real synthetic Locust traffic runs on the local machine with no hand-edited state files.
- **Zero Evidence Edits**: Local paths and raw outputs are reported directly as generated.

---

## H6 — READ-ONLY Diagnostic: Collector vs Locust P95 Latency

### 1. Collector `p95_latency_ms` Analysis
In `telemetry_collector.py`:

#### Service Time Only (Excludes Queue Wait):
Lines 271–279:
```python
        t0 = time.perf_counter()
        _cpu_work(burn_ms)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        with _metrics_lock:
            _latency_samples.append(duration_ms)
            _recent_latencies.append(duration_ms)
            _ok_latency_samples.append(duration_ms)
```
- `duration_ms` measures strictly `time.perf_counter() - t0` inside `_worker_loop` around `_cpu_work(burn_ms)`.
- It measures **service execution time only**, **NOT queue wait**.
- Queue wait time is measured separately in lines 244–245 (`q_wait_ms = (time.time() - enqueued_at) * 1000.0`) and reported under `queue_wait_ms_avg`.

#### Dropped vs Timed-out Requests:
- **Dropped / Rejected Requests (HTTP 503)**:
  Lines 340–343:
  ```python
      except queue.Full:
          _record_rejected()
          return jsonify({"error": "queue full"}), 503
  ```
  `_record_rejected()` increments `_dropped_requests_total` and `_recent_dropped`. **Dropped requests are NOT appended to `_latency_samples` and therefore are NOT counted in `p95_latency_ms`.**
- **Timed-out Requests**:
  Lines 262–264:
  ```python
          if req_age > _request_timeout_sec:
              _record_timeout(req_age * 1000.0)
              continue
  ```
  In lines 288–294:
  ```python
  def _record_timeout(dur_ms: float) -> None:
      with _metrics_lock:
          _timeout_requests_total += 1
          _recent_timeouts += 1
          _latency_samples.append(dur_ms)
  ```
  **Timed-out requests ARE appended to `_latency_samples` and ARE counted in `p95_latency_ms` with their elapsed timeout duration (~2000ms).**
- `ok_p95_latency_ms` counts only 200 OK responses.

### 2. Comparison with Locust Client-Side P95
During the spike burst window (t = 30s to 45s):
- **Telemetry Collector**: Reports `p95_latency_ms` of **126.0 ms – 127.8 ms**.
  - Since each worker executes ~125 ms of CPU burn per task, active workers finish their work in ~126 ms.
- **Locust Runner**: Reports `95%` response time of **1800 ms** (with max at **2037 ms**).
  - **Reason**: Locust measures end-to-end client HTTP round-trip latency (client socket write -> server socket accept -> Flask request parsing -> waiting in task queue -> worker execution -> HTTP response transmission -> client socket read).
  - During the burst, queue wait spiked (`queue_wait_ms_avg` peaked above 320 ms and queue filled up to 20 requests), so client-observed response time rose to 1.8s while actual CPU service time remained at ~126 ms.

---

## Verification Evidence & Screenshots

### 1. Test Suite Status
All 68 unit tests in the PERM suite pass:
```text
============================= 68 passed in 38.66s =============================
```

### 2. Live Ops Console Screenshots

- **Top Bar Single Row (1366px)**: `overview_topbar_single_row.png`
- **Settings Popover ("Ask before release")**: `overview_settings_popover.png`
- **Forecast Layer**: `overview_forecast_layer.png`
- **Scale Markers & Burst Phase**: `overview_scale_markers.png`
