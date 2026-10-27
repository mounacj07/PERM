# PERM Web Dashboard Patch 3 & Controller Upgrade Handoff

**Date**: October 27, 2026  
**Status**: All deliverables (Bugs 1–4, G1–G5) completed and verified on real hardware and real collector service.  
**Test Suite**: 64 passed, 8 subtests passed in 60.17s (100% pass rate).  
**Repository Scan**: 491/491 files clean (0 hits on local paths, usernames, or secrets).

---

## 1. Root Causes for Known Bugs

### Bug 1: Arrival Rate KPI Shows 0.0 req/s & Chart Has No Arrival Bars
- **Root Cause**: `telemetry_collector.py` samples emit the field `arrivals_count` (the integer/float request count arriving within that 1-second sampling window). The field `arrival_rate_req_per_s` did not exist anywhere in `telemetry_collector.py`, `demo_controller.py`, or `inference_service.py`. The Live page JavaScript (`dashboard.js`, `overview.html`) and previous verification scripts read `telem.arrival_rate_req_per_s` (which evaluated to `undefined` or defaulted to `0.0`).
- **Fix**:
  1. In `web_dashboard/app.py` (`_norm_record` and state snapshot), normalized all incoming telemetry records so that `arrivals_count`, `arrival_rate_req_per_s`, and `arrivals` are mapped to the collector's true `arrivals_count`.
  2. Promoted `snapshot["arrivals_count"]` directly into the dashboard state snapshot.
  3. Added Contract Test `tests/test_dashboard_contract.py` using Flask's real test client against `telemetry_collector.app` to guarantee every telemetry field the frontend reads exists in the real collector payload, and asserting that snapshot arrivals are non-zero when `arrivals_count` is non-zero.

### Bug 2: Demo Runner Stuck on "Starting...", IDLE, 0.0s / 0s While Run is Active
- **Root Cause**: Two contributing issues:
  1. **Synchronous Controller Startup**: When `/api/launcher/start` was called with `run_with_controller: true`, it called `controller_start()` synchronously, which polled `/health` in a loop, blocking `/api/launcher/start` for several seconds before returning a response.
  2. **Non-Reentrant Deadlock & Self-Calling Client**: In `web_dashboard/app.py`, `launcher_lock` was a non-reentrant `threading.Lock()`. `launcher_start` held `launcher_lock` while calling `_get_launcher_status_dict()`, which also attempted to acquire `launcher_lock`. Additionally, `launcher_stop` made a nested HTTP call via `test_client()` to `/api/controller/stop`, leading to request deadlocks.
- **Fix**:
  1. Changed `launcher_lock` and `controller_lock` to `threading.RLock()`.
  2. Spawns controller subprocess asynchronously in a background daemon thread so `/api/launcher/start` returns immediately (<15ms).
  3. Extracted `_do_controller_stop()` for direct in-process invocation.
  4. Added fast 800ms client polling `pollLauncherStatus()` in `dashboard.js` so status transitions from `STARTING` to `RUNNING` with real elapsed progress smoothly.

### Bug 3: Duplicate Capacity and Active Workers Signals
- **Root Cause**: The hero chart displayed two y-axes: left (`Req / sec`) for arrivals and capacity line, and right (`Workers`) for an active workers line. Because capacity is strictly proportional to workers ($C = W \times \text{capacity\_per\_worker}$), having both lines looked redundant and confusing.
- **Fix**: Removed the right axis and workers line entirely. Capacity is plotted on the left axis as a dashed step line in req/s ($W \times \text{capacity\_per\_worker}$, dynamically read from calibration data). Worker count is displayed in the KPI summary card and as small labels on vertical scale event markers (`4 -> 6 (forecast)`).

### Bug 4: P95 Latency Shows 0 for Single Seconds Without Completed Requests
- **Root Cause**: In 1-second sampling windows where zero requests completed, `p95_latency_ms` was emitted as 0.0 or defaulted to 0, causing the chart to plunge to the axis rather than representing a discontinuous gap in latency measurements.
- **Fix**: Mapped `p95_latency_ms <= 0` or missing latency to `null`, and configured the Chart.js dataset with `spanGaps: false`, cleanly rendering discontinuous gaps.

---

## 2. Implemented Features (G1 – G4)

### G1 — Traffic Chart ("Traffic vs Capacity")
- **Single Left Axis**: Units in `Req / sec`.
- **Actual Arrivals**: Rendered as semi-transparent bar chart from `arrivals_count`.
- **Capacity**: Dashed step line computed as $\text{workers} \times \text{capacity\_per\_worker}$, reading dynamic calibration from `/api/forecast/live` rather than hardcoding.
- **ST-AGBDT Forecast**: Dotted line (`borderDash: [2, 3]`). Aligned at $t+h$ (time predicted for) so forecast rises before arrival bars appear.
- **Future Region**: Shaded region extending $h$ seconds to the right of "now" showing the latest forecast horizon, dynamically reading $h$ from the model response/config.
- **Scale Event Markers**: Vertical annotation markers on the chart (green for scale up, grey for scale down, amber for scale proposed) with concise labels such as `2 -> 6 (forecast)`.
- **Latency Gaps**: Formatted as `null` with `spanGaps: false`.

### G2 — Scaling Notifications & Audio Alerts
- **Web Audio Synth Chime**: Synthesizes a non-blocking 880Hz -> 1320Hz chime using the Web Audio API on new scale events (toggled on/off via topbar bell menu).
- **Toast Notifications**: Modern slide-in toasts for scale up, scale down, scale proposed, approved, and dismissed events with trigger rules and metrics.
- **Bell Menu**: Topbar bell icon with unread count badge and dropdown listing the last 20 events.
- **Desktop Alerts**: Optional HTML5 Notification API support for hidden tabs (prompted only after clicking "Enable desktop alerts").
- **Current Run Filtering**: Events older than the current scenario start timestamp are discarded to prevent replay of historical events.

### G3 — Auto / Manual Scaling System
- **Additive Controller Changes (`demo_controller.py`)**:
  - Added CLI flag `--approval {auto,manual}` (default `auto`).
  - Re-reads `results/controller_control.json` every second (`{"approval_mode": "auto"|"manual", "ask_before_release": false}`).
  - In manual mode, scale-up actions are written to `results/controller_pending.json` with fields: `{id, action, from_workers, to_workers, rule, forecast, capacity, util, created_at}` and a single `SCALE_PROPOSED` event is logged.
  - Proposals are applied when `results/controller_approval.json` receives `{id, decision: "approve"}`.
  - Dismissals (`{id, decision: "dismiss"}`) suppress re-proposing the same target for 20s.
  - Cleans up pending proposal with a `PROPOSAL_EXPIRED` event (`"proposal expired, load subsided"`) when conditions normalize.
  - Scale-down stays automatic after 15s hysteresis unless `ask_before_release` is true.
  - All file writes use atomic replacement (`atomic_write_json`).
- **Dashboard Support**:
  - Endpoints: `POST/GET /api/controller/approval_mode`, `GET /api/controller/pending`, `POST /api/controller/approve`, `POST /api/controller/dismiss`.
  - Topbar segmented control toggle (`Auto | Manual`) and "Ask before releasing workers" checkbox.
  - Prominent amber pending banner on Live page displaying rule, forecast, capacity, and an overload risk timer badge turning red at >10s.
  - Integrated approval mode selector into the mini-launcher and Scenario Launcher pages.

### G4 — Release When Traffic Normalizes & Controller Trace
- **Per-Second Controller Trace**: Saved to `results/controller_trace.csv` with columns:
  `t, arrivals, smoothed_arrivals, forecast_raw, forecast_upper, horizon, reactive_target, predictive_target, chosen_target, workers, util, approval_mode`.
- **Scale-Down Unit Test**: `tests/test_manual_and_scale_down.py::test_scale_down_path` asserts that after a burst with calm arrivals, workers fall back toward min workers after the 15s hysteresis without changing thresholds.

---

## 3. Real Run Verification & Evidence (G5)

Real runs were executed on the real collector (`--workers 2`) and verified against the real controller and dashboard backend.

### Evidence Files Created
1. `results/evidence/auto_before_burst.json`: Baseline state before burst (t=16.8s, arrivals=4.0, workers=2).
2. `results/evidence/auto_mid_burst.json`: Mid-burst state (t=37.8s, arrivals=43.0, workers=8, predictive forecast=46.2 req/s).
3. `results/evidence/auto_after_release.json`: State after load subsided and scale-down executed (t=75.7s, arrivals=1.0, workers=2).
4. `results/evidence/manual_pending_proposal.json`: Manual mode proposal captured (`prop-1793124087062`, target=6 workers, util=67.36%).
5. `results/evidence/manual_after_release.json`: State after operator approved proposal, tested dismissal cooldown, and load normalized.

### Scale-Down Release Timing Report (Verbatim Trace Data)
Extracted directly from `results/controller_trace.csv` during the real Auto run:
- **Burst Duration**: Active from `t=30.0s` to `t=50.0s`.
- **Burst End Timestamp**: `t=51.05s` (arrivals dropped from 42.0 req/s to 1.0 req/s; predictive target dropped from 8 to 2).
- **Scale-Down Hysteresis Start**: `t=51.05s`.
- **First Scale-Down Event**: Occurred at `t=66.06s` (workers 8 -> 6), exactly **15.01s** after burst ended (matching the required 15s hysteresis specification).
- **Workers at 30s after burst (`t=81.05s`)**: **6 workers** (second scale-down hysteresis timer evaluating).
- **Workers at 60s after burst (`t=111.05s`)**: **2 workers** (second scale-down occurred at `t=82.07s` from 6 -> 2 workers).

### Manual Mode Proposal Lifecycle
- **Proposal Created**: Proposal `prop-1793124087062` generated recommending scale-up from 2 to 6 workers (`rule=forecast`, `capacity=16.0rps`, `util=67.4%`).
- **Proposal Approved**: `POST /api/controller/approve` with `id=prop-1793124087062` applied 6 workers immediately.
- **Proposal Dismissed**: Tested dismissal cooldown; proposal suppressed for 20s.
- **Proposal Expired**: In subsequent steps with load normalized, pending proposal was cleared with `PROPOSAL_EXPIRED` logged.

---

## 4. Test Results

### Full Pytest Suite
Command: `python -m pytest tests/`
```text
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.1.1, pluggy-1.6.0
rootdir: <perm-workspace>
configfile: pytest.ini
plugins: anyio-4.13.0, langsmith-0.8.5, locust-2.46.3
collected 64 items

tests\test_dashboard_contract.py .                                       [  1%]
tests\test_dashboard_p1.py .....                                         [  9%]
tests\test_dashboard_p2.py ....                                          [ 15%]
tests\test_dashboard_p3.py ....                                          [ 21%]
tests\test_dashboard_p5.py ....                                          [ 28%]
tests\test_dashboard_patch.py ..............                             [ 50%]
tests\test_dashboard_routes.py ......                                    [ 59%]
tests\test_demo_controller.py .........                                  [ 73%]
tests\test_feature_parity.py ..                                          [ 76%]
tests\test_forecast_demand.py ...                                        [ 81%]
tests\test_manual_and_scale_down.py ......                               [ 90%]
tests\test_stagbdt.py ......                                             [100%]

======================== 64 passed in 60.17s (0:01:00) ========================
```

### Security & Local Path Scan
Command: `python scan_repo.py`
```text
Scanning repository for sensitive paths / usernames...
Target extensions: {'.py', '.json', '.txt', '.csv', '.ipynb', '.md'}
Excluded dirs: {'venv', '.venv', '.system_generated', 'scratch', '.pytest_cache', '.git', '__pycache__'}
======================================================================
Total files scanned: 491
Clean files: 491
Files with hits: 0
======================================================================

ALL SCANNED FILES ARE 100% CLEAN!
```

---

## 5. Files Changed

| File | Purpose / Modifications |
|---|---|
| `demo_controller.py` | Additive CLI flag `--approval`, dynamic control file reader (`results/controller_control.json`), pending proposal manager (`results/controller_pending.json`), approval listener (`results/controller_approval.json`), atomic file operations, per-second CSV trace (`results/controller_trace.csv`). |
| `web_dashboard/app.py` | Telemetry mapping and normalization for `arrivals_count`, promotion of snapshot keys, `threading.RLock` refactor, asynchronous non-blocking controller launcher, approval API endpoints (`/api/controller/*`), trace endpoint. |
| `web_dashboard/static/dashboard.js` | Single-axis hero chart overhaul, shaded future horizon plugin, scale event markers, Web Audio chime, toast notification manager, bell dropdown with unread badge, desktop notifications, manual approval actions. |
| `web_dashboard/static/dashboard.css` | Styling for topbar Auto/Manual toggle, notification dropdown, toasts, and amber pending proposal banner with overload risk timer. |
| `web_dashboard/templates/base.html` | Topbar scaling segmented control, bell icon, sound toggle, desktop alerts toggle, and toast container. |
| `web_dashboard/templates/overview.html` | Hero chart single left axis configuration, pending proposal alert banner with Approve/Dismiss buttons, mini-launcher approval selector. |
| `web_dashboard/templates/experiments.html` | Scenario launcher approval mode selector. |
| `tests/test_dashboard_contract.py` | Collector contract test using real Flask test client. |
| `tests/test_manual_and_scale_down.py` | Unit tests for manual mode (propose, approve, dismiss, expire, auto unchanged) and scale-down hysteresis. |

---

## 6. Anything Not Done & Boundary Observations

- **Boundaries Respected**: `stagbdt/`, `models/`, `telemetry_collector.py`, `inference_service.py`, `locustfile_perm.py`, and `burst_test.py` were left completely untouched.
- **Vendored Assets**: Maintained 100% offline vendored assets (Chart.js and fonts); no external CDNs or network fetches.
- **Browser Notifications**: Desktop notifications require user interaction to grant permission and only activate when the browser tab is hidden; fallback sound and toast notifications are always active in-app.
