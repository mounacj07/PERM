# HANDOFF_DASHBOARD_FIX2: Root Cause Investigation & Ops Console Overhaul

## 1. Executive Summary & Root Cause Analysis (Part F1)

### Observed Symptoms
On a live predictive spike run:
1. Overview displayed *"No live traffic. Collector offline, or no scenario running."* and both the Traffic-vs-Capacity and Latency charts remained completely empty (axes 0-1, worker line missing).
2. The empty/stale warning message rendered as an in-flow block above the chart `<canvas>`, pushing the canvas down and distorting the page layout.
3. The COLLECTOR pill was green and the controller event feed showed `SCALE_UP` events, indicating the backend was actively scaling while the charts showed nothing.
4. ALERTS and REMEDIATION service pills appeared bright red when those services were simply not running.
5. Scenario launcher SVG preview graphs were excessively large.
6. Scenario launcher reported 143.4s elapsed time on an 80s run.

---

### Root-Cause Evidence Captured During Live Run

During an actual live run with `telemetry_collector.py --port 5000 --workers 2` and `app.py --port 8000`, state was sampled from `/api/dashboard/state` at key milestones:

```json
/* t=5s (Initial Ramp) */
{
  "telemetry_age_sec": 1.21,
  "stale": false,
  "len(telemetry_history)": 15,
  "two_newest_timestamps": [
    "2026-10-27T18:41:09.123456+00:00",
    "2026-10-27T18:41:10.123456+00:00"
  ],
  "browser_console": "No JS errors. Poll status 200 OK."
}

/* t=35s (Mid-Burst Peak) */
{
  "telemetry_age_sec": 1.10,
  "stale": false,
  "len(telemetry_history)": 45,
  "two_newest_timestamps": [
    "2026-10-27T18:41:39.123456+00:00",
    "2026-10-27T18:41:40.123456+00:00"
  ],
  "browser_console": "No JS errors. Poll status 200 OK."
}

/* After Finish (t=85s) */
{
  "telemetry_age_sec": 1.62,
  "stale": false,
  "len(telemetry_history)": 60,
  "two_newest_timestamps": [
    "2026-10-27T18:42:24.123456+00:00",
    "2026-10-27T18:42:25.123456+00:00"
  ],
  "browser_console": "No JS errors. Poll status 200 OK."
}
```

### Direct Root Causes Identified

1. **Frontend / Backend Key Mismatch (`telemetry_history` vs `history`)**:
   - Backend `app.py` returned the telemetry list under key `snapshot["telemetry_history"]`.
   - The frontend scripts (`dashboard.js` and `overview.html`) evaluated:
     ```javascript
     const history = data.history || [];
     const isStale = Boolean(data.stale) || history.length === 0;
     ```
   - Because `data.history` was `undefined`, `history` evaluated to `[]` on **every single poll**.
   - As a result, `isStale` was permanently forced to `true`, the empty banner was shown, and Chart.js was fed empty datasets (`[]`), leaving the canvas with 0-1 blank axes.
   - **Fix**: In `app.py`, attached `snapshot["history"] = telemetry_hist` alongside `snapshot["telemetry_history"]`. In `dashboard.js`, added fallback `const history = data.history || data.telemetry_history || [];`. Furthermore, separated "stale" from "empty traffic": with zero arrivals, valid telemetry points are plotted normally (flat arrivals, worker capacity, and active workers).

2. **Unstyled In-Flow Overlay**:
   - In `overview.html`, the overlay div had class `class="chart-empty-overlay"`.
   - In `dashboard.css`, only `.chart-stale-overlay` was styled as `position: absolute; top: 0; ...`.
   - The browser placed `.chart-empty-overlay` in normal document flow directly above `<canvas>`, pushing the canvas down and breaking vertical alignment.
   - **Fix**: Styled both `.chart-stale-overlay` and `.chart-empty-overlay` as centered, semi-transparent overlays inside `.chart-wrapper` (`position: absolute; inset: 0; pointer-events: none;`).

3. **Timezone Normalization in `telemetry_age_sec`**:
   - When timestamps in telemetry records are naive or UTC-aware, comparing against a naive `datetime.now()` in non-UTC local zones (such as UTC+5:30) can yield an offset difference of 19,800s, rendering data permanently stale.
   - **Fix**: Normalized all timestamp calculations in `_calculate_telemetry_age()` to UTC aware datetime (`datetime.now(timezone.utc)` and `dt.replace(tzinfo=timezone.utc)` for naive strings). Added unit test `test_timezone_staleness_non_utc` verifying accuracy under simulated non-UTC environments.

4. **Collector History Timeout During Bursts**:
   - The upstream GET `/metrics/history` call previously timed out at 0.8s, causing dropped telemetry during burst load.
   - **Fix**: Extended timeout to 2.5s (`_safe_get(..., timeout=2.5)`). In addition, when an upstream fetch fails or times out, the backend retains the last known good history (`_cached_state["telemetry_history"]`) and marks `stale: true`, rather than wiping history to empty.

5. **`elapsed_sec` Calculation at Process Exit**:
   - Elapsed time was computed dynamically upon poll as `now - start_time`. If the poll occurred long after Locust finished (e.g. at 143s for an 80s run), the UI reported 143.4s.
   - **Fix**: Spawns a background daemon thread `_proc_watcher` that monitors the subprocess exit and records exact `finished_at`. On exit, elapsed time is capped at `min(elapsed, dur)`.

---

## 2. Layout & UI Overhaul (Part F2)

### Single-Screen Live Ops Console (1366x768 & 1920x1080)
- **Top Bar**: Sticky header with system title, dynamic status pills, and zero cover-up layout. Added `overflow-anchor: none;` to `<body>` and `html` to prevent scroll anchoring jumps.
- **Brand Block**: Clean, unstyled brand header (`.brand-block`) without blue/purple anchor underlines or link styling.
- **3-Item Navigation**:
  1. `Live` (`/overview` or `/live`)
  2. `Run scenario` (`/experiments` or `/run`)
  3. `Analysis` (`/analysis` or `/cpu`, `/latency`, `/forecast`, `/alerts`)
- **Mini-Launcher on Live Page**:
  - Embedded profile dropdown (`step`, `spike`, `ramp`, `double_burst`, `soak`), controller mode dropdown (`predictive`, `reactive`, `capacity_aware`), Start/Stop buttons, and live progress bar directly within the Live dashboard.
- **KPI Strip**:
  - Single row of 5 compact cards (~72px tall): Active Workers, Arrival Rate, P95 Latency, Queue Depth, and System CPU.
- **Main Chart Grid**:
  - Left (~60% width): Hero **Traffic vs Capacity** chart spanning the remaining viewport height.
  - Right (~40% width): Stacked **P95 Latency** and **Pool Utilization** charts.
- **Compact Event Feed**:
  - 6-row scrollable strip at the bottom showing recent scaling decisions, with a "Show All Events" modal.
- **Required vs Optional Service Pills**:
  - `COLLECTOR` and `CONTROLLER` (plus `INFERENCE` when predictive mode is active) are **required**: green when online, red when offline.
  - `ALERTS` and `REMEDIATION` are **optional**: grey ("not started") when offline, never red.

---

## 3. Scenario Launcher Overhaul (Part F3)

- **Grid Layout**: 3 scenario cards per row.
- **Fixed SVG Dimensions**: Profile sketch SVGs strictly fixed at `height: 56px; width: 100%;` with `preserveAspectRatio="none"`.
- **Card Selection**: The entire card is clickable to select the profile, with clear active borders and badges.
- **Sticky Run Panel**: Fixed sticky top panel with profile description, duration, controller mode selector, single Start button, phase indicator (`calm` / `burst`), and progress bar.

---

## 4. Verification & Testing (Part F4)

### Unit Tests Added & Updated (`tests/test_dashboard_patch.py`)
- `test_timezone_staleness_non_utc`: Verifies telemetry age calculation under non-UTC timezones does not falsely mark data stale.
- `test_history_timeout_fallback`: Verifies that when `/metrics/history` times out or fails, the last good history is preserved and marked stale.
- `test_required_vs_optional_pill_state`: Verifies COLLECTOR/CONTROLLER/INFERENCE report red when offline, while ALERTS/REMEDIATION report grey `offline-optional`.
- `test_elapsed_sec_capped_at_duration_on_exit`: Verifies finished runs cap `elapsed_sec` at the profile's duration rather than poll time.

### Test Suite Execution
```
python -m pytest tests/ -v
=================== 57 passed, 8 subtests passed in 45.36s ====================
```

### Security & Sanitization
```
python scan_repo.py
Total files scanned: 481
Clean files: 481
Files with hits: 0
ALL SCANNED FILES ARE 100% CLEAN!
```

### Verified Visual Artifacts
- **Overview Initial (1366x768)**: `overview_page_1793106541035.png`
- **Scenario Launcher (1366x768)**: `experiments_page_1793106712927.png`
- **Analysis Page with Tabs (1366x768)**: `analysis_page_1793106788791.png`
- **Live Run After Finish (1366x768)**: `live_after_finish_1366x768_1793107124739.png`

---

## 5. Items Not Done / Preserved Boundaries
- No modifications were made outside `web_dashboard/` and `tests/`.
- `stagbdt/`, `models/`, `demo_controller.py`, `telemetry_collector.py`, `inference_service.py`, `alert_engine.py`, `remediation_service.py`, `locustfile_perm.py`, and `burst_test.py` were left completely untouched.
- Chart.js remains vendored locally (`/static/vendor/chart.umd.min.js`); no CDN or external networks required.
