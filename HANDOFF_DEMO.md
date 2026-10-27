# HANDOFF_DEMO: Opt-In Demo Controller & Live Evaluation

## 1. Executive Summary

This deliverable implements the standalone opt-in **Demo Controller** (`demo_controller.py`) for PERM, providing **reactive**, **capacity_aware**, and **predictive** edge resource scaling over a 1-second control loop without altering existing production services (`alert_engine.py`, `remediation_service.py`, worker caps, core pinning, or model code).

All constants (`service_time = 0.126s`, `capacity_per_worker = 7.89 req/s`, `logical_cpus = 16`) are loaded directly from `testbed_calibration.json`. All reported metrics come exclusively from real physical measurements gathered during live Locust runs (seed 44) on the active testbed.

> **Important Notice on Methodology**: The runs reported below are **illustrative single runs, not experimental results**. Single runs on isolated profiles do not provide statistical confidence intervals or generalize across workloads; experimental claims require multi-seed replication across diverse profiles.

---

## 2. Architecture & Deliverables

### D1. Demo Controller (`demo_controller.py`)
- **1-Second Control Loop**: Standalone script querying `/metrics/latest`.
- **Pool Utilization**:
  $$\text{Pool Utilization} = \text{clip}\left(\frac{\text{worker\_cpu\_percent} \times \text{logical\_cpus}}{\text{active\_workers}}, 0, 100\right)$$
  smoothed over a 3-second rolling mean. Displays queue depth, arrivals, failures, and worker state each second.
- **Alert Engine** (`controller_events.csv`):
  - `WARN`: logged when smoothed pool utilization crosses $\ge 70\%$.
  - `CRITICAL`: logged when smoothed pool utilization crosses $\ge 85\%$.
  - `RECOVERY`: logged when smoothed pool utilization drops back $< 70\%$.
- **Modes Supported**:
  1. `reactive`:
     - `reactive_target = ceil(smoothed_arrivals * service_time / 0.7)`, applied when pool utilization $> 70\%$ or queue depth $\ge 5$.
     - If pool utilization $> 85\%$ for $\ge 2\text{s}$, a safety boost of $+2$ workers is added.
     - When not stressed, target is `min(current_workers, base_target)`.
  2. `capacity_aware`:
     - `cap_target = ceil(smoothed_arrivals * service_time / 0.7)` directly from arrival rate only (no utilization threshold required).
     - Separates "uses arrival rate" from "predicts".
     - Immediate scale-up; identical 15s scale-down hysteresis.
  3. `predictive`:
     - Requires at least 10 seconds of arrival history for warm-up. For $t < 10\text{s}$, logs `"predictive warming up"` and uses reactive logic only.
     - For $t \ge 10\text{s}$, queries `/forecast/demand?horizon=20` (enabled via `PERM_FORECASTER=stagbdt`) with arrival history.
     - `predictive_target = ceil(forecast_upper_bound / (capacity_per_worker * 0.7))`.
     - Overall target = $\max(\text{reactive\_target}, \text{predictive\_target})$.
     - If forecast service is unreachable, logs `"forecast offline"` and falls back to reactive target.
  - **Shared Constraints**:
    - All targets strictly clipped to $[2, 8]$ workers.
    - Scale-up is immediate via `POST /api/capacity {"active_workers": n}`.
    - Scale-down requires staying below current worker count continuously for $\ge 15\text{s}$ (hysteresis).
    - Every scale event records the triggering rule (`forecast`, `capacity_aware`, `reactive`, `safety`, or `scale_down`) along with pool utilization before and after.

### D2. Unit Test Suite (`tests/test_demo_controller.py`)
Seven unit tests verifying mock inputs without requiring Locust or live services:
1. `test_target_at_40_arrivals`: 2 workers with 40 req/s arrivals produces target 8 ($\lceil 40 \times 0.126 / 0.7 \rceil = 8$).
2. `test_utilization_above_85_for_2s_adds_2_workers`: Pool utilization $> 85\%$ for 2 consecutive seconds adds $+2$ safety workers.
3. `test_scale_down_hysteresis`: Scale-down is suppressed at $t=10\text{s}$ and $t=14.9\text{s}$, executing only after $15\text{s}$ below current workers.
4. `test_forecast_offline_fallback`: When `/forecast/demand` fails, logs `"forecast offline"` and falls back to reactive target.
5. `test_alert_written_at_70_utilization`: Verifies CSV event emission for `WARN` ($\ge 70\%$), `CRITICAL` ($\ge 85\%$), and `RECOVERY` ($< 70\%$).
6. `test_predictive_warmup_3s_history_not_used`: With 3s of arrival history ($< 10\text{s}$), the predictive endpoint is not queried, `"predictive warming up"` is logged, and only reactive logic is used.
7. `test_capacity_aware_mode`: Computes target directly from arrivals without checking utilization, scales up immediately, and respects 15s scale-down hysteresis.
- Result: **7/7 tests passed**.

---

## 3. How to Run the Demo (Exact Commands)

### Option A: Automated Runner (Runs All 6 Runs Sequentially)
```bash
python run_demo_runs.py --mode all --profile both
```

### Option B: Step-by-Step Manual Commands

#### 1. Start the Telemetry Collector (Terminal 1)
```bash
python telemetry_collector.py --port 5000 --workers 2
```

#### 2. Start the Inference Service (Terminal 2 — Predictive Mode Only)
```bash
# On Linux/macOS:
PERM_FORECASTER=stagbdt python inference_service.py --port 5001 --collector-url http://127.0.0.1:5000

# On Windows PowerShell:
$env:PERM_FORECASTER="stagbdt"; python inference_service.py --port 5001 --collector-url http://127.0.0.1:5000
```

#### 3. Start the Demo Controller (Terminal 3)
```bash
# Predictive Mode:
python demo_controller.py --mode predictive --initial-workers 2

# Capacity-Aware Mode:
python demo_controller.py --mode capacity_aware --initial-workers 2

# Reactive Mode:
python demo_controller.py --mode reactive --initial-workers 2
```

#### 4. Launch the Locust Traffic Generator (Terminal 4)
```bash
# Step Profile:
$env:PERM_PROFILE="step"; $env:PERM_SEED="44"; python -m locust -f locustfile_perm.py --headless -u 1 -r 1 -t 125s --host http://127.0.0.1:5000

# Ramp Profile:
$env:PERM_PROFILE="ramp"; $env:PERM_SEED="44"; python -m locust -f locustfile_perm.py --headless -u 1 -r 1 -t 140s --host http://127.0.0.1:5000
```

---

## 4. Evaluation Results: 6 Honest Re-Runs

The following table presents the measured metrics across all 6 runs executed on the live testbed. All values come directly from the generated JSON artifacts (`results/demo_run_*.json`).

| Run Key | 1st Scale-Up Rel to Burst/Ramp Start | Workers During 1st 25s | Peak Pool Util (%) | Seconds Above 70% | Failed Requests | Worker-Seconds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`reactive_step`** | **+7.76 s** | 2 | 94.2% | 7 s | 68 | 749.8 |
| **`capacity_aware_step`** | **+8.59 s** | 2 | 96.8% | 7 s | 73 | 764.2 |
| **`predictive_step`** | **-19.85 s** | 2, 3, 4, 6, 7, 8 | 73.0% | 3 s | 0 | 923.1 |
| **`reactive_ramp`** | **+14.82 s** | 2 | 92.3% | 11 s | 0 | 782.8 |
| **`capacity_aware_ramp`** | **+11.75 s** | 2 | 83.7% | 16 s | 0 | 772.2 |
| **`predictive_ramp`** | **-19.86 s** | 2, 3, 4, 5, 6, 7 | 72.3% | 3 s | 0 | 1014.6 |

*(Note: Burst and ramp start offsets are at $t = 30.0\text{s}$ from Locust start. Negative values indicate scale-up before the burst/ramp begins).*

---

### Key Observations & Comparison

#### 1. Does `capacity_aware` match `predictive`?
- **On the Step Profile**: No. `capacity_aware` relies on observed arrival rates smoothed over 3 seconds. When a sudden step burst occurs at $t = 30\text{s}$, arrival smoothing lags by 2–3s, and worker processes require ~1.5s cold-start provisioning. By the time `capacity_aware` scaled up at $+8.59\text{s}$ ($t = 38.6\text{s}$), the request queue had already backed up, resulting in **73 dropped requests** and **96.8% peak pool utilization** (comparable to `reactive` at 68 drops and 94.2% peak util). Only `predictive` (scaling at $-19.85\text{s}$ before burst onset) pre-provisioned workers ahead of time, avoiding queue drops.
- **On the Ramp Profile**: Partially. Because the ramp increases demand gradually (from 4 to 40 req/s over 30s), both `capacity_aware` and `reactive` suffered zero failed requests. `capacity_aware` smoothly scaled workers in proportional steps ($2 \to 3 \to 4 \to 5 \to 6 \to 7 \to 8$), lowering peak pool utilization to **83.7%** (compared to 92.3% for reactive) while maintaining lean worker-seconds (**772.2s** vs **1014.6s** for predictive). `predictive` achieved a lower peak utilization (**72.3%**), but used **+31.4% more worker-seconds** by pre-warming all 8 workers 20 seconds in advance.

#### 2. Worker Count During the First 25 Seconds
- For both **`reactive`** and **`capacity_aware`**, workers remained strictly at **2** during the first 25 seconds across both profiles, as baseline demand was calm (~4 req/s).
- For **`predictive`**, workers remained at **2** during the initial 10-second warm-up window ($0.0\text{s} \text{ to } 10.1\text{s}$). Because the forecast horizon is $h = 20\text{s}$, once elapsed time reached $t \approx 10.2\text{s}$, the forward-looking horizon touched the burst/ramp onset at $t = 30.0\text{s}$. As a result, `predictive` progressively scaled workers ($2 \to 3 \to 4 \to 6 \to 7 \to 8$) during $t = 10\text{s}\text{ to }20\text{s}$, meaning workers during the first 25s were `2, 3, 4, 6, 7, 8` rather than remaining at 2.

---

### Detailed Worker Transition Timelines

#### Step Profile Runs (Seed 44)
- **`reactive_step`**:
  `0.0s–37.8s`: 2w (37.7s) $\to$ `37.8s–38.8s`: 3w (1.0s) $\to$ `38.8s–39.8s`: 5w (1.0s) $\to$ `39.8s–111.6s`: 8w (71.8s) $\to$ `111.6s–125.9s`: 6w (14.2s)
- **`capacity_aware_step`**:
  `0.0s–38.6s`: 2w (38.6s) $\to$ `38.6s–39.6s`: 5w (1.0s) $\to$ `39.6s–40.6s`: 7w (1.0s) $\to$ `40.6s–111.6s`: 8w (71.0s) $\to$ `111.6s–125.9s`: 7w (14.3s)
- **`predictive_step`**:
  `0.0s–10.2s`: 2w (10.1s) $\to$ `10.2s–12.2s`: 3w (2.0s) $\to$ `12.2s–16.3s`: 4w (4.1s) $\to$ `16.3s–17.3s`: 6w (1.0s) $\to$ `17.3s–19.3s`: 7w (2.1s) $\to$ `19.3s–125.8s`: 8w (106.5s)

#### Ramp Profile Runs (Seed 44)
- **`reactive_ramp`**:
  `0.0s–44.8s`: 2w (44.8s) $\to$ `44.8s–46.9s`: 3w (2.0s) $\to$ `46.9s–65.1s`: 6w (18.2s) $\to$ `65.1s–73.2s`: 7w (8.1s) $\to$ `73.2s–126.9s`: 8w (53.7s) $\to$ `126.9s–141.1s`: 6w (14.2s)
- **`capacity_aware_ramp`**:
  `0.0s–41.8s`: 2w (41.7s) $\to$ `41.8s–49.9s`: 3w (8.1s) $\to$ `49.9s–53.9s`: 4w (4.0s) $\to$ `53.9s–54.9s`: 5w (1.0s) $\to$ `54.9s–64.0s`: 6w (9.1s) $\to$ `64.0s–69.1s`: 7w (5.0s) $\to$ `69.1s–126.8s`: 8w (57.8s) $\to$ `126.8s–141.0s`: 6w (14.2s)
- **`predictive_ramp`**:
  `0.0s–10.1s`: 2w (10.1s) $\to$ `10.1s–17.2s`: 3w (7.1s) $\to$ `17.2s–18.2s`: 4w (1.0s) $\to$ `18.2s–19.2s`: 5w (1.0s) $\to$ `19.2s–24.3s`: 6w (5.1s) $\to$ `24.3s–29.4s`: 7w (5.1s) $\to$ `29.4s–126.8s`: 8w (97.4s) $\to$ `126.8s–129.8s`: 7w (3.1s) $\to$ `129.8s–141.0s`: 8w (11.2s)

---

## 5. Limitations

1. **8-Worker Cap & Calibrated Capacity**:
   - Each worker sustains a calibrated saturation capacity of $7.89\text{ req/s}$ (`results[0].saturation_rps = 7.89` in `testbed_calibration.json`).
   - At the target utilization threshold of $70\%$, each worker provides an effective capacity of $\approx 5.523\text{ req/s}$.
   - With an 8-worker cap, total pool capacity at 70% is $\approx 44.2\text{ req/s}$. When demand exceeds $\approx 44\text{ req/s}$, the pool is fully expanded to 8 workers, and pool utilization will exceed 70% regardless of controller mode.
2. **Resource Trade-off**:
   - Pre-warming workers ahead of traffic surges consumes additional worker-seconds (+23% on step, +31% on ramp). Under sudden step jumps, this investment prevents queue overflow; under gradual ramps, reactive or capacity-aware scaling can achieve zero drops with lower worker-seconds.

---

## 6. UNVERIFIED Section

1. **Multi-Seed Variance**:
   - The results presented above are single runs on seed 44. Run-to-run Poisson variance and OS scheduling jitter across multiple seeds have not been benchmarked.
2. **Stochastic Workloads (e.g. `random_walk`, `double_burst`)**:
   - Evaluation with the demo controller was performed on canonical `step` and `ramp` profiles. Complex burst shapes with rapid oscillating cycles remain unverified under live closed-loop control.
3. **Lead Time Sensitivity Under Different Lookahead Horizons**:
   - Fixed horizon $h = 20\text{s}$ was evaluated. Shorter horizons ($h = 10\text{s}$) or longer horizons ($h = 30\text{s}$) have not been compared in live closed-loop controller execution.
