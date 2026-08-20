"""
PERM - Predictive Edge Resource Management
Locust Traffic Simulation

Drives the telemetry_collector's /api/process endpoint through three
distinct traffic profiles so we can collect labeled training data for the
Random Forest forecaster:

    1. STEADY-STATE  - constant, low user count, constant per-request load.
                        Baseline CPU/latency behavior.
    2. RAMP          - user count climbs linearly over the phase duration.
                        Captures how CPU/latency trend as load builds.
    3. BURST         - user count spikes sharply, holds briefly, then drops.
                        Captures sudden-shock behavior (the hardest case for
                        the model to forecast 30s ahead).

The four phases run back-to-back and repeat for CYCLES full loops in a
single Locust run via a custom LoadTestShape, so one execution produces
enough raw telemetry to build a real training set (not just one pass).
Each cycle uses freshly randomized per-request load within each phase's
range, so repeated cycles aren't just noisy duplicates of each other.

Each phase transition (name, cycle number, start time, target user count)
is appended to phase_log.csv, so feature_engineering.py can later join
telemetry_log.csv timestamps against this file to label each row with its
traffic profile.

Default config: 5 cycles x ~8.25 min/cycle =~ 41 minutes total, yielding
roughly ~2,400 seconds of raw telemetry -> ~2,300 sliding-window feature
rows after 30s windowing -> ~1,600 / 350 / 350 after a 70/15/15 split.
Override CYCLES via the PERM_LOCUST_CYCLES environment variable if you
want a shorter smoke-test run or a longer one.

Run (headless, full multi-cycle run):
    locust -f locustfile.py --host http://127.0.0.1:5000 --headless

Run a quick 1-cycle smoke test instead:
    # PowerShell
    $env:PERM_LOCUST_CYCLES = 1
    locust -f locustfile.py --host http://127.0.0.1:5000 --headless

Run (with web UI, for manual poking around):
    locust -f locustfile.py --host http://127.0.0.1:5000
    then open http://localhost:8089
"""

import csv
import os
import random
import time

from locust import HttpUser, task, between
from locust import LoadTestShape

# --------------------------------------------------------------------------
# Phase schedule (one cycle)
# --------------------------------------------------------------------------
# Each stage: duration_sec, target_user_count, spawn_rate, phase_name, load_range
# load_range is the (min, max) value sent as "load" to /api/process for that
# phase, letting per-request intensity vary alongside request volume.
STAGES = [
    # -- STEADY-STATE: low, constant concurrency, light/medium load ---------
    {"duration": 180, "users": 10, "spawn_rate": 2, "phase": "steady", "load_range": (2, 4)},

    # -- RAMP: users climb from ~10 to ~50 over the phase ---------------------
    {"duration": 180, "users": 50, "spawn_rate": 1, "phase": "ramp", "load_range": (3, 6)},

    # -- BURST: sharp spike to a high user count, brief hold ------------------
    {"duration": 45, "users": 100, "spawn_rate": 50, "phase": "burst", "load_range": (5, 10)},

    # -- COOLDOWN: drop back down, let the system settle -----------------------
    {"duration": 90, "users": 5, "spawn_rate": 20, "phase": "cooldown", "load_range": (1, 2)},
]

CYCLE_DURATION = sum(stage["duration"] for stage in STAGES)
CYCLES = int(os.environ.get("PERM_LOCUST_CYCLES", 5))

PHASE_LOG_PATH = os.path.join(os.path.dirname(__file__), "phase_log.csv")
PHASE_LOG_FIELDS = ["timestamp", "phase", "cycle", "target_users", "elapsed_sec"]


def _ensure_phase_log_header():
    if not os.path.exists(PHASE_LOG_PATH):
        with open(PHASE_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PHASE_LOG_FIELDS)
            writer.writeheader()


def _log_phase_transition(phase_name: str, cycle: int, target_users: int, elapsed: float):
    _ensure_phase_log_header()
    with open(PHASE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE_LOG_FIELDS)
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "+00:00",
            "phase": phase_name,
            "cycle": cycle,
            "target_users": target_users,
            "elapsed_sec": round(elapsed, 1),
        })


# --------------------------------------------------------------------------
# Current phase state, shared with the HttpUser tasks below so each request
# uses the load_range appropriate to whatever phase is currently active.
# --------------------------------------------------------------------------
_current_load_range = {"min": 2, "max": 4}


class TrafficStagesShape(LoadTestShape):
    """
    Drives Locust through STAGES sequentially, repeating the full cycle
    CYCLES times, so one run yields CYCLES x len(STAGES) phase segments.
    Returns None once all cycles are complete, which stops the test
    automatically.
    """

    def __init__(self):
        super().__init__()
        self._last_logged_key = None  # (cycle, phase) tuple, avoids duplicate log rows

    def tick(self):
        run_time = self.get_run_time()
        total_duration = CYCLE_DURATION * CYCLES

        if run_time >= total_duration:
            return None  # all cycles finished -> stop the test

        # Which cycle are we in, and how far into that cycle?
        cycle_index = int(run_time // CYCLE_DURATION)  # 0-based
        time_in_cycle = run_time % CYCLE_DURATION

        elapsed_boundary = 0
        for stage in STAGES:
            stage_end = elapsed_boundary + stage["duration"]
            if time_in_cycle < stage_end:
                key = (cycle_index, stage["phase"])
                if self._last_logged_key != key:
                    _log_phase_transition(stage["phase"], cycle_index + 1, stage["users"], run_time)
                    self._last_logged_key = key
                    _current_load_range["min"], _current_load_range["max"] = stage["load_range"]

                return (stage["users"], stage["spawn_rate"])

            elapsed_boundary = stage_end

        # Shouldn't be reachable, but stop gracefully if we fall through.
        return None


class PermWorkloadUser(HttpUser):
    """
    Simulated client hitting the PERM telemetry collector's workload
    endpoint. wait_time is short since we want to generate meaningful
    request volume; actual concurrency/user-count is controlled entirely
    by TrafficStagesShape above.
    """
    wait_time = between(0.2, 1.0)

    @task
    def hit_process_endpoint(self):
        load = random.randint(_current_load_range["min"], _current_load_range["max"])
        self.client.post(
            "/api/process",
            json={"load": load},
            name="/api/process",
        )