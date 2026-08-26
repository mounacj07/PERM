"""
PERM - Predictive Edge Resource Management
Locust Traffic Simulation & Live Presentation Driver

Supports 3 execution modes:

  1. FAST DEMO MODE (PERM_FAST_DEMO=1):
     Shortens full cycle to 55 seconds (15s steady, 15s ramp, 15s burst, 10s cooldown).
     Perfect for live presentations so evaluators don't have to wait!

  2. MANUAL WEB UI MODE (PERM_MANUAL_TRAFFIC=1):
     Disables automated shape control completely. Type user counts (10, 50, 100)
     directly into http://localhost:8089 and Locust spawns them immediately!

  3. STANDARD TRAINING MODE (Default):
     Full 8.25-minute multi-cycle automated run for high-density training data collection.

Commands:
  # Fast 55-second demo run (Recommended for Live Demo):
  $env:PERM_FAST_DEMO = 1
  locust -f locustfile.py --host http://127.0.0.1:5000

  # Manual Web UI mode (Type 10, 50, 100 directly into Web UI):
  $env:PERM_MANUAL_TRAFFIC = 1
  locust -f locustfile.py --host http://127.0.0.1:5000
"""

import csv
import os
import random
import time

from locust import HttpUser, task, between, LoadTestShape

# --------------------------------------------------------------------------
# Mode Selection via Environment Variables
# --------------------------------------------------------------------------
IS_MANUAL_MODE = os.environ.get("PERM_MANUAL_TRAFFIC", "0") == "1"
IS_FAST_DEMO = os.environ.get("PERM_FAST_DEMO", "0") == "1"

if IS_FAST_DEMO:
    # ~100s total: short enough to be snappy, long enough for accurate ML windows
    STAGES = [
        {"duration": 20, "users": 8,  "spawn_rate": 3,  "phase": "steady",   "load_range": (2, 3)},
        {"duration": 25, "users": 45, "spawn_rate": 8,  "phase": "ramp",     "load_range": (4, 7)},
        {"duration": 30, "users": 90, "spawn_rate": 25, "phase": "burst",    "load_range": (7, 10)},
        {"duration": 25, "users": 5,  "spawn_rate": 20, "phase": "cooldown", "load_range": (1, 2)},
    ]
else:
    # Standard training schedule (~8.25 min/cycle)
    STAGES = [
        {"duration": 180, "users": 10,  "spawn_rate": 2,  "phase": "steady",   "load_range": (2, 4)},
        {"duration": 180, "users": 50,  "spawn_rate": 1,  "phase": "ramp",     "load_range": (3, 6)},
        {"duration": 45,  "users": 100, "spawn_rate": 50, "phase": "burst",    "load_range": (5, 10)},
        {"duration": 90,  "users": 5,   "spawn_rate": 20, "phase": "cooldown", "load_range": (1, 2)},
    ]

CYCLE_DURATION = sum(stage["duration"] for stage in STAGES)
CYCLES = int(os.environ.get("PERM_LOCUST_CYCLES", 5 if not IS_FAST_DEMO else 1))

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


_current_load_range = {"min": 2, "max": 4}


if not IS_MANUAL_MODE:

    class TrafficStagesShape(LoadTestShape):
        """
        Drives automated traffic stages.
        This class only exists when NOT in manual mode.
        """

        def __init__(self):
            super().__init__()
            self._last_logged_key = None

        def tick(self):
            run_time = self.get_run_time()
            total_duration = CYCLE_DURATION * CYCLES

            if run_time >= total_duration:
                return None

            cycle_index = int(run_time // CYCLE_DURATION)
            time_in_cycle = run_time % CYCLE_DURATION

            elapsed_boundary = 0

            for stage in STAGES:
                stage_end = elapsed_boundary + stage["duration"]

                if time_in_cycle < stage_end:
                    key = (cycle_index, stage["phase"])

                    if self._last_logged_key != key:
                        _log_phase_transition(
                            stage["phase"],
                            cycle_index + 1,
                            stage["users"],
                            run_time
                        )

                        self._last_logged_key = key

                        _current_load_range["min"], \
                        _current_load_range["max"] = stage["load_range"]

                    return (
                        stage["users"],
                        stage["spawn_rate"]
                    )

                elapsed_boundary = stage_end

            return None


class PermWorkloadUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def hit_process_endpoint(self):
        load = random.randint(_current_load_range["min"], _current_load_range["max"])
        self.client.post(
            "/api/process",
            json={"load": load},
            name="/api/process",
        )