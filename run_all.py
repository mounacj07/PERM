import warnings
warnings.filterwarnings("ignore")

"""
PERM - Predictive Edge Resource Management
Universal Orchestrator Launcher

Launches all edge microservices in parallel and manages graceful shutdown:
  - Telemetry Collector:  http://127.0.0.1:5000
  - Inference Service:    http://127.0.0.1:5001
  - Alert Engine:         http://127.0.0.1:5002
  - Remediation Service:  http://127.0.0.1:5003
  - Web Dashboard Demo:   http://127.0.0.1:8000
  - Matplotlib Dashboard: (Optional via --gui)

Each service is health-checked at its /health endpoint before the next
one starts, so startup ordering is guaranteed even on slow machines.

Run:
    python run_all.py
    python run_all.py --gui
"""

import argparse
import os
import subprocess
import sys
import time

import requests

SERVICES = [
    {
        "name":        "Telemetry Collector",
        "cmd":         [sys.executable, "telemetry_collector.py", "--port", "5000"],
        "health_url":  "http://127.0.0.1:5000/health",
    },
    {
        "name":        "Inference Service",
        "cmd":         [sys.executable, "inference_service.py", "--port", "5001",
                        "--collector-url", "http://127.0.0.1:5000"],
        "health_url":  "http://127.0.0.1:5001/health",
    },
    {
        "name":        "Alert Engine",
        "cmd":         [sys.executable, "alert_engine.py", "--port", "5002",
                        "--inference-url", "http://127.0.0.1:5001"],
        "health_url":  "http://127.0.0.1:5002/health",
    },
    {
        "name":        "Remediation Service",
        "cmd":         [sys.executable, "remediation_service.py", "--port", "5003",
                        "--alert-url", "http://127.0.0.1:5002",
                        "--collector-url", "http://127.0.0.1:5000"],
        "health_url":  "http://127.0.0.1:5003/health",
    },
    {
        "name":        "Web Dashboard Demo",
        "cmd":         [sys.executable, "web_dashboard/app.py", "--port", "8000"],
        "health_url":  "http://127.0.0.1:8000/",
    },
]

HEALTH_CHECK_RETRIES = 20   # max attempts before giving up
HEALTH_CHECK_INTERVAL = 0.5  # seconds between attempts


def _wait_for_healthy(name: str, health_url: str) -> bool:
    """
    Poll health_url until it returns HTTP 200 or we exceed retries.
    Returns True if the service became healthy, False if it timed out.
    """
    for attempt in range(HEALTH_CHECK_RETRIES):
        try:
            r = requests.get(health_url, timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    print(f"[!] WARNING: {name} did not become healthy after "
          f"{HEALTH_CHECK_RETRIES * HEALTH_CHECK_INTERVAL:.0f}s — continuing anyway.")
    return False


def main():
    parser = argparse.ArgumentParser(description="PERM System Orchestrator")
    parser.add_argument("--gui", action="store_true",
                        help="Launch Matplotlib GUI dashboard window alongside web dashboard")
    args = parser.parse_args()

    print("=" * 70)
    print(" PERM — Predictive Edge Resource Management System ")
    print("=" * 70)

    # Pass PYTHONWARNINGS=ignore to all subprocesses for clean operational output
    sub_env = os.environ.copy()
    sub_env["PYTHONWARNINGS"] = "ignore"

    processes = []
    try:
        for svc in SERVICES:
            name = svc["name"]
            print(f"[*] Starting {name}...")
            p = subprocess.Popen(svc["cmd"], env=sub_env)
            processes.append((name, p))
            # Wait until the service is actually ready before starting the next one
            _wait_for_healthy(name, svc["health_url"])

        if args.gui:
            print("[*] Starting Matplotlib Operator GUI Dashboard...")
            gui_p = subprocess.Popen([sys.executable, "matplotlib_dashboard.py"], env=sub_env)
            processes.append(("Matplotlib Dashboard", gui_p))

        print("\n" + "=" * 70)
        print(" ALL PERM EDGE SERVICES ARE RUNNING! ")
        print("=" * 70)
        print("  • Edge Web Dashboard Demo : http://127.0.0.1:8000")
        print("  • Telemetry Collector     : http://127.0.0.1:5000")
        print("  • Inference Service       : http://127.0.0.1:5001")
        print("  • Alert Engine            : http://127.0.0.1:5002")
        print("  • Remediation Service     : http://127.0.0.1:5003")
        print("=" * 70)
        print(" Press Ctrl+C to stop all services cleanly.\n")

        while True:
            time.sleep(1.0)
            for name, p in processes:
                if p.poll() is not None:
                    print(f"[!] Warning: {name} terminated unexpectedly with code {p.returncode}")

    except KeyboardInterrupt:
        print("\n[*] Stopping all PERM edge services...")
    finally:
        for name, p in reversed(processes):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("[*] All services stopped.")


if __name__ == "__main__":
    main()
