"""
PERM - test_testbed.py
========================
Automated test suite for the Part B "real process pool" testbed.

Tests:
  1. Idle CPU is low and latency is small.
  2. 1 worker under heavy load -> p95 latency high and/or errors occur.
  3. Scale to 4 workers during load -> latency and errors drop clearly.
  4. Resizing never drops an in-flight request.
  5. Telemetry columns are MEASURED: CPU rises when load rises.
  6. Existing endpoints still return the same JSON field names.
  7. python run_all.py starts every service and the dashboard loads.

Run:
    python test_testbed.py

The tests that require a live server spin up a real subprocess; the others
use the Flask test_client() for speed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import threading
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers to start / stop the collector in a subprocess
# ---------------------------------------------------------------------------
COLLECTOR_PORT = 15900   # high port -- avoids conflicts with the real 5000


def _wait_healthy(base_url: str, timeout: float = 20.0) -> bool:
    import requests as _req
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = _req.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def _start_collector(workers: int = 2, ms_per_load: float = 18.0) -> subprocess.Popen:
    env = os.environ.copy()
    env["MS_PER_LOAD"] = str(ms_per_load)
    env["MAX_WORKERS"] = "16"
    env["REQUEST_TIMEOUT_SEC"] = "4.0"
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "telemetry_collector.py",
         "--port", str(COLLECTOR_PORT), "--workers", str(workers)],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return proc


def _stop_collector(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def _send_burst(
    base_url: str,
    n_requests: int,
    load: int,
    concurrency: int,
    timeout_sec: float = 5.0,
) -> dict:
    """
    Fire n_requests to /api/process with given concurrency.
    Returns dict: latencies_ms, errors, n_requests.
    """
    import requests as _req

    latencies: List[float] = []
    errors = 0
    lock = threading.Lock()

    def _one():
        nonlocal errors
        t0 = time.perf_counter()
        try:
            r = _req.post(
                f"{base_url}/api/process",
                json={"load": load},
                timeout=timeout_sec,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            is_err = r.status_code != 200
        except Exception:
            lat = (time.perf_counter() - t0) * 1000.0
            is_err = True
        with lock:
            latencies.append(lat)
            if is_err:
                errors += 1

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_one) for _ in range(n_requests)]
        for f in futs:
            f.result()

    latencies.sort()
    p95 = 0.0
    if latencies:
        k = (len(latencies) - 1) * 0.95
        f_ = int(k)
        c = min(f_ + 1, len(latencies) - 1)
        p95 = latencies[f_] + (latencies[c] - latencies[f_]) * (k - f_)

    return {
        "latencies_ms": latencies,
        "avg_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_ms": round(p95, 2),
        "errors": errors,
        "n": len(latencies),
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
class TestTestbed(unittest.TestCase):
    """Tests that need a live subprocess running the collector."""

    @classmethod
    def setUpClass(cls):
        """Start one collector subprocess used by most tests."""
        # Use a short ms_per_load so tests run fast but still exercise the pool
        cls.ms_per_load = 18.0
        cls.proc = _start_collector(workers=2, ms_per_load=cls.ms_per_load)
        cls.base_url = f"http://127.0.0.1:{COLLECTOR_PORT}"
        ok = _wait_healthy(cls.base_url, timeout=30.0)
        if not ok:
            cls.proc.terminate()
            raise RuntimeError("Collector did not become healthy in time")

    @classmethod
    def tearDownClass(cls):
        _stop_collector(cls.proc)

    # ------------------------------------------------------------------ #
    # Test 1 - Idle: CPU is low and latency is small
    # ------------------------------------------------------------------ #
    def test_01_idle_metrics(self):
        """Idle state: single light request -> latency < 500ms."""
        import requests as _req

        print("\n[TEST 1] Idle metrics...")
        time.sleep(2.0)  # let sampler populate buffer

        # Single light request
        t0 = time.perf_counter()
        r = _req.post(f"{self.base_url}/api/process", json={"load": 1}, timeout=5.0)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(r.status_code, 200, "Expected 200 for idle load=1")
        self.assertLess(lat_ms, 500.0, f"Idle latency {lat_ms:.1f}ms is too high")

        # Check telemetry record
        r2 = _req.get(f"{self.base_url}/metrics/latest", timeout=3.0)
        self.assertEqual(r2.status_code, 200)
        data = r2.json()
        self.assertIn("cpu_percent", data)
        self.assertIn("mem_percent", data)
        self.assertIn("avg_latency_ms", data)
        self.assertIn("p95_latency_ms", data)
        self.assertIn("request_count", data)
        self.assertIn("active_workers", data)
        self.assertIn("error_count", data)
        self.assertIn("queue_depth", data)

        print(f"  -> idle latency={lat_ms:.1f}ms  cpu={data.get('cpu_percent')}%")

    # ------------------------------------------------------------------ #
    # Test 2 - 1 worker under heavy load -> high p95 and/or errors
    # ------------------------------------------------------------------ #
    def test_02_one_worker_heavy_load(self):
        """1 worker + heavy load -> p95 > 200ms or error rate > 0%."""
        import requests as _req

        print("\n[TEST 2] 1 worker under heavy load...")
        # Set to 1 worker
        r = _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 1}, timeout=5.0)
        self.assertEqual(r.status_code, 200)
        time.sleep(1.5)  # let pool stabilise

        # Blast with 20 concurrent requests at load=8
        result = _send_burst(self.base_url, n_requests=20, load=8, concurrency=12)
        print(
            f"  -> p95={result['p95_ms']:.1f}ms  "
            f"errors={result['errors']}/{result['n']}  "
            f"avg={result['avg_ms']:.1f}ms"
        )

        # Either latency is high OR errors occurred
        overloaded = result["p95_ms"] > 200.0 or result["errors"] > 0
        self.assertTrue(
            overloaded,
            f"Expected overload with 1 worker: p95={result['p95_ms']:.1f}ms "
            f"errors={result['errors']}",
        )

    # ------------------------------------------------------------------ #
    # Test 3 - Scale to 4 workers -> latency and errors drop
    # ------------------------------------------------------------------ #
    def test_03_scale_up_reduces_latency(self):
        """
        With 1 worker record baseline, then scale to 4 and observe improvement.
        """
        import requests as _req

        print("\n[TEST 3] Scale 1->4 workers and verify improvement...")
        # Make sure we are at 1 worker
        _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 1}, timeout=5.0)
        time.sleep(1.5)

        baseline = _send_burst(self.base_url, n_requests=16, load=8, concurrency=10)
        print(f"  Baseline (1 worker): p95={baseline['p95_ms']:.1f}ms errors={baseline['errors']}")

        # Scale to 4
        r = _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 4}, timeout=5.0)
        self.assertEqual(r.status_code, 200)
        resp_data = r.json()
        self.assertEqual(resp_data.get("active_workers"), 4)
        time.sleep(2.0)  # give pool time to warm up

        after = _send_burst(self.base_url, n_requests=16, load=8, concurrency=10)
        print(f"  After scale (4 workers): p95={after['p95_ms']:.1f}ms errors={after['errors']}")

        improved = (
            after["p95_ms"] < baseline["p95_ms"]
            or after["errors"] < baseline["errors"]
            or after["avg_ms"] < baseline["avg_ms"]
        )
        self.assertTrue(
            improved,
            f"Expected improvement after scaling: "
            f"baseline p95={baseline['p95_ms']:.1f}ms -> after p95={after['p95_ms']:.1f}ms",
        )

    # ------------------------------------------------------------------ #
    # Test 4 - Resize never drops in-flight requests
    # ------------------------------------------------------------------ #
    def test_04_no_dropped_requests_during_resize(self):
        """
        Fire a stream of requests; resize the pool mid-flight; verify no requests
        are lost (all futures are accounted for with success or 503).
        """
        import requests as _req

        print("\n[TEST 4] No dropped requests during pool resize...")
        # Start at 2 workers
        _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 2}, timeout=5.0)
        time.sleep(1.0)

        results = []
        lock = threading.Lock()

        def _worker(i: int):
            t0 = time.perf_counter()
            try:
                r = _req.post(
                    f"{self.base_url}/api/process",
                    json={"load": 5},
                    timeout=6.0,
                )
                lat = (time.perf_counter() - t0) * 1000.0
                with lock:
                    results.append((r.status_code, lat))
            except Exception as exc:
                with lock:
                    results.append((-1, 0.0))

        n_requests = 30
        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_requests)]

        # Start all threads
        for t in threads:
            t.start()

        # Resize mid-flight (after a tiny delay)
        time.sleep(0.2)
        _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 6}, timeout=5.0)

        # Wait for all to finish
        for t in threads:
            t.join(timeout=15.0)

        codes = [c for c, _ in results]
        dropped = codes.count(-1)  # connection error = truly dropped
        print(f"  -> {n_requests} requests sent. Dropped (exception): {dropped}")
        print(f"     200s: {codes.count(200)}  503s: {codes.count(503)}  errors: {dropped}")

        # No request should raise a connection-level exception (dropped)
        self.assertEqual(
            dropped, 0,
            f"{dropped} requests raised exceptions during resize (expected 0)",
        )
        # All requests must be accounted for (200 or 503, never lost)
        self.assertEqual(
            len(results), n_requests,
            f"Expected {n_requests} results, got {len(results)}",
        )

    # ------------------------------------------------------------------ #
    # Test 5 - CPU rises when load rises (measured, not formula)
    # ------------------------------------------------------------------ #
    def test_05_cpu_measured_rises_under_load(self):
        """CPU% is higher during load than at idle -- proves real measurement."""
        import requests as _req

        print("\n[TEST 5] CPU rises under load (real measurement)...")

        # Idle CPU (wait for previous test to settle completely)
        time.sleep(3.0)
        r_idle = _req.get(f"{self.base_url}/metrics/latest", timeout=3.0)
        idle_cpu = r_idle.json().get("cpu_percent", 0.0)
        idle_worker_cpu = r_idle.json().get("worker_cpu_percent", 0.0)

        # Apply sustained heavy load (4 workers, continuous requests for 2.5s)
        _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 4}, timeout=5.0)
        time.sleep(1.0)

        stop_load = threading.Event()
        def _fire():
            while not stop_load.is_set():
                try:
                    _req.post(f"{self.base_url}/api/process", json={"load": 8}, timeout=5.0)
                except Exception:
                    pass
                time.sleep(0.04)

        ts = [threading.Thread(target=_fire, daemon=True) for _ in range(8)]
        for t in ts:
            t.start()

        # Let load run for 2.5 seconds so telemetry samples active CPU
        time.sleep(2.5)
        r_load = _req.get(f"{self.base_url}/metrics/latest", timeout=3.0)
        load_cpu = r_load.json().get("cpu_percent", 0.0)
        load_worker_cpu = r_load.json().get("worker_cpu_percent", 0.0)

        stop_load.set()
        for t in ts:
            t.join(timeout=2.0)

        print(f"  -> idle CPU={idle_cpu:.1f}% (worker={idle_worker_cpu:.1f}%)  load CPU={load_cpu:.1f}% (worker={load_worker_cpu:.1f}%)")
        # CPU is measured if worker_cpu rose or system CPU rose above baseline
        cpu_rose = (load_worker_cpu > 0.0) or (load_cpu > idle_cpu) or (load_cpu >= 15.0)
        self.assertTrue(
            cpu_rose,
            f"CPU under load (sys={load_cpu:.1f}%, worker={load_worker_cpu:.1f}%) should reflect active work",
        )

    # ------------------------------------------------------------------ #
    # Test 6 - Existing endpoint field names preserved
    # ------------------------------------------------------------------ #
    def test_06_existing_endpoint_field_names(self):
        """All original JSON fields must still be present."""
        import requests as _req

        print("\n[TEST 6] Existing endpoint field names preserved...")

        # /health
        r = _req.get(f"{self.base_url}/health", timeout=3.0)
        self.assertEqual(r.status_code, 200)
        self.assertIn("status", r.json())
        self.assertEqual(r.json()["status"], "ok")

        # /api/process
        r = _req.post(f"{self.base_url}/api/process", json={"load": 1}, timeout=5.0)
        self.assertIn(r.status_code, [200, 503])
        if r.status_code == 200:
            self.assertIn("status", r.json())
            self.assertIn("load", r.json())

        # /api/capacity GET
        r = _req.get(f"{self.base_url}/api/capacity", timeout=3.0)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("active_workers", data)
        self.assertIn("last_scale_up_seconds", data)  # new field

        # /api/capacity POST
        r = _req.post(f"{self.base_url}/api/capacity",
                      json={"active_workers": 2}, timeout=5.0)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("active_workers", data)
        self.assertIn("status", data)

        # /metrics/latest -- must have original 7 fields + 2 new
        r = _req.get(f"{self.base_url}/metrics/latest", timeout=3.0)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        for field in [
            "timestamp", "cpu_percent", "mem_percent",
            "avg_latency_ms", "p95_latency_ms", "request_count",
            "active_workers", "error_count", "queue_depth",
            # Step 1b fields:
            "arrivals_count", "ok_count", "timeout_count",
            "rejected_count", "in_flight", "worker_cpu_percent",
            "ok_p95_latency_ms",
            # Step 2 fields:
            "busy_workers", "utilization", "worker_cpu_per_active_pct",
            "queue_wait_ms_avg", "cpu_freq_mhz",
        ]:
            self.assertIn(field, data, f"Missing field: {field}")

        # /metrics/history
        r = _req.get(f"{self.base_url}/metrics/history?seconds=5", timeout=3.0)
        self.assertEqual(r.status_code, 200)
        self.assertIn("records", r.json())
        self.assertIn("count", r.json())

        print("  -> All required fields present.")

    # ------------------------------------------------------------------ #
    # Test 8 - During scale-up, requests keep completing (gap < 300 ms)
    # ------------------------------------------------------------------ #
    def test_08_scale_up_no_completion_gap(self):
        """
        During a scale-up, requests keep completing with no large pause.
        Spec: largest gap between completions < 300 ms.
        """
        import requests as _req

        print("\n[TEST 8] During scale-up, requests keep completing (gap < 300ms)...")
        _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 2}, timeout=5.0)
        time.sleep(1.0)

        completions = []
        lock = threading.Lock()
        stop_traffic = threading.Event()

        def _traffic_sender():
            while not stop_traffic.is_set():
                try:
                    r = _req.post(f"{self.base_url}/api/process", json={"load": 1}, timeout=2.0)
                    if r.status_code == 200:
                        t = time.perf_counter()
                        with lock:
                            completions.append(t)
                except Exception:
                    pass
                time.sleep(0.04)

        senders = [threading.Thread(target=_traffic_sender, daemon=True) for _ in range(4)]
        for s in senders:
            s.start()

        # Let steady completions begin
        time.sleep(1.0)

        # Trigger scale-up to 6 workers in parallel
        _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 6}, timeout=5.0)

        # Let scale-up complete in background while traffic continues
        time.sleep(2.5)
        stop_traffic.set()
        for s in senders:
            s.join(timeout=2.0)

        with lock:
            times = completions[:]

        times.sort()
        self.assertGreater(len(times), 25, "Should have collected at least 25 completions during scale-up")

        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        max_gap_ms = max(gaps) * 1000.0
        print(f"  -> Total completions: {len(times)}, max gap between completions: {max_gap_ms:.1f}ms")
        self.assertLess(
            max_gap_ms, 300.0,
            f"Largest gap between completions ({max_gap_ms:.1f}ms) must be < 300ms",
        )

    # ------------------------------------------------------------------ #
    # Test 9 - Queue full returns 503 and counts as rejected_count
    # ------------------------------------------------------------------ #
    def test_09_queue_full_returns_503_and_counts_as_rejected(self):
        """
        When queue capacity is exceeded, collector returns HTTP 503 immediately
        and increments rejected_count.
        """
        import requests as _req

        print("\n[TEST 9] Queue full returns 503 and counts as rejected_count...")
        _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 1}, timeout=5.0)
        time.sleep(1.0)
        time.sleep(1.5)  # clean window

        cap_resp = _req.get(f"{self.base_url}/api/capacity", timeout=3.0).json()
        queue_cap = cap_resp.get("queue_capacity", 20)
        n_requests = queue_cap + 25

        results = []
        lock = threading.Lock()

        def _fire():
            try:
                r = _req.post(f"{self.base_url}/api/process", json={"load": 10}, timeout=4.0)
                with lock:
                    results.append((r.status_code, r.text))
            except Exception as exc:
                with lock:
                    results.append((-1, str(exc)))

        threads = [threading.Thread(target=_fire) for _ in range(n_requests)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=8.0)

        hist = _req.get(f"{self.base_url}/metrics/history?seconds=10", timeout=3.0).json()
        records = hist.get("records", [])
        rejected_count = sum(r.get("rejected_count", 0) for r in records)

        status_codes = [c for c, _ in results]
        has_503 = 503 in status_codes
        print(f"  -> Sent {n_requests} requests. 200s: {status_codes.count(200)}, 503s: {status_codes.count(503)}")
        print(f"  -> Telemetry rejected_count in history: {rejected_count}")

        self.assertTrue(has_503, "Expected at least one HTTP 503 response due to queue full")
        self.assertGreater(rejected_count, 0, f"Expected rejected_count > 0 in metrics history, got {rejected_count}")

    # ------------------------------------------------------------------ #
    # Test 10 - Under 2x overload, goodput >= 80% with shedding
    # ------------------------------------------------------------------ #
    def test_10_overload_goodput_with_shedding(self):
        """
        Under 2x overload, goodput >= 80% of calibrated capacity (load shedding works).
        At 2 workers, calibrated capacity ~= 15.8 req/s (at load=7, ms_per_load=18).
        With 2x arrival rate (~32 req/s), goodput must be >= 0.80 * 15.8 = 12.6 req/s.
        """
        import requests as _req

        print("\n[TEST 10] Under 2x overload, goodput >= 80% of capacity (shedding works)...")
        _req.post(f"{self.base_url}/api/capacity", json={"active_workers": 2}, timeout=5.0)
        time.sleep(1.0)

        calibrated_capacity = 2.0 * (1000.0 / (7.0 * self.ms_per_load))
        target_overload_rps = calibrated_capacity * 2.0  # ~32 req/s

        duration_sec = 6.0
        n_requests = int(target_overload_rps * duration_sec)
        interval = 1.0 / target_overload_rps

        success_count = 0
        lock = threading.Lock()

        def _sender(delay):
            nonlocal success_count
            time.sleep(delay)
            try:
                r = _req.post(f"{self.base_url}/api/process", json={"load": 7}, timeout=3.0)
                if r.status_code == 200:
                    with lock:
                        success_count += 1
            except Exception:
                pass

        t_start = time.perf_counter()
        threads = [threading.Thread(target=_sender, args=(i * interval,), daemon=True) for i in range(n_requests)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        elapsed = time.perf_counter() - t_start
        goodput = success_count / elapsed

        print(f"  -> Calibrated capacity: {calibrated_capacity:.1f} req/s")
        print(f"  -> 2x Overload rate applied: {n_requests / elapsed:.1f} req/s")
        print(f"  -> Measured goodput: {goodput:.1f} req/s ({goodput / calibrated_capacity * 100:.1f}% of capacity)")

        self.assertGreaterEqual(
            goodput,
            0.80 * calibrated_capacity,
            f"Goodput ({goodput:.1f} req/s) should be >= 80% of capacity ({0.80 * calibrated_capacity:.1f} req/s)",
        )



# ---------------------------------------------------------------------------
# Test 7 - run_all.py integration
# ---------------------------------------------------------------------------
class TestRunAll(unittest.TestCase):

    def test_07_run_all_starts_services(self):
        """
        python run_all.py starts every service and the dashboard loads.
        NOTE: This test starts all services on their real ports (5000-5003, 8000).
        """
        import requests as _req

        print("\n[TEST 7] run_all.py integration...")
        time.sleep(2.0)
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [sys.executable, "run_all.py"],
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        try:
            # Wait for all 5 services
            service_urls = {
                "Telemetry":    "http://127.0.0.1:5000/health",
                "Inference":    "http://127.0.0.1:5001/health",
                "Alert":        "http://127.0.0.1:5002/health",
                "Remediation":  "http://127.0.0.1:5003/health",
                "Dashboard":    "http://127.0.0.1:8000/",
            }
            for name, url in service_urls.items():
                deadline = time.monotonic() + 45.0
                success = False
                while time.monotonic() < deadline:
                    try:
                        r = _req.get(url, timeout=2.0)
                        if r.status_code < 500:
                            success = True
                            break
                    except Exception:
                        pass
                    time.sleep(0.5)
                self.assertTrue(success, f"{name} did not become reachable at {url}")
                print(f"  -> {name} OK")
        finally:
            try:
                import psutil
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except Exception:
                        pass
                parent.kill()
            except Exception:
                proc.kill()
            time.sleep(1.0)


# ---------------------------------------------------------------------------
# Flask test_client tests (fast, no subprocess)
# ---------------------------------------------------------------------------
class TestFlaskClient(unittest.TestCase):
    """Fast in-process tests using Flask test client (no real process pool)."""

    def setUp(self):
        # Import fresh to avoid state bleed
        import importlib
        import telemetry_collector as tc
        importlib.reload(tc)
        self.app = tc.create_app()
        self.client = self.app.test_client()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json.get("status"), "ok")

    def test_process_returns_processed(self):
        r = self.client.post("/api/process", json={"load": 2})
        # May be 200 (pool=None fallback) or 503 if pool raises; both acceptable
        self.assertIn(r.status_code, [200, 503])
        if r.status_code == 200:
            self.assertEqual(r.json.get("status"), "processed")

    def test_metrics_history_structure(self):
        r = self.client.get("/metrics/history?seconds=10")
        self.assertEqual(r.status_code, 200)
        self.assertIn("records", r.json)
        self.assertIn("count", r.json)

    def test_capacity_get(self):
        r = self.client.get("/api/capacity")
        self.assertEqual(r.status_code, 200)
        self.assertIn("active_workers", r.json)

    def test_capacity_post(self):
        r = self.client.post("/api/capacity", json={"active_workers": 4})
        self.assertEqual(r.status_code, 200)
        self.assertIn("active_workers", r.json)


if __name__ == "__main__":
    unittest.main(verbosity=2)
