"""
PERM - Predictive Edge Resource Management
Database Persistence Layer (SQLite)

Provides a lightweight, zero-configuration embedded time-series database (`perm_data.db`)
for thread-safe persistence across edge microservices.

Tables:
  1. telemetry       - 1s telemetry metrics
  2. phase_log       - Traffic phase transitions
  3. predictions     - 30s ML forecast history
  4. alerts          - Proactive SLO alerts & lead times
  5. remediation_log - Scaling actions & audit trail

Run:
    python database.py
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = "perm_data.db"
_lock = threading.Lock()
_local = threading.local()  # per-thread connection cache

log = logging.getLogger("database")


def _get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Return a cached per-thread SQLite connection with resilient timeout.
    """
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        _local.conn = conn
    return conn


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Public alias for backward compatibility."""
    return _get_conn(db_path)


def init_db(db_path: str = DB_PATH):
    """Creates database tables if they do not exist."""
    with _lock:
        conn = sqlite3.connect(db_path, timeout=15.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            cursor = conn.cursor()

            # 1. Telemetry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    mem_percent REAL,
                    avg_latency_ms REAL,
                    p95_latency_ms REAL,
                    request_count INTEGER,
                    error_count INTEGER
                );
            """)

            # 2. Phase Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS phase_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    cycle INTEGER,
                    target_users INTEGER,
                    elapsed_sec REAL
                );
            """)

            # 3. Predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    target_timestamp_30s TEXT,
                    current_cpu REAL,
                    current_latency_ms REAL,
                    predicted_cpu_30s REAL,
                    predicted_latency_30s REAL
                );
            """)

            # 4. Alerts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    current_value REAL,
                    forecast_value_30s REAL,
                    lead_time_sec REAL
                );
            """)

            # 5. Remediation Audit Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS remediation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    action TEXT NOT NULL,
                    workers_before INTEGER,
                    workers_after INTEGER,
                    approved_by TEXT
                );
            """)

            conn.commit()
            print(f"[Database] SQLite database initialized at {os.path.abspath(db_path)}")
        finally:
            conn.close()


def record_telemetry_sample(sample: dict, db_path: str = DB_PATH):
    """Inserts a single 1s telemetry row into SQLite with retry."""
    with _lock:
        for _ in range(3):
            try:
                conn = _get_conn(db_path)
                conn.execute("""
                    INSERT INTO telemetry (timestamp, cpu_percent, mem_percent, avg_latency_ms, p95_latency_ms, request_count, error_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    sample.get("cpu_percent", 0.0),
                    sample.get("mem_percent", 0.0),
                    sample.get("avg_latency_ms", 0.0),
                    sample.get("p95_latency_ms", 0.0),
                    sample.get("request_count", 0),
                    sample.get("error_count", 0),
                ))
                conn.commit()
                break
            except sqlite3.OperationalError:
                pass


def record_alert_event(alert: dict, db_path: str = DB_PATH):
    """Inserts a proactive SLO alert into SQLite with retry."""
    with _lock:
        for _ in range(3):
            try:
                conn = _get_conn(db_path)
                conn.execute("""
                    INSERT INTO alerts (timestamp, metric, severity, current_value, forecast_value_30s, lead_time_sec)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    alert.get("metric", "cpu_percent"),
                    alert.get("severity", "CRITICAL"),
                    alert.get("current_value", 0.0),
                    alert.get("forecast_value_30s", 0.0),
                    alert.get("lead_time_sec", 25.5),
                ))
                conn.commit()
                break
            except sqlite3.OperationalError:
                pass


def record_remediation_event(event: dict, db_path: str = DB_PATH):
    """Inserts a scaling action into SQLite with retry."""
    with _lock:
        for _ in range(3):
            try:
                conn = _get_conn(db_path)
                conn.execute("""
                    INSERT INTO remediation_log (timestamp, mode, action, workers_before, workers_after, approved_by)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    event.get("mode", "AUTO"),
                    event.get("action", "SCALE_UP"),
                    event.get("workers_before", 2),
                    event.get("workers_after", 4),
                    event.get("approved_by", "SYSTEM_AUTO"),
                ))
                conn.commit()
                break
            except sqlite3.OperationalError:
                pass


if __name__ == "__main__":
    init_db()
