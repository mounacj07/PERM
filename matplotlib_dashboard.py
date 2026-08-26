"""
PERM - Predictive Edge Resource Management
Real-Time Matplotlib Operator Dashboard

Live graphical dashboard displaying real-time system metrics, 30s-ahead Random Forest
forecasts, SLO threshold boundaries, and proactive alerts with lead times.

Run:
    python matplotlib_dashboard.py
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import requests

# --------------------------------------------------------------------------
# Service URLs
# --------------------------------------------------------------------------
COLLECTOR_URL = "http://127.0.0.1:5000"
INFERENCE_URL = "http://127.0.0.1:5001"
ALERT_URL = "http://127.0.0.1:5002"
REMEDIATION_URL = "http://127.0.0.1:5003"


# --------------------------------------------------------------------------
# Dashboard Visualization Class
# --------------------------------------------------------------------------
class PermMatplotlibDashboard:
    def __init__(self):
        plt.style.use("dark_background")
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.suptitle(
            "PERM — Predictive Edge Resource Management (Real-Time Operator Dashboard)",
            fontsize=15, fontweight="bold", color="#00f2fe"
        )
        self.fig.patch.set_facecolor("#0b0f19")

        # Color Palette
        self.c_cpu_actual = "#38bdf8"    # Cyan
        self.c_cpu_pred = "#f43f5e"      # Rose / Red-pink
        self.c_lat_actual = "#4ade80"    # Green
        self.c_lat_pred = "#fbbf24"      # Amber
        self.c_warn = "#f59e0b"          # Warning Orange
        self.c_crit = "#ef4444"          # Critical Red

        for row in self.axs:
            for ax in row:
                ax.set_facecolor("#111827")
                ax.tick_params(colors="#9ca3af")
                for spine in ax.spines.values():
                    spine.set_color("#374151")

    def fetch_data(self):
        telemetry = []
        try:
            r = requests.get(f"{COLLECTOR_URL}/metrics/history?seconds=90", timeout=1.0)
            if r.status_code == 200:
                telemetry = r.json().get("records", [])
        except Exception:
            pass

        predictions = []
        try:
            r = requests.get(f"{INFERENCE_URL}/predictions/history?seconds=90", timeout=1.0)
            if r.status_code == 200:
                predictions = r.json().get("predictions", [])
        except Exception:
            pass

        alerts = []
        try:
            r = requests.get(f"{ALERT_URL}/alerts/history?count=10", timeout=1.0)
            if r.status_code == 200:
                alerts = r.json().get("alerts", [])
        except Exception:
            pass

        remediation = {}
        try:
            r = requests.get(f"{REMEDIATION_URL}/remediation/status", timeout=1.0)
            if r.status_code == 200:
                remediation = r.json()
        except Exception:
            pass

        return telemetry, predictions, alerts, remediation

    def update(self, frame):
        telemetry, predictions, alerts, remediation = self.fetch_data()

        ax_cpu = self.axs[0, 0]
        ax_lat = self.axs[0, 1]
        ax_req = self.axs[1, 0]
        ax_alt = self.axs[1, 1]

        ax_cpu.clear()
        ax_lat.clear()
        ax_req.clear()
        ax_alt.clear()

        # Re-apply dark subplot styling
        for ax in (ax_cpu, ax_lat, ax_req, ax_alt):
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#9ca3af")
            for spine in ax.spines.values():
                spine.set_color("#374151")

        # 1. CPU Plot
        ax_cpu.set_title("CPU Utilization (%) — Actual vs 30s Forecast", fontsize=11, color="#e5e7eb", loc="left")
        ax_cpu.set_ylim(0, 105)
        ax_cpu.axhline(70.0, color=self.c_warn, linestyle="--", alpha=0.7, label="SLO Warning (70%)")
        ax_cpu.axhline(85.0, color=self.c_crit, linestyle="--", alpha=0.8, label="SLO Critical (85%)")

        if telemetry:
            cpu_actuals = [r["cpu_percent"] for r in telemetry]
            x_act = list(range(len(cpu_actuals)))
            ax_cpu.plot(x_act, cpu_actuals, color=self.c_cpu_actual, linewidth=2, label="Telemetry (1s)")

            if predictions:
                pred_map = {}
                for p in predictions:
                    pred_map[p.get("timestamp")] = p.get("predicted_cpu_30s")
                
                # Match to telemetry timeline
                cpu_preds = []
                last_p = predictions[-1].get("predicted_cpu_30s", 0.0)
                for r in telemetry:
                    # Find closest prediction
                    t_str = r.get("timestamp")
                    match = pred_map.get(t_str)
                    if match is not None:
                        last_p = match
                    cpu_preds.append(last_p)
                
                ax_cpu.plot(x_act, cpu_preds, color=self.c_cpu_pred, linestyle=":", linewidth=2, label="RF Forecast (+30s)")

        ax_cpu.legend(loc="upper left", fontsize=8, facecolor="#1f2937", edgecolor="none")
        ax_cpu.grid(True, linestyle=":", alpha=0.2)

        # 2. Latency Plot
        ax_lat.set_title("Response Latency (ms) — Actual vs 30s Forecast", fontsize=11, color="#e5e7eb", loc="left")
        max_lat = 350
        if telemetry:
            lats = [r["avg_latency_ms"] for r in telemetry]
            if lats:
                max_lat = max(max_lat, max(lats) * 1.2)
        ax_lat.set_ylim(0, max_lat)
        ax_lat.axhline(150.0, color=self.c_warn, linestyle="--", alpha=0.7, label="SLO Warning (150ms)")
        ax_lat.axhline(300.0, color=self.c_crit, linestyle="--", alpha=0.8, label="SLO Critical (300ms)")

        if telemetry:
            lat_actuals = [r["avg_latency_ms"] for r in telemetry]
            ax_lat.plot(list(range(len(lat_actuals))), lat_actuals, color=self.c_lat_actual, linewidth=2, label="Telemetry (1s)")

            if predictions:
                pred_lat_map = {p.get("timestamp"): p.get("predicted_latency_30s") for p in predictions}
                lat_preds = []
                last_lp = predictions[-1].get("predicted_latency_30s", 0.0)
                for r in telemetry:
                    t_str = r.get("timestamp")
                    match = pred_lat_map.get(t_str)
                    if match is not None:
                        last_lp = match
                    lat_preds.append(last_lp)

                ax_lat.plot(list(range(len(lat_actuals))), lat_preds, color=self.c_lat_pred, linestyle=":", linewidth=2, label="RF Forecast (+30s)")

        ax_lat.legend(loc="upper left", fontsize=8, facecolor="#1f2937", edgecolor="none")
        ax_lat.grid(True, linestyle=":", alpha=0.2)

        # 3. Request Traffic Profile
        ax_req.set_title("Synthetic Workload RPS & Edge Worker Allocation", fontsize=11, color="#e5e7eb", loc="left")
        if telemetry:
            reqs = [r["request_count"] for r in telemetry]
            ax_req.bar(list(range(len(reqs))), reqs, color="#a855f7", alpha=0.6, width=0.8, label="Requests/sec")

        active_workers = remediation.get("active_workers", 2)
        mode = remediation.get("mode", "AUTO")
        ax_req.text(
            0.02, 0.88,
            f"Active Cores: {active_workers} | Mode: {mode}",
            transform=ax_req.transAxes,
            color="#38bdf8", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.8)
        )
        ax_req.grid(True, linestyle=":", alpha=0.2)

        # 4. Proactive Alert Log Panel
        ax_alt.axis("off")
        ax_alt.set_title("SLO Proactive Alert Stream (~25s Lead Time)", fontsize=11, color="#e5e7eb", loc="left")

        status_text = "SYSTEM NORMAL"
        status_color = "#22c55e"
        if alerts:
            latest = alerts[-1]
            if latest.get("severity") == "CRITICAL":
                status_text = "PROACTIVE CRITICAL ALERT"
                status_color = "#ef4444"
            elif latest.get("severity") == "WARNING":
                status_text = "PROACTIVE WARNING ALERT"
                status_color = "#f59e0b"

        # Active Status Badge
        ax_alt.text(
            0.02, 0.85, f"STATUS: {status_text}",
            transform=ax_alt.transAxes,
            color=status_color, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1f2937", edgecolor=status_color, alpha=0.9)
        )

        # Alert List
        y_pos = 0.68
        ax_alt.text(0.02, y_pos, "Recent Proactive Alerts:", transform=ax_alt.transAxes, color="#9ca3af", fontsize=9, fontweight="bold")
        y_pos -= 0.08

        recent_alerts = list(reversed(alerts[-4:])) if alerts else []
        if not recent_alerts:
            ax_alt.text(0.04, y_pos, "No active alerts. Forecast within SLO parameters.", transform=ax_alt.transAxes, color="#6b7280", fontsize=8.5, linestyle="italic")
        else:
            for alt in recent_alerts:
                sev = alt.get("severity", "WARN")
                c = "#ef4444" if sev == "CRITICAL" else "#f59e0b"
                msg = alt.get("message", "")
                if len(msg) > 65:
                    msg = msg[:62] + "..."
                ax_alt.text(0.04, y_pos, f"• [{sev}] {msg}", transform=ax_alt.transAxes, color=c, fontsize=8)
                y_pos -= 0.12

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def main():
    dashboard = PermMatplotlibDashboard()
    ani = animation.FuncAnimation(dashboard.fig, dashboard.update, interval=1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
