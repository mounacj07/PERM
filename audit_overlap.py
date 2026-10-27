"""
audit_overlap.py
================
F4 Overlap audit:
Checks all run_command executions during the collection session
against the start/end timestamps of all 48 runs.
"""

import json
import os
import glob
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
user_home = os.environ.get("USERPROFILE", os.path.expanduser("~"))
TRANSCRIPT_PATH = os.path.join(user_home, ".gemini", "antigravity-ide", "brain", "417d6938-36a4-4c17-b5b2-3bc3c54aa8e5", ".system_generated", "logs", "transcript.jsonl")


def parse_iso(ts_str):
    if not ts_str:
        return None
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


def main():
    # 1. Load run start and end times from CSVs
    run_times = {}
    csv_files = glob.glob(os.path.join(RUNS_DIR, "*.csv"))
    run_csvs = [f for f in csv_files if not f.endswith("_stats.csv") and not f.endswith("_failures.csv") and not f.endswith("_exceptions.csv") and not f.endswith("_stats_history.csv")]

    for fpath in run_csvs:
        run_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            df = pd.read_csv(fpath)
            if "timestamp" in df.columns and len(df) > 0:
                t_start = parse_iso(df["timestamp"].iloc[0])
                t_end = parse_iso(df["timestamp"].iloc[-1])
                run_times[run_id] = {
                    "csv": os.path.basename(fpath),
                    "start": t_start,
                    "end": t_end,
                    "rows": len(df),
                }
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    # 2. Extract run_command tool calls from transcript
    executed_commands = []
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                created_at = parse_iso(obj.get("created_at"))
                for tc in obj.get("tool_calls", []):
                    if tc.get("name") == "run_command":
                        cmd = tc.get("args", {}).get("CommandLine", "")
                        # Ignore the collection command itself
                        if "collect_dataset.py --generator locust" in cmd and "--profiles" not in cmd:
                            continue
                        executed_commands.append({
                            "step_index": obj.get("step_index"),
                            "time": created_at,
                            "cmd": cmd,
                        })
            except Exception:
                pass

    print(f"Total auxiliary commands identified in session: {len(executed_commands)}")

    # 3. Check overlaps
    # For each run, see if any command's execution timestamp falls within [run.start, run.end]
    overlapped_runs = {}
    for run_id, rinfo in sorted(run_times.items(), key=lambda x: x[1]["start"] if x[1]["start"] else datetime.min):
        r_start = rinfo["start"]
        r_end = rinfo["end"]
        if not r_start or not r_end:
            continue

        overlapping_cmds = []
        for cmd_info in executed_commands:
            c_time = cmd_info["time"]
            if not c_time:
                continue
            # Check if command executed between run start and end (with 1s margin)
            if r_start <= c_time <= r_end:
                overlapping_cmds.append(cmd_info)

        if overlapping_cmds:
            overlapped_runs[run_id] = {
                "rinfo": rinfo,
                "cmds": overlapping_cmds,
            }

    print("\n" + "=" * 80)
    print(f"OVERLAP AUDIT RESULTS: {len(overlapped_runs)} of {len(run_times)} runs overlapped with commands")
    print("=" * 80)

    for run_id, oinfo in overlapped_runs.items():
        r = oinfo["rinfo"]
        print(f"\nRun: {run_id}")
        print(f"  Window: {r['start'].strftime('%H:%M:%S')} to {r['end'].strftime('%H:%M:%S')} UTC")
        print(f"  Overlapping commands ({len(oinfo['cmds'])}):")
        for c in oinfo["cmds"]:
            print(f"    [{c['time'].strftime('%H:%M:%S')} UTC] (Step {c['step_index']}): {c['cmd'][:70]}")


if __name__ == "__main__":
    main()
