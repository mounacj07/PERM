"""
move_and_rerun_overlapped.py
============================
F4: Move overlapped runs to data/runs_locust/_superseded/
and prepare manifest.json for re-running them cleanly.
"""

import os
import glob
import shutil
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
SUPERSEDED_DIR = os.path.join(RUNS_DIR, "_superseded")
MANIFEST_PATH = os.path.join(RUNS_DIR, "manifest.json")

OVERLAPPED_RUN_IDS = [
    "step_seed43",
    "step_seed44",
    "ramp_seed42",
    "ramp_seed43",
    "random_mix_seed5",
    "random_mix_seed6",
]


def main():
    os.makedirs(SUPERSEDED_DIR, exist_ok=True)
    print(f"Created superseded directory: {SUPERSEDED_DIR}")

    moved_files = []
    for run_id in OVERLAPPED_RUN_IDS:
        # Match run_id.csv and run_id_locust_stats_*.csv
        patterns = [
            os.path.join(RUNS_DIR, f"{run_id}.csv"),
            os.path.join(RUNS_DIR, f"{run_id}_locust_stats_*.csv"),
        ]
        for p in patterns:
            for fpath in glob.glob(p):
                dest = os.path.join(SUPERSEDED_DIR, os.path.basename(fpath))
                shutil.move(fpath, dest)
                moved_files.append(os.path.basename(fpath))
                print(f"Moved {os.path.basename(fpath)} -> _superseded/")

    print(f"\nMoved {len(moved_files)} files to _superseded/.")

    # Update manifest.json: mark these 6 runs as completed: false
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)

        comp_count = 0
        for r in m.get("runs", []):
            if r["run_id"] in OVERLAPPED_RUN_IDS:
                r["completed"] = False
                r["rows_logged"] = 0
                r["superseded"] = True
            if r.get("completed"):
                comp_count += 1

        m["completed_runs"] = comp_count
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

        print(f"Updated manifest.json: completed_runs set to {comp_count}/48.")


if __name__ == "__main__":
    main()
