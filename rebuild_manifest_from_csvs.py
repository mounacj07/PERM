"""
rebuild_manifest_from_csvs.py
=============================
Rebuilds data/runs_locust/manifest.json by reading all 48 completed CSV files
using collect_dataset.build_manifest.
"""

import os
import collect_dataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
MANIFEST_PATH = os.path.join(RUNS_DIR, "manifest.json")

def main():
    fixed_profiles = ["step", "ramp", "spike", "double_burst", "sawtooth", "random_walk"]
    fixed_seeds = [42, 43, 44]
    mix_seeds = list(range(30))

    all_run_specs = []

    # 1. Fixed profiles
    for prof in fixed_profiles:
        for s in fixed_seeds:
            dur = collect_dataset.get_profile_duration(prof, s)
            csv_path = os.path.join(RUNS_DIR, f"{prof}_seed{s}.csv")
            all_run_specs.append({
                "generator": "locust",
                "profile": prof,
                "seed": s,
                "duration": dur,
                "csv_path": csv_path,
                "completed": os.path.exists(csv_path),
            })

    # 2. Random mix
    for s in mix_seeds:
        dur = collect_dataset.get_profile_duration("random_mix", s)
        csv_path = os.path.join(RUNS_DIR, f"random_mix_seed{s}.csv")
        all_run_specs.append({
            "generator": "locust",
            "profile": "random_mix",
            "seed": s,
            "duration": dur,
            "csv_path": csv_path,
            "completed": os.path.exists(csv_path),
        })

    print(f"Constructed {len(all_run_specs)} run specs. Calling build_manifest...")
    collect_dataset.build_manifest(all_run_specs, MANIFEST_PATH, "locust")
    print("build_manifest finished.")


if __name__ == "__main__":
    main()
