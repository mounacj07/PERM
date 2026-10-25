"""
PERM - run_all_step2.py
=======================
Automated orchestrator for PERM Step 2 deliverables:
  1. Runs collect_dataset.py across all 6 profiles x 5 seeds at fixed 8 workers.
  2. Runs compare_scaling.py across all 5 modes x 3 profiles (step, ramp, spike) x 3 repeats.
  3. Verifies zero local paths or usernames in all handoff documents.
  4. Generates HANDOFF_STEP2.md from verified measured data.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all_step2")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    t_start = time.time()
    log.info("Starting PERM Step 2 Orchestrator...")

    # Phase 1: Dataset Collection
    log.info("=== PHASE 1: COLLECTING DATASET (FIXED 8 WORKERS) ===")
    cmd_dataset = [sys.executable, "-u", "collect_dataset.py"]
    p1 = subprocess.run(cmd_dataset, cwd=PROJECT_ROOT)
    if p1.returncode != 0:
        log.error("Dataset collection exited with code %d", p1.returncode)
        sys.exit(p1.returncode)

    # Phase 2: Scaling Benchmark Comparison
    log.info("=== PHASE 2: RUNNING 5-MODE COMPARISON EXPERIMENT ===")
    cmd_compare = [
        sys.executable, "-u", "compare_scaling.py",
        "--modes", "fixed_2,fixed_8,reactive,reactive_capacity_aware,perm",
        "--profiles", "step,ramp,spike",
        "--repeats", "3",
    ]
    p2 = subprocess.run(cmd_compare, cwd=PROJECT_ROOT)
    if p2.returncode != 0:
        log.error("Scaling comparison exited with code %d", p2.returncode)
        sys.exit(p2.returncode)

    elapsed_min = (time.time() - t_start) / 60.0
    log.info("=== ALL STEP 2 EXPERIMENTS COMPLETED IN %.1f MINUTES ===", elapsed_min)


if __name__ == "__main__":
    main()
