"""
diag/locustfile_diag.py
=======================
Thin Locustfile wrapper for diagnostic runs.
Reuses the exact PermOpenLoopUser from locustfile_perm.py without modifying
locustfile_perm.py or burst_test.py.
Patches burst_test.generate_schedule to point to diag_profiles.generate_schedule
so diagnostic profiles (flat_4, step_late, ramp_late) can be loaded.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import burst_test
import diag.diag_profiles as diag_profiles

# Monkey-patch schedule generator to support diagnostic profiles
burst_test.generate_schedule = diag_profiles.generate_schedule

# Import and expose PermOpenLoopUser so Locust CLI detects it
from locustfile_perm import PermOpenLoopUser, _DISPATCHER_SEMAPHORE
