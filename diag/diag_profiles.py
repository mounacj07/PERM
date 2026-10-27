"""
diag/diag_profiles.py
=====================
Traffic profile definitions for warm-up / scaling diagnostics.
Provides:
  - flat_4: 4 req/s for 120s
  - step_late: 4 req/s for 60s, then 40 req/s for 60s
  - ramp_late: 4 req/s for 60s, ramp 4->40 over 30s, then hold 30s at 40 req/s
  - step, ramp, etc.: delegated to burst_test.generate_schedule
"""

from __future__ import annotations

import random
from typing import List, Tuple
import burst_test

_ORIGINAL_BURST_TEST_SCHEDULE = burst_test.generate_schedule


def generate_schedule(profile: str, seed: int) -> Tuple[List[float], float, float]:
    """
    Generate schedule of request dispatch offsets (in seconds from t0) using a Poisson process.
    Returns:
        (dispatch_offsets, burst_start_offset, total_duration)
    """
    if profile in ["step", "ramp", "spike", "double_burst", "sawtooth", "random_walk", "random_mix"]:
        return _ORIGINAL_BURST_TEST_SCHEDULE(profile, seed)

    rng = random.Random(seed)
    offsets: List[float] = []

    if profile == "flat_4":
        # 4 req/s for 120s (no burst)
        stages = [
            (120.0, 4.0, 4.0, False),
        ]
    elif profile == "step_late":
        # 4 req/s for 60s, then 40 req/s for 60s
        stages = [
            (60.0, 4.0, 4.0, False),
            (60.0, 40.0, 40.0, True),
        ]
    elif profile == "ramp_late":
        # 4 req/s for 60s, ramp 4->40 over 30s, then hold 30s at 40 req/s (total 120s)
        stages = [
            (60.0, 4.0, 4.0, False),
            (30.0, 4.0, 40.0, True),
            (30.0, 40.0, 40.0, False),
        ]
    else:
        raise ValueError(f"Unknown diagnostic profile: {profile}")

    current_t = 0.0
    stage_start_t = 0.0
    burst_start_offset = 9999.0
    total_duration = sum(d for d, _, _, _ in stages)

    for dur, r_start, r_end, is_burst in stages:
        stage_end = stage_start_t + dur
        if is_burst and burst_start_offset == 9999.0:
            burst_start_offset = stage_start_t

        current_t = max(current_t, stage_start_t)
        while current_t < stage_end:
            if dur > 0:
                fraction = min(1.0, max(0.0, (current_t - stage_start_t) / dur))
            else:
                fraction = 1.0
            current_rate = r_start + fraction * (r_end - r_start)
            current_rate = max(0.1, current_rate)

            # Poisson arrival interval ~ Exponential(lambda)
            dt = rng.expovariate(current_rate)
            current_t += dt
            if current_t < stage_end:
                offsets.append(current_t)

        stage_start_t = stage_end

    offsets.sort()
    if burst_start_offset == 9999.0:
        burst_start_offset = total_duration

    return offsets, burst_start_offset, total_duration
