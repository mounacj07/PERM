"""
PERM - ramp_burst.py
====================
Workload profile with a precursor ramp followed by a high burst:
  - 30s calm @ 4 req/s
  - 20s precursor ramp: 4 -> 20 req/s
  - 20s burst jump @ 45 req/s
  - 20s calm return @ 4 req/s
Total duration = 90.0s

Note: Created as a standalone module to preserve immutability boundaries of burst_test.py.
Card text: "A step with no precursor cannot be forecast from history; ramp_burst has one."
"""

from __future__ import annotations

import logging
import random
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ramp_burst")


def generate_schedule(profile: str = "ramp_burst", seed: int = 42) -> Tuple[List[float], float, float]:
    """
    Generate schedule of request dispatch offsets (in seconds from t0) using a Poisson process.
    Returns:
        (dispatch_offsets, burst_start_offset, total_duration)
    """
    rng = random.Random(seed)
    offsets: List[float] = []

    # Stages definition:
    # 1. 30s calm @ 4 req/s
    # 2. 20s ramp 4 -> 20 req/s (precursor ramp)
    # 3. 20s jump @ 45 req/s
    # 4. 20s calm return @ 4 req/s
    stages = [
        (30.0, 4.0, 4.0, False),
        (20.0, 4.0, 20.0, True),
        (20.0, 45.0, 45.0, False),
        (20.0, 4.0, 4.0, False),
    ]

    current_t = 0.0
    stage_start_t = 0.0
    burst_start_offset = 30.0
    total_duration = sum(d for d, _, _, _ in stages)

    for dur, r_start, r_end, is_burst in stages:
        stage_end = stage_start_t + dur
        if is_burst:
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
    return offsets, burst_start_offset, total_duration
