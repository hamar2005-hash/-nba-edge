"""Small, transparent probability helpers used by NBA Edge."""

import math


def probability_at_least(mean: float, residual_sd: float, cutoff: int) -> float:
    """Approximate P(count >= cutoff) with a continuity-corrected normal tail.

    The residual SD must come from the market's held-out model errors. This is
    an approximation for ranking, not a calibrated player-level probability.
    """
    sd = max(float(residual_sd), 0.5)
    boundary = float(cutoff) - 0.5
    z = (boundary - float(mean)) / (sd * math.sqrt(2.0))
    probability = 0.5 * math.erfc(z)
    return min(max(probability, 0.0), 1.0)
