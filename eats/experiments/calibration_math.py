"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from eats.algorithms.calibrated_glr import GeometricBernoulliGLRState
import math
import numpy as np


class GLRComparisonError(RuntimeError):
    """Raised when a calibration, identity, or policy invariant fails."""


def _xlogx_table(length: int) -> np.ndarray:
    indices = np.arange(length + 1, dtype=float)
    output = np.zeros(length + 1, dtype=float)
    output[1:] = indices[1:] * np.log(indices[1:])
    return output


def maximum_glr_on_trace(
    actions: np.ndarray, rewards: np.ndarray, n_arms: int
) -> float:
    """Maximum familywise GLR score on one fixed predictable trajectory."""

    played = np.asarray(actions, dtype=np.int64)
    outcomes = np.asarray(rewards, dtype=np.uint8)
    if (
        played.ndim != 1
        or played.size == 0
        or outcomes.shape != played.shape
        or np.any((played < 0) | (played >= n_arms))
        or np.any((outcomes != 0) & (outcomes != 1))
    ):
        raise ValueError("invalid action/reward trace")
    table = _xlogx_table(int(played.size))
    states = [
        GeometricBernoulliGLRState(xlogx_table=table) for _ in range(n_arms)
    ]
    maximum = 0.0
    for arm_value, reward_value in zip(played, outcomes, strict=True):
        maximum = max(
            maximum,
            states[int(arm_value)].update(int(reward_value)),
        )
    return maximum


def _matched_threshold(scores: np.ndarray, target_count: int) -> float:
    """Choose a strict interior cutoff with exactly ``target_count`` crossings."""

    values = np.sort(np.asarray(scores, dtype=float))[::-1]
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("calibration scores must be finite and nonempty")
    if not 0 <= target_count < values.size:
        raise ValueError("target crossing count is outside the calibratable range")
    if target_count == 0:
        threshold = float(np.nextafter(values[0], math.inf))
    else:
        upper = float(values[target_count - 1])
        lower = float(values[target_count])
        if not upper > lower:
            raise GLRComparisonError(
                "a GLR score tie prevents exact false-alarm count matching"
            )
        threshold = lower + 0.5 * (upper - lower)
    if int(np.count_nonzero(values >= threshold)) != target_count:
        raise GLRComparisonError("calibrated threshold does not match target count")
    return threshold
