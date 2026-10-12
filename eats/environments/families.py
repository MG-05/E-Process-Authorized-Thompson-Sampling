"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from eats.algorithms.types import FloatArray
import numpy as np


def _validate_horizon_n_arms(
    horizon: int,
    n_arms: int,
    *,
    min_arms: int,
) -> None:
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_arms < min_arms:
        raise ValueError(f"n_arms must be >= {min_arms}.")


def _validate_prob(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")


def slow_varying_sinusoid_means(
    horizon: int = 5000,
    n_arms: int = 4,
    period: int = 1000,
    offsets: FloatArray | None = None,
) -> FloatArray:
    """
    Slow-varying sinusoidal means used in the paper's non-stationary setup.
    Mean rewards are constrained to [0, 1] via (sin(.) + 1) / 2.
    """

    _validate_horizon_n_arms(horizon, n_arms, min_arms=2)
    if period <= 0:
        raise ValueError("period must be > 0.")

    if offsets is None:
        offsets = np.linspace(0.0, 2.0 * np.pi, num=n_arms, endpoint=False)
    else:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (n_arms,):
            raise ValueError("offsets must have shape (n_arms,).")

    t = np.arange(horizon, dtype=float)
    phase = (2.0 * np.pi * t[:, None] / float(period)) + offsets[None, :]
    means = 0.5 + 0.5 * np.sin(phase)
    return np.clip(means, 0.0, 1.0)


def _segment_lengths_with_minimum(
    horizon: int,
    n_segments: int,
    min_segment: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    required = n_segments * min_segment
    if required > horizon:
        raise ValueError("horizon is too small for requested min_segment and breakpoints.")
    lengths = np.full(n_segments, min_segment, dtype=np.int64)
    slack = horizon - required
    if slack > 0:
        lengths += rng.multinomial(slack, np.full(n_segments, 1.0 / float(n_segments)))
    return lengths


def random_breakpoint_means(
    horizon: int = 5000,
    n_arms: int = 4,
    min_breakpoints: int = 3,
    max_breakpoints: int = 10,
    min_segment: int = 20,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
    min_jump: float = 0.08,
    seed: int | None = None,
) -> FloatArray:
    """
    Piecewise-constant benchmark with random breakpoint locations and levels.
    Useful for testing adaptation to abrupt regime changes.
    """

    _validate_horizon_n_arms(horizon, n_arms, min_arms=1)
    if min_breakpoints < 0:
        raise ValueError("min_breakpoints must be >= 0.")
    if max_breakpoints < min_breakpoints:
        raise ValueError("max_breakpoints must be >= min_breakpoints.")
    if min_segment <= 0:
        raise ValueError("min_segment must be > 0.")
    _validate_prob("min_mean", min_mean)
    _validate_prob("max_mean", max_mean)
    if max_mean <= min_mean:
        raise ValueError("max_mean must be > min_mean.")
    if min_jump < 0.0:
        raise ValueError("min_jump must be >= 0.")
    if min_jump > (max_mean - min_mean):
        raise ValueError("min_jump cannot exceed max_mean - min_mean.")
    if (max_breakpoints + 1) * min_segment > horizon:
        raise ValueError("horizon too short for max_breakpoints and min_segment.")

    rng = np.random.default_rng(seed)
    means = np.zeros((horizon, n_arms), dtype=float)

    for arm in range(n_arms):
        n_breaks = int(rng.integers(min_breakpoints, max_breakpoints + 1))
        n_segments = n_breaks + 1
        lengths = _segment_lengths_with_minimum(
            horizon=horizon,
            n_segments=n_segments,
            min_segment=min_segment,
            rng=rng,
        )

        levels = np.empty(n_segments, dtype=float)
        levels[0] = rng.uniform(min_mean, max_mean)
        for idx in range(1, n_segments):
            candidate = float(rng.uniform(min_mean, max_mean))
            attempts = 0
            while abs(candidate - levels[idx - 1]) < min_jump and attempts < 32:
                candidate = float(rng.uniform(min_mean, max_mean))
                attempts += 1
            if abs(candidate - levels[idx - 1]) < min_jump:
                prev = levels[idx - 1]
                if prev < 0.5 * (min_mean + max_mean):
                    candidate = min(max_mean, prev + min_jump)
                else:
                    candidate = max(min_mean, prev - min_jump)
            levels[idx] = candidate

        start = 0
        for seg_idx, seg_len in enumerate(lengths):
            end = start + int(seg_len)
            means[start:end, arm] = levels[seg_idx]
            start = end

    return np.clip(means, 0.0, 1.0)


def random_drift_amplitude_means(
    horizon: int = 5000,
    n_arms: int = 4,
    amplitude_min: float = 0.05,
    amplitude_max: float = 0.30,
    amplitude_step_std: float = 0.01,
    period_min: int = 80,
    period_max: int = 400,
    baseline_low: float = 0.20,
    baseline_high: float = 0.80,
    drift_step_std: float = 0.004,
    seed: int | None = None,
) -> FloatArray:
    """
    Smoothly non-stationary benchmark where sinusoid amplitude drifts over time.
    This avoids tuning to a single fixed-amplitude cartoon trajectory.
    """

    _validate_horizon_n_arms(horizon, n_arms, min_arms=1)
    if amplitude_min < 0.0:
        raise ValueError("amplitude_min must be >= 0.")
    if amplitude_max <= amplitude_min:
        raise ValueError("amplitude_max must be > amplitude_min.")
    if amplitude_step_std < 0.0:
        raise ValueError("amplitude_step_std must be >= 0.")
    if period_min <= 1:
        raise ValueError("period_min must be > 1.")
    if period_max < period_min:
        raise ValueError("period_max must be >= period_min.")
    _validate_prob("baseline_low", baseline_low)
    _validate_prob("baseline_high", baseline_high)
    if baseline_high <= baseline_low:
        raise ValueError("baseline_high must be > baseline_low.")
    if drift_step_std < 0.0:
        raise ValueError("drift_step_std must be >= 0.")

    rng = np.random.default_rng(seed)
    t_grid = np.arange(horizon, dtype=float)
    means = np.zeros((horizon, n_arms), dtype=float)

    for arm in range(n_arms):
        base = rng.uniform(baseline_low, baseline_high)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        period = float(rng.integers(period_min, period_max + 1))
        trend_scale = rng.uniform(0.02, 0.12)

        amp = np.empty(horizon, dtype=float)
        amp[0] = rng.uniform(amplitude_min, amplitude_max)
        if horizon > 1:
            amp_noise = rng.normal(loc=0.0, scale=amplitude_step_std, size=horizon - 1)
            for t in range(1, horizon):
                amp[t] = np.clip(amp[t - 1] + amp_noise[t - 1], amplitude_min, amplitude_max)

        trend = np.cumsum(rng.normal(loc=0.0, scale=drift_step_std, size=horizon))
        trend_norm = np.max(np.abs(trend))
        if trend_norm > 0.0:
            trend = trend / trend_norm

        means[:, arm] = base + amp * np.sin((2.0 * np.pi * t_grid / period) + phase) + trend_scale * trend

    return np.clip(means, 0.0, 1.0)


def global_switching_means(
    horizon: int = 2000,
    n_arms: int = 4,
    switch_prob: float = 0.02,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
    seed: int | None = None,
) -> FloatArray:
    """
    Global switching environment inspired by 1302.3721:
    all arms redraw their Bernoulli means at each changepoint.
    """
    _validate_horizon_n_arms(horizon, n_arms, min_arms=1)
    if not (0.0 <= switch_prob <= 1.0):
        raise ValueError("switch_prob must be in [0, 1].")
    if not (0.0 <= min_mean < max_mean <= 1.0):
        raise ValueError("Require 0 <= min_mean < max_mean <= 1.")

    rng = np.random.default_rng(seed)
    means = np.zeros((horizon, n_arms), dtype=float)

    current = rng.uniform(min_mean, max_mean, size=n_arms)
    means[0] = current
    for t in range(1, horizon):
        if rng.random() < switch_prob:
            current = rng.uniform(min_mean, max_mean, size=n_arms)
        means[t] = current
    return means


def abrupt_means(
    horizon: int = 1000,
    n_arms: int = 4,
    cycle_len: int = 250,
    change_times: list[int] | None = None,
    levels: list[float] | None = None,
) -> FloatArray:
    """Return the periodic abrupt environment used in Raj and Kalyani (2017)."""
    if change_times is None:
        change_times = [50, 100, 150, 200]
    if levels is None:
        levels = [0.10, 0.37, 0.63, 0.90]
    if len(change_times) != n_arms or len(levels) != n_arms:
        raise ValueError("change_times and levels must have length n_arms.")
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if cycle_len <= 0:
        raise ValueError("cycle_len must be > 0.")

    means = np.zeros((horizon, n_arms), dtype=float)
    for t in range(horizon):
        tau = t % cycle_len
        for arm in range(n_arms):
            if tau >= change_times[arm]:
                means[t, arm] = levels[arm]
    return means


def per_arm_switching_means(
    horizon: int = 2000,
    n_arms: int = 4,
    switch_prob: float = 0.02,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
    seed: int | None = None,
) -> FloatArray:
    """
    Per-arm switching environment inspired by 1302.3721:
    each arm independently redraws its Bernoulli mean at changepoints.
    """
    _validate_horizon_n_arms(horizon, n_arms, min_arms=1)
    if not (0.0 <= switch_prob <= 1.0):
        raise ValueError("switch_prob must be in [0, 1].")
    if not (0.0 <= min_mean < max_mean <= 1.0):
        raise ValueError("Require 0 <= min_mean < max_mean <= 1.")

    rng = np.random.default_rng(seed)
    means = np.zeros((horizon, n_arms), dtype=float)

    current = rng.uniform(min_mean, max_mean, size=n_arms)
    means[0] = current
    for t in range(1, horizon):
        switches = rng.random(n_arms) < switch_prob
        if np.any(switches):
            current = current.copy()
            current[switches] = rng.uniform(min_mean, max_mean, size=int(np.sum(switches)))
        means[t] = current
    return means
