"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from eats.environments.types import EnvironmentSpec
from typing import Any
import numpy as np


def _segment_lengths_with_minimum(
    *,
    horizon: int,
    n_segments: int,
    min_segment: int,
    rng: np.random.Generator,
) -> np.ndarray:
    required = n_segments * min_segment
    if required > horizon:
        raise ValueError("horizon is too small for the requested event spacing.")
    lengths = np.full(n_segments, int(min_segment), dtype=np.int64)
    slack = int(horizon - required)
    if slack > 0:
        lengths += rng.multinomial(slack, np.full(n_segments, 1.0 / float(n_segments)))
    return lengths


def build_piecewise_stationary_k32_environment(
    *,
    horizon: int = 20_000,
    n_arms: int = 32,
    seed: int = 20260426,
    min_events: int = 3,
    max_events: int = 5,
    min_segment: int | None = None,
    affected_arms_per_event: int = 8,
    key: str | None = None,
    name: str | None = None,
) -> EnvironmentSpec:
    """
    Controlled K=32 piecewise-stationary benchmark.

    Each event is local: the incumbent oracle arm, the new oracle arm, and a
    small subset of distractors jump abruptly while the rest of the arms remain
    unchanged. The oracle arm is forced to change at every registered event.
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_arms < 4:
        raise ValueError("n_arms must be >= 4.")
    if min_events < 0:
        raise ValueError("min_events must be >= 0.")
    if max_events < min_events:
        raise ValueError("max_events must be >= min_events.")
    if affected_arms_per_event < 2:
        raise ValueError("affected_arms_per_event must be >= 2.")
    affected_arms_per_event = min(int(affected_arms_per_event), max(2, n_arms - 1))

    rng = np.random.default_rng(seed)
    max_feasible_events = min(max_events, n_arms - 1)
    if min_events > max_feasible_events:
        raise ValueError("min_events is too large for n_arms when oracle arms must be unique.")
    n_events = int(rng.integers(min_events, max_feasible_events + 1))
    if min_segment is None:
        min_segment = max(20, horizon // 10)
    lengths = _segment_lengths_with_minimum(
        horizon=horizon,
        n_segments=n_events + 1,
        min_segment=min_segment,
        rng=rng,
    )
    event_times = np.cumsum(lengths[:-1]).astype(int).tolist()

    means = np.zeros((horizon, n_arms), dtype=float)
    current = rng.uniform(0.12, 0.58, size=n_arms)
    oracle_path = rng.choice(n_arms, size=n_events + 1, replace=False).astype(int).tolist()
    current[oracle_path[0]] = float(rng.uniform(0.80, 0.90))
    for challenger in oracle_path[1:]:
        current[challenger] = float(rng.uniform(0.35, 0.62))

    events: list[dict[str, Any]] = []
    start = 0
    current_oracle = int(oracle_path[0])
    for event_idx, segment_len in enumerate(lengths):
        end = start + int(segment_len)
        means[start:end] = current
        if event_idx >= n_events:
            break

        event_time = int(end)
        next_oracle = int(oracle_path[event_idx + 1])
        candidate_pool = [arm for arm in range(n_arms) if arm not in {current_oracle, next_oracle}]
        distractor_count = max(0, min(affected_arms_per_event - 2, len(candidate_pool)))
        distractors = (
            rng.choice(candidate_pool, size=distractor_count, replace=False).astype(int).tolist()
            if distractor_count > 0
            else []
        )
        affected = [current_oracle, next_oracle, *distractors]
        before = {str(arm): float(current[arm]) for arm in affected}

        updated = current.copy()
        updated[current_oracle] = float(rng.uniform(0.25, 0.55))
        updated[next_oracle] = float(rng.uniform(0.82, 0.93))
        for arm in distractors:
            direction = -1.0 if rng.random() < 0.5 else 1.0
            jump = direction * float(rng.uniform(0.12, 0.30))
            updated[arm] = float(np.clip(updated[arm] + jump, 0.05, 0.78))
        current = np.clip(updated, 0.0, 1.0)
        after = {str(arm): float(current[arm]) for arm in affected}

        events.append(
            {
                "event_index": event_idx,
                "time": event_time,
                "pre_oracle_arm": current_oracle,
                "post_oracle_arm": next_oracle,
                "affected_arms": affected,
                "affected_fraction": float(len(affected) / n_arms),
                "pre_means": before,
                "post_means": after,
            }
        )
        current_oracle = next_oracle
        start = end

    actual_oracle = np.argmax(means, axis=1)
    actual_changes = np.flatnonzero(actual_oracle[1:] != actual_oracle[:-1]) + 1
    metadata = {
        "benchmark_type": "synthetic_piecewise_stationary_local",
        "pre_registered_protocol": {
            "horizon": horizon,
            "n_arms": n_arms,
            "min_events": min_events,
            "max_events": max_events,
            "affected_arms_per_event": affected_arms_per_event,
            "min_segment": min_segment,
            "oracle_change_forced_at_each_registered_event": True,
        },
        "seed": seed,
        "event_times": event_times,
        "change_events": events,
        "oracle_argmax_change_times": actual_changes.astype(int).tolist(),
        "oracle_argmax_changes": int(actual_changes.size),
    }
    return EnvironmentSpec(
        key=key or f"piecewise_k{n_arms}_seed{seed}",
        name=name or f"Piecewise stationary local shocks (K={n_arms}, seed={seed})",
        means_tk=means,
        arm_labels=tuple(f"Arm {arm}" for arm in range(n_arms)),
        metadata=metadata,
    )
