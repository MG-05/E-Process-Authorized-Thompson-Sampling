"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from eats.environments.families import abrupt_means
from eats.environments.families import global_switching_means
from eats.environments.families import per_arm_switching_means
from eats.environments.families import random_breakpoint_means
from eats.environments.families import random_drift_amplitude_means
from eats.environments.families import slow_varying_sinusoid_means
from eats.environments.local_shocks import build_piecewise_stationary_k32_environment
from eats.environments.types import EnvironmentSpec
from typing import Any
import numpy as np


SCALE_K_VALUES: tuple[int, ...] = (4, 32)


SCALE_FAMILY_ORDER: tuple[str, ...] = (
    "slow",
    "fast",
    "abrupt",
    "mixed",
    "random_breakpoints",
    "random_drift",
    "global_switching",
    "per_arm_switching",
    "local_shocks",
)


SCALE_FAMILY_NAMES: dict[str, str] = {
    "slow": "Slow sine",
    "fast": "Fast sine",
    "abrupt": "Abrupt cycle",
    "mixed": "Mixed regime",
    "random_breakpoints": "Random breakpoints",
    "random_drift": "Random drift amplitude",
    "global_switching": "Global switching",
    "per_arm_switching": "Per-arm switching",
    "local_shocks": "Forced-oracle local shocks",
}


PULLS_PER_ARM = 625


REFERENCE_K = 4


SCALE_ENVIRONMENT_DESIGN_ID = "synthetic_scale_suite_k4_k32_v1"


def scale_horizon(n_arms: int) -> int:
    if n_arms not in SCALE_K_VALUES:
        raise ValueError(f"Scale suite requires K in {SCALE_K_VALUES}.")
    return PULLS_PER_ARM * n_arms


def local_shock_affected_arms(n_arms: int) -> int:
    """Return max(2, round(0.25*K)) using explicit half-up rounding."""

    if n_arms < 2:
        raise ValueError("n_arms must be >= 2.")
    return max(2, int(np.floor(0.25 * n_arms + 0.5)))


def _scale_int(reference_value: int, n_arms: int) -> int:
    return max(1, int(round(reference_value * n_arms / REFERENCE_K)))


def _scaled_diffusion(reference_std: float, n_arms: int) -> float:
    return float(reference_std / np.sqrt(n_arms / REFERENCE_K))


def _oracle_change_events(means_tk: np.ndarray) -> list[dict[str, Any]]:
    oracle = np.argmax(means_tk, axis=1)
    times = np.flatnonzero(oracle[1:] != oracle[:-1]) + 1
    return [
        {
            "event_index": index,
            "time": int(time),
            "pre_oracle_arm": int(oracle[time - 1]),
            "post_oracle_arm": int(oracle[time]),
            "source": "oracle_argmax_change",
        }
        for index, time in enumerate(times)
    ]


def _row_change_events(means_tk: np.ndarray, *, source: str) -> list[dict[str, Any]]:
    changed = np.any(means_tk[1:] != means_tk[:-1], axis=1)
    times = np.flatnonzero(changed) + 1
    return [
        {"event_index": index, "time": int(time), "source": source}
        for index, time in enumerate(times)
    ]


def _scale_consistent_mixed_means(
    *,
    horizon: int,
    n_arms: int,
    seed: int,
) -> np.ndarray:
    """Repeat four declared arm roles instead of changing mechanism after arm 4."""

    if n_arms % 4:
        raise ValueError("The scale-consistent mixed family requires K divisible by 4.")
    rng = np.random.default_rng(seed)
    means = np.zeros((horizon, n_arms), dtype=float)
    group_count = n_arms // 4
    burst_probability = 0.20 / n_arms
    burst_duration = _scale_int(20, n_arms)
    switch_interval = _scale_int(200, n_arms)

    for group in range(group_count):
        fraction = (group + 0.5) / group_count
        stable_arm = 4 * group
        drift_arm = stable_arm + 1
        burst_arm = stable_arm + 2
        switch_arm = stable_arm + 3

        means[:, stable_arm] = 0.52 + 0.26 * fraction
        drift_low = 0.15 + 0.12 * fraction
        drift_high = 0.88 - 0.12 * fraction
        if group % 2:
            drift_low, drift_high = drift_high, drift_low
        means[:, drift_arm] = np.linspace(drift_low, drift_high, horizon)

        burst_base = 0.08 + 0.10 * fraction
        burst = np.full(horizon, burst_base, dtype=float)
        time = 0
        while time < horizon:
            if rng.random() < burst_probability:
                length = int(rng.integers(1, burst_duration + 1))
                end = min(horizon, time + length)
                burst[time:end] = float(rng.uniform(0.72, 0.96))
            time += 1
        means[:, burst_arm] = burst

        low = 0.12 + 0.08 * fraction
        high = 0.90 - 0.08 * fraction
        time = 0
        level = high if group % 2 else low
        while time < horizon:
            minimum = max(2, switch_interval // 2)
            maximum = max(minimum + 1, int(1.5 * switch_interval) + 1)
            length = int(rng.integers(minimum, maximum))
            end = min(horizon, time + length)
            means[time:end, switch_arm] = level
            level = low if level == high else high
            time = end

    return np.clip(means, 0.0, 1.0)


def _base_metadata(
    *,
    family: str,
    n_arms: int,
    horizon: int,
    seed: int,
    parameters: dict[str, Any],
    change_events: list[dict[str, Any]],
    preflight: bool,
) -> dict[str, Any]:
    return {
        "benchmark_type": "synthetic_scale_generalization",
        "environment_design_id": SCALE_ENVIRONMENT_DESIGN_ID,
        "family": family,
        "seed": int(seed),
        "scaling_contract": {
            "reference_k": REFERENCE_K,
            "pulls_per_arm": PULLS_PER_ARM,
            "horizon_rule": "T=625*K",
            "temporal_rule": (
                "periods and segment lengths multiply by K/4; hazards and "
                "per-step diffusion divide by K/4 and sqrt(K/4), respectively"
            ),
        },
        "parameters": parameters,
        "preflight_truncation": bool(preflight),
        "event_metric_windows": [
            _scale_int(window, n_arms) for window in (250, 500, 1000)
        ],
        # An explicit empty list intentionally means that this family has no
        # preregistered discrete events; do not infer events from smooth noise.
        "change_events": change_events,
        "horizon": int(horizon),
        "n_arms": int(n_arms),
    }


def build_scale_environment(
    *,
    family: str,
    n_arms: int,
    seed: int,
    horizon: int | None = None,
    preflight: bool = False,
) -> EnvironmentSpec:
    if family not in SCALE_FAMILY_ORDER:
        raise ValueError(f"Unknown scale-suite family {family!r}.")
    expected_horizon = scale_horizon(n_arms)
    if horizon is None:
        horizon = expected_horizon
    if horizon != expected_horizon and not preflight:
        raise ValueError(
            f"Full scale geometry requires horizon={expected_horizon} for K={n_arms}."
        )
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    parameters: dict[str, Any]
    if family in {"slow", "fast"}:
        reference_period = 1000 if family == "slow" else 100
        period = _scale_int(reference_period, n_arms)
        means = slow_varying_sinusoid_means(
            horizon=horizon,
            n_arms=n_arms,
            period=period,
        )
        parameters = {"period": period}
        events = []
    elif family == "abrupt":
        cycle_length = _scale_int(250, n_arms)
        change_times = [
            int(round((arm + 1) * cycle_length / (n_arms + 1)))
            for arm in range(n_arms)
        ]
        if len(set(change_times)) != n_arms:
            raise RuntimeError("Abrupt-cycle scaling produced duplicate change times.")
        levels = np.linspace(0.10, 0.90, n_arms).tolist()
        means = abrupt_means(
            horizon=horizon,
            n_arms=n_arms,
            cycle_len=cycle_length,
            change_times=change_times,
            levels=levels,
        )
        parameters = {
            "cycle_length": cycle_length,
            "change_times": change_times,
            "levels": levels,
        }
        events = _row_change_events(means, source="abrupt_cycle_change")
    elif family == "mixed":
        means = _scale_consistent_mixed_means(
            horizon=horizon,
            n_arms=n_arms,
            seed=seed + 11,
        )
        parameters = {
            "roles": ("stable", "linear_drift", "bursty", "switching"),
            "role_assignment": "arm_index_modulo_4",
            "burst_probability": 0.20 / n_arms,
            "burst_duration_max": _scale_int(20, n_arms),
            "switch_interval": _scale_int(200, n_arms),
        }
        events = []
    elif family == "random_breakpoints":
        min_segment = _scale_int(max(10, scale_horizon(REFERENCE_K) // 150), n_arms)
        if preflight:
            min_segment = min(min_segment, max(1, horizon // 3))
        max_breakpoints = min(16, max(1, horizon // min_segment - 1))
        min_breakpoints = min(4, max_breakpoints)
        means = random_breakpoint_means(
            horizon=horizon,
            n_arms=n_arms,
            min_breakpoints=min_breakpoints,
            max_breakpoints=max_breakpoints,
            min_segment=min_segment,
            min_jump=0.10,
            seed=seed + 21,
        )
        parameters = {
            "min_breakpoints_per_arm": min_breakpoints,
            "max_breakpoints_per_arm": max_breakpoints,
            "min_segment": min_segment,
            "min_jump": 0.10,
        }
        events = _oracle_change_events(means)
    elif family == "random_drift":
        period_min = _scale_int(80, n_arms)
        period_max = max(_scale_int(150, n_arms), horizon // 5)
        amplitude_step_std = _scaled_diffusion(0.015, n_arms)
        drift_step_std = _scaled_diffusion(0.006, n_arms)
        means = random_drift_amplitude_means(
            horizon=horizon,
            n_arms=n_arms,
            amplitude_min=0.05,
            amplitude_max=0.30,
            amplitude_step_std=amplitude_step_std,
            period_min=period_min,
            period_max=period_max,
            drift_step_std=drift_step_std,
            seed=seed + 31,
        )
        parameters = {
            "amplitude_min": 0.05,
            "amplitude_max": 0.30,
            "amplitude_step_std": amplitude_step_std,
            "period_min": period_min,
            "period_max": period_max,
            "drift_step_std": drift_step_std,
        }
        events = []
    elif family in {"global_switching", "per_arm_switching"}:
        switch_probability = 0.08 / n_arms
        generator = (
            global_switching_means
            if family == "global_switching"
            else per_arm_switching_means
        )
        means = generator(
            horizon=horizon,
            n_arms=n_arms,
            switch_prob=switch_probability,
            seed=seed + (41 if family == "global_switching" else 51),
        )
        parameters = {"switch_probability_per_step": switch_probability}
        events = (
            _row_change_events(means, source=family)
            if family == "global_switching"
            else _oracle_change_events(means)
        )
    else:
        affected = local_shock_affected_arms(n_arms)
        local = build_piecewise_stationary_k32_environment(
            horizon=horizon,
            n_arms=n_arms,
            seed=seed + 61,
            affected_arms_per_event=affected,
            key=family,
            name=SCALE_FAMILY_NAMES[family],
        )
        means = local.means_tk
        parameters = {
            "affected_arms_rule": "max(2, round_half_up(0.25*K))",
            "affected_arms_per_event": affected,
            "affected_fraction": affected / n_arms,
            "k4_minimum_disclosure": (
                "K=4 affects 2/4 arms because two arms are required to force "
                "an oracle change."
            ),
        }
        events = list((local.metadata or {}).get("change_events", []))

    metadata = _base_metadata(
        family=family,
        n_arms=n_arms,
        horizon=horizon,
        seed=seed,
        parameters=parameters,
        change_events=events,
        preflight=preflight,
    )
    return EnvironmentSpec(
        key=family,
        name=f"{SCALE_FAMILY_NAMES[family]} (K={n_arms}, seed={seed})",
        means_tk=np.asarray(means, dtype=float),
        arm_labels=tuple(f"Arm {arm}" for arm in range(n_arms)),
        metadata=metadata,
    )
