"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Literal
import numpy as np


FloatArray = NDArray[np.float64]


SamplingMode = Literal["convex", "mixture"]


@dataclass(frozen=True)
class VGDTSV3Params:
    """Public parameter surface shared by VG-dTS v3-A and v3-B."""

    gamma_min: float = 0.95
    gamma_max: float = 0.995
    dual_memory_mix_power: float = 0.30
    local_takeover_scale: float = 0.40
    global_takeover_scale: float = 0.0


def _validate_params(params: VGDTSV3Params) -> None:
    if not (0.0 < params.gamma_min <= params.gamma_max <= 1.0):
        raise ValueError("Require 0 < gamma_min <= gamma_max <= 1.")
    if params.dual_memory_mix_power <= 0.0:
        raise ValueError("dual_memory_mix_power must be > 0.")
    if not (0.0 <= params.local_takeover_scale <= 1.0):
        raise ValueError("local_takeover_scale must be in [0, 1].")
    if not (0.0 <= params.global_takeover_scale <= 1.0):
        raise ValueError("global_takeover_scale must be in [0, 1].")


def _validate_inputs(
    mu: np.ndarray,
    returns: np.ndarray | None,
) -> tuple[FloatArray, FloatArray | None]:
    means = np.asarray(mu, dtype=float)
    if means.ndim != 2 or means.shape[0] == 0:
        raise ValueError("mu must have shape (K, T) with K > 0.")
    if np.any((means < 0.0) | (means > 1.0)):
        raise ValueError("mu entries must be in [0, 1].")
    if returns is None:
        return means, None
    observed = np.asarray(returns, dtype=float)
    if observed.shape != means.shape:
        raise ValueError("returns must have shape (K, T) matching mu.")
    if np.any((observed != 0.0) & (observed != 1.0)):
        raise ValueError("Bernoulli returns entries must be binary 0/1.")
    return means, observed


def _pull(
    mu: FloatArray,
    returns: FloatArray | None,
    arm: int,
    t: int,
    rng: np.random.Generator,
) -> int:
    if returns is not None:
        return int(returns[arm, t])
    return int(rng.random() < mu[arm, t])


def _draw_scores(
    *,
    alpha_short: FloatArray,
    beta_short: FloatArray,
    alpha_long: FloatArray,
    beta_long: FloatArray,
    short_weight: FloatArray,
    sampling: SamplingMode,
    rng: np.random.Generator,
) -> tuple[FloatArray, int]:
    mean_short = alpha_short / (alpha_short + beta_short)
    mean_long = alpha_long / (alpha_long + beta_long)
    active = short_weight > 0.0

    # This branch is exactly Optimistic TS and consumes no short-channel RNG.
    if not np.any(active):
        sample = rng.beta(alpha_long, beta_long)
        return np.maximum(sample, mean_long), 0

    if sampling == "convex":
        sample_long = rng.beta(alpha_long, beta_long)
        sample = sample_long.copy()
        sample_short = rng.beta(alpha_short[active], beta_short[active])
        sample[active] = (
            short_weight[active] * sample_short
            + (1.0 - short_weight[active]) * sample_long[active]
        )
        mean = short_weight * mean_short + (1.0 - short_weight) * mean_long
        return np.maximum(sample, mean), 0

    choose_short = rng.random(len(short_weight)) < short_weight
    sample = np.empty_like(short_weight)
    mean = np.where(choose_short, mean_short, mean_long)
    sample[choose_short] = rng.beta(alpha_short[choose_short], beta_short[choose_short])
    choose_long = ~choose_short
    sample[choose_long] = rng.beta(alpha_long[choose_long], beta_long[choose_long])
    return np.maximum(sample, mean), int(np.count_nonzero(choose_short))
