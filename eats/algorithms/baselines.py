"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from array import array
from collections import deque
from eats.algorithms.parameters import AFFTSParams
from eats.algorithms.parameters import AFF_TS_IMPLEMENTATION_ID
from eats.algorithms.parameters import BetaSWTSParams
from eats.algorithms.parameters import CUSUMUCBParams
from eats.algorithms.parameters import CUSUM_UCB_IMPLEMENTATION_ID
from eats.algorithms.parameters import DLinUCBParams
from eats.algorithms.parameters import DTSParams
from eats.algorithms.parameters import DUCBParams
from eats.algorithms.parameters import GAMMA_SWGTS_IMPLEMENTATION_ID
from eats.algorithms.parameters import GLOBAL_CTS_IMPLEMENTATION_ID
from eats.algorithms.parameters import GLR_KLUCB_IMPLEMENTATION_ID
from eats.algorithms.parameters import GLR_KLUCB_PREFIX_OPTIMIZED_IMPLEMENTATION_ID
from eats.algorithms.parameters import GLR_KLUCB_VECTORIZED_INDEX_IMPLEMENTATION_ID
from eats.algorithms.parameters import GLRklUCBParams
from eats.algorithms.parameters import GammaSWGTSParams
from eats.algorithms.parameters import GlobalCTSParams
from eats.algorithms.parameters import MUCBParams
from eats.algorithms.parameters import M_UCB_IMPLEMENTATION_ID
from eats.algorithms.parameters import REXP3Params
from eats.algorithms.parameters import SWUCBParams
from eats.algorithms.parameters import TSParams
from eats.algorithms.parameters import dOTSParams
from eats.algorithms.parameters import dTSParams
from eats.algorithms.types import AFFTSRunDiagnostics
from eats.algorithms.types import CUSUMUCBRunDiagnostics
from eats.algorithms.types import GLRklUCBRunDiagnostics
from eats.algorithms.types import GammaSWGTSRunDiagnostics
from eats.algorithms.types import GlobalCTSRunDiagnostics
from eats.algorithms.types import MUCBRunDiagnostics
import math
import numpy as np
import sys


DYNAMIC_TS_BETA_SHAPE_FLOOR = 1e-12


def _action_trace_buffer(
    diagnostics: dict[str, object] | None,
    horizon: int,
) -> np.ndarray | None:
    record = (
        bool(diagnostics.pop("_record_played_arms", False))
        if diagnostics is not None
        else False
    )
    return np.full(horizon, -1, dtype=np.int64) if record else None


def _store_action_trace(
    diagnostics: dict[str, object] | None,
    played_arms: np.ndarray | None,
) -> None:
    if diagnostics is not None and played_arms is not None:
        diagnostics["played_arms"] = played_arms


def run_TS(
    mu: np.ndarray,
    params: TSParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    K, T = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    alpha = np.full(K, params.alpha0, dtype=float)
    beta = np.full(K, params.beta0, dtype=float)
    rewards = np.zeros(T, dtype=int)
    played_arms = _action_trace_buffer(diagnostics, T)

    for t in range(T):
        theta = rng.beta(alpha, beta)
        arm = int(np.argmax(theta))
        if played_arms is not None:
            played_arms[t] = arm
        r = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = r
        alpha[arm] += r
        beta[arm] += (1 - r)

    _store_action_trace(diagnostics, played_arms)
    return rewards


def run_DTS(
    mu: np.ndarray,
    params: DTSParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    K, T = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if params.C <= 0.0:
        raise ValueError("C must be > 0.")

    alpha = np.full(K, params.alpha0, dtype=float)
    beta = np.full(K, params.beta0, dtype=float)
    rewards = np.zeros(T, dtype=int)
    played_arms = _action_trace_buffer(diagnostics, T)

    disc = params.C / (params.C + 1.0)

    for t in range(T):
        theta = rng.beta(
            np.maximum(alpha, DYNAMIC_TS_BETA_SHAPE_FLOOR),
            np.maximum(beta, DYNAMIC_TS_BETA_SHAPE_FLOOR),
        )
        arm = int(np.argmax(theta))
        if played_arms is not None:
            played_arms[t] = arm
        r = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = r

        # Gupta et al.'s Dynamic TS update: below C, use the ordinary
        # conjugate update; at/above C, multiply the *updated* shapes by
        # C/(C+1). Scaling before adding the reward is a different recursion.
        if (alpha[arm] + beta[arm]) < params.C:
            alpha[arm] += r
            beta[arm] += (1 - r)
        else:
            alpha[arm] = (alpha[arm] + r) * disc
            beta[arm] = (beta[arm] + (1 - r)) * disc
            alpha[arm] = max(alpha[arm], DYNAMIC_TS_BETA_SHAPE_FLOOR)
            beta[arm] = max(beta[arm], DYNAMIC_TS_BETA_SHAPE_FLOOR)

    _store_action_trace(diagnostics, played_arms)
    return rewards


def run_dTS(
    mu: np.ndarray,
    params: dTSParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    K, T = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if not (0.0 < params.gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")
    S = np.zeros(K, dtype=float)
    F = np.zeros(K, dtype=float)
    rewards = np.zeros(T, dtype=int)
    played_arms = _action_trace_buffer(diagnostics, T)

    for t in range(T):
        theta = rng.beta(S + params.alpha0, F + params.beta0)
        arm = int(np.argmax(theta))
        if played_arms is not None:
            played_arms[t] = arm
        r = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = r

        # discount all arms
        S *= params.gamma
        F *= params.gamma

        # then update played arm
        S[arm] += r
        F[arm] += (1 - r)

    _store_action_trace(diagnostics, played_arms)
    return rewards


def run_dOTS(
    mu: np.ndarray,
    params: dOTSParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    *,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    K, T = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if not (0.0 < params.gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")
    S = np.zeros(K, dtype=float)
    F = np.zeros(K, dtype=float)
    rewards = np.zeros(T, dtype=int)
    played_arms = (
        np.empty(T, dtype=np.int64)
        if diagnostics is not None and diagnostics.get("_record_played_arms", False)
        else None
    )

    for t in range(T):
        a = S + params.alpha0
        b = F + params.beta0
        tilde = rng.beta(a, b)
        mean = a / (a + b)
        theta = np.maximum(mean, tilde)  # Eq. (6)
        arm = int(np.argmax(theta))
        if played_arms is not None:
            played_arms[t] = arm
        r = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = r

        S *= params.gamma
        F *= params.gamma
        S[arm] += r
        F[arm] += (1 - r)

    if diagnostics is not None and played_arms is not None:
        diagnostics["played_arms"] = played_arms
    return rewards


def _kl_bernoulli(p: float, q: float, eps: float = 1e-12) -> float:
    p_clip = float(np.clip(p, eps, 1.0 - eps))
    q_clip = float(np.clip(q, eps, 1.0 - eps))
    return p_clip * math.log(p_clip / q_clip) + (1.0 - p_clip) * math.log((1.0 - p_clip) / (1.0 - q_clip))


def run_REXP3(
    mu: np.ndarray,
    params: REXP3Params,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    """Run Rexp3 as stated by Besbes, Gur, and Zeevi.

    Log weights are an algebraically identical representation of the paper's
    multiplicative update and prevent numerical overflow on long batches.
    ``gamma`` and ``Delta`` remain explicit inputs because the theoretical
    coupling requires a variation budget that is not generally known.
    """
    K, T = _validate_mu(mu)
    if K < 2:
        raise ValueError("Rexp3 requires at least two arms.")
    if not (0.0 < params.gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")
    if params.Delta <= 0:
        raise ValueError("Delta must be > 0.")
    rewards = np.zeros(T, dtype=int)
    played_arms = _action_trace_buffer(diagnostics, T)

    log_w = np.zeros(K, dtype=float)

    def reset():
        log_w.fill(0.0)

    for t in range(T):
        # restart each batch
        if t % params.Delta == 0 and t != 0:
            reset()

        shifted = log_w - float(np.max(log_w))
        normalized_w = np.exp(shifted)
        normalized_w /= float(np.sum(normalized_w))
        p = (1.0 - params.gamma) * normalized_w + params.gamma / K

        arm = int(rng.choice(K, p=p))
        if played_arms is not None:
            played_arms[t] = arm
        r = float(_pull_from_returns_or_mu(mu, returns, arm, t, rng))
        rewards[t] = int(r)

        # importance-weighted reward estimate
        xhat = r / p[arm]

        # Paper update: w[a] <- w[a] exp(gamma * xhat[a] / K).
        log_w[arm] += (params.gamma / K) * xhat

    _store_action_trace(diagnostics, played_arms)
    return rewards


def _validate_mu(mu: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(mu, dtype=float)
    if arr.ndim != 2:
        raise ValueError("mu must have shape (K, T).")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError("mu must contain at least one arm and one round.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("mu entries must be finite.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("mu entries must be in [0, 1].")
    return int(arr.shape[0]), int(arr.shape[1])


def _pull_from_mu(mu: np.ndarray, arm: int, t: int, rng: np.random.Generator) -> int:
    return int(rng.random() < float(mu[arm, t]))


def _pull_from_returns_or_mu(
    mu: np.ndarray,
    returns: np.ndarray | None,
    arm: int,
    t: int,
    rng: np.random.Generator,
) -> int:
    if returns is None:
        return _pull_from_mu(mu, arm, t, rng)
    returns_arr = np.asarray(returns)
    if returns_arr.shape != mu.shape:
        raise ValueError("returns must have shape (K, T) matching mu.")
    value = float(returns_arr[arm, t])
    if value < -1e-12 or value > 1.0 + 1e-12:
        raise ValueError("Bernoulli returns entries must lie in [0, 1].")
    rounded = round(value)
    if not math.isclose(value, float(rounded), abs_tol=1e-8):
        raise ValueError("Bernoulli returns entries must be binary 0/1.")
    return int(rounded)


def _kl_ucb_bernoulli(mean: float, n: int, bonus: float) -> float:
    if n <= 0:
        return 1.0
    if bonus <= 0.0:
        return float(np.clip(mean, 0.0, 1.0))

    mean_clip = float(np.clip(mean, 0.0, 1.0))
    lo = mean_clip
    hi = 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if n * _kl_bernoulli(mean_clip, mid) <= bonus:
            lo = mid
        else:
            hi = mid
    return float(np.clip(lo, 0.0, 1.0))


def _kl_bernoulli_vectorized(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Evaluate the scalar Bernoulli-KL expression lane-wise in float64."""

    p_values, q_values = np.broadcast_arrays(
        np.asarray(p, dtype=np.float64),
        np.asarray(q, dtype=np.float64),
    )
    p_clip = np.clip(p_values, eps, 1.0 - eps)
    q_clip = np.clip(q_values, eps, 1.0 - eps)
    return p_clip * np.log(p_clip / q_clip) + (1.0 - p_clip) * np.log(
        (1.0 - p_clip) / (1.0 - q_clip)
    )


def _kl_ucb_bernoulli_vectorized(
    means: np.ndarray,
    counts: np.ndarray,
    bonuses: np.ndarray,
) -> np.ndarray:
    """Vectorize the scalar 40-step klUCB bisection across independent arms."""

    mean_values, count_values, bonus_values = np.broadcast_arrays(
        np.asarray(means, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        np.asarray(bonuses, dtype=np.float64),
    )
    scores = np.ones(mean_values.shape, dtype=np.float64)
    observed = count_values > 0
    nonpositive_bonus = observed & (bonus_values <= 0.0)
    scores[nonpositive_bonus] = np.clip(
        mean_values[nonpositive_bonus], 0.0, 1.0
    )

    active = observed & ~nonpositive_bonus
    if not np.any(active):
        return scores
    active_means = np.clip(mean_values[active], 0.0, 1.0)
    active_counts = count_values[active]
    active_bonuses = bonus_values[active]
    lo = active_means.copy()
    hi = np.ones_like(lo)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        within = (
            active_counts * _kl_bernoulli_vectorized(active_means, mid)
            <= active_bonuses
        )
        lo = np.where(within, mid, lo)
        hi = np.where(within, hi, mid)
    scores[active] = np.clip(lo, 0.0, 1.0)
    return scores


def _bernoulli_glr_max_statistic_from_prefix(
    cumulative_successes: array,
    *,
    split_step: int = 1,
) -> tuple[float, int]:
    """Evaluate the same GLR statistic from retained cumulative successes.

    The prefix representation removes the avoidable conversion and cumulative
    sum from each detector call.  It does not remove the exact maximization
    over the retained candidate split points.
    """
    if split_step <= 0:
        raise ValueError("split_step must be > 0.")
    n = len(cumulative_successes)
    splits = np.arange(split_step, n, split_step, dtype=np.int64)
    if splits.size == 0:
        return 0.0, -1

    prefix = np.frombuffer(cumulative_successes, dtype=np.uint64)
    total = float(prefix[-1])
    pooled = total / float(n)
    if pooled <= 0.0 or pooled >= 1.0:
        return 0.0, int(splits[0])

    left_sum = prefix[splits - 1].astype(float, copy=False)
    right_count = n - splits
    right_sum = total - left_sum
    p_left = left_sum / splits
    p_right = right_sum / right_count

    def kl_vector(p: np.ndarray, q: float) -> np.ndarray:
        result = np.zeros_like(p, dtype=float)
        positive = p > 0.0
        below_one = p < 1.0
        result[positive] += p[positive] * np.log(p[positive] / q)
        result[below_one] += (1.0 - p[below_one]) * np.log(
            (1.0 - p[below_one]) / (1.0 - q)
        )
        return result

    statistics = splits * kl_vector(p_left, pooled)
    statistics += right_count * kl_vector(p_right, pooled)
    best_position = int(np.argmax(statistics))
    return float(statistics[best_position]), int(splits[best_position])


def _glr_practical_threshold(n: int, delta: float) -> float:
    """Practical Bernoulli threshold beta(n, delta)=log(n^(3/2)/delta)."""
    if n <= 0:
        raise ValueError("n must be > 0.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    return 1.5 * math.log(float(n)) + math.log(1.0 / delta)


def _klucb_exploration_function(age: int, coefficient: float) -> float:
    """The paper's f(age)=log(age)+c log(log(age)), clipped at zero."""
    if age <= 0:
        raise ValueError("age must be > 0.")
    if coefficient < 0.0:
        raise ValueError("coefficient must be >= 0.")
    log_age = math.log(float(age))
    if log_age <= 0.0:
        return 0.0
    return max(0.0, log_age + coefficient * math.log(log_age))


def _stratified_optimal_resample(
    weights: np.ndarray,
    target_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Fearnhead-Clifford stratified optimal resampling for discrete weights."""
    probabilities = np.asarray(weights, dtype=np.longdouble)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array.")
    if np.any(probabilities < 0.0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("weights must be finite and non-negative.")
    mass = np.sum(probabilities, dtype=np.longdouble)
    if mass <= 0.0:
        raise ValueError("weights must have positive mass.")
    probabilities = probabilities / mass
    if target_size <= 0:
        raise ValueError("target_size must be > 0.")
    if target_size > int(np.count_nonzero(probabilities > 0.0)):
        raise ValueError("target_size cannot exceed the positive-weight support.")
    if target_size >= probabilities.size:
        normalized = np.asarray(probabilities, dtype=float)
        normalized /= float(np.sum(normalized))
        return np.arange(probabilities.size, dtype=int), normalized

    # Solve M = sum_i min(1, p_i / kappa) by moving every p_i > kappa
    # into the deterministic set. Long-double arithmetic prevents tiny
    # residual masses from disappearing during the threshold calculation.
    deterministic = np.zeros(probabilities.size, dtype=bool)
    while True:
        stochastic_count = target_size - int(np.count_nonzero(deterministic))
        if stochastic_count <= 0:
            raise RuntimeError("Optimal resampling exhausted its stochastic particle budget.")
        residual = probabilities[~deterministic]
        residual_mass = np.sum(residual, dtype=np.longdouble)
        kappa = residual_mass / np.longdouble(stochastic_count)
        newly_deterministic = (~deterministic) & (probabilities > kappa)
        if not np.any(newly_deterministic):
            break
        deterministic |= newly_deterministic

    deterministic_indices = np.flatnonzero(deterministic)
    residual_indices = np.flatnonzero(~deterministic)
    residual_probabilities = probabilities[residual_indices]
    residual_mass = np.sum(residual_probabilities, dtype=np.longdouble)
    kappa = residual_mass / np.longdouble(stochastic_count)

    offset = np.longdouble(rng.random()) * kappa
    thresholds = offset + kappa * np.arange(stochastic_count, dtype=np.longdouble)
    cumulative = np.cumsum(residual_probabilities, dtype=np.longdouble)
    cumulative[-1] = residual_mass
    selected_positions = np.searchsorted(cumulative, thresholds, side="right")
    stochastic_indices = residual_indices[selected_positions]

    retained = np.concatenate((deterministic_indices, stochastic_indices))
    retained_weights = np.concatenate(
        (
            probabilities[deterministic_indices],
            np.full(stochastic_count, kappa, dtype=np.longdouble),
        )
    )
    order = np.argsort(retained)
    retained = retained[order]
    retained_weights = retained_weights[order]

    if retained.size != target_size or np.unique(retained).size != target_size:
        raise RuntimeError(
            "Stratified optimal resampling produced an invalid particle count: "
            f"expected {target_size}, observed {np.unique(retained).size}."
        )

    next_weights = np.asarray(retained_weights, dtype=float)
    next_weights /= float(np.sum(next_weights))
    return retained.astype(int), next_weights


def run_global_cts(
    mu: np.ndarray,
    params: GlobalCTSParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: GlobalCTSRunDiagnostics | None = None,
) -> np.ndarray:
    """
    Bounded Global-CTS following Mellor-Shapiro Algorithm 1.

    The reset run-length particle is the unupdated Beta prior, and support is
    bounded with the paper's stratified optimal resampling rule. Here
    ``max_runlengths`` is the hard support cap N; overflow is resampled to
    ceil(N * resample_target_fraction), making the paper's unspecified M
    budget explicit.
    """
    k, horizon = _validate_mu(mu)
    if not (0.0 <= params.hazard <= 1.0):
        raise ValueError("hazard must be in [0, 1].")
    if params.max_runlengths <= 0:
        raise ValueError("max_runlengths must be > 0.")
    if not (0.0 < params.resample_target_fraction < 1.0):
        raise ValueError("resample_target_fraction must be in (0, 1) for bounded optimal resampling.")
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")

    record_final_state = (
        bool(diagnostics.pop("_record_final_state", False))
        if diagnostics is not None
        else False
    )
    record_played_arms = (
        bool(diagnostics.pop("_record_played_arms", False))
        if diagnostics is not None
        else False
    )
    rewards = np.zeros(horizon, dtype=int)
    played_arms = (
        np.full(horizon, -1, dtype=np.int64) if record_played_arms else None
    )
    weights = np.array([1.0], dtype=float)
    alpha = np.full((1, k), params.alpha0, dtype=float)
    beta = np.full((1, k), params.beta0, dtype=float)
    max_support = 1
    resampling_count = 0
    peak_model_state_bytes = int(weights.nbytes + alpha.nbytes + beta.nbytes)

    for t in range(horizon):
        state_idx = int(rng.choice(len(weights), p=weights))
        theta = rng.beta(alpha[state_idx], beta[state_idx])
        arm = int(np.argmax(theta))
        if played_arms is not None:
            played_arms[t] = arm
        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        denom = alpha[:, arm] + beta[:, arm]
        likelihood = (
            alpha[:, arm] / np.maximum(denom, 1e-12)
            if reward == 1
            else beta[:, arm] / np.maximum(denom, 1e-12)
        )
        growth_weights = (1.0 - params.hazard) * weights * likelihood
        cp_weight = params.hazard * float(np.sum(weights * likelihood))
        next_weights = np.concatenate(([cp_weight], growth_weights), axis=0)

        n_states_next = len(next_weights)
        next_alpha = np.empty((n_states_next, k), dtype=float)
        next_beta = np.empty((n_states_next, k), dtype=float)
        next_alpha[0] = params.alpha0
        next_beta[0] = params.beta0
        next_alpha[1:] = alpha
        next_beta[1:] = beta
        next_alpha[1:, arm] += reward
        next_beta[1:, arm] += (1 - reward)

        mass = float(np.sum(next_weights))
        if mass <= 0.0 or not np.isfinite(mass):
            next_weights = np.full(n_states_next, 1.0 / float(n_states_next), dtype=float)
        else:
            next_weights = next_weights / mass

        positive = next_weights > 0.0
        if not np.all(positive):
            next_weights = next_weights[positive]
            next_alpha = next_alpha[positive]
            next_beta = next_beta[positive]
            n_states_next = len(next_weights)

        # Mellor--Shapiro Algorithm 1 resamples when |R_t| reaches N, not
        # after allowing a transient N+1 support state.
        if n_states_next >= params.max_runlengths:
            target_size = max(
                1,
                min(
                    params.max_runlengths,
                    int(math.ceil(params.max_runlengths * params.resample_target_fraction)),
                ),
            )
            keep, resampled_weights = _stratified_optimal_resample(
                next_weights,
                target_size,
                rng,
            )
            next_weights = resampled_weights
            next_alpha = next_alpha[keep]
            next_beta = next_beta[keep]
            resampling_count += 1

        weights = next_weights
        alpha = next_alpha
        beta = next_beta
        max_support = max(max_support, int(len(weights)))
        peak_model_state_bytes = max(
            peak_model_state_bytes,
            int(weights.nbytes + alpha.nbytes + beta.nbytes),
        )

    if diagnostics is not None:
        diagnostics["implementation_id"] = GLOBAL_CTS_IMPLEMENTATION_ID
        diagnostics["max_particle_support"] = int(max_support)
        diagnostics["resampling_count"] = int(resampling_count)
        diagnostics["resample_target_fraction"] = float(params.resample_target_fraction)
        diagnostics["peak_persistent_model_state_bytes"] = int(peak_model_state_bytes)
        diagnostics["reward_output_bytes"] = int(rewards.nbytes)
        if record_final_state:
            diagnostics["final_weights"] = weights.copy()
            diagnostics["final_alpha"] = alpha.copy()
            diagnostics["final_beta"] = beta.copy()
        if played_arms is not None:
            diagnostics["played_arms"] = played_arms

    return rewards


def run_aff_ts(
    mu: np.ndarray,
    params: AFFTSParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: AFFTSRunDiagnostics | None = None,
) -> np.ndarray:
    """Run AFF-TS exactly as Algorithm 4 and equations (3)--(9), (16)--(17),
    and (24)--(25) of Lu, Adams, and Kantas (2019).

    The base AFF moments are updated only when an arm is observed. At decision
    time, the paper's separate idle-arm discount flattens an unobserved arm's
    posterior by ``lambda ** (idle_time / K)``. The first pull of each arm is
    the initialization required by Algorithm 4.
    """
    k, horizon = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if not (0.0 < params.eta < 1.0):
        raise ValueError("eta must be in (0, 1), as required by Algorithm 4.")
    if not (0.0 <= params.lambda_init <= 1.0):
        raise ValueError("lambda_init must be in [0, 1].")
    if params.eps <= 0.0:
        raise ValueError("eps must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    record_played_arms = (
        bool(diagnostics.get("_record_played_arms", False))
        if diagnostics is not None
        else False
    )
    record_state_history = (
        bool(diagnostics.get("_record_state_history", False))
        if diagnostics is not None
        else False
    )
    played_arms = np.full(horizon, -1, dtype=int) if record_played_arms else None
    lambda_history = (
        np.full((horizon, k), np.nan, dtype=float) if record_state_history else None
    )

    lam = np.full(k, params.lambda_init, dtype=float)
    m = np.zeros(k, dtype=float)
    w = np.zeros(k, dtype=float)
    m_dot = np.zeros(k, dtype=float)
    w_dot = np.zeros(k, dtype=float)
    last_selected = np.full(k, -1, dtype=int)

    init_rounds = min(k, horizon)
    for t in range(init_rounds):
        arm = t
        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward
        if played_arms is not None:
            played_arms[t] = arm
        m[arm] = float(reward)
        w[arm] = 1.0
        m_dot[arm] = 0.0
        w_dot[arm] = 0.0
        last_selected[arm] = t
        if lambda_history is not None:
            lambda_history[t] = lam

    for t in range(init_rounds, horizon):
        # The posterior used at round t is the time-(t-1) posterior. Therefore
        # an arm selected at t-1 has zero idle time, as equations (16)--(17)
        # require. (Global time is zero-based here.)
        state_time = t - 1
        idle_time = np.maximum(state_time - last_selected, 0)
        decay = np.power(lam, idle_time.astype(float) / float(k))
        m_tilde = decay * m
        w_tilde = decay * w

        alpha = params.alpha0 + m_tilde
        beta = params.beta0 + np.maximum(w_tilde - m_tilde, 0.0)

        samples = rng.beta(alpha, beta)
        arm = int(np.argmax(samples))
        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward
        if played_arms is not None:
            played_arms[t] = arm

        prev_m = m[arm]
        prev_w_raw = w[arm]
        prev_w = max(prev_w_raw, params.eps)
        prev_mean = prev_m / prev_w
        lam_old = lam[arm]

        m_new = lam_old * prev_m + reward
        w_new = lam_old * prev_w_raw + 1.0

        # Equation (9) differentiates the one-step-ahead squared prediction
        # error (hat Y_{j-1} - Y_j)^2. Y_j is the observed reward, not the
        # newly updated AFF mean.
        grad_term = (m_dot[arm] - w_dot[arm] * prev_mean) / prev_w
        gradient = 2.0 * (prev_mean - float(reward)) * grad_term

        m_dot_new = lam_old * m_dot[arm] + prev_m
        w_dot_new = lam_old * w_dot[arm] + prev_w_raw
        lam_new = float(np.clip(lam_old - params.eta * gradient, 0.0, 1.0))

        m[arm] = m_new
        w[arm] = w_new
        m_dot[arm] = m_dot_new
        w_dot[arm] = w_dot_new
        lam[arm] = lam_new
        last_selected[arm] = t
        if lambda_history is not None:
            lambda_history[t] = lam

    if diagnostics is not None:
        diagnostics["implementation_id"] = AFF_TS_IMPLEMENTATION_ID
        diagnostics["final_lambda"] = lam.copy()
        diagnostics["final_m"] = m.copy()
        diagnostics["final_w"] = w.copy()
        diagnostics["final_m_dot"] = m_dot.copy()
        diagnostics["final_w_dot"] = w_dot.copy()
        diagnostics["final_last_selected"] = last_selected.copy()
        if played_arms is not None:
            diagnostics["played_arms"] = played_arms
        if lambda_history is not None:
            diagnostics["lambda_history"] = lambda_history

    return rewards


def run_dlinucb_onehot(
    mu: np.ndarray,
    params: DLinUCBParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    """
    D-LinUCB (1909.09146) with one-hot arm features for K-armed Bernoulli setup.
    """
    k, horizon = _validate_mu(mu)
    if not (0.0 < params.gamma < 1.0):
        raise ValueError("gamma must be in (0, 1).")
    if params.lambda_reg <= 0.0:
        raise ValueError("lambda_reg must be > 0.")
    if not (0.0 < params.delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    if params.sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if params.action_norm_bound <= 0.0 or params.theta_norm_bound <= 0.0:
        raise ValueError("action_norm_bound and theta_norm_bound must be > 0.")
    if params.jitter <= 0.0:
        raise ValueError("jitter must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    played_arms = _action_trace_buffer(diagnostics, horizon)

    eye = np.eye(k, dtype=float)
    v = params.lambda_reg * eye.copy()
    v_tilde = params.lambda_reg * eye.copy()
    b = np.zeros(k, dtype=float)
    theta_hat = np.zeros(k, dtype=float)

    denom_gamma = max(1.0 - params.gamma**2, 1e-12)
    for t in range(horizon):
        inv_v = np.linalg.inv(v + params.jitter * eye)
        radius_term = 1.0 + (
            (params.action_norm_bound**2) * (1.0 - params.gamma ** (2 * t))
        ) / (params.lambda_reg * k * denom_gamma)
        beta_t = np.sqrt(params.lambda_reg) * params.theta_norm_bound
        beta_t += params.sigma * np.sqrt(
            2.0 * np.log(1.0 / params.delta) + k * np.log(max(radius_term, 1.0))
        )

        conf_mat = inv_v @ v_tilde @ inv_v
        conf = np.sqrt(np.clip(np.diag(conf_mat), 0.0, None))
        scores = theta_hat + beta_t * conf
        arm = int(np.argmax(scores))
        if played_arms is not None:
            played_arms[t] = arm

        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        x = eye[arm]
        outer = np.outer(x, x)
        v = params.gamma * v + outer + (1.0 - params.gamma) * params.lambda_reg * eye
        v_tilde = (
            (params.gamma**2) * v_tilde
            + outer
            + (1.0 - params.gamma**2) * params.lambda_reg * eye
        )
        b = params.gamma * b + reward * x
        theta_hat = np.linalg.solve(v + params.jitter * eye, b)

    _store_action_trace(diagnostics, played_arms)
    return rewards


def run_beta_swts(
    mu: np.ndarray,
    params: BetaSWTSParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    """
    Beta-SWTS from 2409.05181 for Bernoulli rewards.
    """
    k, horizon = _validate_mu(mu)
    if params.alpha0 <= 0.0 or params.beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be > 0.")
    if params.tau <= 0:
        raise ValueError("tau must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    played_arms = _action_trace_buffer(diagnostics, horizon)
    pulls = np.zeros(k, dtype=int)
    successes = np.zeros(k, dtype=float)
    history: list[tuple[int, int]] = []

    for t in range(horizon):
        alpha = params.alpha0 + successes
        beta = params.beta0 + np.maximum(pulls.astype(float) - successes, 0.0)
        theta = rng.beta(alpha, beta)
        arm = int(np.argmax(theta))
        if played_arms is not None:
            played_arms[t] = arm

        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        pulls[arm] += 1
        successes[arm] += reward
        history.append((arm, reward))

        if len(history) > params.tau:
            old_arm, old_reward = history.pop(0)
            pulls[old_arm] -= 1
            successes[old_arm] -= old_reward

    _store_action_trace(diagnostics, played_arms)
    return rewards


def run_gamma_swgts(
    mu: np.ndarray,
    params: GammaSWGTSParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: GammaSWGTSRunDiagnostics | None = None,
) -> np.ndarray:
    r"""Run the published Gaussian :math:`\gamma`-SWGTS Algorithm 2.

    The sliding window is over the last ``tau`` *global rounds*.  Following
    the paper, every arm is first played once.  Later, an arm absent from the
    window is forced; otherwise the arm score is sampled from
    ``Normal(S_i/N_i, 1/(gamma*N_i))``.
    """
    k, horizon = _validate_mu(mu)
    if params.tau <= 0:
        raise ValueError("tau must be > 0.")
    if params.gamma <= 0.0:
        raise ValueError("gamma must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    pulls = np.zeros(k, dtype=int)
    successes = np.zeros(k, dtype=float)
    history: deque[tuple[int, int]] = deque()
    record_played_arms = bool(diagnostics.pop("_record_played_arms", False)) if diagnostics is not None else False
    played_arms = np.full(horizon, -1, dtype=int) if record_played_arms else None
    forced_zero_count_plays = 0

    for t in range(horizon):
        if t < k:
            arm = t
        elif np.any(pulls == 0):
            arm = int(np.flatnonzero(pulls == 0)[0])
            forced_zero_count_plays += 1
        else:
            means = successes / pulls
            stddev = np.sqrt(1.0 / (params.gamma * pulls))
            theta = rng.normal(means, stddev)
            arm = int(np.argmax(theta))

        if played_arms is not None:
            played_arms[t] = arm
        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        pulls[arm] += 1
        successes[arm] += reward
        history.append((arm, reward))

        if len(history) > params.tau:
            old_arm, old_reward = history.popleft()
            pulls[old_arm] -= 1
            successes[old_arm] -= old_reward

    if diagnostics is not None:
        diagnostics["implementation_id"] = GAMMA_SWGTS_IMPLEMENTATION_ID
        diagnostics["forced_zero_count_plays"] = forced_zero_count_plays
        diagnostics["final_window_pulls"] = pulls.copy()
        diagnostics["final_window_successes"] = successes.copy()
        if played_arms is not None:
            diagnostics["played_arms"] = played_arms.copy()

    return rewards


def run_sw_ucb(
    mu: np.ndarray,
    params: SWUCBParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    """
    Sliding-Window UCB baseline for bounded rewards in [0,1].
    """
    k, horizon = _validate_mu(mu)
    if params.tau <= 0:
        raise ValueError("tau must be > 0.")
    if params.xi <= 0.0:
        raise ValueError("xi must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    played_arms = _action_trace_buffer(diagnostics, horizon)
    pulls = np.zeros(k, dtype=float)
    successes = np.zeros(k, dtype=float)
    history: list[tuple[int, int]] = []

    for t in range(horizon):
        if np.any(pulls <= 0.0):
            arm = int(np.where(pulls <= 0.0)[0][0])
        else:
            means = successes / np.maximum(pulls, 1e-12)
            log_term = math.log(max(2.0, float(min(t + 1, params.tau))))
            bonus = np.sqrt(params.xi * log_term / np.maximum(pulls, 1e-12))
            arm = int(np.argmax(means + bonus))

        if played_arms is not None:
            played_arms[t] = arm

        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        pulls[arm] += 1.0
        successes[arm] += float(reward)
        history.append((arm, reward))

        if len(history) > params.tau:
            old_arm, old_reward = history.pop(0)
            pulls[old_arm] -= 1.0
            successes[old_arm] -= float(old_reward)

    _store_action_trace(diagnostics, played_arms)
    return rewards


def run_d_ucb(
    mu: np.ndarray,
    params: DUCBParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> np.ndarray:
    """
    Discounted UCB baseline with exponentially discounted sufficient statistics.
    """
    k, horizon = _validate_mu(mu)
    if not (0.0 < params.gamma < 1.0):
        raise ValueError("gamma must be in (0, 1).")
    if params.xi <= 0.0:
        raise ValueError("xi must be > 0.")
    if params.bonus_scale <= 0.0:
        raise ValueError("bonus_scale must be > 0.")

    rewards = np.zeros(horizon, dtype=int)
    played_arms = _action_trace_buffer(diagnostics, horizon)
    discounted_pulls = np.zeros(k, dtype=float)
    discounted_successes = np.zeros(k, dtype=float)
    n_gamma = 0.0

    for t in range(horizon):
        if np.any(discounted_pulls <= 1e-12):
            arm = int(np.where(discounted_pulls <= 1e-12)[0][0])
        else:
            means = discounted_successes / np.maximum(discounted_pulls, 1e-12)
            log_term = math.log(max(1.0, n_gamma))
            bonus = params.bonus_scale * np.sqrt(
                params.xi * log_term / np.maximum(discounted_pulls, 1e-12)
            )
            arm = int(np.argmax(means + bonus))

        if played_arms is not None:
            played_arms[t] = arm

        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        discounted_pulls *= params.gamma
        discounted_successes *= params.gamma
        discounted_pulls[arm] += 1.0
        discounted_successes[arm] += float(reward)
        n_gamma = params.gamma * n_gamma + 1.0

    _store_action_trace(diagnostics, played_arms)
    return rewards


def _cusum_ucb_scores(
    pulls: np.ndarray,
    successes: np.ndarray,
    xi: float,
) -> np.ndarray:
    """Equation (4) UCB indices using only post-restart observations."""
    pulls_array = np.asarray(pulls, dtype=float)
    successes_array = np.asarray(successes, dtype=float)
    if pulls_array.shape != successes_array.shape or np.any(pulls_array <= 0.0):
        raise ValueError("pulls and successes must have equal shape and positive pulls.")
    if xi <= 0.0:
        raise ValueError("xi must be > 0.")
    n_valid = float(np.sum(pulls_array))
    means = successes_array / pulls_array
    bonus = np.sqrt(xi * math.log(max(1.0, n_valid)) / pulls_array)
    return means + bonus


def run_cusum_ucb(
    mu: np.ndarray,
    params: CUSUMUCBParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: CUSUMUCBRunDiagnostics | None = None,
) -> np.ndarray:
    """Run CUSUM-UCB Algorithm 3 with arm-wise local restarts.

    Each detector is supplied its first ``M=warmup`` observations immediately,
    both initially and after an alarm.  UCB uses the paper's total number of
    observations valid since the arms' respective local restarts.  The reward
    that triggers an alarm is excluded from the restarted statistics because
    Algorithm 3 sets :math:`\tau_i=t+1`.
    """
    k, horizon = _validate_mu(mu)
    if params.xi <= 0.0:
        raise ValueError("xi must be > 0.")
    if params.epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")
    if params.threshold <= 0.0:
        raise ValueError("threshold must be > 0.")
    if params.warmup <= 1:
        raise ValueError("warmup must be > 1.")
    if not (0.0 <= params.random_explore < 1.0):
        raise ValueError("random_explore must be in [0, 1).")

    rewards = np.zeros(horizon, dtype=int)
    pulls = np.zeros(k, dtype=float)
    successes = np.zeros(k, dtype=float)
    detector_count = np.zeros(k, dtype=int)
    reference_sum = np.zeros(k, dtype=float)
    ref_mean = np.zeros(k, dtype=float)
    g_plus = np.zeros(k, dtype=float)
    g_minus = np.zeros(k, dtype=float)
    countdown = np.full(k, params.warmup, dtype=int)
    record_played_arms = bool(diagnostics.pop("_record_played_arms", False)) if diagnostics is not None else False
    played_arms = np.full(horizon, -1, dtype=int) if record_played_arms else None
    restart_rounds: list[int] = []
    restart_arms: list[int] = []
    forced_warmup_count = 0

    for t in range(horizon):
        if np.any(countdown > 0):
            arm = int(np.flatnonzero(countdown > 0)[0])
            countdown[arm] -= 1
            forced_warmup_count += 1
        elif rng.random() < params.random_explore:
            arm = int(rng.integers(0, k))
        else:
            arm = int(np.argmax(_cusum_ucb_scores(pulls, successes, params.xi)))

        if played_arms is not None:
            played_arms[t] = arm
        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        pulls[arm] += 1.0
        successes[arm] += float(reward)

        detector_count[arm] += 1
        if detector_count[arm] <= params.warmup:
            reference_sum[arm] += float(reward)
            if detector_count[arm] == params.warmup:
                ref_mean[arm] = reference_sum[arm] / float(params.warmup)
            continue

        drift_pos = float(reward) - ref_mean[arm] - params.epsilon
        drift_neg = ref_mean[arm] - float(reward) - params.epsilon
        g_plus[arm] = max(0.0, g_plus[arm] + drift_pos)
        g_minus[arm] = max(0.0, g_minus[arm] + drift_neg)

        if max(g_plus[arm], g_minus[arm]) >= params.threshold:
            restart_rounds.append(t)
            restart_arms.append(arm)
            pulls[arm] = 0.0
            successes[arm] = 0.0
            detector_count[arm] = 0
            reference_sum[arm] = 0.0
            ref_mean[arm] = 0.0
            g_plus[arm] = 0.0
            g_minus[arm] = 0.0
            countdown[arm] = params.warmup

    if diagnostics is not None:
        diagnostics["implementation_id"] = CUSUM_UCB_IMPLEMENTATION_ID
        diagnostics["restart_count"] = len(restart_rounds)
        diagnostics["restart_rounds"] = np.asarray(restart_rounds, dtype=int)
        diagnostics["restart_arms"] = np.asarray(restart_arms, dtype=int)
        diagnostics["forced_warmup_count"] = forced_warmup_count
        diagnostics["final_valid_pulls"] = pulls.copy()
        diagnostics["final_detector_counts"] = detector_count.copy()
        diagnostics["final_countdowns"] = countdown.copy()
        if played_arms is not None:
            diagnostics["played_arms"] = played_arms.copy()

    return rewards


def run_m_ucb(
    mu: np.ndarray,
    params: MUCBParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: MUCBRunDiagnostics | None = None,
) -> np.ndarray:
    """
    M-UCB Algorithm 2 from Cao et al. (2019).

    ``xi=1`` gives the published UCB1 index.  The forced-exploration period is
    ``floor(K/gamma)`` as written in Algorithm 2, and a detector alarm causes a
    global restart that excludes the alarm reward from the new episode.
    """
    k, horizon = _validate_mu(mu)
    if params.xi <= 0.0:
        raise ValueError("xi must be > 0.")
    if params.window <= 0 or params.window % 2 != 0:
        raise ValueError("window must be a positive even integer.")
    if params.threshold <= 0.0:
        raise ValueError("threshold must be > 0.")
    if not (0.0 < params.gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")

    rewards = np.zeros(horizon, dtype=int)
    record_played_arms = bool(diagnostics.pop("_record_played_arms", False)) if diagnostics is not None else False
    played_arms = np.full(horizon, -1, dtype=int) if record_played_arms else None
    restart_rounds: list[int] = []
    episode_start = 0
    pulls = np.zeros(k, dtype=float)
    successes = np.zeros(k, dtype=float)
    histories: list[list[int]] = [[] for _ in range(k)]

    forced_cycle = int(math.floor(k / params.gamma))

    for t in range(horizon):
        if forced_cycle > 0:
            offset = (t - episode_start) % forced_cycle
        else:
            offset = k

        if offset < k:
            arm = int(offset)
        elif np.any(pulls <= 0.0):
            arm = int(np.where(pulls <= 0.0)[0][0])
        else:
            means = successes / np.maximum(pulls, 1e-12)
            log_term = math.log(max(2.0, float(t - episode_start + 1)))
            bonus = np.sqrt(2.0 * params.xi * log_term / np.maximum(pulls, 1e-12))
            arm = int(np.argmax(means + bonus))

        if played_arms is not None:
            played_arms[t] = arm
        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        pulls[arm] += 1.0
        successes[arm] += float(reward)
        histories[arm].append(int(reward))
        if len(histories[arm]) > params.window:
            histories[arm].pop(0)

        hist = histories[arm]
        if len(hist) < params.window:
            continue

        split = params.window // 2
        left_sum = float(sum(hist[:split]))
        right_sum = float(sum(hist[split:]))
        if abs(right_sum - left_sum) > params.threshold:
            restart_rounds.append(t)
            episode_start = t + 1
            pulls.fill(0.0)
            successes.fill(0.0)
            histories = [[] for _ in range(k)]

    if diagnostics is not None:
        diagnostics["implementation_id"] = M_UCB_IMPLEMENTATION_ID
        diagnostics["forced_cycle"] = forced_cycle
        diagnostics["restart_count"] = len(restart_rounds)
        diagnostics["restart_rounds"] = np.asarray(restart_rounds, dtype=int)
        if played_arms is not None:
            diagnostics["played_arms"] = played_arms.copy()

    return rewards


def _run_glr_klucb_prefix_core(
    mu: np.ndarray,
    params: GLRklUCBParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: GLRklUCBRunDiagnostics | None = None,
    *,
    vectorized_index: bool,
) -> np.ndarray:
    """Shared prefix-state runner for the scalar and vectorized-index variants.

    Running success counts replace ``mean(history)`` in the klUCB indices.
    Each retained arm history stores cumulative successes in a packed unsigned
    64-bit array, so detector calls can read prefix sums without rebuilding
    them.  The detector still evaluates every registered split point.
    """
    n_arms, horizon = _validate_mu(mu)
    if params.exploration_alpha0 <= 0.0:
        raise ValueError("exploration_alpha0 must be > 0.")
    if params.delta is not None and not (0.0 < params.delta < 1.0):
        raise ValueError("delta must be in (0, 1) when supplied.")
    if params.delta_exponent <= 0.0:
        raise ValueError("delta_exponent must be > 0.")
    if params.restart_mode not in {"local", "global"}:
        raise ValueError("restart_mode must be 'local' or 'global'.")
    if params.klucb_loglog_coefficient < 0.0:
        raise ValueError("klucb_loglog_coefficient must be >= 0.")
    if params.detection_interval <= 0:
        raise ValueError("detection_interval must be > 0.")
    if params.split_step <= 0:
        raise ValueError("split_step must be > 0.")

    effective_horizon = max(horizon, 2)
    resolved_delta = (
        float(params.delta)
        if params.delta is not None
        else float(effective_horizon ** (-params.delta_exponent))
    )
    resolved_delta = min(resolved_delta, float(np.nextafter(1.0, 0.0)))

    record_played_arms = (
        bool(diagnostics.get("_record_played_arms", False))
        if diagnostics is not None
        else False
    )
    record_final_histories = (
        bool(diagnostics.get("_record_final_histories", False))
        if diagnostics is not None
        else False
    )
    record_allocated_state = (
        bool(diagnostics.get("_record_allocated_state", False))
        if diagnostics is not None
        else False
    )
    rewards = np.zeros(horizon, dtype=int)
    played_arms = np.full(horizon, -1, dtype=int) if record_played_arms else None
    forced_flags = np.zeros(horizon, dtype=bool) if record_played_arms else None

    histories = [array("Q") for _ in range(n_arms)]
    successes = np.zeros(n_arms, dtype=np.int64)
    restart_times = np.zeros(n_arms, dtype=np.int64)
    history_itemsize = histories[0].itemsize
    fixed_logical_bytes = successes.nbytes + restart_times.nbytes
    fixed_allocated_bytes = (
        sys.getsizeof(successes) + sys.getsizeof(restart_times)
        if record_allocated_state
        else 0
    )
    history_allocated_bytes = (
        sys.getsizeof(histories)
        + sum(sys.getsizeof(history) for history in histories)
        if record_allocated_state
        else 0
    )
    retained_observations = 0
    peak_retained_observations = 0
    peak_selected_history_length = 0
    peak_logical_bytes = fixed_logical_bytes
    peak_allocated_bytes = (
        fixed_allocated_bytes + history_allocated_bytes
        if record_allocated_state
        else 0
    )

    episode = 1
    forced_count = 0
    detector_calls = 0
    split_candidates_examined = 0
    restart_rounds: list[int] = []
    restart_arms: list[int] = []
    zero_mean_index_evaluations = 0
    one_mean_index_evaluations = 0

    def update_state_peaks() -> None:
        nonlocal peak_retained_observations
        nonlocal peak_logical_bytes
        nonlocal peak_allocated_bytes
        peak_retained_observations = max(
            peak_retained_observations,
            retained_observations,
        )
        peak_logical_bytes = max(
            peak_logical_bytes,
            fixed_logical_bytes + retained_observations * history_itemsize,
        )
        if record_allocated_state:
            peak_allocated_bytes = max(
                peak_allocated_bytes,
                fixed_allocated_bytes + history_allocated_bytes,
            )

    for t in range(horizon):
        round_number = t + 1
        exploration_probability = params.exploration_alpha0 * math.sqrt(
            episode * n_arms * math.log(float(effective_horizon))
            / float(effective_horizon)
        )
        exploration_probability = min(
            max(exploration_probability, float(np.nextafter(0.0, 1.0))),
            float(np.nextafter(1.0, 0.0)),
        )
        forced_cycle = int(math.floor(n_arms / exploration_probability))
        forced_remainder = round_number % forced_cycle
        is_forced = 1 <= forced_remainder <= n_arms

        if is_forced:
            arm = forced_remainder - 1
            forced_count += 1
        else:
            if vectorized_index:
                counts = np.fromiter(
                    (len(history) for history in histories),
                    dtype=np.int64,
                    count=n_arms,
                )
                means = np.zeros(n_arms, dtype=np.float64)
                bonuses = np.zeros(n_arms, dtype=np.float64)
                observed = counts > 0
                means[observed] = successes[observed] / counts[observed]
                for arm_idx in np.flatnonzero(observed):
                    age = max(
                        round_number - int(restart_times[arm_idx]),
                        1,
                    )
                    bonuses[arm_idx] = _klucb_exploration_function(
                        age,
                        params.klucb_loglog_coefficient,
                    )
                scores = _kl_ucb_bernoulli_vectorized(
                    means,
                    counts,
                    bonuses,
                )
            else:
                scores = np.empty(n_arms, dtype=float)
                means = np.zeros(n_arms, dtype=np.float64)
                observed = np.zeros(n_arms, dtype=bool)
                for arm_idx, history in enumerate(histories):
                    n = len(history)
                    if n == 0:
                        scores[arm_idx] = 1.0
                        continue
                    observed[arm_idx] = True
                    mean = float(successes[arm_idx]) / float(n)
                    means[arm_idx] = mean
                    age = max(round_number - int(restart_times[arm_idx]), 1)
                    bonus = _klucb_exploration_function(
                        age,
                        params.klucb_loglog_coefficient,
                    )
                    scores[arm_idx] = _kl_ucb_bernoulli(
                        mean=mean,
                        n=n,
                        bonus=bonus,
                    )
            if diagnostics is not None:
                zero_mean_index_evaluations += int(
                    np.count_nonzero(observed & (means == 0.0))
                )
                one_mean_index_evaluations += int(
                    np.count_nonzero(observed & (means == 1.0))
                )
            maximum = float(np.max(scores))
            tied = np.flatnonzero(scores == maximum)
            arm = int(rng.choice(tied))

        if played_arms is not None:
            played_arms[t] = arm
            forced_flags[t] = is_forced
        reward = _pull_from_returns_or_mu(mu, returns, arm, t, rng)
        rewards[t] = reward

        history = histories[arm]
        allocated_before = (
            sys.getsizeof(history) if record_allocated_state else 0
        )
        successes[arm] += reward
        history.append(int(successes[arm]))
        if record_allocated_state:
            history_allocated_bytes += (
                sys.getsizeof(history) - allocated_before
            )
        retained_observations += 1
        peak_selected_history_length = max(
            peak_selected_history_length,
            len(history),
        )
        update_state_peaks()

        detector_age = max(round_number - int(restart_times[arm]), 1)
        if (detector_age - 1) % params.detection_interval != 0:
            continue

        detector_calls += 1
        n = len(history)
        split_candidates_examined += (n - 1) // params.split_step
        statistic, _best_split = _bernoulli_glr_max_statistic_from_prefix(
            history,
            split_step=params.split_step,
        )
        threshold = _glr_practical_threshold(n, resolved_delta)
        if statistic < threshold:
            continue

        restart_rounds.append(round_number)
        restart_arms.append(arm if params.restart_mode == "local" else -1)
        episode += 1
        if params.restart_mode == "local":
            old_history = histories[arm]
            old_length = len(old_history)
            old_allocated = (
                sys.getsizeof(old_history) if record_allocated_state else 0
            )
            histories[arm] = array("Q")
            if record_allocated_state:
                history_allocated_bytes += (
                    sys.getsizeof(histories[arm]) - old_allocated
                )
            retained_observations -= old_length
            successes[arm] = 0
            restart_times[arm] = round_number
        else:
            if record_allocated_state:
                history_allocated_bytes -= sum(
                    sys.getsizeof(item) for item in histories
                )
            retained_observations = 0
            histories = [array("Q") for _ in range(n_arms)]
            if record_allocated_state:
                history_allocated_bytes += sum(
                    sys.getsizeof(item) for item in histories
                )
            successes.fill(0)
            restart_times.fill(round_number)

    if diagnostics is not None:
        final_logical_bytes = (
            fixed_logical_bytes + retained_observations * history_itemsize
        )
        final_allocated_bytes = (
            fixed_allocated_bytes + history_allocated_bytes
            if record_allocated_state
            else 0
        )
        diagnostics["implementation_id"] = (
            GLR_KLUCB_VECTORIZED_INDEX_IMPLEMENTATION_ID
            if vectorized_index
            else GLR_KLUCB_PREFIX_OPTIMIZED_IMPLEMENTATION_ID
        )
        diagnostics["reference_implementation_id"] = GLR_KLUCB_IMPLEMENTATION_ID
        diagnostics["index_computation"] = (
            "float64_vectorized_40_step_bisection"
            if vectorized_index
            else "scalar_40_step_bisection"
        )
        if vectorized_index:
            diagnostics["optimization_reference_implementation_id"] = (
                GLR_KLUCB_PREFIX_OPTIMIZED_IMPLEMENTATION_ID
            )
        diagnostics["resolved_delta"] = resolved_delta
        diagnostics["restart_mode"] = params.restart_mode
        diagnostics["klucb_index"] = (
            "log_age"
            if params.klucb_loglog_coefficient == 0.0
            else f"log_age_plus_{params.klucb_loglog_coefficient:g}_loglog_age"
        )
        diagnostics["detection_interval"] = int(params.detection_interval)
        diagnostics["split_step"] = int(params.split_step)
        diagnostics["episode_count"] = int(episode)
        diagnostics["restart_count"] = len(restart_rounds)
        diagnostics["forced_exploration_count"] = int(forced_count)
        diagnostics["zero_mean_index_evaluations"] = int(
            zero_mean_index_evaluations
        )
        diagnostics["one_mean_index_evaluations"] = int(
            one_mean_index_evaluations
        )
        diagnostics["detector_call_count"] = int(detector_calls)
        diagnostics["split_candidates_examined"] = int(
            split_candidates_examined
        )
        diagnostics["peak_selected_history_length"] = int(
            peak_selected_history_length
        )
        diagnostics["peak_retained_observations"] = int(
            peak_retained_observations
        )
        diagnostics["final_retained_observations"] = int(
            retained_observations
        )
        diagnostics["history_itemsize_bytes"] = int(history_itemsize)
        diagnostics["state_representation"] = (
            "uint64 cumulative-success prefix per retained observation"
        )
        diagnostics["final_logical_model_state_bytes"] = int(
            final_logical_bytes
        )
        diagnostics["peak_logical_model_state_bytes"] = int(
            peak_logical_bytes
        )
        if record_allocated_state:
            diagnostics["final_allocated_model_state_bytes"] = int(
                final_allocated_bytes
            )
            diagnostics["peak_allocated_model_state_bytes"] = int(
                peak_allocated_bytes
            )
        diagnostics["restart_rounds"] = np.asarray(restart_rounds, dtype=int)
        diagnostics["restart_arms"] = np.asarray(restart_arms, dtype=int)
        diagnostics["final_restart_times"] = restart_times.copy()
        diagnostics["final_history_lengths"] = np.asarray(
            [len(history) for history in histories],
            dtype=int,
        )
        if record_final_histories:
            diagnostics["final_histories"] = tuple(
                np.diff(
                    np.concatenate(
                        (
                            np.zeros(1, dtype=np.int64),
                            np.asarray(history, dtype=np.int64),
                        )
                    )
                ).astype(np.int8)
                for history in histories
            )
        if played_arms is not None:
            diagnostics["played_arms"] = played_arms
            diagnostics["forced_exploration"] = forced_flags

    return rewards


def run_glr_klucb_vectorized_index(
    mu: np.ndarray,
    params: GLRklUCBParams,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: GLRklUCBRunDiagnostics | None = None,
) -> np.ndarray:
    """Run prefix-state GLR-klUCB with its arm-index bisection vectorized.

    The separately identified scalar implementation remains the fidelity
    oracle.  This implementation changes only how independent arm indices are
    evaluated; forced exploration, tie-breaking, detection, and restart state
    share the prefix runner above.
    """

    return _run_glr_klucb_prefix_core(
        mu,
        params,
        rng,
        returns=returns,
        diagnostics=diagnostics,
        vectorized_index=True,
    )
