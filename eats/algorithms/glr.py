"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from dataclasses import dataclass
from eats.algorithms.calibrated_glr import GeometricBernoulliGLRState
from eats.algorithms.common import _draw_scores
from eats.algorithms.common import _pull
from eats.algorithms.common import _validate_inputs
from eats.algorithms.e_ats import DEPLOYED_RESTART_ETA
from eats.algorithms.e_ats import E_ATS_AUTHORIZATION_ALPHA
from eats.algorithms.e_ats import E_ATS_DETECTOR_DECAY_CAP
from eats.algorithms.e_ats import E_ATS_DUAL_MEMORY_MIX_POWER
from eats.algorithms.e_ats import E_ATS_GLOBAL_TAKEOVER_SCALE
from eats.algorithms.e_ats import E_ATS_LOCAL_TAKEOVER_SCALE
from eats.algorithms.e_ats import E_ATS_MEAN_NORMALIZATION_BUDGET
from eats.algorithms.e_ats import E_ATS_PRIOR_ALPHA
from eats.algorithms.e_ats import E_ATS_PRIOR_BETA
from eats.algorithms.e_ats import E_ATS_SHORT_DISCOUNT
from eats.algorithms.e_ats import MEAN_NUMERICAL_EPSILON
from eats.algorithms.e_ats import RESTART_FIRST_CANDIDATE_PULL
from eats.algorithms.e_ats import detector_decay
from eats.algorithms.e_ats import update_full_budget_normalized_ramp
from eats.algorithms.scaling import derive_vgdts_v22_secondary_params
from functools import lru_cache
from typing import Any
from typing import Literal
import math
import numpy as np


BoundaryKind = Literal["besson_lemma2", "besson_practical"]


TIME_UNIFORM_GLR_IMPLEMENTATION_ID = (
    "e_ats_time_uniform_glr_authorized_full_budget_normalized_ramp_v1"
)


PRACTICAL_GLR_IMPLEMENTATION_ID = (
    "e_ats_practical_glr_authorized_full_budget_normalized_ramp_v1"
)


@lru_cache(maxsize=None)
def h_inverse(value: float) -> float:
    """Invert ``h(u)=u-log(u)`` on its increasing branch ``u>=1``.

    A bracketed Newton iteration avoids an optional Lambert-W dependency and
    is deterministic across the registered Python environment.  It retains a
    monotone bracket, so the derivative's flatness near one cannot send an
    iterate to the wrong inverse branch.
    """

    if not math.isfinite(value) or value < 1.0:
        raise ValueError("h_inverse requires a finite value at least one")
    if value == 1.0:
        return 1.0
    lower = 1.0
    upper = max(2.0, value + math.log(value) + 1.0)
    while upper - math.log(upper) < value:
        upper *= 2.0
    current = upper
    for _ in range(16):
        function = current - math.log(current) - value
        if abs(function) <= 2.0 * math.ulp(value):
            return current
        derivative = 1.0 - 1.0 / current
        candidate = current - function / derivative
        if not lower < candidate < upper:
            candidate = lower + 0.5 * (upper - lower)
        candidate_value = candidate - math.log(candidate)
        if abs(candidate_value - value) <= 2.0 * math.ulp(value):
            return candidate
        if candidate_value < value:
            lower = candidate
        else:
            upper = candidate
        current = candidate
    for _ in range(8):
        midpoint = lower + 0.5 * (upper - lower)
        if midpoint - math.log(midpoint) < value:
            lower = midpoint
        else:
            upper = midpoint
    return lower + 0.5 * (upper - lower)


def _tilde_h(value: float) -> float:
    """Kaufmann--Koolen calibration function used by Besson et al."""

    if not math.isfinite(value) or value < 1.0:
        raise ValueError("tilde_h requires a finite value at least one")
    transition_u = 1.0 / math.log(1.5)
    transition = transition_u - math.log(transition_u)
    if value >= transition:
        inverse = h_inverse(value)
        return math.exp(1.0 / inverse) * inverse
    return 1.5 * (value - math.log(math.log(1.5)))


@lru_cache(maxsize=None)
def kaufmann_koolen_T(value: float) -> float:
    """Return the function T in Besson et al. (2022), Equation (3)."""

    if not math.isfinite(value) or value < 0.0:
        raise ValueError("T requires a finite nonnegative value")
    argument = 0.5 * (
        h_inverse(1.0 + value) + math.log(math.pi * math.pi / 3.0)
    )
    return 2.0 * _tilde_h(argument)


@lru_cache(maxsize=None)
def besson_time_uniform_glr_boundary(local_n: int, delta: float) -> float:
    """The theorem-backed Bernoulli-GLR boundary from Besson et al., Lemma 2."""

    if local_n < 1:
        raise ValueError("local_n must be positive")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    log_argument = math.log(3.0 / delta) + 1.5 * math.log(local_n)
    return (
        2.0 * kaufmann_koolen_T(0.5 * log_argument)
        + 6.0 * math.log1p(math.log(local_n))
    )


@lru_cache(maxsize=None)
def besson_practical_glr_boundary(local_n: int, delta: float) -> float:
    """The calibration-free practical boundary recommended in Besson et al."""

    if local_n < 1:
        raise ValueError("local_n must be positive")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    return 1.5 * math.log(local_n) + math.log(1.0 / delta)


def glr_boundary(local_n: int, delta: float, kind: BoundaryKind) -> float:
    """Dispatch one frozen GLR boundary specification."""

    if kind == "besson_lemma2":
        return besson_time_uniform_glr_boundary(local_n, delta)
    if kind == "besson_practical":
        return besson_practical_glr_boundary(local_n, delta)
    raise ValueError(f"unknown GLR boundary kind: {kind}")


@dataclass(frozen=True)
class TimeUniformGLRAuthorizationConfig:
    """Calibration-free GLR authorization configuration."""

    boundary_kind: BoundaryKind
    familywise_alpha: float = E_ATS_AUTHORIZATION_ALPHA
    eta: float = DEPLOYED_RESTART_ETA

    def __post_init__(self) -> None:
        if self.boundary_kind not in ("besson_lemma2", "besson_practical"):
            raise ValueError("unsupported GLR boundary kind")
        if not math.isfinite(self.familywise_alpha) or not (
            0.0 < self.familywise_alpha < 1.0
        ):
            raise ValueError("familywise_alpha must lie strictly between zero and one")
        if self.eta != DEPLOYED_RESTART_ETA:
            raise ValueError(
                f"the comparator is frozen at eta={DEPLOYED_RESTART_ETA}"
            )


def time_uniform_glr_contract(
    config: TimeUniformGLRAuthorizationConfig, n_arms: int
) -> dict[str, Any]:
    """Return the behavior-complete detector/controller contract."""

    if n_arms <= 0:
        raise ValueError("n_arms must be positive")
    theorem_backed = config.boundary_kind == "besson_lemma2"
    return {
        "implementation_id": (
            TIME_UNIFORM_GLR_IMPLEMENTATION_ID
            if theorem_backed
            else PRACTICAL_GLR_IMPLEMENTATION_ID
        ),
        "authorization_detector": "two_segment_bernoulli_glr",
        "glr_formula": "max_split_log_mle_two_segments_over_mle_pooled",
        "candidate_grid": "same_geometric_grid_as_canonical_e_ats",
        "boundary_kind": config.boundary_kind,
        "boundary_reference": (
            "Besson_Kaufmann_Maillard_Seznec_2022_Lemma_2_Equation_4"
            if theorem_backed
            else "Besson_Kaufmann_Maillard_Seznec_2022_Section_3_3_practical"
        ),
        "analytic_anytime_false_alarm_guarantee": theorem_backed,
        "analytic_null_scope": (
            "every_fixed_stationary_Bernoulli_mean"
            if theorem_backed
            else "not_claimed"
        ),
        "familywise_alpha": config.familywise_alpha,
        "per_arm_delta": config.familywise_alpha / n_arms,
        "arm_multiplicity_correction": "Bonferroni_union_bound",
        "empirical_threshold_parameters": 0,
        "horizon_required_by_boundary": False,
        "restart_eta": config.eta,
        "restart_first_candidate_pull": RESTART_FIRST_CANDIDATE_PULL,
        "authorization_coupling": "absorbing_historical_latch",
        "mean_normalization_budget": E_ATS_MEAN_NORMALIZATION_BUDGET,
        "mean_numerical_epsilon": MEAN_NUMERICAL_EPSILON,
        "detector_decay": "clip_1_minus_inverse_sqrt_horizon_0_cap",
        "detector_decay_cap": E_ATS_DETECTOR_DECAY_CAP,
        "short_discount": E_ATS_SHORT_DISCOUNT,
        "short_discount_clock": "every_global_round",
        "dual_memory_mix_power": E_ATS_DUAL_MEMORY_MIX_POWER,
        "local_takeover_scale": E_ATS_LOCAL_TAKEOVER_SCALE,
        "global_takeover_scale": E_ATS_GLOBAL_TAKEOVER_SCALE,
        "sampling": "convex",
        "score": "optimistic_max_sample_and_posterior_mean",
        "gate_timing": "reward_at_t_updates_gate_for_action_t_plus_1",
    }


def run_e_ats_time_uniform_glr_full_budget_ramp(
    mu: np.ndarray,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, Any] | None = None,
    *,
    authorization: TimeUniformGLRAuthorizationConfig,
) -> np.ndarray:
    """Run e-ATS with only its historical authorization detector replaced."""

    means, observed = _validate_inputs(mu, returns)
    n_arms, horizon = means.shape
    if horizon <= 0:
        raise ValueError("mu must have a positive horizon")
    derived = derive_vgdts_v22_secondary_params(
        alpha0=E_ATS_PRIOR_ALPHA,
        beta0=E_ATS_PRIOR_BETA,
        horizon=horizon,
        n_arms=n_arms,
    )
    reliability_mass = float(derived["shock_takeover_reliability_mass"])
    decay = detector_decay(horizon)
    per_arm_delta = authorization.familywise_alpha / n_arms

    alpha_short = np.full(n_arms, E_ATS_PRIOR_ALPHA, dtype=float)
    beta_short = np.full(n_arms, E_ATS_PRIOR_BETA, dtype=float)
    alpha_long = np.full(n_arms, E_ATS_PRIOR_ALPHA, dtype=float)
    beta_long = np.full(n_arms, E_ATS_PRIOR_BETA, dtype=float)
    pull_counts = np.zeros(n_arms, dtype=np.int64)
    mean_evidence = np.zeros(n_arms, dtype=float)
    mean_range_sq = np.zeros(n_arms, dtype=float)
    deployed_gate = np.zeros(n_arms, dtype=float)
    indices = np.arange(horizon + 1, dtype=float)
    xlogx_table = np.zeros(horizon + 1, dtype=float)
    xlogx_table[1:] = indices[1:] * np.log(indices[1:])
    glr_states = [
        GeometricBernoulliGLRState(
            authorization.eta,
            xlogx_table=xlogx_table,
        )
        for _ in range(n_arms)
    ]
    glr_authorized = np.zeros(n_arms, dtype=bool)

    rewards = np.zeros(horizon, dtype=np.int64)
    played_arms = np.full(horizon, -1, dtype=np.int64)
    played_gate_before = np.zeros(horizon, dtype=float)
    played_ramp_after = np.zeros(horizon, dtype=float)
    played_gate_after = np.zeros(horizon, dtype=float)
    played_glr_after = np.full(horizon, np.nan, dtype=float)
    played_boundary_after = np.full(horizon, np.nan, dtype=float)
    first_glr = np.full(n_arms, -1, dtype=np.int64)
    first_ramp = np.full(n_arms, -1, dtype=np.int64)
    first_deployed = np.full(n_arms, -1, dtype=np.int64)
    minimum_mean_zero_distance = math.inf
    minimum_mean_saturation_distance = math.inf
    minimum_glr_boundary_distance = math.inf

    for timestep in range(horizon):
        n_eff_short = np.maximum(
            alpha_short + beta_short - E_ATS_PRIOR_ALPHA - E_ATS_PRIOR_BETA,
            0.0,
        )
        reliability = n_eff_short / (n_eff_short + reliability_mass)
        base_weight = np.power(deployed_gate, E_ATS_DUAL_MEMORY_MIX_POWER)
        short_weight = base_weight + (
            E_ATS_LOCAL_TAKEOVER_SCALE
            * (1.0 - base_weight)
            * deployed_gate
            * reliability
        )
        short_weight = np.clip(short_weight, 0.0, 1.0)
        scores, _ = _draw_scores(
            alpha_short=alpha_short,
            beta_short=beta_short,
            alpha_long=alpha_long,
            beta_long=beta_long,
            short_weight=short_weight,
            sampling="convex",
            rng=rng,
        )
        arm = int(np.argmax(scores))
        reward = _pull(means, observed, arm, timestep, rng)
        played_arms[timestep] = arm
        rewards[timestep] = reward
        played_gate_before[timestep] = deployed_gate[arm]

        prediction = float(alpha_long[arm] / (alpha_long[arm] + beta_long[arm]))
        strength, mean_boundary = update_full_budget_normalized_ramp(
            arm=arm,
            reward=reward,
            prediction=prediction,
            decay=decay,
            n_arms=n_arms,
            pull_counts=pull_counts,
            mean_evidence=mean_evidence,
            mean_range_sq=mean_range_sq,
        )
        magnitude = abs(float(mean_evidence[arm]))
        minimum_mean_zero_distance = min(
            minimum_mean_zero_distance,
            abs(magnitude - MEAN_NUMERICAL_EPSILON),
        )
        minimum_mean_saturation_distance = min(
            minimum_mean_saturation_distance,
            abs(magnitude - mean_boundary),
        )
        if strength > 0.0 and first_ramp[arm] < 0:
            first_ramp[arm] = timestep

        if not glr_authorized[arm]:
            statistic = glr_states[arm].update(reward)
            local_n = glr_states[arm].pull_count
            boundary = glr_boundary(
                local_n,
                per_arm_delta,
                authorization.boundary_kind,
            )
            played_glr_after[timestep] = statistic
            played_boundary_after[timestep] = boundary
            minimum_glr_boundary_distance = min(
                minimum_glr_boundary_distance,
                abs(statistic - boundary),
            )
            if statistic >= boundary:
                glr_authorized[arm] = True
                first_glr[arm] = timestep
        deployed_gate[arm] = strength if glr_authorized[arm] else 0.0
        if deployed_gate[arm] > 0.0 and first_deployed[arm] < 0:
            first_deployed[arm] = timestep
        played_ramp_after[timestep] = strength
        played_gate_after[timestep] = deployed_gate[arm]

        alpha_short = E_ATS_PRIOR_ALPHA + E_ATS_SHORT_DISCOUNT * (
            alpha_short - E_ATS_PRIOR_ALPHA
        )
        beta_short = E_ATS_PRIOR_BETA + E_ATS_SHORT_DISCOUNT * (
            beta_short - E_ATS_PRIOR_BETA
        )
        alpha_short[arm] += reward
        beta_short[arm] += 1 - reward
        alpha_long[arm] += reward
        beta_long[arm] += 1 - reward

    if diagnostics is not None:
        theorem_backed = authorization.boundary_kind == "besson_lemma2"
        diagnostics.update(
            {
                "implementation_id": (
                    TIME_UNIFORM_GLR_IMPLEMENTATION_ID
                    if theorem_backed
                    else PRACTICAL_GLR_IMPLEMENTATION_ID
                ),
                "contract": time_uniform_glr_contract(authorization, n_arms),
                "played_arms": played_arms,
                "played_gate_before": played_gate_before,
                "played_ramp_after": played_ramp_after,
                "played_gate_after": played_gate_after,
                "played_glr_after": played_glr_after,
                "played_glr_boundary_after": played_boundary_after,
                "first_glr_crossing_update_timestep": first_glr,
                "first_ramp_positive_update_timestep": first_ramp,
                "first_deployed_gate_update_timestep": first_deployed,
                "final_glr_authorized": glr_authorized,
                "authorized_arm_count": int(np.count_nonzero(glr_authorized)),
                "minimum_abs_mean_zero_distance": minimum_mean_zero_distance,
                "minimum_abs_mean_saturation_distance": (
                    minimum_mean_saturation_distance
                ),
                "minimum_abs_glr_boundary_distance": (
                    minimum_glr_boundary_distance
                ),
                "live_segment_components": sum(
                    len(state.starts) for state in glr_states
                ),
                "packed_detector_state_bytes": sum(
                    state.state_size_bytes() for state in glr_states
                ),
            }
        )
    return rewards
