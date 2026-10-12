"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from eats.algorithms.common import _draw_scores
from eats.algorithms.common import _pull
from eats.algorithms.common import _validate_inputs
from eats.algorithms.e_ats import DEPLOYED_RESTART_ETA
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
from typing import Any
import math
import numpy as np


E_ATS_GLR_FULL_BUDGET_RAMP_IMPLEMENTATION_ID = (
    "e_ats_glr_authorized_full_budget_normalized_ramp_v1"
)


def _bernoulli_maximized_log_likelihood(successes: int, trials: int) -> float:
    """Return the Bernoulli log likelihood at its MLE, including endpoints."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes and trials must describe a nonempty block")
    failures = trials - successes
    value = 0.0
    if successes:
        value += successes * math.log(successes / trials)
    if failures:
        value += failures * math.log(failures / trials)
    return value


def bernoulli_two_segment_glr(
    *,
    prefix_successes: int,
    prefix_trials: int,
    total_successes: int,
    total_trials: int,
) -> float:
    """Log GLR for two Bernoulli means versus one pooled Bernoulli mean."""

    suffix_trials = total_trials - prefix_trials
    suffix_successes = total_successes - prefix_successes
    if prefix_trials <= 0 or suffix_trials <= 0:
        raise ValueError("both GLR segments must be nonempty")
    if not (
        0 <= prefix_successes <= prefix_trials
        and 0 <= suffix_successes <= suffix_trials
    ):
        raise ValueError("success counts are inconsistent")
    statistic = (
        _bernoulli_maximized_log_likelihood(prefix_successes, prefix_trials)
        + _bernoulli_maximized_log_likelihood(suffix_successes, suffix_trials)
        - _bernoulli_maximized_log_likelihood(total_successes, total_trials)
    )
    # The exact statistic is nonnegative.  Each block calculation is O(n), so
    # use an explicit binary64 forward-error allowance that scales with the
    # number of summed log-likelihood contributions.
    tolerance = 64.0 * np.finfo(np.float64).eps * total_trials
    if statistic < -tolerance:
        raise ArithmeticError(
            f"negative Bernoulli GLR statistic: {statistic} < {-tolerance}"
        )
    return max(0.0, statistic)


@dataclass
class GeometricBernoulliGLRState:
    """Arm-local GLR over the same geometric split grid as canonical e-ATS."""

    eta: float = DEPLOYED_RESTART_ETA
    pull_count: int = 0
    total_successes: int = 0
    starts: list[int] = field(default_factory=list)
    prefix_successes: list[int] = field(default_factory=list)
    next_start: int = RESTART_FIRST_CANDIDATE_PULL
    maximum_statistic: float = 0.0
    xlogx_table: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.eta != DEPLOYED_RESTART_ETA:
            raise ValueError(
                f"the comparator is frozen at eta={DEPLOYED_RESTART_ETA}"
            )
        if self.xlogx_table is not None:
            table = np.asarray(self.xlogx_table, dtype=float)
            if table.ndim != 1 or table.size < 2 or table[0] != 0.0:
                raise ValueError("xlogx_table must contain x*log(x) from x=0")
            self.xlogx_table = table

    def update(self, reward: int) -> float:
        """Update one arm-local reward and return the current maximum split GLR."""

        if reward not in (0, 1):
            raise ValueError("reward must be binary")
        current_pull = self.pull_count + 1
        if self.next_start == current_pull:
            self.starts.append(current_pull)
            self.prefix_successes.append(self.total_successes)
            self.next_start = max(
                current_pull + 1,
                math.ceil((1.0 + self.eta) * current_pull),
            )
        self.total_successes += reward
        self.pull_count = current_pull
        current_maximum = 0.0
        pooled_log_likelihood = None
        if self.xlogx_table is not None:
            if current_pull >= self.xlogx_table.size:
                raise ValueError("xlogx_table is shorter than the reward prefix")
            pooled_log_likelihood = (
                self.xlogx_table[self.total_successes]
                + self.xlogx_table[current_pull - self.total_successes]
                - self.xlogx_table[current_pull]
            )
        for start, prefix_successes in zip(
            self.starts, self.prefix_successes, strict=True
        ):
            if self.xlogx_table is None:
                statistic = bernoulli_two_segment_glr(
                    prefix_successes=prefix_successes,
                    prefix_trials=start - 1,
                    total_successes=self.total_successes,
                    total_trials=current_pull,
                )
            else:
                prefix_trials = start - 1
                suffix_trials = current_pull - prefix_trials
                suffix_successes = self.total_successes - prefix_successes
                statistic = (
                    self.xlogx_table[prefix_successes]
                    + self.xlogx_table[prefix_trials - prefix_successes]
                    - self.xlogx_table[prefix_trials]
                    + self.xlogx_table[suffix_successes]
                    + self.xlogx_table[suffix_trials - suffix_successes]
                    - self.xlogx_table[suffix_trials]
                    - float(pooled_log_likelihood)
                )
                if statistic < 0.0:
                    # The x*log(x) identity subtracts O(n log n) terms and can
                    # lose a few ulps when the true GLR is near zero.  Recompute
                    # only those cases from blockwise likelihoods rather than
                    # accepting or clipping a scale-dependent negative value.
                    statistic = bernoulli_two_segment_glr(
                        prefix_successes=prefix_successes,
                        prefix_trials=prefix_trials,
                        total_successes=self.total_successes,
                        total_trials=current_pull,
                    )
            current_maximum = max(current_maximum, statistic)
        self.maximum_statistic = max(self.maximum_statistic, current_maximum)
        return current_maximum

    def state_size_bytes(self) -> int:
        return 5 * 8 + len(self.starts) * 2 * 8


@dataclass(frozen=True)
class GLRAuthorizationConfig:
    """A frozen scalar threshold for one registered ``(K, B)`` geometry."""

    threshold: float
    calibration_id: str
    eta: float = DEPLOYED_RESTART_ETA

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("the calibrated GLR threshold must be finite and positive")
        if not self.calibration_id:
            raise ValueError("calibration_id must be nonempty")
        if self.eta != DEPLOYED_RESTART_ETA:
            raise ValueError(
                f"the comparator is frozen at eta={DEPLOYED_RESTART_ETA}"
            )


def glr_comparator_contract(config: GLRAuthorizationConfig) -> dict[str, Any]:
    """Return the behavior-complete policy contract for audit artifacts."""

    return {
        "implementation_id": E_ATS_GLR_FULL_BUDGET_RAMP_IMPLEMENTATION_ID,
        "authorization_detector": "classical_two_segment_bernoulli_glr",
        "glr_formula": "max_split_log_mle_two_segments_over_mle_pooled",
        "candidate_grid": "same_geometric_grid_as_canonical_e_ats",
        "restart_eta": config.eta,
        "restart_first_candidate_pull": RESTART_FIRST_CANDIDATE_PULL,
        "authorization_coupling": "absorbing_historical_latch",
        "threshold": config.threshold,
        "calibration_id": config.calibration_id,
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


def run_e_ats_glr_full_budget_ramp(
    mu: np.ndarray,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, Any] | None = None,
    *,
    authorization: GLRAuthorizationConfig,
) -> np.ndarray:
    """Run canonical e-ATS with only its e-process latch replaced by GLR."""

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
    first_glr = np.full(n_arms, -1, dtype=np.int64)
    first_ramp = np.full(n_arms, -1, dtype=np.int64)
    first_deployed = np.full(n_arms, -1, dtype=np.int64)
    minimum_mean_zero_distance = math.inf
    minimum_mean_saturation_distance = math.inf
    minimum_glr_threshold_distance = math.inf

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
        strength, boundary = update_full_budget_normalized_ramp(
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
            abs(magnitude - boundary),
        )
        if strength > 0.0 and first_ramp[arm] < 0:
            first_ramp[arm] = timestep

        if not glr_authorized[arm]:
            glr = glr_states[arm].update(reward)
            played_glr_after[timestep] = glr
            minimum_glr_threshold_distance = min(
                minimum_glr_threshold_distance,
                abs(glr - authorization.threshold),
            )
            if glr >= authorization.threshold:
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
        diagnostics.update(
            {
                "implementation_id": E_ATS_GLR_FULL_BUDGET_RAMP_IMPLEMENTATION_ID,
                "contract": glr_comparator_contract(authorization),
                "played_arms": played_arms,
                "played_gate_before": played_gate_before,
                "played_ramp_after": played_ramp_after,
                "played_gate_after": played_gate_after,
                "played_glr_after": played_glr_after,
                "first_glr_crossing_update_timestep": first_glr,
                "first_ramp_positive_update_timestep": first_ramp,
                "first_deployed_gate_update_timestep": first_deployed,
                "final_glr_authorized": glr_authorized,
                "authorized_arm_count": int(np.count_nonzero(glr_authorized)),
                "minimum_abs_mean_zero_distance": minimum_mean_zero_distance,
                "minimum_abs_mean_saturation_distance": (
                    minimum_mean_saturation_distance
                ),
                "minimum_abs_glr_threshold_distance": (
                    minimum_glr_threshold_distance
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
