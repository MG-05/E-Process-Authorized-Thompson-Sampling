"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from eats.algorithms.common import VGDTSV3Params
from eats.algorithms.common import _draw_scores
from eats.algorithms.common import _pull
from eats.algorithms.common import _validate_inputs
from eats.algorithms.common import _validate_params
from eats.algorithms.scaling import DEFAULT_VGDTS_V22_HORIZON
from eats.algorithms.scaling import DEFAULT_VGDTS_V22_N_ARMS
from eats.algorithms.scaling import derive_vgdts_v22_secondary_params
from typing import Any
import math
import numpy as np


E_ATS_FULL_BUDGET_RAMP_IMPLEMENTATION_ID = (
    "e_ats_full_budget_normalized_ramp_v1"
)


E_ATS_FULL_BUDGET_RAMP_NO_CE_ABLATION_ID = (
    "e_ats_full_budget_normalized_ramp_no_ce_v1"
)


DEPLOYED_RESTART_ETA = 0.25


RESTART_FIRST_CANDIDATE_PULL = 2


PRIOR_PREDICTIVE_NULL = "independent_per_arm_beta_1_1_bernoulli"


MEAN_CERTIFICATE_FALSE_ALARM_BUDGET = 0.05


MEAN_NUMERICAL_EPSILON = 1e-12


E_ATS_AUTHORIZATION_ALPHA = 0.05


E_ATS_MEAN_NORMALIZATION_BUDGET = MEAN_CERTIFICATE_FALSE_ALARM_BUDGET


E_ATS_PRIOR_ALPHA = 1.0


E_ATS_PRIOR_BETA = 1.0


E_ATS_DETECTOR_DECAY_CAP = 0.999


E_ATS_SHORT_DISCOUNT = 0.995


E_ATS_DUAL_MEMORY_MIX_POWER = 0.30


E_ATS_LOCAL_TAKEOVER_SCALE = 0.80


E_ATS_GLOBAL_TAKEOVER_SCALE = 0.0


def _logsumexp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _start_weight(index: int) -> float:
    if index <= 0:
        raise ValueError("restart component index must be positive.")
    return 1.0 / (index * (index + 1.0))


def _future_capital(active_component_count: int) -> float:
    if active_component_count < 0:
        raise ValueError("active component count must be nonnegative.")
    return 1.0 / (active_component_count + 1.0)


@dataclass
class RestartEProcessState:
    """Production log-space restart e-process for one arm in pull time."""

    eta: float
    pull_count: int = 0
    total_successes: int = 0
    total_failures: int = 0
    starts: list[int] = field(default_factory=list)
    component_log_bfs: list[float] = field(default_factory=list)
    segment_successes: list[int] = field(default_factory=list)
    segment_failures: list[int] = field(default_factory=list)
    next_start: int = RESTART_FIRST_CANDIDATE_PULL

    def __post_init__(self) -> None:
        if self.eta != DEPLOYED_RESTART_ETA:
            raise ValueError(
                f"the deployed construction is frozen at eta={DEPLOYED_RESTART_ETA}."
            )

    def log_evidence(self) -> float:
        terms = tuple(
            math.log(_start_weight(index)) + log_bf
            for index, log_bf in enumerate(self.component_log_bfs, start=1)
        )
        return _logsumexp(
            (*terms, math.log(_future_capital(len(self.starts))))
        )

    def update(self, reward: int) -> float:
        if reward not in (0, 1):
            raise ValueError("reward must be binary.")
        current_pull = self.pull_count + 1
        if self.next_start == current_pull:
            self.starts.append(current_pull)
            self.component_log_bfs.append(0.0)
            self.segment_successes.append(0)
            self.segment_failures.append(0)
            self.next_start = max(
                current_pull + 1,
                math.ceil((1.0 + self.eta) * current_pull),
            )
        for component in range(len(self.starts)):
            if reward:
                segment_numerator = self.segment_successes[component] + 1.0
                null_numerator = self.total_successes + 1.0
            else:
                segment_numerator = self.segment_failures[component] + 1.0
                null_numerator = self.total_failures + 1.0
            segment_denominator = (
                self.segment_successes[component]
                + self.segment_failures[component]
                + 2.0
            )
            null_denominator = self.total_successes + self.total_failures + 2.0
            self.component_log_bfs[component] += math.log(
                segment_numerator / segment_denominator
            ) - math.log(null_numerator / null_denominator)
            self.segment_successes[component] += reward
            self.segment_failures[component] += 1 - reward
        self.total_successes += reward
        self.total_failures += 1 - reward
        self.pull_count = current_pull
        return self.log_evidence()

    def state_size_bytes(self) -> int:
        """Return a deterministic packed-state size, excluding Python overhead."""

        scalar_bytes = 5 * 8
        component_bytes = len(self.starts) * (8 + 8 + 8 + 8)
        return scalar_bytes + component_bytes


@dataclass(frozen=True)
class EValueAuthorizationConfig:
    eta: float = DEPLOYED_RESTART_ETA
    alpha: float = E_ATS_AUTHORIZATION_ALPHA

    def __post_init__(self) -> None:
        if self.eta != DEPLOYED_RESTART_ETA:
            raise ValueError(
                f"the deployed construction is frozen at eta={DEPLOYED_RESTART_ETA}."
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0,1).")

    def threshold(self, n_arms: int) -> float:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        return n_arms / self.alpha


def detector_decay(horizon: int) -> float:
    return float(
        np.clip(
            1.0 - 1.0 / math.sqrt(max(horizon, 1)),
            0.0,
            E_ATS_DETECTOR_DECAY_CAP,
        )
    )


def frozen_e_ats_contract() -> dict[str, Any]:
    """Return the behavior-complete contract of the frozen e-ATS candidate."""

    return {
        "implementation_id": E_ATS_FULL_BUDGET_RAMP_IMPLEMENTATION_ID,
        "no_ce_ablation_implementation_id": (
            E_ATS_FULL_BUDGET_RAMP_NO_CE_ABLATION_ID
        ),
        "prior_alpha": E_ATS_PRIOR_ALPHA,
        "prior_beta": E_ATS_PRIOR_BETA,
        "authorization_alpha": E_ATS_AUTHORIZATION_ALPHA,
        "armwise_e_threshold": "n_arms_over_authorization_alpha",
        "restart_eta": DEPLOYED_RESTART_ETA,
        "restart_first_candidate_pull": RESTART_FIRST_CANDIDATE_PULL,
        "restart_candidate_rule": "max_previous_plus_1_ceil_1_plus_eta_times_previous",
        "restart_component_weight": "1_over_j_times_j_plus_1",
        "restart_future_capital": "1_over_active_component_count_plus_1",
        "authorization_coupling": "absorbing_historical_latch",
        "mean_normalization_budget": E_ATS_MEAN_NORMALIZATION_BUDGET,
        "mean_spending_schedule": "beta_over_k_times_6_over_pi2_n2",
        "mean_update": "lambda_m_plus_1_minus_lambda_times_reward_minus_long_prediction",
        "mean_range_update": "lambda_squared_q_plus_1_minus_lambda_squared",
        "mean_boundary": "sqrt_0.5_q_log_2_over_delta",
        "relevance_transform": "min_1_abs_mean_evidence_over_boundary",
        "boundary_scale": 1.0,
        "mean_numerical_epsilon": MEAN_NUMERICAL_EPSILON,
        "detector_decay": "clip_1_minus_inverse_sqrt_horizon_0_cap",
        "detector_decay_cap": E_ATS_DETECTOR_DECAY_CAP,
        "short_discount": E_ATS_SHORT_DISCOUNT,
        "short_discount_clock": "every_global_round",
        "long_discount": 1.0,
        "dual_memory_mix_power": E_ATS_DUAL_MEMORY_MIX_POWER,
        "local_takeover_scale": E_ATS_LOCAL_TAKEOVER_SCALE,
        "global_takeover_scale": E_ATS_GLOBAL_TAKEOVER_SCALE,
        "reliability_mass": "max_2_prior_mass_2_kref_prior_mass_times_sqrt_arm_local_budget_ratio",
        "reliability_reference_horizon": DEFAULT_VGDTS_V22_HORIZON,
        "reliability_reference_n_arms": DEFAULT_VGDTS_V22_N_ARMS,
        "short_reliability": "short_effective_mass_over_short_effective_mass_plus_reliability_mass",
        "short_weight": "gate_power_plus_local_scale_times_1_minus_gate_power_times_gate_times_reliability",
        "sampling": "convex",
        "score": "optimistic_max_sample_and_posterior_mean",
        "tie_breaking": "numpy_argmax_lowest_index",
        "gate_timing": "reward_at_t_updates_gate_for_action_t_plus_1",
    }


def update_full_budget_normalized_ramp(
    *,
    arm: int,
    reward: int,
    prediction: float,
    decay: float,
    n_arms: int,
    pull_counts: np.ndarray,
    mean_evidence: np.ndarray,
    mean_range_sq: np.ndarray,
) -> tuple[float, float]:
    """Update the frozen mean discrepancy and return its normalized ramp.

    The full declared mean budget appears directly in ``channel_alpha``.  No
    half-budget multiplier, boundary-scale argument, or transform argument is
    exposed by the canonical implementation.
    """

    if reward not in (0, 1):
        raise ValueError("reward must be binary.")
    if n_arms <= 0:
        raise ValueError("n_arms must be positive.")
    if not 0 <= arm < n_arms:
        raise ValueError("arm must lie in [0, n_arms).")
    if not 0.0 <= decay <= E_ATS_DETECTOR_DECAY_CAP:
        raise ValueError(
            f"decay must lie in [0, {E_ATS_DETECTOR_DECAY_CAP}]."
        )
    if not MEAN_NUMERICAL_EPSILON < prediction < 1.0 - MEAN_NUMERICAL_EPSILON:
        raise ValueError(
            "prediction clipping would bind and invalidate exact mean innovations."
        )

    p = float(prediction)
    one_minus_decay = 1.0 - decay
    mean_evidence[arm] = (
        decay * mean_evidence[arm] + one_minus_decay * (reward - p)
    )
    mean_range_sq[arm] = (
        decay * decay * mean_range_sq[arm]
        + one_minus_decay * one_minus_decay
    )
    pull_counts[arm] += 1
    n = int(pull_counts[arm])
    channel_alpha = (
        (E_ATS_MEAN_NORMALIZATION_BUDGET / float(n_arms))
        * (6.0 / (math.pi * math.pi * float(n * n)))
    )
    boundary = math.sqrt(
        0.5 * mean_range_sq[arm] * math.log(2.0 / channel_alpha)
    )
    magnitude = abs(float(mean_evidence[arm]))
    strength = (
        0.0
        if magnitude <= MEAN_NUMERICAL_EPSILON
        else float(np.clip(magnitude / boundary, 0.0, 1.0))
    )
    return strength, boundary


def _frozen_e_ats_controller_params() -> VGDTSV3Params:
    return VGDTSV3Params(
        gamma_min=E_ATS_SHORT_DISCOUNT,
        gamma_max=E_ATS_SHORT_DISCOUNT,
        dual_memory_mix_power=E_ATS_DUAL_MEMORY_MIX_POWER,
        local_takeover_scale=E_ATS_LOCAL_TAKEOVER_SCALE,
        global_takeover_scale=E_ATS_GLOBAL_TAKEOVER_SCALE,
    )


def _run_e_ats_full_budget_ramp(
    mu: np.ndarray,
    rng: np.random.Generator,
    returns: np.ndarray | None,
    diagnostics: dict[str, Any] | None,
    *,
    require_e_authorization: bool,
) -> np.ndarray:
    params = _frozen_e_ats_controller_params()
    _validate_params(params)
    means, observed = _validate_inputs(mu, returns)
    n_arms, horizon = means.shape
    if horizon <= 0:
        raise ValueError("mu must have a positive horizon.")
    derived = derive_vgdts_v22_secondary_params(
        alpha0=E_ATS_PRIOR_ALPHA,
        beta0=E_ATS_PRIOR_BETA,
        horizon=horizon,
        n_arms=n_arms,
    )
    reliability_mass = float(derived["shock_takeover_reliability_mass"])
    decay = detector_decay(horizon)
    authorization = EValueAuthorizationConfig(
        eta=DEPLOYED_RESTART_ETA,
        alpha=E_ATS_AUTHORIZATION_ALPHA,
    )
    log_threshold = math.log(authorization.threshold(n_arms))

    alpha_short = np.full(n_arms, E_ATS_PRIOR_ALPHA, dtype=float)
    beta_short = np.full(n_arms, E_ATS_PRIOR_BETA, dtype=float)
    alpha_long = np.full(n_arms, E_ATS_PRIOR_ALPHA, dtype=float)
    beta_long = np.full(n_arms, E_ATS_PRIOR_BETA, dtype=float)
    pull_counts = np.zeros(n_arms, dtype=np.int64)
    mean_evidence = np.zeros(n_arms, dtype=float)
    mean_range_sq = np.zeros(n_arms, dtype=float)
    deployed_gate = np.zeros(n_arms, dtype=float)
    e_states = (
        [RestartEProcessState(authorization.eta) for _ in range(n_arms)]
        if require_e_authorization
        else []
    )
    e_authorized = np.zeros(n_arms, dtype=bool)

    rewards = np.zeros(horizon, dtype=np.int64)
    played_arms = np.full(horizon, -1, dtype=np.int64)
    played_gate_before = np.zeros(horizon, dtype=float)
    played_ramp_after = np.zeros(horizon, dtype=float)
    played_gate_after = np.zeros(horizon, dtype=float)
    played_log_e_after = np.full(horizon, np.nan, dtype=float)
    first_e = np.full(n_arms, -1, dtype=np.int64)
    first_ramp = np.full(n_arms, -1, dtype=np.int64)
    first_deployed = np.full(n_arms, -1, dtype=np.int64)
    minimum_mean_zero_distance = math.inf
    minimum_mean_saturation_distance = math.inf
    minimum_e_threshold_distance = math.inf

    for timestep in range(horizon):
        n_eff_short = np.maximum(
            alpha_short + beta_short - E_ATS_PRIOR_ALPHA - E_ATS_PRIOR_BETA,
            0.0,
        )
        reliability = n_eff_short / (n_eff_short + reliability_mass)
        base_weight = np.power(
            deployed_gate,
            E_ATS_DUAL_MEMORY_MIX_POWER,
        )
        short_weight = base_weight + (
            E_ATS_LOCAL_TAKEOVER_SCALE
            * (1.0 - base_weight)
            * deployed_gate
            * reliability
        )
        short_weight = np.clip(short_weight, 0.0, 1.0)
        scores, _selected_short = _draw_scores(
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

        if require_e_authorization:
            log_e = e_states[arm].update(reward)
            played_log_e_after[timestep] = log_e
            minimum_e_threshold_distance = min(
                minimum_e_threshold_distance,
                abs(log_e - log_threshold),
            )
            if not e_authorized[arm] and log_e >= log_threshold:
                e_authorized[arm] = True
                first_e[arm] = timestep
            deployed_gate[arm] = strength if e_authorized[arm] else 0.0
        else:
            deployed_gate[arm] = strength
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
        implementation_id = (
            E_ATS_FULL_BUDGET_RAMP_IMPLEMENTATION_ID
            if require_e_authorization
            else E_ATS_FULL_BUDGET_RAMP_NO_CE_ABLATION_ID
        )
        diagnostics.update(
            {
                "implementation_id": implementation_id,
                "contract": frozen_e_ats_contract(),
                "requires_e_authorization": require_e_authorization,
                "played_arms": played_arms,
                "played_gate_before": played_gate_before,
                "played_ramp_after": played_ramp_after,
                "played_gate_after": played_gate_after,
                "played_log_e_after": played_log_e_after,
                "first_e_crossing_update_timestep": first_e,
                "first_ramp_positive_update_timestep": first_ramp,
                "first_deployed_gate_update_timestep": first_deployed,
                "final_e_authorized": e_authorized,
                "authorized_arm_count": int(np.count_nonzero(e_authorized)),
                "minimum_abs_mean_zero_distance": (
                    minimum_mean_zero_distance
                ),
                "minimum_abs_mean_saturation_distance": (
                    minimum_mean_saturation_distance
                ),
                "minimum_abs_e_threshold_distance": (
                    minimum_e_threshold_distance
                    if require_e_authorization
                    else None
                ),
                "live_segment_components": sum(
                    len(state.starts) for state in e_states
                ),
                "prior_predictive_null": PRIOR_PREDICTIVE_NULL,
                "uniform_fixed_p_validity_claimed": False,
            }
        )
    return rewards


def run_e_ats_full_budget_ramp(
    mu: np.ndarray,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    """Run the frozen full-budget normalized-ramp e-ATS policy."""

    return _run_e_ats_full_budget_ramp(
        mu,
        rng,
        returns,
        diagnostics,
        require_e_authorization=True,
    )


def run_e_ats_full_budget_ramp_no_ce(
    mu: np.ndarray,
    rng: np.random.Generator,
    returns: np.ndarray | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    """Run the exact direct ablation that removes only e-authorization."""

    return _run_e_ats_full_budget_ramp(
        mu,
        rng,
        returns,
        diagnostics,
        require_e_authorization=False,
    )
