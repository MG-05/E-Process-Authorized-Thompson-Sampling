from __future__ import annotations

import math

import numpy as np
import pytest

from eats.algorithms.glr import (
    PRACTICAL_GLR_IMPLEMENTATION_ID,
    TIME_UNIFORM_GLR_IMPLEMENTATION_ID,
    TimeUniformGLRAuthorizationConfig,
    besson_practical_glr_boundary,
    besson_time_uniform_glr_boundary,
    h_inverse,
    run_e_ats_time_uniform_glr_full_budget_ramp,
    time_uniform_glr_contract,
)


def test_h_inverse_returns_increasing_branch() -> None:
    for value in (1.0, 1.000001, 2.0, 5.0, 20.0):
        inverse = h_inverse(value)
        assert inverse >= 1.0
        assert inverse - math.log(inverse) == pytest.approx(
            value, rel=2e-15, abs=2e-15
        )


def test_published_boundaries_have_expected_order_and_values() -> None:
    # Independently transcribed reference values for delta=0.0125.
    expected = {
        2: 28.734067807638482,
        10: 36.11924859956585,
        625: 49.00832518744198,
        2500: 52.59522318861653,
    }
    previous = -math.inf
    for local_n, reference in expected.items():
        certified = besson_time_uniform_glr_boundary(local_n, 0.0125)
        practical = besson_practical_glr_boundary(local_n, 0.0125)
        assert certified == pytest.approx(reference, rel=2e-15, abs=2e-15)
        assert certified > practical
        assert certified > previous
        previous = certified


def test_contract_separates_theorem_backed_and_practical_claims() -> None:
    certified = time_uniform_glr_contract(
        TimeUniformGLRAuthorizationConfig("besson_lemma2"), 8
    )
    practical = time_uniform_glr_contract(
        TimeUniformGLRAuthorizationConfig("besson_practical"), 8
    )
    assert certified["implementation_id"] == TIME_UNIFORM_GLR_IMPLEMENTATION_ID
    assert certified["analytic_anytime_false_alarm_guarantee"] is True
    assert certified["per_arm_delta"] == pytest.approx(0.05 / 8)
    assert certified["empirical_threshold_parameters"] == 0
    assert practical["implementation_id"] == PRACTICAL_GLR_IMPLEMENTATION_ID
    assert practical["analytic_anytime_false_alarm_guarantee"] is False
    assert practical["analytic_null_scope"] == "not_claimed"


@pytest.mark.parametrize("kind", ["besson_lemma2", "besson_practical"])
def test_policy_is_reproducible_and_reward_table_coupled(kind: str) -> None:
    horizon = 120
    means = np.vstack(
        [
            np.full(horizon, 0.3),
            np.full(horizon, 0.7),
        ]
    )
    returns = np.zeros_like(means, dtype=np.uint8)
    returns[1] = 1
    first: dict[str, object] = {}
    second: dict[str, object] = {}
    config = TimeUniformGLRAuthorizationConfig(kind)  # type: ignore[arg-type]
    rewards_first = run_e_ats_time_uniform_glr_full_budget_ramp(
        means,
        np.random.default_rng(8123),
        returns=returns,
        diagnostics=first,
        authorization=config,
    )
    rewards_second = run_e_ats_time_uniform_glr_full_budget_ramp(
        means,
        np.random.default_rng(8123),
        returns=returns,
        diagnostics=second,
        authorization=config,
    )
    np.testing.assert_array_equal(rewards_first, rewards_second)
    np.testing.assert_array_equal(first["played_arms"], second["played_arms"])
    arms = np.asarray(first["played_arms"], dtype=np.int64)
    np.testing.assert_array_equal(rewards_first, returns[arms, np.arange(horizon)])


def test_certified_detector_can_authorize_on_large_abrupt_change() -> None:
    horizon = 500
    means = np.full((1, horizon), 0.5)
    returns = np.concatenate(
        [np.zeros(250, dtype=np.uint8), np.ones(250, dtype=np.uint8)]
    )[None, :]
    diagnostics: dict[str, object] = {}
    run_e_ats_time_uniform_glr_full_budget_ramp(
        means,
        np.random.default_rng(99),
        returns=returns,
        diagnostics=diagnostics,
        authorization=TimeUniformGLRAuthorizationConfig("besson_lemma2"),
    )
    assert diagnostics["authorized_arm_count"] == 1
    first = np.asarray(
        diagnostics["first_glr_crossing_update_timestep"], dtype=np.int64
    )
    assert 250 <= first[0] < horizon


def test_configuration_rejects_invalid_claim_inputs() -> None:
    with pytest.raises(ValueError):
        TimeUniformGLRAuthorizationConfig("unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TimeUniformGLRAuthorizationConfig("besson_lemma2", familywise_alpha=1.0)
    with pytest.raises(ValueError):
        besson_time_uniform_glr_boundary(0, 0.05)
    with pytest.raises(ValueError):
        besson_practical_glr_boundary(10, 0.0)
