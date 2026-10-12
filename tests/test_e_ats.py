import inspect
import math
import numpy as np
import pytest
from eats.algorithms.e_ats import (
    E_ATS_AUTHORIZATION_ALPHA,
    E_ATS_DUAL_MEMORY_MIX_POWER,
    E_ATS_FULL_BUDGET_RAMP_IMPLEMENTATION_ID,
    E_ATS_FULL_BUDGET_RAMP_NO_CE_ABLATION_ID,
    E_ATS_LOCAL_TAKEOVER_SCALE,
    E_ATS_MEAN_NORMALIZATION_BUDGET,
    E_ATS_SHORT_DISCOUNT,
    detector_decay,
    frozen_e_ats_contract,
    run_e_ats_full_budget_ramp,
    run_e_ats_full_budget_ramp_no_ce,
    update_full_budget_normalized_ramp,
)

def test_full_budget_ramp_has_no_tunable_policy_coordinate() -> None:
    expected = {"mu", "rng", "returns", "diagnostics"}
    assert set(inspect.signature(run_e_ats_full_budget_ramp).parameters) == expected
    assert (
        set(inspect.signature(run_e_ats_full_budget_ramp_no_ce).parameters)
        == expected
    )
    contract = frozen_e_ats_contract()
    assert contract["implementation_id"] == (
        E_ATS_FULL_BUDGET_RAMP_IMPLEMENTATION_ID
    )
    assert contract["no_ce_ablation_implementation_id"] == (
        E_ATS_FULL_BUDGET_RAMP_NO_CE_ABLATION_ID
    )
    assert contract["authorization_alpha"] == E_ATS_AUTHORIZATION_ALPHA == 0.05
    assert contract["mean_normalization_budget"] == (
        E_ATS_MEAN_NORMALIZATION_BUDGET
    ) == 0.05
    assert contract["short_discount"] == E_ATS_SHORT_DISCOUNT == 0.995
    assert contract["dual_memory_mix_power"] == (
        E_ATS_DUAL_MEMORY_MIX_POWER
    ) == 0.30
    assert contract["local_takeover_scale"] == (
        E_ATS_LOCAL_TAKEOVER_SCALE
    ) == 0.80
    assert "mean_budget_fraction" not in contract
    assert "c" not in contract

def test_full_budget_ramp_uses_the_complete_declared_schedule() -> None:
    pull_counts = np.zeros(4, dtype=np.int64)
    evidence = np.zeros(4)
    range_sq = np.zeros(4)
    decay = detector_decay(2_500)
    strength, boundary = update_full_budget_normalized_ramp(
        arm=0,
        reward=1,
        prediction=0.5,
        decay=decay,
        n_arms=4,
        pull_counts=pull_counts,
        mean_evidence=evidence,
        mean_range_sq=range_sq,
    )
    channel_alpha = (0.05 / 4.0) * (6.0 / (math.pi * math.pi))
    expected_boundary = math.sqrt(
        0.5 * range_sq[0] * math.log(2.0 / channel_alpha)
    )
    assert boundary == expected_boundary
    assert strength == abs(evidence[0]) / boundary
    assert pull_counts[0] == 1

def test_ce_is_the_only_difference_between_frozen_policy_and_ablation() -> None:
    n_arms, horizon = 4, 2
    means = np.full((n_arms, horizon), 0.5)
    returns = np.zeros((n_arms, horizon), dtype=np.uint8)
    e_diagnostics: dict[str, object] = {}
    no_ce_diagnostics: dict[str, object] = {}
    run_e_ats_full_budget_ramp(
        means,
        np.random.default_rng(1618),
        returns,
        e_diagnostics,
    )
    run_e_ats_full_budget_ramp_no_ce(
        means,
        np.random.default_rng(1618),
        returns,
        no_ce_diagnostics,
    )
    assert e_diagnostics["implementation_id"] == (
        E_ATS_FULL_BUDGET_RAMP_IMPLEMENTATION_ID
    )
    assert no_ce_diagnostics["implementation_id"] == (
        E_ATS_FULL_BUDGET_RAMP_NO_CE_ABLATION_ID
    )
    assert e_diagnostics["contract"] == no_ce_diagnostics["contract"]
    assert e_diagnostics["requires_e_authorization"] is True
    assert no_ce_diagnostics["requires_e_authorization"] is False
    assert e_diagnostics["played_ramp_after"][0] > 0.0
    assert no_ce_diagnostics["played_ramp_after"][0] > 0.0
    assert e_diagnostics["played_gate_after"][0] == 0.0
    assert no_ce_diagnostics["played_gate_after"][0] > 0.0

def test_frozen_e_ats_is_deterministic_and_fails_on_empty_horizon() -> None:
    means = np.full((3, 32), 0.5)
    returns = np.random.default_rng(2718).integers(
        0,
        2,
        size=means.shape,
        dtype=np.uint8,
    )
    first: dict[str, object] = {}
    second: dict[str, object] = {}
    first_rewards = run_e_ats_full_budget_ramp(
        means,
        np.random.default_rng(314),
        returns,
        first,
    )
    second_rewards = run_e_ats_full_budget_ramp(
        means,
        np.random.default_rng(314),
        returns,
        second,
    )
    assert np.array_equal(first_rewards, second_rewards)
    assert np.array_equal(first["played_arms"], second["played_arms"])
    with pytest.raises(ValueError, match="positive horizon"):
        run_e_ats_full_budget_ramp(
            np.empty((2, 0)),
            np.random.default_rng(1),
        )

def test_frozen_e_authorization_is_absorbing_not_current_level() -> None:
    rewards = np.concatenate(
        (
            np.zeros(20, dtype=np.uint8),
            np.ones(20, dtype=np.uint8),
            np.zeros(1_000, dtype=np.uint8),
        )
    )
    diagnostics: dict[str, object] = {}
    run_e_ats_full_budget_ramp(
        np.full((1, rewards.size), 0.5),
        np.random.default_rng(1),
        rewards.reshape(1, -1),
        diagnostics,
    )
    crossing = int(diagnostics["first_e_crossing_update_timestep"][0])
    log_threshold = math.log(1.0 / E_ATS_AUTHORIZATION_ALPHA)
    below_after_crossing = np.flatnonzero(
        (np.arange(rewards.size) > crossing)
        & (diagnostics["played_log_e_after"] < log_threshold)
    )
    assert crossing >= 0
    assert below_after_crossing.size > 0
    timestep = int(below_after_crossing[0])
    assert diagnostics["played_ramp_after"][timestep] > 0.0
    assert diagnostics["played_gate_after"][timestep] == diagnostics[
        "played_ramp_after"
    ][timestep]
    assert diagnostics["final_e_authorized"][0]
