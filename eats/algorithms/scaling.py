"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

import math


DEFAULT_VGDTS_V22_HORIZON = 5000


DEFAULT_VGDTS_V22_N_ARMS = 4


def _v22_prior_mass(alpha0: float, beta0: float) -> float:
    return max(float(alpha0 + beta0), 1.0)


def _v22_reference_budget_scale(*, horizon: int, n_arms: int) -> float:
    horizon_eff = max(int(horizon), 1)
    n_arms_eff = max(int(n_arms), 1)
    # Anchor secondary time-constants at the frozen paper reference regime
    # (T=5000, K=4), then scale with the square root of the arm-local budget.
    # This is a fixed calibration rule, not a theory claim.
    return math.sqrt(
        (float(horizon_eff) / float(DEFAULT_VGDTS_V22_HORIZON))
        * (float(DEFAULT_VGDTS_V22_N_ARMS) / float(n_arms_eff))
    )


def _effective_v22_window(*, horizon: int, n_arms: int, alpha0: float, beta0: float) -> float:
    prior_mass = _v22_prior_mass(alpha0, beta0)
    reference_window = math.sqrt(float(DEFAULT_VGDTS_V22_HORIZON)) / float(DEFAULT_VGDTS_V22_N_ARMS)
    scale = _v22_reference_budget_scale(horizon=horizon, n_arms=n_arms)
    # Keep the benchmark reference regime (T=5000, K=4) unchanged, then scale
    # the secondary timescales with the square root of the arm-local budget so
    # the volatility and shock channels remain active at moderate K.
    return max(2.0 * prior_mass, reference_window * scale)


def derive_vgdts_v22_lambda_vol(*, horizon: int, n_arms: int, alpha0: float, beta0: float) -> float:
    window = _effective_v22_window(horizon=horizon, n_arms=n_arms, alpha0=alpha0, beta0=beta0)
    return float(min(max(1.0 - 1.0 / window, 0.0), 0.999))


def derive_vgdts_v22_shock_half_life(*, horizon: int, n_arms: int, alpha0: float, beta0: float) -> float:
    return max(2.0, 0.5 * _effective_v22_window(horizon=horizon, n_arms=n_arms, alpha0=alpha0, beta0=beta0))


def derive_vgdts_v22_shock_score_decay(*, horizon: int, n_arms: int, alpha0: float, beta0: float) -> float:
    half_life = derive_vgdts_v22_shock_half_life(horizon=horizon, n_arms=n_arms, alpha0=alpha0, beta0=beta0)
    return float(math.exp(-math.log(2.0) / half_life))


def derive_vgdts_v22_shock_takeover_reliability_mass(
    *,
    alpha0: float,
    beta0: float,
    horizon: int,
    n_arms: int,
) -> float:
    prior_mass = _v22_prior_mass(alpha0, beta0)
    reference_mass = 2.0 * float(DEFAULT_VGDTS_V22_N_ARMS) * prior_mass
    scale = _v22_reference_budget_scale(horizon=horizon, n_arms=n_arms)
    return float(max(2.0 * prior_mass, reference_mass * scale))


def derive_vgdts_v22_secondary_params(
    *,
    alpha0: float,
    beta0: float,
    horizon: int,
    n_arms: int,
) -> dict[str, float]:
    return {
        "lambda_vol": derive_vgdts_v22_lambda_vol(horizon=horizon, n_arms=n_arms, alpha0=alpha0, beta0=beta0),
        "shock_score_decay": derive_vgdts_v22_shock_score_decay(
            horizon=horizon,
            n_arms=n_arms,
            alpha0=alpha0,
            beta0=beta0,
        ),
        "shock_takeover_reliability_mass": derive_vgdts_v22_shock_takeover_reliability_mass(
            alpha0=alpha0,
            beta0=beta0,
            horizon=horizon,
            n_arms=n_arms,
        ),
    }
