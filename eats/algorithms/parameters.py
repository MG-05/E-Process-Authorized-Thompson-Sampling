"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import TypeAlias


GLRRestartMode: TypeAlias = Literal["local", "global"]


@dataclass
class TSParams:
    alpha0: float = 1.0
    beta0: float = 1.0


@dataclass
class DTSParams:
    """Dynamic Thompson Sampling parameters (Gupta et al., Algorithm 2)."""

    alpha0: float = 2.0
    beta0: float = 2.0
    C: float = 250.0


@dataclass
class dTSParams:
    """Discounted Thompson Sampling parameters."""

    alpha0: float = 1.0
    beta0: float = 1.0
    gamma: float = 0.75


@dataclass
class dOTSParams(dTSParams):
    """Discounted Optimistic Thompson Sampling parameters."""


@dataclass
class REXP3Params:
    """Rexp3 inputs from Besbes, Gur, and Zeevi (2014), Algorithm Rexp3."""

    gamma: float = 0.1136
    Delta: int = 250


@dataclass
class GlobalCTSParams:
    alpha0: float = 1.0
    beta0: float = 1.0
    hazard: float = 0.02
    max_runlengths: int = 200
    resample_target_fraction: float = 0.50


GLOBAL_CTS_IMPLEMENTATION_ID = "mellor_shapiro_alg1_resample_at_cap_v2"


@dataclass
class AFFTSParams:
    """AFF-TS Algorithm 4 parameters (Lu, Adams, and Kantas, 2019)."""

    # Algorithm 4 is stated for a generic Beta prior, but every Thompson-
    # sampling experiment in the paper uses Beta(2, 2).
    alpha0: float = 2.0
    beta0: float = 2.0
    eta: float = 0.001
    lambda_init: float = 1.0
    eps: float = 1e-12


AFF_TS_IMPLEMENTATION_ID = "lu_adams_kantas_aff_ts_alg4_v2"


@dataclass
class DLinUCBParams:
    gamma: float = 0.98
    lambda_reg: float = 1.0
    delta: float = 0.05
    sigma: float = 0.5
    action_norm_bound: float = 1.0
    theta_norm_bound: float = 1.0
    jitter: float = 1e-9


@dataclass
class BetaSWTSParams:
    alpha0: float = 1.0
    beta0: float = 1.0
    tau: int = 200


@dataclass
class SWUCBParams:
    tau: int = 200
    xi: float = 0.5


@dataclass
class DUCBParams:
    gamma: float = 0.99
    xi: float = 0.5
    bonus_scale: float = 2.0


@dataclass
class CUSUMUCBParams:
    """CUSUM-UCB Algorithm 3 parameters.

    ``random_explore`` is the paper's :math:`\alpha`: the probability of a
    uniformly random arm after all detector countdowns have reached zero.
    ``warmup`` is the paper's :math:`M`, and ``threshold`` is :math:`h`.
    """

    xi: float = 0.5
    epsilon: float = 0.05
    threshold: float = 8.0
    warmup: int = 50
    random_explore: float = 0.05


CUSUM_UCB_IMPLEMENTATION_ID = "liu_lee_shroff_cusum_ucb_alg3_v1"


@dataclass
class MUCBParams:
    """Monitored-UCB Algorithm 2 parameters.

    The paper's UCB1 index is recovered at the required default ``xi=1``.
    Other values are retained only for explicitly labelled sensitivity work.
    """

    xi: float = 1.0
    window: int = 200
    threshold: float = 4.0
    gamma: float = 0.10


M_UCB_IMPLEMENTATION_ID = "cao_wen_kveton_xie_m_ucb_alg2_v1"


@dataclass
class GLRklUCBParams:
    """Bernoulli GLR-klUCB Algorithm 1 parameters.

    ``exploration_alpha0`` scales the paper's horizon-aware schedule
    alpha_k = alpha0 * sqrt(k K log(T) / T). If ``delta`` is omitted, the
    paper's experimental choice delta = 1/sqrt(T) is used. The publication
    defaults reproduce the empirical construction: Global Restart, the
    authors' ``f(x)=log(x)`` klUCB index, Algorithm 1's
    ``floor(K / alpha_k)`` forced-exploration cycle, a detector check every
    ten local time steps, and split-point down-sampling by five. Set both
    strides to one for the literal, unaccelerated Algorithm 1 scan.
    """

    exploration_alpha0: float = 1.0
    delta: float | None = None
    delta_exponent: float = 0.5
    restart_mode: GLRRestartMode = "global"
    klucb_loglog_coefficient: float = 0.0
    detection_interval: int = 10
    split_step: int = 5


GLR_KLUCB_IMPLEMENTATION_ID = "besson_kaufmann_maillard_seznec_glr_klucb_alg1_v3"


GLR_KLUCB_PREFIX_OPTIMIZED_IMPLEMENTATION_ID = (
    "besson_kaufmann_maillard_seznec_glr_klucb_alg1_prefix_v1"
)


GLR_KLUCB_VECTORIZED_INDEX_IMPLEMENTATION_ID = (
    "besson_kaufmann_maillard_seznec_glr_klucb_alg1_prefix_vectorized_index_v1"
)


@dataclass
class GammaSWGTSParams:
    """Gaussian sliding-window Thompson sampling (Algorithm 2).

    The published algorithm has no Beta prior parameters.  It samples arm
    means from ``Normal(S/N, 1/(gamma*N))`` and forces an arm whenever its
    count in the global-round window is zero.
    """

    tau: int = 200
    gamma: float = 0.7


GAMMA_SWGTS_IMPLEMENTATION_ID = "fiandri_gamma_swgts_alg2_v1"
