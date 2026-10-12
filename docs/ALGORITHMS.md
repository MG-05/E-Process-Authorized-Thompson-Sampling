# Algorithms and fixed settings

All methods receive a `(K,T)` mean array for environment simulation, an RNG,
and optionally the shared binary potential-reward table. Action traces are
available through `diagnostics['played_arms']`. The mean array is not used to
choose actions; it supplies a reward only when no potential table is passed.

## Paper controller

| CLI name | Definition | Implementation |
|---|---|---|
| `e-ats` | Full-history optimistic TS anchor, arm-local e-process authorization latch, normalized relevance ramp, short-memory posterior | `eats/algorithms/e_ats.py:run_e_ats_full_budget_ramp` |
| `open-ramp` | Same controller with only the authorization latch removed | `eats/algorithms/e_ats.py:run_e_ats_full_budget_ramp_no_ce` |
| `ots` | Optimistic TS with Beta(1,1) prior and no discount | `eats/algorithms/baselines.py:run_dOTS`, gamma=1 |
| `certified-glr` | Same controller; geometric-split GLR with Besson et al. Lemma 2 boundary | `eats/algorithms/glr.py`, `besson_lemma2` |
| `practical-glr` | Same controller; boundary log(n^(3/2)/delta), no finite-sample guarantee claimed | `eats/algorithms/glr.py`, `besson_practical` |
| `calibrated-glr` | Same controller; geometry-specific count-matched scalar threshold; appendix sensitivity only | `eats/algorithms/calibrated_glr.py` |

Canonical constants: alpha_E=0.05, short discount=0.995 per global round,
long discount=1, normalized-ramp budget=0.05, mixing power=0.30, local takeover
scale=0.80, global takeover=0. Candidate starts begin at pull 2 and increase
by ceil(1.25 × previous); mixture weights are 1/[j(j+1)] with unreached capital
1/(J+1). The long-posterior draw occurs first; no short-posterior RNG is consumed
when its weight is zero. Reward at round t changes the gate for round t+1.
Ties select the smallest arm index. See Algorithm 1 and the exact update
equations in `paper/appendix.tex`; `frozen_e_ats_contract()` provides the full
machine-readable contract.

The files `common.py` and `scaling.py` preserve shared numerical helpers from
the development implementation. Their historical symbol names do not expose
retired policies. The public `from eats import run_e_ats, run_open_ramp` API
has no controller-tuning surface.

## Literature replay baselines

All implementations below are in `eats/algorithms/baselines.py`. Their exact
dataclass parameters at K=4 and K=32 are in
`data/baseline_configurations.json`. `params` is the actual executed setting;
`best_params` is the original tuning/provenance metadata. Do not replace the
frozen parameters with class defaults when trying to reproduce the paper.

| CLI name | Paper name | Frozen row | Function | Source citation key |
|---|---|---|---|---|
| `dts` | dTS | historical_04 | run_dTS | raj2017taming |
| `dots` | dOTS | historical_05 | run_dOTS | raj2017taming |
| `aff-ts` | AFF-TS | historical_06 | run_aff_ts | lu2017adaptive |
| `rexp3` | REXP3 | historical_07 | run_REXP3 | besbes2014rexp3 |
| `dynamic-ts` | Dynamic TS | historical_08 | run_DTS | gupta2011dynamic |
| `sw-ucb` | SW-UCB | historical_09 | run_sw_ucb | garivier2008ucb |
| `d-ucb` | D-UCB | historical_10 | run_d_ucb | garivier2008ucb |
| `m-ucb` | M-UCB | historical_11 | run_m_ucb | cao2018mucb |
| `cusum-ucb` | CUSUM-UCB | historical_12 | run_cusum_ucb | liu2017cusum |
| `glr-klucb` | GLR-klUCB | historical_13 | run_glr_klucb_vectorized_index | besson2022glr |
| `beta-swts` | Beta-SWTS | historical_14 | run_beta_swts | fiandri2024swts |
| `gamma-swgts` | gamma-SWGTS | historical_15 | run_gamma_swgts | fiandri2024swts |
| `global-cts` | Global-CTS | historical_16 | run_global_cts | mellor2013globalcts |
| `d-linucb` | D-LinUCB one-hot | historical_17 | run_dlinucb_onehot | russac2019weighted |
| `ots` | OTS | historical_18 | run_dOTS (gamma=1) | may2012optimistic |
| `ts` | TS | historical_19 | run_TS | thompson1933 |

Citation entries are in `paper/references.bib`. Settings were selected in earlier
development; this repository does not rerun tuning or claim equal tuning budgets.
The replay leaderboard excludes four retired internal rows from the historical
21-row artifact and recomputes ranks among e-ATS and these 16 baselines.
