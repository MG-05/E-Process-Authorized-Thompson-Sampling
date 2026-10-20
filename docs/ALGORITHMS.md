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
has no controller-tuning surface. That API restriction does not imply that its
constants were never tuned during development.

### Origin of controller settings

e-ATS inherits controller constants tuned for its predecessor VG-dTS by the
same staged validation on the nine replay families, with seeds disjoint from
replay. The primary 288-point search selected:

| Selection geometry | Short discount γ | Mixing power ν | Local takeover ζ | Grid index |
|---|---:|---:|---:|---:|
| K=4, T=2500 | 0.9925 | 0.30 | 0.80 | 199 |
| K=32, T=20000 | 0.995 | 0.30 | 0.80 | 263 |

e-ATS uses the K=32 vector **(0.995,0.30,0.80) at every geometry**, rather than
the geometry-specific predecessor winners. All three K=32 coordinates lie on
the search-grid boundary; K=4's selected γ is interior. Selection minimized
validation mean normalized dynamic pseudo-regret, with ties broken by lowest
grid index. The records establish the transfer, not its optimality or a causal
explanation for the difference between K=4 and K=32 replay ranks.

The authorization threshold **K/alpha_E is not regret-tuned**: it follows the
declared error budget alpha_E=0.05 and equal allocation across arms under the
prior-predictive model. Other construction choices did involve development
comparisons: split spacing eta=0.25 was selected from {0.25,0.5,1}; the normalized
clipped ramp was selected over a soft ratio; and the absorbing latch was retained
after comparison with a current-level gate. The relevance scale 0.05 and the
reliability formula were inherited calibrations. Structural, algebraically
derived, empirical, and numerical constants are distinguished in the recovered
[constant-origin audit](../provenance/freeze_notes/e_value_ce_normalized_ramp_magic_number_audit_2026-08-23.md).

[Controller selection provenance](../provenance/controller_selection.json), the
[original selection record](../provenance/synthetic_validation.json), and the
[freeze manifest](../provenance/e_ats_full_budget_ramp_freeze_manifest.json)
preserve the evidence. These historical records refer to the private development
repository; their original paths are not public links.

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
| `sw-ucb` | SW-UCB (τ=T; full history in these runs) | historical_09 | run_sw_ucb | garivier2008ucb |
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

Citation entries are in `paper/references.bib`. The 14 tunable baselines used
geometry-specific configurations selected by staged grid search and validation
on disjoint seeds of the same nine families. TS and OTS used fixed configurations;
e-ATS used the inherited K=32 controller settings described above, frozen across
all 18 cells. This repository does
not rerun tuning or claim equal tuning budgets. The
[selection and replay contract](REPRODUCIBILITY.md#baseline-selection-and-replay-scope)
records the splits and the deterministic-family qualification.
The replay leaderboard excludes four retired internal rows from the historical
21-row artifact and recomputes ranks among e-ATS and these 16 baselines.

### UCB memory settings

| Geometry (K,T) | SW-UCB window τ | SW-UCB exploration ξ | D-UCB discount γ | D-UCB memory scale 1/(1−γ) |
|---|---:|---:|---:|---:|
| (4,2500) | 2500 | 2.0 | 0.9996 | 2500 |
| (32,20000) | 20000 | 1.0 | 0.99995 | 20000 |

SW-UCB removes an observation only when the stored history exceeds τ. Since
τ=T, this never happens during either evaluation horizon: the executed policy
is full-history UCB with the exploration coefficient shown above. Its lowest
overall mean regret and second-best mean rank therefore do not demonstrate a
benefit from windowing. The frozen row and CLI names retain the source algorithm's
name for reproducibility.

D-UCB multiplies accumulated counts and successes by γ each round. Its
conventional geometric memory scale equals T (up to floating-point rounding),
but it still downweights older observations. It is not identical to full-history
UCB. Both geometries use `xi=0.5` and `bonus_scale=2.0` for D-UCB.
