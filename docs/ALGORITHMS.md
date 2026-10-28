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

Use `from eats import run_e_ats, run_open_ramp` for the public API. Both use
the paper's fixed settings; their development history is described below.

### Origin of controller settings

e-ATS inherits controller constants tuned for its predecessor VG-dTS by the
same staged validation on the nine replay families, with seeds disjoint from
replay. The primary 288-point search selected:

| Selection geometry | Short discount γ | Mixing power ν | Local takeover ζ | Grid index |
|---|---:|---:|---:|---:|
| K=4, T=2500 | 0.9925 | 0.30 | 0.80 | 199 |
| K=32, T=20000 | 0.995 | 0.30 | 0.80 | 263 |

e-ATS uses the K=32 settings **(0.995,0.30,0.80) at every geometry**. All three
values lie at the search-grid boundary; K=4's selected γ is interior. Selection
minimized validation mean normalized dynamic pseudo-regret, with ties broken
by lowest grid index. The K=32-to-K=4 transfer is one difference to consider
when comparing performance across geometries; its effect was not isolated.

The threshold **K/alpha_E was not regret-tuned**. It follows alpha_E=0.05 and
equal allocation across arms under the independent Beta(1,1)–Bernoulli
prior-predictive model. Development comparisons selected split spacing
eta=0.25 from {0.25,0.5,1} and the clipped ramp over a soft ratio; the absorbing
latch was retained after comparison with a current-level gate. The relevance scale 0.05 and reliability
formula were inherited calibrations. The original
[constant-origin audit](../provenance/freeze_notes/e_value_ce_normalized_ramp_magic_number_audit_2026-08-23.md)
records the source of each choice.

[Controller selection provenance](../provenance/controller_selection.json), the
[original selection record](../provenance/synthetic_validation.json), and the
[freeze manifest](../provenance/e_ats_full_budget_ramp_freeze_manifest.json)
preserve the original evidence from the private development repository.

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

Citation entries are in `paper/references.bib`. The 14 tunable baselines were
selected per geometry by staged search and validation on disjoint seeds of
the same nine families; TS and OTS used fixed settings. Tuning budgets were
not equal. See the [selection history](REPRODUCIBILITY.md#baseline-selection-and-replay-scope)
for seed splits and mean-trajectory reuse. Ranks are recomputed among e-ATS
and these 16 baselines; four retired internal methods are excluded.

### UCB memory settings

| Geometry (K,T) | SW-UCB window τ | SW-UCB exploration ξ | D-UCB discount γ | D-UCB memory scale 1/(1−γ) |
|---|---:|---:|---:|---:|
| (4,2500) | 2500 | 2.0 | 0.9996 | 2500 |
| (32,20000) | 20000 | 1.0 | 0.99995 | 20000 |

With τ=T, SW-UCB never drops an observation during these runs: it acts as
full-history UCB with the exploration coefficient shown above. Its lowest
overall mean regret and second-best mean rank do not demonstrate a benefit
from forgetting. The CLI retains the source algorithm's name.

D-UCB discounts counts and successes each round. Its memory scale is T, but
older observations still receive less weight. Both geometries use `xi=0.5`
and `bonus_scale=2.0`.
