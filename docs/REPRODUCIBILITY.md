# Reproducing the paper

This guide maps the paper's results to the released code and records how the
experiments were selected and analyzed. Start with the
[README commands](../README.md#reproduction) to recompute results or run a suite.

## Paper-to-code map

`python -m eats reproduce --bootstrap --no-figures` recomputes all reported
tables and confidence intervals from saved runs. On macOS ARM64, omit
`--no-figures` to regenerate the three figures as well.

| Result | Saved input | Full experiment command |
|---|---|---|
| Registered regret and gate tables; primary interval | `data/registered.jsonl.gz` | `python -m eats run registered --full` |
| Replay gate comparison, primary interval, 17-method table, and ranks | `data/replay.jsonl.gz` | `python -m eats run replay --full` |
| Calibrated-GLR sensitivity | Registered outcomes and `data/calibration.jsonl.gz` | `python -m eats run calibration --full`, then `python -m eats run registered --full --policies calibrated-glr` |
| Null crossing counts: 4/4000, 1/4000, 4/4000 | `data/null.jsonl.gz` | `python -m eats run null --full` |

Figures are built by `eats/figures.py`. The registered atlas uses the nine
`data/fixtures/*_k4_b625.npy` visualization examples; the replay atlas uses seed
1840201. The summary figure uses saved evaluation outcomes.

## Statistical procedures

Normalized dynamic pseudo-regret is
`mean(max(mu[:, t]) - mu[action[t], t])`. Policies in a group share the same
mean trajectories, potential rewards, and initial policy RNG state. Each
policy observes only the reward of its chosen arm. The registered and replay
suites use different generators and are analyzed separately.

**Registered.** The primary analysis uses 91 random environments in each of
180 nonstationary cells. The bootstrap resamples within cells and keeps the
paired source indices for isolated incumbent decreases and challenger
increases together. The primary contrast is e-ATS minus open ramp: its upper
one-sided 95% bound must be below −0.0002. Tables report the opposite sign,
comparator minus e-ATS. Stationary and fixed stress groups are separate.
Evaluation uses source indices 5–95 for random environments, 5–31 for stress
environments, and reward tape 2.

**Replay.** There are 250 comparisons in each of 18 family–geometry cells.
Slow, fast, and abrupt each use a fixed mean trajectory at each geometry;
the other six families use 50 trajectories with five reward tapes each.
This gives 606 distinct trajectories. Rank is one plus the number of strictly
lower cell means among the 17 displayed methods, with equal weight for each cell.

The replay primary interval resamples individual comparisons in fixed-mean
cells and trajectory clusters, keeping all five tapes together, in random-mean
cells. This grouping was corrected after outcomes were available, as disclosed
in the paper. The GLR intervals retain their original five-tape grouping in
all cells; the calibrated-GLR sensitivity uses its original independently
stratified interval. All intervals use the recorded seeds and 50,000 resamples;
`--bootstrap` checks the recomputed endpoints against the paper.

**Calibration and null checks.** Calibration uses 5,460 stationary runs over
20 geometries, on tapes 0, 1, and 3, separate from evaluation tape 2. Its GLR
cutoffs match the e-process's familywise crossing counts along OTS traces;
17 geometries have zero crossings. `verify` rederives the cutoffs from the saved
statistics. The null experiment draws arm means independently from Beta(1,1)
and monitors crossings along OTS. Before authorization, e-ATS follows the same
path, so this measures its first-authorization event under that model.

## Baseline selection and replay scope

The 14 tunable baselines were selected separately at each geometry by staged
grid search, followed by validation of a shortlist. Selection minimized mean
normalized dynamic pseudo-regret across the nine families, with ties broken
by lowest grid index. TS and OTS used fixed configurations.

| Phase | Environment-seed labels | Reward/policy streams per label and family |
|---|---|---:|
| Search stage 1 | 1830001–1830003 | 1 |
| Search stage 2 | 1830004–1830008 | 1 |
| Validation | 1830101–1830106 | 5 |
| Replay | 1840201–1840250 | 5 |

All 28 selected baseline configurations match their original grid indices
and parameter hashes. The recorded seed audit found no replay-seed overlap
with the checked development history. See the
[selection summary](../provenance/baseline_selection.json) and
[original validation record](../provenance/validation_selection.json).

e-ATS inherits **(γ,ν,ζ)=(0.995,0.30,0.80)** from the K=32 selection for its
predecessor VG-dTS, using the same staged validation and disjoint replay seeds.
It applies those settings at every geometry, although the predecessor's K=4
selection used γ=0.9925. All three K=32 values lie at the search-grid boundary.
The threshold K/α_E follows the declared error budget and was not regret-tuned.
The [algorithm guide](ALGORITHMS.md#origin-of-controller-settings) describes
other development choices and links the original records.

For slow, fast, and abrupt, baselines and the predecessor were tuned on the
same mean trajectories used in replay; only reward and policy randomness
differed. The other families vary with the environment seed. Settings were
not selected on replay outcomes, but baseline outcomes were known before the
replay protocol was frozen. Replay is therefore a historical comparison;
tuning budgets were not equalized.

SW-UCB's selected window is the full horizon, and D-UCB's discount gives a
horizon-length memory scale. Their results should be interpreted with those
[settings](ALGORITHMS.md#ucb-memory-settings) in mind.

## Platforms and CI

Use Python 3.12.13 and the pinned dependencies. Saved-result analysis is
covered by CI on Ubuntu 24.04 x86-64 and macOS 14 ARM64. Exact policy reruns
and figure regeneration are supported on **macOS ARM64**.

| Check | Linux and macOS | macOS ARM64 only |
|---|---|---|
| Release integrity and reported results | `python -m eats verify` | |
| Tables and intervals | `python -m eats reproduce --bootstrap --no-figures` | |
| Tests | `make test` | `make test-reference` |
| Experiment smoke runs | | `python -m eats run registered --smoke`; `python -m eats run replay --smoke` |

Without `make`, use `python -m pytest -m 'not reference'` for portable tests
and `python -m pytest -m reference` for exact trajectory tests. Plain
`python -m pytest` runs both. Smoke runs cover every mechanism or family but
are small subsets of the full experiments.

Linux has a confirmed mean-table mismatch in the registered gradual-drift
smoke case. Matching Python and NumPy versions does not ensure identical
floating-point calculations across platforms. The runner stops at a mismatch;
use saved-result analysis with `--no-figures` on Linux. Two historical
NumPy-version differences have explicit fixtures; these do not resolve the
broader platform limitation.

## Outputs and integrity checks

Outputs go to the Git-ignored `results/` directory. Repeat the same run command
to resume; use `--output` for a separate run. Checkpoints identify the selected
groups, policies, runtime, and release manifest. A changed selection, runtime,
or manifest automatically starts a separate checkpoint. Reruns check generated means,
rewards, action traces, regret, and authorization counts against the archive.
The released inputs are never overwritten.

`provenance/release.json` records file checksums, which commands check before
running. It includes the documentation, so documentation edits also require
updated checksums. `provenance/code.json` records the extracted numerical
function bodies; their source hashes are tested.

The development repository is private. Original paths in provenance files are
historical references, not public links. The release includes the inputs
needed to reproduce the paper, both original selection records, and four
unaltered freeze notes. The larger development run manifests remain private.
The dated `provenance/validation*.json` files record past checks; current CI
results should be read from GitHub Actions for the commit being released.

## Relation to arXiv v1

The bundled manuscript is based on
[arXiv:2610.03646v1](https://arxiv.org/abs/2610.03646v1). It clarifies tuning,
baseline memory settings, rank positions, and the limits of the comparisons;
it also corrects the author spelling to Wei-Chen Au. Results, algorithm
parameters, tables, and figures are unchanged. The arXiv version has not been
replaced.

The source uses `[preprint]`, as arXiv v1 does; this fixes the repository's
earlier submission-format extraction. Upstream LaTeX package sources and
notices are bundled as described in [third-party notices](../THIRD_PARTY_NOTICES.md).
