# Reproduction contract

This repository reproduces the results reported in
[arXiv:2610.03646v1](https://arxiv.org/abs/2610.03646v1), including its appendix.
The repository is a paper-specific extraction, not the entire development
history. Older e-authorized threshold policies, unused ablations, tuning
searches, real datasets, and unused figures are excluded.

## Relation to arXiv v1

The bundled manuscript is based on the
[arXiv v1 source](https://arxiv.org/src/2610.03646v1). The arXiv record has not
been replaced by these local repository corrections.

The earlier repository extraction used the submission option. It now uses
`[preprint]`, matching arXiv v1 and removing the submission notice and line
numbers. This repairs the extraction; it is not a difference from arXiv v1.

The intentional differences from the arXiv v1 source are:

- `main.tex` and `appendix.tex` clarify the overall rank positions. The appendix
  also identifies baseline tuning, e-ATS's inherited tuned constants and other
  development choices, the full-horizon SW-UCB window, D-UCB's memory scale,
  and the limits of the strongest-cell and geometry comparisons.
- The manuscript's author spelling is corrected to Wei-Chen Au.
- `trimspaces.sty` is the unmodified CTAN v1.1 file. Its upstream attribution and
  complete source distribution are bundled under `paper/vendor/trimspaces/`;
  the complete upstream `environ` archive is also bundled. See
  [third-party notices](../THIRD_PARTY_NOTICES.md).

These are build, attribution, and explanatory corrections. Algorithms, frozen
parameters, evaluation data, numerical table entries, and figures are unchanged.
The bundled source is not claimed to be byte-identical to arXiv v1; the arXiv
record remains the version of record for that submission.

## Paper-to-code map

| Paper result | Inputs | Recompute | Execute policies again |
|---|---|---|---|
| Main regret table; appendix gate table; registered primary interval | `data/registered.jsonl.gz` | `python -m eats reproduce --bootstrap` | `python -m eats run registered --full` |
| Replay gate reversal and corrected primary interval | `data/replay.jsonl.gz` | same | `python -m eats run replay --full --policies e-ats open-ramp ots certified-glr practical-glr` |
| 17-method replay table and geometry ranks | `data/replay.jsonl.gz`, baseline configurations | same | `python -m eats run replay --full` |
| Calibrated-GLR sensitivity | registered outcomes, `data/calibration.jsonl.gz`, calibrated thresholds | same; thresholds are rederived from saved calibration statistics | `python -m eats run calibration --full`, then `python -m eats run registered --full --policies calibrated-glr` |
| Null crossing counts 4/4000, 1/4000, 4/4000 | `data/null.jsonl.gz` | same | `python -m eats run null --full` |
| Registered environment atlas | nine `data/fixtures/*_k4_b625.npy` tables | `eats/figures.py` via `reproduce` | fixed visualization fixtures, distinct from evaluation |
| Replay environment atlas | `eats/environments/replay.py`, seed 1840201 | same | deterministic generator |
| Authorization/rank summary figure | all saved evaluation groups | same | the corresponding suite commands |

## Pairing, populations, and inference

Regret is `mean(max(mu[:, t]) - mu[action[t], t])`. A group shares the entire
mean table, binary potential rewards, and initial policy RNG descriptor across
policies. Only the selected arm's reward is observed by a policy. The two
suites use different generators; their absolute regret levels are not a paired
comparison.

Registered evaluation uses source indices 5–95 for random environments and
5–31 for fixed stress environments, with reward tape 2. The primary analysis
includes 91 environments in each of 180 nonstationary mechanism/geometry cells.
Its bootstrap resamples within cells and preserves the paired source indices
of isolated incumbent decreases and challenger increases. Stress results are
a separate census; stationary groups are separate from nonstationary means.
The primary contrast is e-ATS minus open ramp, with an upper one-sided 95%
bound required below -0.0002. Table gate differences use comparator minus e-ATS.

Replay has 18 cells, 250 comparisons per cell. Six cells (slow, fast, abrupt,
each at K=4 and K=32) use one fixed mean trajectory. The other twelve cells
use 50 mean trajectories with five reward tapes each. The nominal 900 task
descriptors therefore represent 606 unique mean tables, not 900 independent
environments. Rank is 1 plus the count of strictly lower cell means, recomputed
among exactly the 17 displayed policies; cells receive equal weight.

The corrected replay primary interval resamples 250 individual comparisons
in each fixed-mean cell, and 50 trajectory clusters retaining five tapes each
in random-mean cells. This was a post-outcome correction, as disclosed in the
paper. The GLR intervals preserve the original grouping of five tapes in all
cells. The separately fitted GLR sensitivity retains its original independently
stratified interval. No interval is silently replaced with a new procedure.
The original seeds, iteration order, quantile conventions and 50,000 resamples
are retained. `--bootstrap` verifies each result against the frozen endpoint.

Calibration uses stationary tapes 0, 1, and 3, disjoint from evaluation tape 2:
5,460 runs over 20 geometries. The cutoff exactly matches the e-process's
familywise crossing count along OTS traces. In 17 geometries that count is
zero. `calibration` recomputes those traces and statistics; `verify` rederives
all 20 cutoffs from the saved statistics. The null experiment independently
draws arm means from Beta(1,1), then follows OTS and monitors the e-process.
The reported endpoint is any threshold crossing, which agrees with e-ATS's
first-authorization event under the coupled preauthorization path.

## Baseline selection and replay scope

The 14 tunable baselines used geometry-specific configurations selected by
staged grid search and validation on disjoint seeds of the same nine families.
TS and OTS used fixed configurations. Candidates were scored by mean normalized dynamic pseudo-regret
across the nine families at each geometry. The search retained a shortlist,
then selected its best validation candidate, with ties broken by lowest grid
index; it did not exhaustively validate
every grid point.

| Phase | Environment-seed labels | Reward/policy streams per label and family |
|---|---|---:|
| Search stage 1 | 1830001–1830003 | 1 |
| Search stage 2 | 1830004–1830008 | 1 |
| Validation | 1830101–1830106 | 5 |
| Historical final bank used for this replay | 1840201–1840250 | 5 |

The final bank reused the earlier validation-selected baseline configurations.
All 28 released tunable configurations (14 methods at two geometries) match
their original selected grid indices and candidate-signature hashes. The
recorded final-bank seed audit passed with no reported final environment-seed
collisions against the checked development history.
[Selection provenance](../provenance/baseline_selection.json) records those
identities, split ranges, and source-artifact hashes; it is an extraction of the
historical records, not a new execution of the tuning search. The complete
[baseline validation selection](../provenance/validation_selection.json) is
bundled byte-for-byte, so the selected indices and scores can be inspected
without access to the private development repository.

e-ATS inherits controller constants tuned for its predecessor VG-dTS by the
same staged validation on these families (seeds disjoint from replay):
(γ,ν,ζ)=(0.995,0.30,0.80), using the K=32 selection at every geometry. The K=4
predecessor winner used γ=0.9925 with the same ν and ζ. All three K=32 values
lie at the search-grid boundary. Frozen for evaluation therefore does not mean
untuned in development. Its authorization threshold K/alpha_E is not
regret-tuned; it follows the declared prior-predictive error budget.
[Controller provenance](../provenance/controller_selection.json) links the
byte-identical [predecessor selection record](../provenance/synthetic_validation.json)
and four freeze notes recovered from private-source commit `6997d27`, whose
hashes match the [freeze manifest](../provenance/e_ats_full_budget_ramp_freeze_manifest.json).
The notes also disclose empirical choices of detector spacing, ramp form, and
latch behavior; see [the setting history](ALGORITHMS.md#origin-of-controller-settings).

For slow, fast and abrupt, baselines were tuned on the same mean trajectories
used in replay; only reward and policy randomness differed. This also applies
to the predecessor's controller tuning. The six
other families vary with the environment seed. No settings were selected using
the replay outcomes, but the replay is a secondary historical comparison:
baseline outcomes had been seen before its protocol freeze, and equal tuning
budgets or an untouched confirmatory superiority test are not claimed.

The [UCB settings](ALGORITHMS.md#ucb-memory-settings) also distinguish the
full-history SW-UCB configuration from D-UCB's horizon-scale discounting.

## Integrity and numerical scope

`provenance/export.json` identifies the source commit and working-tree input
hashes in the **private development repository**. Its paths, and the original
paths in selection records and freeze notes, are historical identifiers rather
than publicly accessible links. Public selection records and recovered notes
are bundled under `provenance/`; the larger original run manifests remain private.
Every exported SQLite record's original digest was verified. Only
paper-relevant fields and policy outcomes were retained; no private database
is required. `provenance/code.json` identifies the original source file, line,
and SHA-256 of each extracted function/class/constant. Tests compare those
hashes against the released bodies. Imports were relocated to `eats`; the
numerical bodies, draw order, tie-breaking, and implementation IDs are unchanged.

`provenance/release.json` hashes this release's code and inputs. All commands
check it before analysis or simulation. The historical protocol JSON files
are preserved as provenance; their paths and whole-file hashes refer to the
original repository, not the extracted layout. They are not relabeled as newly
prospective registrations. The release hash manifest is the current integrity
contract. Documentation changes also require a deliberate release-manifest update.

The reference validation platform is macOS ARM64 with Python 3.12.13 and
NumPy 2.5.2; the original replay environment generation used NumPy 2.4.4.
Two mean tables that differ across
these runtimes are supplied by the historical hash-verified bridge. The runner
first regenerates every table and uses a fixture only for a declared bridge
record. It checks the mean table, potential uniform stream, binary reward
table, policy action trace, available reward trace, regret and authorization
count. A disagreement is an error; no tolerance is applied to action hashes.
Regret allows only an absolute 2e-13 reduction tolerance.

Binary64 agreement on these tested paths is not an all-path proof of the
exact-real theorem. Platform math libraries and future runtime versions may
change paths; such mismatches are reported rather than waived. Full runs can
be lengthy and need no network after installation. Only small frozen synthetic
fixtures are shipped; no external empirical dataset is used in this manuscript.

## Platforms and CI

There are two distinct validation scopes:

| Scope | CI platforms | Commands |
|---|---|---|
| Release integrity, mathematical invariants, saved-result tables and all reported intervals | Ubuntu 24.04 x86-64 and macOS 14 ARM64 | `python -m eats verify`; `python -m pytest -m 'not reference'`; `python -m eats reproduce --bootstrap --no-figures` |
| Exact archived trajectories and complete registered/replay smoke suites | macOS 14 ARM64 | `python -m pytest -m reference`; both `python -m eats run <suite> --smoke` commands |

The `reference` pytest marker identifies tests that require exact archived
traces. An ordinary `python -m pytest` still runs them; they are not silently
skipped based on the host. CI selects and labels each scope explicitly.
Neither smoke suite is a complete scientific rerun.
`make test` runs the portable tests; `make test-reference` checks Darwin ARM64
before running the reference tests. Use both targets on the reference platform.

A Linux x86-64 CI run using Python 3.12.13 and NumPy 2.5.2 reported a mean-table
hash mismatch for `D:gradual_drift:k4:b250:random_environment:005:tape0002`.
The same group, including its five gate policies, matches the archive on the
macOS ARM64 reference runtime. This establishes a portability limitation. The
Linux error did not include the generated table, so it does not establish the
size of the numerical difference or whether policy outcomes would agree.

The generator uses floating-point transcendental functions. NumPy selects
[CPU-specific kernels](https://numpy.org/doc/stable/reference/simd/index.html),
and its [random-stream compatibility conditions](https://numpy.org/doc/stable/reference/random/compatibility.html)
also depend on the build and environment. Package version pins are insufficient
to promise byte identity across operating systems and CPU architectures.
No hash is weakened, rounded, or waived to accommodate the Linux mismatch.

Exact full-suite regeneration on Linux is currently unsupported. For Linux,
use the saved-result commands above. Figure regeneration also includes an
exact hash check of regenerated replay means; use `--no-figures` for analysis
on other platforms and consult the original images in `paper/figures/`.
Exact cross-platform regeneration would require a separately validated solution;
a green Linux analysis job does not establish that capability.

## Output and resuming

`results/` is ignored by Git. Each run's identity includes the exact group and
policy selection, Python/NumPy versions, OS/architecture/Python-build details,
and release manifest hash. Completed rows are checked against frozen outcomes
on resume; a final interrupted partial append
is discarded. Runs never write into the released data, source or paper
directories. A run using `--smoke`, `--k`, `--family` or `--limit` is explicitly
marked partial. For scientific reporting use the complete saved-data analysis
or complete relevant suite, not a smoke subset.

`provenance/validation.json` is explicitly a historical snapshot of the
2026-10-11 release preparation. Its test count, CI configuration and execution
status, and installation checks describe that date, not the current repository.
It distinguishes complete saved-result checks from the subset of policies
reexecuted then. Later local checks are recorded separately in
`provenance/validation_2026-10-19.json`; neither record substitutes for a current
GitHub Actions run on the commit being released.
