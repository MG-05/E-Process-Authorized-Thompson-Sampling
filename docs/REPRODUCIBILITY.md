# Reproduction contract

The scope is the exact `E_Values_Workshop_2026/main.tex` requested for this
release and its included `appendix.tex`. The manuscript is copied unchanged.
The repository is a paper-specific extraction, not the entire development
history. Older e-authorized threshold policies, unused ablations, tuning
searches, real datasets, and unused figures are excluded.

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

## Integrity and numerical scope

`provenance/export.json` identifies the source commit and working-tree input
hashes. Every exported SQLite record's original digest was verified. Only
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

The reference runtime is Python 3.12.13, NumPy 2.5.2; the original replay
environment generation used NumPy 2.4.4. Two mean tables that differ across
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

## Output and resuming

`results/` is ignored by Git. Each run's identity includes the exact group and
policy selection, runtime and release manifest hash. Completed rows are
checked against frozen outcomes on resume; a final interrupted partial append
is discarded. Runs never write into the released data, source or paper
directories. A run using `--smoke`, `--k`, `--family` or `--limit` is explicitly
marked partial. For scientific reporting use the complete saved-data analysis
or complete relevant suite, not a smoke subset.

The release validation record in `provenance/validation.json` distinguishes
complete saved-result checks from the subset of policies reexecuted while
preparing the release.
