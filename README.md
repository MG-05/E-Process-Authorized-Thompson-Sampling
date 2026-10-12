# E-process-authorized Thompson sampling (e-ATS)

Reproduction code for **When May a Bandit Leave Its Anchor? E-Process-Authorized
Thompson Sampling under Non-stationarity**, by Mayand Gulati, Kerong Wang, and
WeiChen Au. The source manuscript is [paper/main.tex](paper/main.tex).

This release contains the experiments reported by that manuscript, including
its appendix: the registered authorization comparison, the literature-derived
replay, the two GLR gates, the reported calibrated-GLR sensitivity, and the
prior-predictive crossing experiment. It includes saved per-run results, exact
seed descriptors, source hashes, and code to regenerate the experiments.

## Install

Use **Python 3.12.13** for the reference runtime. From this checkout:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
```

On Windows, activate with `.venv\Scripts\activate`. Keep this checkout: the
experiment commands use its `data/`, `paper/`, and `provenance/` directories.
The editable install is intentional; this is a repository reproduction bundle,
not a standalone wheel containing the experiment data.

## Reproduce the paper from saved runs

```sh
python -m eats verify
python -m eats reproduce --bootstrap
python -m pytest
```

`verify` checks release hashes, suite censuses, and numerical paper claims.
`reproduce --bootstrap` recomputes tables, all reported confidence intervals
(50,000 resamples), and the three figures used by the paper. Outputs go to
`results/paper/`. No saved policy is executed by these two commands.
Omit `--bootstrap` for a faster descriptive readout, or use `--no-figures`.

The headline means should be:

| Policy | Registered nonstationary | Replay |
|---|---:|---:|
| OTS | 0.0494 | 0.2157 |
| Open ramp | 0.0540 | 0.1414 |
| e-ATS | 0.0390 | 0.1530 |
| Certified GLR | 0.0418 | 0.1750 |
| Practical GLR | 0.0392 | 0.1529 |

Values are normalized dynamic pseudo-regret; lower is better. The columns are
different populations. Authorization helps on average in the registered suite
and hurts on replay. e-ATS has mean rank **9.72 of 17** replay methods.

## Rerun the policies

Start with small subsets at the original horizons:

```sh
python -m eats run registered --smoke --workers 4
python -m eats run replay --smoke --workers 4
```

The registered smoke run covers all ten mechanisms at K=4, B=250. Replay covers
all nine families at K=4 and runs all 17 paper methods plus the three gate
comparators. Neither smoke run is a complete scientific rerun.

Full commands:

```sh
python -m eats run registered --full --workers 4
python -m eats run replay --full --workers 4
python -m eats run calibration --full --workers 4
python -m eats run registered --full --policies calibrated-glr --workers 4
python -m eats run null --full --workers 4
```

Every rerun regenerates the means and rewards, executes the selected policies,
and checks action hashes and regret against the archived outcomes. A mismatch
stops the run. Outputs are resumable under `results/<suite>/`; repeat the exact
command to resume. `--output`, `--k`, `--family`, `--limit`, and `--policies`
support explicit subsets. See `python -m eats run --help`.

Full runs are substantial CPU jobs, not quick demos: registered evaluation has
23,600 paired groups, horizons up to 160,000, and several billion policy rounds.
More workers trade RAM for throughput. Results can differ across numerical
platforms even with pinned versions; hash checks detect that rather than hiding
it. The two known historical NumPy mean-table differences have included,
hash-verified fixtures.

## Two suites, two roles

| Suite | Purpose | Geometry | Groups |
|---|---|---|---:|
| Registered | Contribution of authorization | K ∈ {4,8,16,32,64}; B=T/K ∈ {250,625,1250,2500} | 16,380 random nonstationary; 1,820 random stationary; 5,400 fixed stress |
| Replay | Comparison with 16 literature baselines and transfer of the gate comparison | (K,T)=(4,2500), (32,20000) | 4,500 on 606 distinct mean tables |

Stress groups are a separate breadth census, excluded from the registered
primary mean and interval. Replay contains no stationary regret population.
The calibration and null commands are separate experiments, not extra replay
families. [Experiment details and result map](docs/REPRODUCIBILITY.md).

## Layout and algorithm API

```text
eats/algorithms/     e-ATS, open ramp, GLR variants, 16 baseline implementations
eats/environments/   registered and replay generators
eats/experiments/    exact seeded reruns, null and calibration support
data/               compressed per-run records, configurations, small fixtures
paper/              unchanged manuscript sources and its three figures
provenance/         extraction hashes and original prospective protocols
tests/              mathematical, coupling, trace, and release checks
```

```python
import numpy as np
from eats import run_e_ats

mu = np.full((4, 1000), 0.5)  # shape: arms × rounds
returns = (np.random.default_rng(1).random(mu.shape) < mu).astype(np.uint8)
diagnostics = {}
rewards = run_e_ats(mu, np.random.default_rng(2), returns, diagnostics)
actions = diagnostics['played_arms']
```

The canonical policy has no tuning arguments. [Algorithm names, definitions,
and baseline settings](docs/ALGORITHMS.md) explain each implementation. The
numerical function bodies were extracted unchanged; original helper names and
implementation IDs are retained for traceability.

To compile the manuscript, install TeX Live with `algorithm` and `algorithmic`,
then run `make paper`. The tracked paper figures are sufficient to compile;
regenerated copies appear under `results/paper/figures/`.

## Scope of the claims and licensing

The e-process departure guarantee is **Beta(1,1)–Bernoulli prior-predictive**,
not uniform over fixed means. The exact-real theorem has no all-path binary64
equivalence proof. The experiments do not establish regret optimality or
universal superiority. Baselines are local implementations with previously
chosen, geometry-specific settings; tuning effort was not equalized.

Code and repository documentation use the [MIT license](LICENSE).
The manuscript and bundled styles retain their existing rights; see
[third-party notices](THIRD_PARTY_NOTICES.md). Please cite the paper using
[CITATION.cff](CITATION.cff).
