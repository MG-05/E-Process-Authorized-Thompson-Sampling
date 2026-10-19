<div align="center">

<h1>When May a Bandit Leave Its Anchor?</h1>
<h3>E-Process-Authorized Thompson Sampling under Non-stationarity</h3>

<p>
  <strong>Mayand Gulati<sup>1</sup> &nbsp;·&nbsp; Kerong Wang<sup>1</sup> &nbsp;·&nbsp; WeiChen Au<sup>2</sup></strong><br>
  <sup>1</sup>UC Santa Barbara &nbsp;&nbsp; <sup>2</sup>Purdue University
</p>

<p>
  <strong><a href="https://e-values-workshop.github.io/">NeurIPS 2026 Workshop · E-Values: From Statistics to ML</a></strong><br>
  Accepted poster &nbsp;·&nbsp; <a href="https://neurips.cc/virtual/2026/loc/paris/workshop/137611">Official NeurIPS workshop page</a>
</p>

<p>
  <a href="https://arxiv.org/abs/2610.03646"><img src="https://img.shields.io/badge/arXiv-2610.03646-b31b1b.svg" alt="arXiv: 2610.03646"></a>
  <a href="https://arxiv.org/pdf/2610.03646"><img src="https://img.shields.io/badge/Paper-PDF-365D8D.svg" alt="Read the paper PDF"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-59656E.svg" alt="Code license: MIT"></a>
</p>

<p>
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#reproduction">Reproduction</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

</div>

## Overview

**e-ATS** uses an anytime-valid e-process to decide when a bandit may begin
forgetting. Each arm maintains full-history and discounted Beta states. After
authorization, a reversible relevance score controls the discounted state's
influence; before authorization, the policy follows its optimistic Thompson
sampling (OTS) anchor exactly.

This repository accompanies the [paper](https://arxiv.org/abs/2610.03646) and
contains its algorithms, registered and replay experiments, saved per-run
results, exact seed descriptors, and verification tools. The manuscript and
appendix are included in [paper/](paper/main.tex), with
[documented clarifications relative to arXiv v1](docs/REPRODUCIBILITY.md#relation-to-arxiv-v1).

## Results

<p align="center">
  <a href="paper/figures/authorization_and_rank_summary.png"><img src="paper/figures/authorization_and_rank_summary.png" width="1000" alt="Authorization effect by arm count in the registered and replay suites, alongside e-ATS ranks among 17 methods in nine replay families."></a>
</p>

*Left: open-ramp regret minus e-ATS regret; positive values favor e-ATS.
Right: e-ATS ranks among 17 methods in each replay family; lower is better.*

Removing authorization increased mean normalized dynamic pseudo-regret by
**38.4% on the registered suite** and reduced it by **7.5% on replay**, relative
to e-ATS. Authorization's empirical benefit depends on the environment.

| Policy | Registered nonstationary | Replay |
|---|---:|---:|
| OTS | 0.0494 | 0.2157 |
| Open ramp | 0.0540 | 0.1414 |
| **e-ATS** | **0.0390** | **0.1530** |
| Certified GLR | 0.0418 | 0.1750 |
| Practical GLR | 0.0392 | 0.1529 |

Lower is better. The columns represent different populations. Bold identifies
our method. Across the 18 replay family–geometry cells, e-ATS has mean rank
**9.72**, placing **12th of 17 by mean rank and 13th by mean regret**.
Among these 17 methods, it ranks **first in mixed regime and second in random
drift** at (K,T)=(32,20000). These descriptive ranks do not establish a benefit
from authorization: the open ramp has lower regret in that random-drift cell.

SW-UCB has the lowest overall mean regret and second-best mean rank. Its frozen
window is **τ=T**, so it uses the full observation history in these runs.
D-UCB still discounts, with memory scale 1/(1−γ)=T.
[Exact baseline settings and their interpretation →](docs/ALGORITHMS.md#ucb-memory-settings)

### Registered and replay suites

| Suite | Role | Geometry | Evaluation groups |
|---|---|---|---:|
| **Registered** | Measure the contribution of authorization | K ∈ {4,8,16,32,64}; B=T/K ∈ {250,625,1250,2500} | 16,380 random nonstationary; 1,820 random stationary; 5,400 fixed stress |
| **Replay** | Compare with 16 literature baselines and examine transfer | (K,T)=(4,2500), (32,20000) | 4,500 on 606 distinct mean tables |

Stress groups are excluded from the registered primary mean and interval.
Replay has no stationary regret population. Calibration and prior-predictive
null crossing are separate experiments. [Full experiment and result map →](docs/REPRODUCIBILITY.md)

<details>
<summary><strong>Registered suite · Environment atlas</strong></summary>

![Nine registered nonstationary mechanisms, with four arm means and the instantaneous oracle mean.](paper/figures/primary_environment_atlas.png)

One frozen visualization environment per nonstationary mechanism at K=4,
B=625, T=2,500. Colors identify arms; the dashed line is the instantaneous
oracle mean. These visualization fixtures are distinct from evaluation groups.

</details>

<details>
<summary><strong>Replay suite · Environment atlas</strong></summary>

![Nine literature-derived replay families, including sinusoidal, switching, drift, and local-shock environments.](paper/figures/historical_environment_atlas.png)

One frozen example per replay family at K=4, T=2,500, using visualization seed
1,840,201. The fast-sine panel displays rounds 0–1,250; other panels show the
full horizon. These examples illustrate constructions, not empirical frequency.

</details>

## Installation

Use **Python 3.12.13** and the pinned dependencies. From this checkout:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
```

If Python 3.12.13 is unavailable, install it with
[uv](https://docs.astral.sh/uv/getting-started/installation/): run
`uv python install` and `uv venv --managed-python --seed`, then activate the
environment and run the two pip commands above. Both uv commands read
`.python-version`. CI uses uv 0.11.21 because GitHub's `setup-python` has no
macOS build for Python 3.12.13.

On Windows, activate with `.venv\Scripts\activate`. Keep the checkout: the
commands use its `data/`, `paper/`, and `provenance/` directories. The editable
install is intentional; experiment data are not packaged in a standalone wheel.

## Reproduction

### Recompute the paper from saved runs

```sh
python -m eats verify
python -m eats reproduce --bootstrap --no-figures
python -m pytest -m 'not reference'
```

These commands check release integrity and paper claims, recompute the tables
and all reported confidence intervals with **50,000 bootstrap resamples**, and
run mathematical and integrity tests. Results appear in `results/paper/`.
The analysis reads saved outcomes; it does not execute policies again.
Omit `--bootstrap` for a faster descriptive readout.

**Reference platform: macOS ARM64.** Exact seeded regeneration and figure
regeneration were validated there with the pinned runtime. Run
`python -m pytest` for all tests and omit `--no-figures` to rebuild the three
paper figures. The original images are included in [paper/figures/](paper/figures/).

Linux x86-64 has a confirmed mean-table hash mismatch in a registered smoke
case; exact regeneration is currently unsupported there. Version pins alone
do not guarantee bitwise equality across platforms. Linux CI checks saved-result
tables/intervals and mathematical invariants; macOS CI additionally checks
exact archived trajectories and both smoke suites.
[Platform limits and validation scope →](docs/REPRODUCIBILITY.md#platforms-and-ci)

### Execute the registered and replay suites

On the reference platform, start with small subsets at the original horizons:

```sh
python -m eats run registered --smoke --workers 4
python -m eats run replay --smoke --workers 4
```

Registered smoke covers all ten mechanisms at K=4, B=250. Replay smoke covers
all nine families at K=4, with all 17 paper methods and three gate comparators.
Neither is a complete scientific rerun.

<details>
<summary><strong>Full experiment commands</strong></summary>

```sh
python -m eats run registered --full --workers 4
python -m eats run replay --full --workers 4
python -m eats run calibration --full --workers 4
python -m eats run registered --full --policies calibrated-glr --workers 4
python -m eats run null --full --workers 4
```

Full registered evaluation has 23,600 paired groups, horizons up to 160,000,
and several billion policy rounds. These are substantial CPU jobs; more workers
trade RAM for throughput.

</details>

Each rerun regenerates means and rewards, executes policies, and checks action
hashes and regret against the archive. A mismatch stops the run. Repeat the
same command to resume under `results/<suite>/`. Use `--output`, `--k`,
`--family`, `--limit`, or `--policies` for explicit subsets; see
`python -m eats run --help`. The two known historical NumPy mean-table
differences have declared, hash-verified fixtures.

## Code and algorithms

```text
eats/algorithms/     e-ATS, open ramp, GLR variants, 16 baseline implementations
eats/environments/   registered and replay generators
eats/experiments/    exact seeded reruns, null and calibration support
data/               compressed per-run results, configurations, small fixtures
paper/              manuscript sources and its three figures
provenance/         extraction hashes and original prospective protocols
tests/              mathematical, coupling, trace, and release checks
```

The canonical e-ATS policy has no tuning arguments. See
[algorithm definitions and baseline settings](docs/ALGORITHMS.md).
Numerical function bodies were extracted unchanged; original implementation
IDs and helper names are retained for traceability.

<details>
<summary><strong>Minimal algorithm example</strong></summary>

```python
import numpy as np
from eats import run_e_ats

mu = np.full((4, 1000), 0.5)  # arms × rounds
returns = (np.random.default_rng(1).random(mu.shape) < mu).astype(np.uint8)
diagnostics = {}
rewards = run_e_ats(mu, np.random.default_rng(2), returns, diagnostics)
actions = diagnostics['played_arms']
```

</details>

To compile the manuscript, install TeX Live with `algorithm` and `algorithmic`,
then run `make paper`. The included figures are sufficient to compile;
regenerated copies appear under `results/paper/figures/`.

## Scope of the claims

The departure guarantee is **Beta(1,1)–Bernoulli prior-predictive**, not uniform
over fixed means. The exact-real theorem has no all-path binary64 equivalence
proof. Experiments do not establish regret optimality or universal superiority.
The 14 tunable baselines used geometry-specific configurations selected by
staged grid search and validation on disjoint seeds of the same nine families;
TS and OTS used fixed configurations, and e-ATS used one frozen configuration
across all 18 cells. Tuning effort was not equalized. The three deterministic
families reuse fixed mean trajectories despite their separate stochastic runs.
[Selection and replay scope →](docs/REPRODUCIBILITY.md#baseline-selection-and-replay-scope)

## Citation

If you use this code or its results, please cite the
[paper](https://arxiv.org/abs/2610.03646):

```bibtex
@misc{gulati2026eats,
  title         = {When May a Bandit Leave Its Anchor? {E}-Process-Authorized {Thompson} Sampling under Non-stationarity},
  author        = {Mayand Gulati and Kerong Wang and WeiChen Au},
  year          = {2026},
  eprint        = {2610.03646},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.03646}
}
```

Machine-readable citation metadata are available in [CITATION.cff](CITATION.cff).

## License

Code and repository documentation use the [MIT license](LICENSE). The
manuscript and bundled styles retain their existing rights; see
[third-party notices](THIRD_PARTY_NOTICES.md).
