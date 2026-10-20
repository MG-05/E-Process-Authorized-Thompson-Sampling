# Canonical freeze: full-budget normalized-ramp e-ATS

Freeze date: 2026-08-24

Machine-readable freeze:
`project_publication/e_ats_full_budget_ramp_freeze_manifest.json`

Numerical-provenance audit:
`notes/e_value_ce_normalized_ramp_magic_number_audit_2026-08-23.md`

## Decision and status

The full-budget normalized-ramp policy is frozen as the canonical e-ATS
candidate for the next prospectively specified benchmark. Its canonical entry
point is

```text
src.adts.e_value_vgdts.run_e_ats_full_budget_ramp
```

with implementation ID

```text
e_ats_full_budget_normalized_ramp_v1
```

The exact direct ablation that removes only e-process authorization is

```text
src.adts.e_value_vgdts.run_e_ats_full_budget_ramp_no_ce
```

with implementation ID

```text
e_ats_full_budget_normalized_ramp_no_ce_v1
```

This is a **code and protocol freeze**, not a claim that the policy is
universally regret-optimal or already prospectively confirmed. No scientific
suite, validation bank, or baseline comparison was run while creating this
freeze. Only deterministic unit and invariant checks were permitted.

## Behavior-complete frozen contract

The public canonical entry point has no policy-coordinate argument. It accepts
only the environment mean table, a NumPy random generator, an optional binary
potential-reward table, and an optional diagnostics dictionary. In particular,
it exposes no (c), mean-budget fraction, boundary scale, relevance transform,
authorization coupling, discount, controller weight, or restart-grid value.

The frozen behavior is:

| Coordinate or rule | Frozen value |
|---|---|
| reward model | Bernoulli |
| long and short priors | Beta\((1,1)\) |
| restart candidate spacing | \(\eta=0.25\), first candidate pull \(s_1=2\) |
| restart weights | \(w_j=1/[j(j+1)]\), future mass \(1/(J+1)\) |
| e-process level | \(\alpha_E=0.05\) |
| armwise e-threshold | \(K/\alpha_E=20K\) |
| e-authorization coupling | absorbing historical latch |
| mean-normalization budget | \(\beta_M=0.05\) |
| mean spending | \(\delta_{k,n}=(\beta_M/K)6/(\pi^2n^2)\) |
| detector decay | \(\operatorname{clip}(1-T^{-1/2},0,0.999)\) |
| relevance | \(r_{k,n}=0\) for \(|M_{k,n}|\le10^{-12}\), otherwise \(\min\{1,|M_{k,n}|/b_{k,n}\}\) |
| deployed gate | \(C^E_{k,n}r_{k,n}\) |
| short-state discount | \(0.995\) on every global round |
| long-state discount | \(1\) |
| dual-memory exponent | \(0.30\) |
| local takeover scale | \(0.80\) |
| global takeover scale | \(0\) |
| reliability mass | inherited v2.2 square-root arm-local-budget rule |
| posterior mixing | convex short/long mixture |
| action score | maximum of mixed posterior sample and mixed posterior mean |
| tie breaking | NumPy `argmax`, hence the lowest index on an exact tie |
| update timing | reward at round (t) can change only actions from (t+1) onward |

The no-\(C^E\) ablation calls the same private kernel and changes exactly one
Boolean construction choice: it deploys (r_{k,n}) without requiring
e-authorization. It does not have a separately copied controller that could
drift.

## Why the half-budget multiplier is absent

The old factor (1/2) split a declared budget between mean and dispersion
channels. The canonical policy has no dispersion channel. Because

\[
\sum_{n\ge1}\frac{6}{\pi^2n^2}=1,
\]

the frozen schedule spends exactly

\[
\sum_{k=1}^K\sum_{n\ge1}\delta_{k,n}=\beta_M.
\]

This derives the removal of the obsolete multiplier as a budget-accounting
and identifiability decision. It does **not** derive regret optimality. The
post hoc same-bank sensitivity found that the full-budget policy's mean regret
was (9.5016\times10^{-5}) higher than the validated half-budget ramp on the
nonstationary subset, with uncertainty spanning zero. That result is retained
in `notes/e_value_ce_full_mean_budget_sensitivity_results_2026-08-24.md` and
must not be rewritten as an empirical win for the full-budget value.

The full-budget version is frozen because it removes a scientifically
unidentified multiplier and makes the declared budget accounting exact—not
because it won a regret comparison.

## What is and is not mathematically guaranteed

Under the declared independent per-arm Beta\((1,1)\)--Bernoulli
prior-predictive null, the restart mixture is an e-process. Ville's inequality
and equal arm allocation bound the probability of any e-authorization by
\(\alpha_E\). Before authorization, the canonical policy is exactly on its OTS
action branch. The evidence-budget and baseline-capacity results for the
unchanged restart mixture also remain applicable.

The following stronger claims are not justified:

- uniform fixed-\(p\) false-authorization control over every stationary
  environment;
- the old \(\min\{\alpha,\varepsilon_{K,T}\}\) departure statement, because
  the ramp is positive below the former Hoeffding boundary;
- a requirement that the current e-value remain above threshold after the
  historical latch has fired;
- environment-wise or arm-count-wise regret dominance over the no-\(C^E\)
  ablation, OTS, or any other baseline;
- universal regret optimality of the full-budget normalization; or
- absence of empirical coordinates. Several controller values were tuned or
  inherited, and their exact provenance is recorded in the numerical audit.

## Static and deterministic release gates

The freeze is valid only while all of the following remain true:

1. the canonical source and both policy dependencies match the SHA-256 values
   in the machine-readable manifest;
2. the runtime contract returned by `frozen_e_ats_contract()` matches the
   manifest exactly;
3. the public function signatures expose no policy coordinate;
4. the restart e-process remains exactly equal to its independent reference on
   deterministic reward sequences;
5. the full-budget update uses the complete (6/(\pi^2n^2)) schedule;
6. the canonical functions deterministically reproduce their action arrays
   under identical inputs and RNG state;
7. the canonical source reproduces the already specified development-wrapper
   semantics on small deterministic unit fixtures; and
8. the e-authorized and no-\(C^E\) functions retain distinct implementation
   IDs while sharing the same private kernel.

These checks establish identity and implementation consistency. They are not
an experiment and do not estimate regret.

## Change control

After this freeze, any change to an equation, numerical value, update order,
random-draw order, e-process construction, latch semantics, posterior state,
score, tie rule, function parameter surface, or direct-ablation semantics
requires:

1. a new implementation ID;
2. a new freeze manifest;
3. a written reason for the change; and
4. a newly declared evaluation protocol.

Documentation, diagnostics, and tests may change under the current ID only if
they are proved not to alter policy behavior. Any change to a file pinned by
this manifest—including provenance documentation, diagnostics in the policy
source, the verifier, or the runtime pin—still requires a new freeze-manifest
identity, even when the policy implementation ID may remain unchanged. A
future benchmark runner must verify this freeze before generating any result.
The benchmark protocol, environment census, baseline set, randomization,
estimand, aggregation, and multiplicity rules are separate scientific choices
and are deliberately not invented by this algorithm freeze.
