# Full mean-budget constant: derivation and post hoc sensitivity result

Date: 2026-08-24

Protocol:
`notes/e_value_ce_full_mean_budget_sensitivity_protocol_2026-08-24.md`

Executable:
`project_publication/e_value_ce_full_mean_budget_sensitivity.py`

Artifacts:
`E_Values_Workshop_2026/results/e_value_ce_full_mean_budget_sensitivity_v1/`

Report identity:
`2450d7b7d5eb4a600c288a85777f1f408a6074eae7ab699ba76c9031b1952f0a`

## Bottom line

There is a mathematical derivation for replacing the inherited factor
\(1/2\) by \(1\), but it is **not a regret-optimality derivation**.  The value
\(1\) is the unique endpoint that uses the complete former mean-channel
concentration budget after the dispersion channel was removed.  On the same
1,600-group validation suite, this full-budget value produced slightly
**higher**, not lower, nonstationary regret than the validated \(1/2\) ramp:

\[
\bar R(c=1)-\bar R(c=1/2)
=9.50163\times10^{-5}.
\]

The paired standard error was \(6.14876\times10^{-5}\), and the path counts
were 362 lower, 695 equal, and 383 higher for \(c=1\).  A descriptive normal
interval is approximately
\([-2.55\times10^{-5},2.16\times10^{-4}]\).  Because this is a post hoc reuse
of a seen bank, that interval is not a new confirmatory claim.

The full-budget row still substantially outperformed its exact full-budget
no-\(C^E\) ablation, old thresholded e-ATS, and OTS.  Thus the result preserves
the empirical value of \(C^E\), but supplies no reason to replace the validated
factor \(1/2\).

## What mathematics does derive

Write the allocation as

\[
\delta_{k,n}(c)
=c\frac{\beta_M}{K}\frac{6}{\pi^2n^2},
\qquad c>0,
\]

where \(\beta_M=0.05\).  Since
\(\sum_{n\ge1}6/(\pi^2n^2)=1\),

\[
\sum_{k=1}^K\sum_{n\ge1}\delta_{k,n}(c)=c\beta_M.
\]

Therefore the old concentration-budget constraint

\[
\sum_{k,n}\delta_{k,n}(c)\le\beta_M
\]

is equivalent to \(c\le1\).  For

\[
b_{k,n}(c)
=\sqrt{\frac12Q_{k,n}
\log\!\frac{2}{c(\beta_M/K)6/(\pi^2n^2)}},
\]

holding the current state fixed gives

\[
\frac{\partial b_{k,n}(c)}{\partial c}
=-\frac{Q_{k,n}}{4c\,b_{k,n}(c)}<0.
\]

Thus \(c=1\) uniquely minimizes every boundary scale within the old budget
constraint and pointwise maximizes
\(\min\{1,|M_{k,n}|/b_{k,n}(c)\}\).  This is a valid non-waste or maximum-local-
sensitivity statement.

It is not a regret statement.  Once the gate changes an action, future observed
rewards, posterior states, residuals, and later actions also change.  Dynamic
pseudo-regret is therefore not monotone in the pointwise gate strength.  No
environment-independent derivative of regret with respect to \(c\) follows
from the boundary algebra, and the empirical heterogeneity below directly
contradicts a common regret direction.

## Same-suite nonstationary comparison

The analysis reused the exact 1,440 nonstationary and 160 stationary groups
from the completed held-out validation.  It ran only the derived \(c=1\)
e-ATS and its matching \(c=1\) no-\(C^E\) ablation, then joined the frozen
comparators by group identity.

| Policy | Mean normalized dynamic pseudo-regret |
|---|---:|
| Full-budget ramp e-ATS, \(c=1\) | 0.038537193 |
| Full-budget ramp without \(C^E\), \(c=1\) | 0.054293759 |
| Validated ramp e-ATS, \(c=1/2\) | **0.038442177** |
| Validated ramp without \(C^E\), \(c=1/2\) | 0.053785903 |
| Old thresholded e-ATS | 0.043848284 |
| OTS | 0.047352580 |

The paired contrasts for full-budget e-ATS were:

| Comparator | Mean difference | Paired SE | Lower/equal/higher |
|---|---:|---:|---:|
| Full-budget no-\(C^E\) | -0.015756566 | 0.000755557 | 1008/0/432 |
| Validated \(c=1/2\) e-ATS | +0.000095016 | 0.000061488 | 362/695/383 |
| Old thresholded e-ATS | -0.005311090 | 0.000312003 | 685/443/312 |
| OTS | -0.008815387 | 0.000434145 | 742/443/255 |

Relative to the comparator means, full-budget e-ATS was 29.02% below its own
no-\(C^E\) ablation, 12.11% below old e-ATS, and 18.62% below OTS.  It was
0.247% above the validated \(c=1/2\) e-ATS.

The apparent \(C^E\) advantage grows from 0.015343726 at \(c=1/2\) to
0.015756566 at \(c=1\), but this is not an improvement of the candidate:
full-budget e-ATS worsens by 0.000095016 while its direct ablation worsens more,
by 0.000507856.  Choosing \(c=1\) because it makes the ablation gap larger
would reward damage to the ablation and is scientifically invalid.

## Heterogeneity

Among nonstationary groups, the full-budget-minus-validated e-ATS mean by arm
count was:

| K | Mean difference \(R(c=1)-R(c=1/2)\) |
|---:|---:|
| 4 | +0.000105277 |
| 8 | -0.000039163 |
| 16 | +0.000062197 |
| 32 | +0.000176860 |
| 64 | +0.000169911 |

The sign also varied by mechanism: \(c=1\) improved asynchronous switching,
random breakpoints, and sparse sustained shocks; tied isolated challenger-up;
and worsened global switching, gradual drift, incumbent-down, mixed
abrupt-plus-drift, and rapid dense switching.  This is exactly the
environment-dependence that prevents a universal regret-optimal derivation.

At \(K=4\), full-budget e-ATS remained worse than its exact no-\(C^E\)
ablation by 0.001499047 on average.  The pooled value of \(C^E\) therefore
still is not an environment-wise or arm-count-wise dominance statement.

## Stationary behavior and trajectory changes

Across the 160 stationary groups:

| Policy | Mean normalized regret |
|---|---:|
| Full-budget e-ATS | 0.010984404 |
| Validated \(c=1/2\) e-ATS | 0.010985955 |
| Old e-ATS / OTS | 0.011001943 |
| Full-budget no-\(C^E\) | 0.027197081 |

Full-budget and validated e-ATS had identical action hashes on 159/160
stationary groups; the one changed group happened to improve by enough to move
the mean by \(-1.55\times10^{-6}\).  This single realization is not evidence
that the stronger ramp improves stationary behavior.

On the 1,440 nonstationary groups, \(c=1\) and \(c=1/2\) e-ATS had identical
action hashes on 685 groups and different hashes on 755.  Their no-\(C^E\)
counterparts matched on only 30 groups.  The factor is therefore behaviorally
active even though its aggregate e-ATS regret effect is small.

## Numerical and identity audit

- selected groups: 1,600, with the original validation-manifest identity;
- binary64/100-digit mean-reference invocations: 3;
- binary64/reference mean disagreements: 0;
- minimum mean zero-decision distance: \(2.1582\times10^{-10}\);
- minimum ramp-saturation kink distance: \(3.4367\times10^{-8}\);
- minimum e-threshold distance: \(1.6831\times10^{-5}\);
- report identity: shown above.

## Decision

Do not replace \(1/2\) by \(1\) in the submitted validated algorithm.  Doing
so would define a new policy, discard the existing confirmation identity, and
slightly worsen the same-suite nonstationary mean.

The honest manuscript treatment is:

> The inherited factor \(1/2\) originated from an obsolete two-channel budget
> split.  Allocating the full remaining channel budget gives the principled
> endpoint \(c=1\), but a post hoc same-bank sensitivity produced slightly
> higher mean regret.  We therefore retain the validated frozen value and do
> not claim it is regret-optimal.

This removes any pretense that \(1/2\) is mathematically optimal.  It remains
an inherited frozen coordinate whose empirical result is supported at the
validated value, not a theorem-derived regret constant.
