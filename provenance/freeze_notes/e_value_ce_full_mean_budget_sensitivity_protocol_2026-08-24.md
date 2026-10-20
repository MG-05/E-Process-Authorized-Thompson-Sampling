# Post hoc sensitivity protocol: full mean-channel budget

Date fixed: 2026-08-24

## Status and purpose

This is a post-validation sensitivity analysis requested after the outcomes of
the normalized-ramp validation were known.  It is not a new confirmation
experiment and has no authority to replace the validated candidate.

The current mean-scale allocation is

\[
\delta_{k,n}(c)
=c\frac{\beta_M}{K}\frac{6}{\pi^2n^2},
\qquad \beta_M=0.05,
\]

with inherited value \(c=1/2\).  The removed dispersion channel was the only
reason for dividing the budget in half.  If the sole mathematical objective is
to use the largest mean-channel allocation satisfying

\[
\sum_{k=1}^K\sum_{n\ge1}\delta_{k,n}(c)\le\beta_M,
\]

then \(0<c\le1\) and the unique largest admissible value is \(c=1\).  This
choice minimizes every Hoeffding scale \(b_{k,n}(c)\) within that accounting
constraint.  It is called the **full-budget value**, not a regret-optimal
value.  The normalized ramp no longer uses this boundary as a rejection rule,
and no environment-independent regret theorem selects \(c\).

## Frozen comparison

Evaluate exactly one new value, \(c=1\), on the same 1,600 groups used by the
completed normalized-ramp validation.  Run:

1. full-budget ramp e-ATS, \(C^E\min\{1,|M|/b(c=1)\}\);
2. its exact no-\(C^E\) ablation, \(\min\{1,|M|/b(c=1)\}\).

Join these results by group ID to the already frozen rows for:

3. validated ramp e-ATS at \(c=1/2\);
4. validated ramp without \(C^E\) at \(c=1/2\);
5. old thresholded e-ATS; and
6. OTS.

The potential-reward tables, policy RNG descriptors, e-process level 0.05,
restart construction, historical latch, posterior states, controller,
environment weighting, and all other coordinates remain unchanged.

## Readouts

Report equal-group mean normalized dynamic pseudo-regret for all six rows on
the 1,440 nonstationary and 160 stationary groups.  For the new full-budget
e-ATS row, report paired mean differences, standard errors, lower/equal/higher
counts, and breakdowns by mechanism and K against:

- its full-budget no-\(C^E\) ablation;
- validated \(c=1/2\) ramp e-ATS;
- old thresholded e-ATS; and
- OTS.

Also report exact action-hash equality counts and numerical-guard diagnostics.
No candidate selection, pass/fail threshold, or claim of fresh validation is
permitted.  An unfavorable result must be retained and reported.

## Interpretation ceiling

The value \(c=1\) is mathematically derived only as the non-wasteful endpoint
of the former concentration-budget accounting.  Its regret is an empirical
property of this already-seen suite.  Even if it improves the suite mean, it
must not be called mathematically regret-optimal, prospectively validated, or
uniformly superior across environments.
