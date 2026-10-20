# Full-budget normalized-ramp e-ATS: algorithm and numerical provenance

Original audit date: 2026-08-23

Updated for the full-budget candidate: 2026-08-24

## Status and scope

This note gives the complete full-budget normalized-ramp e-ATS algorithm and
traces every behavior-affecting numerical value to one of the following
categories:

1. a mathematical identity or consequence of a declared allocation;
2. a structural or probabilistic modeling choice;
3. a user-declared error budget;
4. an empirically selected coordinate, with its selection artifact;
5. an inherited fixed calibration, with its historical source;
6. a numerical safeguard; or
7. an experimental-analysis constant that is not part of the policy.

The repository does **not** support the claim that every value is theorem-
derived or regret-optimal. It supports the narrower, auditable statement that
the extra multiplier \(c\) can be removed by using the full declared mean
budget, while the remaining empirical and inherited coordinates are disclosed
as such.

There is one essential status distinction:

- the prospectively validated ramp used \(c=1/2\);
- the full-budget \(c=1\) successor was evaluated only in a post hoc same-bank
  sensitivity study.

Therefore this note defines the **full-budget successor candidate**, now frozen
as the canonical implementation for the next prospectively specified
benchmark. It is not yet a prospectively performance-confirmed replacement
for the validated half-budget policy, so the manuscript must not call it the
final validated algorithm before that separate benchmark is completed.

The canonical entry points are
`src.adts.e_value_vgdts.run_e_ats_full_budget_ramp` and its exact direct
ablation `src.adts.e_value_vgdts.run_e_ats_full_budget_ramp_no_ce`, with new
implementation IDs and no public policy-coordinate argument. Their source and
dependencies are hash-bound in
`project_publication/e_ats_full_budget_ramp_freeze_manifest.json`; the written
freeze is `notes/e_ats_full_budget_ramp_freeze_2026-08-24.md`. The configurable
development wrapper remains only as historical specification and audit code.

The inherited controller vector used below is the `primary_3`
`e_authorized_mean_gate` vector in
`report/project_publication/e_value_vgdts_layer3_regret_atlas_v1/run_manifest_v5.json`.
The history of each non-gate coordinate is traced separately in the ledger;
the manifest proves what was run, not why a value was chosen.

## Complete algorithm

Consider \(K\) Bernoulli arms over a declared horizon \(T\). All state below is
arm-local except for action selection. The gate used to choose action \(A_t\)
is predictable: it is computed only from rewards observed before round \(t\).

### 1. Long and short Beta states

For each arm \(k\), initialize

\[
(a^L_{k,0},b^L_{k,0})=(1,1),
\qquad
(a^S_{k,0},b^S_{k,0})=(1,1).
\]

The long state is undiscounted. The short state is discounted every global
round toward its Beta\((1,1)\) prior with

\[
\gamma_S=0.995.
\]

Because the frozen controller has

\[
\gamma_{\min}=\gamma_{\max}=0.995,
\]

the gate does not change the discount factor. It changes only the mixture of
short- and long-state action scores. The short-state geometric mass scale is
\(1/(1-0.995)=200\) global rounds and its half-life is approximately \(138.3\)
rounds; neither number is an arm-local pull count.

### 2. Restart e-process and absorbing authorization

For arm \(k\), maintain the deployed Beta\((1,1)\) restart mixture. Candidate
change locations are arm-local pull indices. They begin at \(s_1=2\), meaning
a possible change immediately before the second observed reward, and follow

\[
s_{j+1}=\max\{s_j+1,\lceil(1+\eta)s_j\rceil\},
\qquad \eta=0.25.
\]

Each live component compares a Beta\((1,1)\) posterior-predictive model for the
post-candidate segment with the arm's pooled Beta\((1,1)\) predictive model.
Its weight is

\[
w_j=\frac{1}{j(j+1)}.
\]

After \(J\) candidates have become evaluable, the capital reserved for all
future candidates is

\[
r_J=\frac{1}{J+1}.
\]

This follows exactly from

\[
\sum_{j=1}^{J}\frac{1}{j(j+1)}=1-\frac{1}{J+1}.
\]

Writing \(L^{(j)}_{k,n}\) for component \(j\)'s Bayes factor after \(n\)
arm-local observations, the e-process is

\[
E_{k,n}=\sum_{j=1}^{J(n)}w_jL^{(j)}_{k,n}+r_{J(n)}.
\]

Use the declared family-wise authorization level

\[
\alpha_E=0.05
\]

and equal arm allocation \(\alpha_E/K\). The per-arm threshold is therefore

\[
E_{k,n}\ge \frac{K}{\alpha_E}=20K.
\]

Authorization is absorbing:

\[
C^E_{k,n}
=
\mathbf 1\!\left\{
\max_{m\le n}E_{k,m}\ge K/\alpha_E
\right\}.
\]

Thus an e-value may subsequently fall below threshold without removing the
historical authorization flag.

### 3. Current mean discrepancy and full-budget normalization

Let \(n=N_k(t)\) be the selected arm's pull count after observing its reward at
round \(t\). Before adding that reward to the long posterior, form the long-
state prediction

\[
\widehat p^L_{k,n-1}
=
\frac{a^L_{k,n-1}}{a^L_{k,n-1}+b^L_{k,n-1}}.
\]

The detector decay is

\[
\lambda_D
=
\operatorname{clip}\!\left(1-T^{-1/2},0,0.999\right).
\]

Update

\[
M_{k,n}
=
\lambda_D M_{k,n-1}
+(1-\lambda_D)
\bigl(X_{k,n}-\widehat p^L_{k,n-1}\bigr),
\]

and

\[
Q_{k,n}
=
\lambda_D^2Q_{k,n-1}+(1-\lambda_D)^2,
\]

from \(M_{k,0}=Q_{k,0}=0\).

The proposed full-budget spending schedule is

\[
\delta_{k,n}
=
\frac{\beta_M}{K}\frac{6}{\pi^2n^2},
\qquad \beta_M=0.05.
\]

The normalization boundary is

\[
b_{k,n}
=
\sqrt{\frac{Q_{k,n}}{2}
\log\!\frac{2}{\delta_{k,n}}}.
\]

It is used as a scale, not as a rejection threshold. With the binary64 guard

\[
\varepsilon_M=10^{-12},
\]

the normalized ramp is

\[
r_{k,n}
=
\begin{cases}
0, & |M_{k,n}|\le \varepsilon_M,\\[2mm]
\min\{1,|M_{k,n}|/b_{k,n}\}, & |M_{k,n}|>\varepsilon_M.
\end{cases}
\]

The deployed gate is

\[
g_{k,t+1}=C^E_{k,n}r_{k,n}.
\]

Hence the policy is exactly on its OTS branch before any e-process crossing,
but the ramp can be positive below the former Hoeffding rejection boundary
after authorization.

### 4. Gate-to-score controller

For each arm before action selection, set

\[
n^S_{k,t}=\max\{a^S_{k,t}+b^S_{k,t}-2,0\}.
\]

The subtraction of \(2\) removes the Beta\((1,1)\) prior mass. Define

\[
\rho(T,K)
=
\max\!\left\{
4,
16\sqrt{\frac{T}{5000}\frac{4}{K}}
\right\},
\qquad
R_{k,t}=\frac{n^S_{k,t}}{n^S_{k,t}+\rho(T,K)}.
\]

The base and final short-state weights are

\[
u_{k,t}=g_{k,t}^{q_{\rm mix}},
\qquad q_{\rm mix}=0.30,
\]

and

\[
\omega_{k,t}
=
u_{k,t}
+\zeta(1-u_{k,t})g_{k,t}R_{k,t},
\qquad \zeta=0.80.
\]

The implementation clips \(\omega_{k,t}\) to \([0,1]\), although the declared
inputs already make that interval the intended range. Global gate spillover
is disabled:

\[
\zeta_{\rm global}=0.
\]

Independently draw

\[
\widetilde\theta^S_{k,t}\sim
\operatorname{Beta}(a^S_{k,t},b^S_{k,t}),
\qquad
\widetilde\theta^L_{k,t}\sim
\operatorname{Beta}(a^L_{k,t},b^L_{k,t}).
\]

Form the convex mixed sample and mean

\[
\widetilde\theta_{k,t}
=
\omega_{k,t}\widetilde\theta^S_{k,t}
+(1-\omega_{k,t})\widetilde\theta^L_{k,t},
\]

\[
\widehat p_{k,t}
=
\omega_{k,t}\frac{a^S_{k,t}}{a^S_{k,t}+b^S_{k,t}}
+(1-\omega_{k,t})
\frac{a^L_{k,t}}{a^L_{k,t}+b^L_{k,t}}.
\]

The optimistic score is

\[
\Theta_{k,t}=\max\{\widetilde\theta_{k,t},\widehat p_{k,t}\},
\qquad
A_t=\arg\max_k\Theta_{k,t}.
\]

There is no additional confidence bonus. When every \(g_{k,t}=0\), the code
executes a separate exact OTS branch and consumes no short-state random draws.
NumPy's `argmax` supplies deterministic lowest-index tie breaking.

### 5. Posterior update order

After observing \(X_{A_t,t}\in\{0,1\}\), update the discrepancy and e-process
for the played arm as above. Their new gate affects the next action, not the
current one. Then discount every short state toward the prior:

\[
a^S_{k,t+1}=1+0.995(a^S_{k,t}-1),
\qquad
b^S_{k,t+1}=1+0.995(b^S_{k,t}-1),
\]

and add the current reward only to the played arm:

\[
a^S_{A_t,t+1}\mathrel{+}=X_{A_t,t},
\qquad
b^S_{A_t,t+1}\mathrel{+}=1-X_{A_t,t}.
\]

The long state is not discounted:

\[
a^L_{A_t,t+1}=a^L_{A_t,t}+X_{A_t,t},
\qquad
b^L_{A_t,t+1}=b^L_{A_t,t}+1-X_{A_t,t},
\]

with unplayed long states unchanged.

## Why the full-budget value is \(c=1\)

The validated ramp inherited

\[
\delta_{k,n}(c)
=
c\frac{\beta_M}{K}\frac{6}{\pi^2n^2}
\]

with \(c=1/2\). That half was the former equal split between mean and
dispersion channels. The revised policy contains no dispersion channel.

Since

\[
\sum_{n=1}^{\infty}\frac{6}{\pi^2n^2}=1,
\]

we have

\[
\sum_{k=1}^{K}\sum_{n=1}^{\infty}\delta_{k,n}(c)
=c\beta_M.
\]

Under the inherited constraint that this sum not exceed \(\beta_M\), the
admissible range is \(0<c\le1\). Moreover,

\[
\frac{\partial b_{k,n}(c)}{\partial c}
=-\frac{Q_{k,n}}{4c\,b_{k,n}(c)}<0.
\]

Thus \(c=1\) is the unique budget-saturating value and the pointwise most
sensitive boundary within that constraint. It removes a redundant multiplier
because only \(c\beta_M\) enters the boundary.

This derivation is a non-waste and identifiability argument, **not** a universal
regret-optimality theorem. Closed-loop regret need not be monotone in \(c\).
The post hoc sensitivity found

\[
\bar R(c=1)-\bar R(c=1/2)=9.5016\times10^{-5}
\]

on 1,440 nonstationary groups, with paired standard error
\(6.1488\times10^{-5}\) and a descriptive normal interval approximately
\([-2.55\times10^{-5},2.16\times10^{-4}]\). This is no credible evidence of a
regret difference, but the reused bank cannot establish prospective
equivalence. The result is recorded in
`notes/e_value_ce_full_mean_budget_sensitivity_results_2026-08-24.md`.

## Numerical provenance ledger

### Evidence layer

| Value or rule | Classification | Exact provenance and limitation |
|---|---|---|
| Bernoulli rewards | Modeling choice | Required by the deployed Beta--Bernoulli posterior and e-process. Not tuned. |
| Beta\((1,1)\) long, short, pooled-null, and segment priors | Modeling choice | Fixed throughout the OTS/e-process program. It gives the prior-predictive null used by the theorem; it is not uniform fixed-\(p\) validity and was not regret-tuned. |
| \(\alpha_E=0.05\) | User-declared error budget | Frozen before the fresh e-value studies in `notes/e_value_workshop_calibration_preregistration_2026-08-11.md`. It was not tuned for regret. |
| equal arm budget \(\alpha_E/K\) | Design choice with derived threshold | Once equal allocation is declared, Ville plus the union bound give threshold \(K/\alpha_E=20K\). Equal allocation is not uniquely required. |
| absorbing historical latch | Inherited and development-retained | A current-level alternative was predeclared and failed its development rule in `notes/e_value_ce_current_level_coupling_development_results_2026-08-23.md`; the historical latch was retained. This is empirical development history, not a theorem that latching minimizes regret. |
| \(s_1=2\) | Structural | Earliest candidate yielding nonempty pre- and post-candidate blocks. No free scalar. |
| \(\eta=0.25\) | Empirically selected detector construction | Selected from \(\{0.25,0.5,1\}\) by the final scale-robust v8 construction protocol in `notes/e_value_restart_construction_selection_preregistration_v8_2026-08-10.md`; the authoritative result is `report/project_publication/e_value_restart_construction_selection_v8/results/construction_selection.json`. Only \(0.25\) was eligible across all registered scale domains. This tuned detection construction, not bandit regret. Earlier v6 selection of \(1\) was invalidated and is not authority. |
| \(w_j=1/[j(j+1)]\) | Normalized design choice | Telescopes to total mass one. Mathematically convenient but not unique and not tuned. |
| future capital \(1/(J+1)\) | Derived | Exact remaining mass implied by the selected weight rule. |

### Mean-discrepancy layer

| Value or rule | Classification | Exact provenance and limitation |
|---|---|---|
| \(\beta_M=0.05\) | Inherited fixed calibration | The old VG-dTS mean-channel budget. It was fixed, not tuned, in the symmetric retuning protocol. A July OFAT development sensitivity evaluated \(0.01\) and \(0.10\) around the frozen \(0.05\) center (`notes/vgdts_v3a_ofat_sensitivity_dev_results_2026-07-18.md`), but that study explicitly prohibited reselection and used the old thresholded/two-channel controller. No artifact establishes \(0.05\) as ramp-regret-optimal. |
| equality \(\beta_M=\alpha_E=0.05\) | Historical coincidence | The proof does not require the two budgets to match. They must remain separately named: \(\alpha_E\) controls e-authorization; \(\beta_M\) controls ramp scale. |
| \(c=1\) | Mathematically derived endpoint | Unique value exhausting the declared mean budget after dispersion removal. Not empirically tuned and not claimed regret-optimal. It still requires fresh policy confirmation because the available evaluation was post hoc. |
| \(6/(\pi^2n^2)\) | Partly derived, partly chosen | The factor \(6/\pi^2\) exactly normalizes the chosen \(n^{-2}\) spending schedule. The exponent \(2\) is an inherited summable schedule choice, not a theorem-unique exponent and not tuned. |
| \(Q/2\) and \(\log(2/\delta)\) | Concentration algebra inherited as scale | The old two-sided Hoeffding boundary supplies these factors. The ramp uses the resulting boundary as normalization, so these factors are now performance-bearing even though the ramp does not obtain the old rejection guarantee. |
| \(\lambda_D=\operatorname{clip}(1-T^{-1/2},0,0.999)\) | Inherited horizon calibration | The \(\sqrt T\) window was fixed in the v3 construction, not regret-derived. The July OFAT study tested timescale multipliers \(1/2\) and \(2\) under the old controller: halving was worse, doubling inconclusive. That is descriptive local sensitivity, not ramp-specific tuning. |
| cap \(0.999\) | Numerical/engineering safeguard | Ensures \(1-\lambda_D\ge0.001\). It is inactive on the registered \(T\le160000\), where \(\lambda_D\le0.9975\). Not tuned. |
| boundary scale \(1\) | Frozen structural scale | The ramp-development protocol prohibited an extra multiplier. Earlier exploratory looseness values do not carry into the selected ramp. |
| ramp \(\min\{1,\lvert M\rvert/b\}\) | Empirically selected discrete transform | Selected over the soft ratio \(\lvert M\rvert/(\lvert M\rvert+b)\) by the predeclared rule on 400 development groups; both qualified, and the ramp had lower nonstationary e-authorized regret. See `notes/e_value_ce_effect_size_relevance_development_protocol_2026-08-23.md` and `notes/e_value_ce_effect_size_relevance_development_results_2026-08-23.md`. The unchanged half-budget ramp then passed a separate 1,600-group validation. |
| ramp cap \(1\) | Structural | Keeps relevance in \([0,1]\) and marks the former boundary as full relevance. It is not a fitted scalar once the ramp form is declared. |
| \(\varepsilon_M=10^{-12}\) | Binary64 safeguard | Inherited numerical zero guard. It was not tuned and has no statistical interpretation. The implementation uses the same value both to make a sufficiently tiny discrepancy exactly inactive and to reject a long-state prediction within \(10^{-12}\) of 0 or 1 rather than silently clipping its innovation. It therefore has a tiny exact-versus-floating-point semantic effect. |

### Downstream controller

| Value or rule | Classification | Exact provenance and limitation |
|---|---|---|
| convex short/long score mixing | Inherited structural choice | The VG-dTS v3-A architecture. Not retuned in the e-ATS ramp study. |
| optimistic score \(\max(\widetilde\theta,\widehat p)\) | Inherited structural choice | OTS rule retained from the historical algorithm. No new scalar. |
| confidence bonus \(0\) | Structural zero | The no-confidence-bonus v3-A construction; no posterior-standard-deviation bonus is added. |
| \(\gamma_{\max}=0.995\) | Inherited fixed coordinate | Formalized as fixed, not tunable, in the simplified v2.2 surface (`notes/vgdts_workshop_log_2026-04-07.md`). No artifact establishes it as a global optimum. |
| \(\gamma_{\min}=0.995\) | Empirically selected then transferred | The August symmetric primary-288 grid searched \(\{0.55,0.80,0.90,0.95,0.97,0.99,0.9925,0.994,0.995\}\) and selected \(0.995\) at \(K=32,T=20000\). The e-value program transferred that K=32 coordinate to all geometries rather than retuning by \(K,T\). See `notes/vgdts_v3_symmetric_retuning_preregistration_2026-08-07.md` and `notes/vgdts_v3_symmetric_retuning_results_2026-08-08.md`. |
| \(q_{\rm mix}=0.30\) | Empirically selected boundary coordinate | Selected at both K=4 and K=32 by the same primary-288 validation pipeline from \(\{0.30,0.45,0.60,0.80\}\). It was the lower grid face, which was not expanded; no interior-optimum claim is justified. It was inherited unchanged by the ramp experiment. |
| \(\zeta=0.80\) | Empirically selected boundary coordinate | Selected at both K=4 and K=32 from \(\{0,0.10,0.20,0.30,0.40,0.50,0.60,0.80\}\). It was the upper grid face; no unconstrained-optimum claim is justified. It was inherited unchanged by the ramp experiment. |
| \(\rho=\max\{4,16\sqrt{(T/5000)(4/K)}\}\) | Inherited fixed calibration | Descends from the v2.2 rule \(2K_{\rm ref}(\alpha_0+\beta_0)\) at \((T_{\rm ref},K_{\rm ref})=(5000,4)\): the Beta\((1,1)\) prior mass is 2, the reference arm count is 4, and an inherited leading multiplier 2 gives \(16\). That leading 2 is a historical calibration, not a theorem-derived optimum. The exact current square-root arm-local-budget rule is in `src/adts/vgdts_v22_paper.py` and was frozen as the OFAT center in `notes/vgdts_v3a_ofat_sensitivity_dev_preregistration_2026-07-17.md`. The July OFAT varied an overall multiplier and the scaling exponent and found negligible aggregate effects, but explicitly prohibited tuning. None of these values was chosen by the ramp study. |
| reliability floor \(4\) | Derived from inherited reference-prior rule, not performance theory | Equals twice the Beta\((1,1)\) prior mass. It is inactive in all registered atlas cells. |
| reference \(T=5000,K=4\) | Historical calibration geometry | Fixed in the v2.2 paper-facing construction. It is not a universal operating point or a regret theorem. |
| global takeover \(0\) | Structural restriction | e-ATS is arm-local. Global spillover was held at zero throughout the e-value program and was not ramp-tuned. |
| subtraction \(2\) in \(n^S=a^S+b^S-2\) | Derived | Removes the Beta\((1,1)\) prior mass \(1+1=2\); not tunable. |
| clipping gates/weights to \([0,1]\) | Structural/numerical | Required for convex weights. Endpoints are probability-mixture endpoints, not performance tunes. |

## Tuning and inheritance chronology

The numerical history is:

1. **April 2026, v2.2 simplification.** Beta\((1,1)\),
   \(\gamma_{\max}=0.995\), the reference geometry \((5000,4)\), and the
   analytic reliability-calibration lineage were fixed in the paper-facing
   construction. These values were not all tuned; several were explicitly
   removed from the tuning surface.
2. **July 2026, development OFAT.** The exact current reliability formula,
   false-alarm budget, and detector timescale were frozen as the study center
   before receiving local sensitivity perturbations. The protocol forbade
   using outcomes to retune that center.
3. **August 7--8, symmetric VG-dTS retuning.** The primary-288 pipeline tuned
   \(\gamma_{\min},q_{\rm mix},\zeta\). The K=32 winner was
   \((0.995,0.30,0.80)\); all three were boundary selections. The e-value
   program later inherited this fixed coordinate at every geometry.
4. **August 10, restart construction selection v8.** The final scale-robust
   detector study selected \(\eta=0.25\) from \(\{0.25,0.5,1\}\).
5. **August 11, e-value calibration freeze.** \(\alpha_E=0.05\), equal arm
   allocation, the prior-predictive null, and the inherited controller were
   fixed before fresh e-process outcomes.
6. **August 23, ramp development and validation.** The normalized ramp was
   selected over the soft ratio on development data and then validated without
   alteration at inherited \(c=1/2\). No controller coordinate was retuned.
7. **August 24, full-budget sensitivity.** \(c=1\) was derived as the unique
   budget-saturating endpoint and tested post hoc on the already seen validation
   bank. It was not prospectively selected or confirmed.

## Registered-geometry values implied by inherited rules

For the registered relation \(T=KH\),

\[
\rho=\max\{4,16\sqrt{4H/5000}\}.
\]

The floor is inactive, giving:

| \(H=T/K\) | \(\rho\) |
|---:|---:|
| 250 | 7.155418 |
| 625 | 11.313708 |
| 1250 | 16.000000 |
| 2500 | 22.627417 |

For the full-budget ramp, the first raw relevance values after one binary
reward are:

| \(K\) | full-budget first raw ramp |
|---:|---:|
| 4 | 0.299533 |
| 8 | 0.282481 |
| 16 | 0.268044 |
| 32 | 0.255616 |
| 64 | 0.244771 |

These values are consequences of the declared equations; they were not tuned.
With \(q_{\rm mix}=0.30\), small positive gates are amplified by the inherited
power map. This explains why the normalization constants remain behaviorally
important even though \(c=1\) versus \(1/2\) changed average regret only
slightly.

## Proof scope for the full-budget ramp

### Results that survive

1. **Anytime e-certified anchor inheritance.** The deployed gate is exactly
   zero before an e-process crossing. Under the declared independent per-arm
   Beta\((1,1)\)--Bernoulli prior-predictive null, Ville's inequality and the
   armwise union bound give family-wise activation probability at most
   \(\alpha_E\). This result is independent of \(c,\beta_M,\lambda_D\), the
   ramp shape, \(0.995,0.30,0.80\), and \(\rho\), provided the zero-before-
   crossing branch is exact.
2. **Restart evidence budget and baseline capacity.** These results concern
   the unchanged restart e-process and retain their explicit dependence on
   \(K,\alpha_E,\eta,w_j\), and Beta-mixture coding redundancy.
3. **Immediate uncontrolled-adaptation prevention.** The no-\(C^E\) ramp can
   activate after the first binary reward, whereas e-ATS remains exactly on its
   OTS branch because the e-process is still one and \(K/\alpha_E>1\).

### Results that do not survive

The old uniform fixed-mean departure theorem and
\(\min\{\alpha,\varepsilon_{K,T}\}\) statement require positive relevance to
imply \(|M|>b\). The ramp is positive below \(b\), so those results do not hold
for either \(c=1/2\) or \(c=1\).

The historical latch also means that a past e-process crossing permanently
authorizes later nonzero ramps. The accurate description is:

> current discrepancy scales adaptation after anytime-valid historical
> authorization.

It is not accurate to say that the current e-value itself must remain above
threshold or that the ramp performs a new significance test at every round.

## Numerical and experimental constants that are not hyperparameters

The following values affect implementation auditing or statistical analysis,
not the mathematical policy definition:

- the binary64 ambiguity guard \(5\times10^{-10}\), 100-digit decimal
  fallback, and diagnostic bins \(0,0.25,0.5,0.75,1\) are audit machinery in
  `project_publication/e_value_ce_relevance_looseness_study.py` and
  `layer3_regret_atlas_v1/numerical_guard.py`;
- the practical development margins \(0.0002\) and \(0.0005\), the 400-group
  development bank, and the selection rule were declared in
  `notes/e_value_ce_effect_size_relevance_development_protocol_2026-08-23.md`;
- the 1,600-group validation census, 20,000 bootstrap replicates, seed
  2026082301, and Bonferroni-adjusted one-sided 97.5% limits are recorded in
  `notes/e_value_ce_effect_size_relevance_validation_results_2026-08-23.md`;
  and
- the post hoc \(c=1\) sensitivity report's descriptive normal interval is
  recorded in
  `notes/e_value_ce_full_mean_budget_sensitivity_results_2026-08-24.md`.

These numbers require protocol provenance, but none should be presented as an
e-ATS tuning parameter.

## Freeze status and remaining work before paper adoption

The canonical implementation, separate implementation IDs, closed parameter
surface, exact direct ablation, deterministic equivalence checks, and
machine-readable source freeze are complete. No scientific experiment was run
as part of that work.

Remaining work is:

1. Specify the future benchmark protocol before running it, including the
   environment census, baseline identities, randomness, estimand, aggregation,
   multiplicity rules, and numerical abort conditions.
2. Make that benchmark runner verify the canonical freeze and call the
   canonical functions directly; it must not call the configurable development
   wrapper.
3. Execute the frozen benchmark with full-budget e-ATS, its exact full-budget
   no-\(C^E\) ablation, and the predeclared baselines. The strength of any
   empirical claim must reflect whether the environments and data are genuinely
   outcome-unseen.
4. Update the manuscript only after that benchmark. Replace
   \(\delta=(1/2)(\beta_M/K)6/(\pi^2n^2)\) by the full-budget expression, while
   continuing to label \(\beta_M=0.05\), \(0.995,0.30,0.80\), and \(\rho\)
   according to their actual provenance.
5. Remove the obsolete fixed-\(p\) and \(\min\{\alpha,\varepsilon\}\) claims.
6. Do not describe the method as parameter-free, universally regret-optimal,
   or uniformly dominant over environments.

## Defensible final wording

If the fresh full-budget confirmation succeeds, the paper may say:

> We use the complete declared mean-normalization budget, eliminating an
> obsolete half-budget multiplier inherited from a removed dispersion channel.
> The restart construction and several controller coordinates were selected or
> inherited in earlier frozen studies, whose provenance is reported explicitly;
> the anytime authorization guarantee is controller-agnostic, while regret
> performance is empirical over the declared environment distribution.

The paper should not say that \(c=1\) is regret-optimal or that every remaining
constant is mathematically derived. Both claims would be contradicted by this
ledger.
