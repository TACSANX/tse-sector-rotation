# Prospective hypotheses — `topix17_1306_core_tilt_v1`

Frozen: **2026-08-21**. These evaluation rules are registered before the first eligible prospective month-end signal.

## Primary purpose

The model is treated as a **risk-adjusted overlay**, not as historically proven alpha. Its intended behavior is to preserve most broad-TOPIX participation while reducing damage in adverse equity months.

## Primary prospective hypotheses

1. **Down-month protection** — During months in which the 100% 1306 comparator has a negative return, the candidate's average return difference versus 1306 should be positive.
2. **Drawdown protection** — Over the same prospective observation window, the candidate's maximum drawdown should be shallower than the 100% 1306 comparator.

Both conditions must be reported; one cannot be substituted for the other after observing results.

## Secondary hypotheses

3. **Sector-selection increment** — Candidate return minus the cash-matched sector-equal-weight control measures the incremental effect of sector identity/ranking while preserving the exact candidate CASH schedule. A positive cumulative/average difference supports retaining the ranking layer.
4. **Opportunity cost** — Candidate minus 1306 during 1306-positive months measures the historical/potential insurance premium. This must be reported alongside down-month protection.
5. **All-month return** — CAGR/total return versus 100% 1306 is descriptive. Historical research does not justify claiming a reliably positive all-month alpha.

## Formal review gate

Do not alter `topix17_1306_core_tilt_v1` merely because of short-term underperformance or outperformance.

The first formal model review requires **all** of the following:

- at least 36 completed prospective signal months;
- at least 12 months in which the 1306 comparator is negative; and
- at least one prospective 1306 peak-to-trough drawdown of 10% or more.

If these conditions are not met after 36 months, tracking continues unchanged until they are met.

## Model-change rule

Any change to universe, weights, lookbacks, ranking formula, absolute filter, failed-slot treatment, data source, or rebalance/execution timing creates a **new model ID/version**. The v1 signal and performance history remain immutable and continue to be available as the control for the new version.

## Historical context only

Phase 18 historical diagnostics suggested an insurance-like profile: better relative returns in 1306 down/tail months and some opportunity cost in up months. Those results were observed on data used after substantial research and are not the prospective confirmation of these hypotheses.
