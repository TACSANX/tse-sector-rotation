# Prospective test — `topix17_1306_core_tilt_v1`

This directory is the audit log for the model frozen on **2026-08-21**.

Primary model:

- 75% NEXT FUNDS TOPIX (1306)
- 25% TOPIX-17 tilt budget
- monthly relative momentum: `0.35 * z(relative 6m) + 0.65 * z(relative 12-1m)`
- rank the 17 sector ETFs and take the Top 3 slots
- each selected slot must be above its 200-observation moving average and have a positive 252-observation return
- a failed slot becomes CASH; it is **not** backfilled with a lower-ranked sector
- signal: last official NAV observation of a completed calendar month
- paper execution: first official NAV observation after the signal date
- research cost assumption: 10 bp per risky side

## Files

- `signals.csv` — immutable prospective signal captures. Existing signal months are never recalculated or overwritten by the tracker.
- `performance.csv` — reconstructed forward returns from the recorded signals only.
- `status.json` — current data-through date, latest signal, and performance summary.

## Comparators

The tracker records the frozen candidate against:

1. 100% 1306
2. 75% 1306 + 25% equal-weight TOPIX-17, no CASH
3. the exact same CASH schedule as the candidate, but equal-weighting the risky sector sleeve

The third comparator isolates the incremental value of sector identity/ranking from the effect of holding CASH.

## Research discipline

Historical research was used to select the frozen model, so historical performance is not treated as confirmatory evidence. Prospective observations after the freeze date are the primary evidence going forward.

Any change to universe, weights, lookbacks, ranking formula, absolute filter, failed-slot treatment, data source, or rebalance timing must create a **new model ID/version**. `config/frozen_model_v1.json` must remain unchanged.
