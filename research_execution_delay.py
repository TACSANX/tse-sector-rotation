#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6: execution-delay sensitivity for the frozen TOPIX-17 candidate.

Unlike Phase 5, the month-end signal is NOT changed. Only execution is delayed.
This separates calendar signal sensitivity from practical implementation delay.

Standard accounting in research_portfolio executes at the first NAV close after
the signal. Here delay=0 means that standard convention; delay=1 means one
additional NAV day later, etc. The target itself remains the original month-end
target, so no future information is introduced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, load_official_prices, build_factors, perf, sliced
from research_dual_momentum import signal_models, absolute_masks, rebalance_dates, choose, era_metrics
from research_core_tilt import blend_targets
from research_portfolio import simulate_self_financing, equal_weight_signal


def delay_signal_index(signal_target: pd.DataFrame, trading_dates: pd.DatetimeIndex, extra_delay: int) -> pd.DataFrame:
    if extra_delay < 0:
        raise ValueError(extra_delay)
    if extra_delay == 0:
        return signal_target.copy()
    rows = []
    idx = []
    for d, row in signal_target.iterrows():
        future = trading_dates[trading_dates > d]
        # Move the artificial signal timestamp to the day immediately before the
        # desired execution close. Row contents still come from original d.
        if len(future) >= extra_delay:
            idx.append(future[extra_delay - 1])
            rows.append(row.values)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=signal_target.columns)


def rolling_three_year_alpha(strategy: pd.Series, benchmark: pd.Series, window: int = 756) -> dict:
    x = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    log_rel = np.log1p(x["s"]) - np.log1p(x["b"])
    ann = np.expm1(log_rel.rolling(window).sum() * (252.0 / window)).dropna()
    if ann.empty:
        return {"rolling3y_positive_alpha_share": np.nan, "rolling3y_median_ann_alpha": np.nan, "rolling3y_worst_ann_alpha": np.nan}
    return {
        "rolling3y_positive_alpha_share": float((ann > 0).mean()),
        "rolling3y_median_ann_alpha": float(ann.median()),
        "rolling3y_worst_ann_alpha": float(ann.min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_execution_delay"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    close = close.dropna(how="any").sort_index()
    dr = close.pct_change(fill_method=None)
    base = build_factors(close)
    model = signal_models(close, base)["classic_6_12minus1"]
    mask = absolute_masks(close)["ma200_r12"]
    dates = rebalance_dates(close.index, "monthly", args.signal_start)

    core0 = equal_weight_signal(dates, TICKERS)
    tilt0 = choose(model, mask, dates, top_k=3)
    core_shares = [0.25, 0.50, 0.75]
    delays = [0, 1, 2, 3, 5]
    costs = [10.0, 20.0]

    rows = []
    equity = {}
    for delay in delays:
        core = delay_signal_index(core0, close.index, delay)
        benchmarks = {}
        for cost in costs:
            rb, wb, tb = simulate_self_financing(close, dr, core, TICKERS, cost)
            benchmarks[cost] = rb
            if cost == 10.0:
                equity[f"benchmark_delay{delay}_10bp"] = (1.0 + rb).cumprod()

        for core_share in core_shares:
            base_target = blend_targets(core_share, core0, tilt0)
            target = delay_signal_index(base_target, close.index, delay)
            for cost in costs:
                r, weights, turnover = simulate_self_financing(close, dr, target, TICKERS, cost)
                b = benchmarks[cost]
                full = perf(r)
                pre = sliced(r, end=args.holdout_start)
                hold = sliced(r, start=args.holdout_start)
                eras = era_metrics(r)
                roll = rolling_three_year_alpha(r, b)
                bf = perf(b)
                bh = sliced(b, start=args.holdout_start)
                name = f"delay{delay}_core{int(core_share*100):03d}_{cost:g}bp"
                rows.append({
                    "variant": name,
                    "extra_execution_delay_nav_days": delay,
                    "core_share": core_share,
                    "cost_bps": cost,
                    **{f"full_{k}": v for k, v in full.items()},
                    **{f"pre2018_{k}": v for k, v in pre.items()},
                    **{f"holdout_{k}": v for k, v in hold.items()},
                    **eras,
                    **roll,
                    "cash_exposure_mean_weight": float(weights["CASH"].mean()),
                    "annualized_turnover": float(turnover.mean() * 252),
                    "full_cagr_alpha_vs_same_delay_ew": float(full["cagr"] - bf["cagr"]),
                    "holdout_cagr_alpha_vs_same_delay_ew": float(hold["cagr"] - bh["cagr"]),
                    "full_mdd_improvement_vs_same_delay_ew": float(full["max_drawdown"] - bf["max_drawdown"]),
                    "holdout_mdd_improvement_vs_same_delay_ew": float(hold["max_drawdown"] - bh["max_drawdown"]),
                })
                if cost == 10.0:
                    equity[name] = (1.0 + r).cumprod()

    result = pd.DataFrame(rows)
    era_cagr = ["2009_2013_cagr", "2014_2017_cagr", "2018_2021_cagr", "2022_2026_cagr"]
    result["positive_era_count"] = (result[era_cagr] > 0).sum(axis=1)
    result["worst_era_cagr"] = result[era_cagr].min(axis=1)
    result = result.sort_values(["cost_bps", "core_share", "extra_execution_delay_nav_days"])
    result.to_csv(args.out_dir / "grid_results.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    summary = (
        result[result["cost_bps"] == 10.0]
        .groupby("core_share", as_index=False)
        .agg(
            min_full_cagr=("full_cagr", "min"),
            max_full_cagr=("full_cagr", "max"),
            median_full_cagr=("full_cagr", "median"),
            min_holdout_cagr=("holdout_cagr", "min"),
            max_holdout_cagr=("holdout_cagr", "max"),
            median_holdout_cagr=("holdout_cagr", "median"),
            worst_full_mdd=("full_max_drawdown", "min"),
            best_full_mdd=("full_max_drawdown", "max"),
            min_rolling3y_positive_alpha_share=("rolling3y_positive_alpha_share", "min"),
            median_rolling3y_positive_alpha_share=("rolling3y_positive_alpha_share", "median"),
            min_positive_era_count=("positive_era_count", "min"),
            max_turnover=("annualized_turnover", "max"),
        )
    )
    summary.to_csv(args.out_dir / "delay_robustness_10bp.csv", index=False)

    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "holdout_cagr", "holdout_max_drawdown", "holdout_sharpe",
        "positive_era_count", "worst_era_cagr",
        "rolling3y_positive_alpha_share", "rolling3y_median_ann_alpha", "rolling3y_worst_ann_alpha",
        "annualized_turnover", "full_cagr_alpha_vs_same_delay_ew", "holdout_cagr_alpha_vs_same_delay_ew",
        "full_mdd_improvement_vs_same_delay_ew", "holdout_mdd_improvement_vs_same_delay_ew",
    ]
    report = [
        "# TOPIX-17 Phase 6 — Execution Delay Sensitivity",
        "",
        "Month-end signal is frozen; only execution is delayed. Official NEXT FUNDS total-return NAV; self-financing accounting.",
        "",
        "## 10bp delay robustness by core share",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Detailed 10bp results",
        "",
        result[result["cost_bps"] == 10.0][cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "If a month-end signal is genuinely implementable, delaying execution by a few NAV days should not destroy its return/drawdown profile. This test does not alter the information set used to choose sectors.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "phase": 6,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "signal_family": "month-end classic 6m + 12-1m; Top3; MA200 + positive 12m",
        "extra_execution_delays_nav_days": delays,
        "core_shares": core_shares,
        "costs_bps": costs,
        "signal_start": args.signal_start,
        "holdout_start": args.holdout_start,
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(result[result["cost_bps"] == 10.0][cols].to_string(index=False))


if __name__ == "__main__":
    main()
