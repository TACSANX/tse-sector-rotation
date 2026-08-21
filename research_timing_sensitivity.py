#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 5: rebalance-timing sensitivity for the frozen TOPIX-17 candidate.

Goal: verify that the core+tilt result is not a month-end artifact.

Frozen signal family:
- classic 6m + 12-1m relative momentum
- Top3
- MA200 + positive 12m absolute filter
- core shares 25/50/75%
- self-financing accounting on official NEXT FUNDS distribution-reinvested NAV

Only the monthly signal observation date is shifted earlier within each month.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, load_official_prices, build_factors, perf, sliced
from research_dual_momentum import signal_models, absolute_masks, choose, era_metrics
from research_core_tilt import blend_targets
from research_portfolio import simulate_self_financing, equal_weight_signal


def monthly_dates_with_offset(index: pd.DatetimeIndex, start: str, offset: int) -> pd.DatetimeIndex:
    idx = index[index >= pd.Timestamp(start)]
    s = pd.Series(idx, index=idx)
    out = []
    for _, g in s.groupby(idx.to_period("M")):
        vals = pd.DatetimeIndex(g.values)
        if len(vals) > offset:
            out.append(vals[-1 - offset])
    return pd.DatetimeIndex(out)


def rolling_three_year_alpha(strategy: pd.Series, benchmark: pd.Series, window: int = 756) -> dict:
    x = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if len(x) < window + 10:
        return {"rolling3y_positive_alpha_share": np.nan, "rolling3y_median_ann_alpha": np.nan, "rolling3y_worst_ann_alpha": np.nan}
    log_rel = np.log1p(x["s"]) - np.log1p(x["b"])
    ann_alpha = np.expm1(log_rel.rolling(window).sum() * (252.0 / window)).dropna()
    return {
        "rolling3y_positive_alpha_share": float((ann_alpha > 0).mean()),
        "rolling3y_median_ann_alpha": float(ann_alpha.median()),
        "rolling3y_worst_ann_alpha": float(ann_alpha.min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_timing_sensitivity"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    close = close.dropna(how="any").sort_index()
    dr = close.pct_change(fill_method=None)
    base = build_factors(close)
    model = signal_models(close, base)["classic_6_12minus1"]
    mask = absolute_masks(close)["ma200_r12"]

    offsets = [0, 3, 5, 10]
    core_shares = [0.25, 0.50, 0.75]
    costs = [10.0, 20.0]
    rows = []
    equity = {}

    for offset in offsets:
        dates = monthly_dates_with_offset(close.index, args.signal_start, offset)
        core = equal_weight_signal(dates, TICKERS)
        tilt = choose(model, mask, dates, top_k=3)
        benchmarks = {}
        for cost in costs:
            rb, wb, tb = simulate_self_financing(close, dr, core, TICKERS, cost)
            benchmarks[cost] = rb
            if cost == 10.0:
                equity[f"benchmark_offset{offset}_10bp"] = (1.0 + rb).cumprod()

        for core_share in core_shares:
            target = blend_targets(core_share, core, tilt)
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
                name = f"offset{offset}_core{int(core_share*100):03d}_{cost:g}bp"
                rows.append({
                    "variant": name,
                    "offset_trade_days_before_month_end": offset,
                    "core_share": core_share,
                    "cost_bps": cost,
                    **{f"full_{k}": v for k, v in full.items()},
                    **{f"pre2018_{k}": v for k, v in pre.items()},
                    **{f"holdout_{k}": v for k, v in hold.items()},
                    **eras,
                    **roll,
                    "cash_exposure_mean_weight": float(weights["CASH"].mean()),
                    "annualized_turnover": float(turnover.mean() * 252),
                    "full_cagr_alpha_vs_same_offset_ew": float(full["cagr"] - bf["cagr"]),
                    "holdout_cagr_alpha_vs_same_offset_ew": float(hold["cagr"] - bh["cagr"]),
                    "full_mdd_improvement_vs_same_offset_ew": float(full["max_drawdown"] - bf["max_drawdown"]),
                    "holdout_mdd_improvement_vs_same_offset_ew": float(hold["max_drawdown"] - bh["max_drawdown"]),
                })
                if cost == 10.0:
                    equity[name] = (1.0 + r).cumprod()

    result = pd.DataFrame(rows)
    era_cagr = ["2009_2013_cagr", "2014_2017_cagr", "2018_2021_cagr", "2022_2026_cagr"]
    result["positive_era_count"] = (result[era_cagr] > 0).sum(axis=1)
    result["worst_era_cagr"] = result[era_cagr].min(axis=1)
    result = result.sort_values(
        ["cost_bps", "core_share", "offset_trade_days_before_month_end"]
    )
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
    summary.to_csv(args.out_dir / "offset_robustness_10bp.csv", index=False)

    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "holdout_cagr", "holdout_max_drawdown", "holdout_sharpe",
        "positive_era_count", "worst_era_cagr",
        "rolling3y_positive_alpha_share", "rolling3y_median_ann_alpha", "rolling3y_worst_ann_alpha",
        "annualized_turnover", "full_cagr_alpha_vs_same_offset_ew", "holdout_cagr_alpha_vs_same_offset_ew",
        "full_mdd_improvement_vs_same_offset_ew", "holdout_mdd_improvement_vs_same_offset_ew",
    ]
    report = [
        "# TOPIX-17 Phase 5 — Rebalance Timing Sensitivity",
        "",
        "Frozen classic Top3 + MA200/12m filter core+tilt family. Official NEXT FUNDS total-return NAV; self-financing accounting.",
        "",
        "## 10bp offset robustness by core share",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Detailed 10bp results",
        "",
        result[result["cost_bps"] == 10.0][cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "A candidate is timing-robust only if shifting the signal several trading days before month-end preserves the broad return/drawdown tradeoff and does not collapse rolling 3-year relative performance.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 5,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "signal_family": "classic 6m + 12-1m; Top3; MA200 + positive 12m",
        "offsets_trade_days_before_month_end": offsets,
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
