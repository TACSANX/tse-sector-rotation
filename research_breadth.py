#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 9: sector-selection breadth for the frozen TOPIX-17 family.

Phase 8 showed that sector selection contributes return but slightly worsens
max drawdown versus a same-CASH equal-weight control. This study tests one
structural remedy only: diversify the tilt from Top3 to Top5/Top7.

Frozen dimensions:
- classic 6m + 12-1m relative momentum
- MA200 + positive 12m absolute filter; failed slots become CASH
- monthly signal
- core shares 50% and 75%
- Top3 / Top5 / Top7 equal-slot tilt
- 10 / 20 bps
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
from research_attribution import cash_matched_equal_weight, rolling_three_year_relative
from research_portfolio import simulate_self_financing, equal_weight_signal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_breadth"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    close = close.dropna(how="any").sort_index()
    dr = close.pct_change(fill_method=None)
    base = build_factors(close)
    model = signal_models(close, base)["classic_6_12minus1"]
    mask = absolute_masks(close)["ma200_r12"]
    dates = rebalance_dates(close.index, "monthly", args.signal_start)
    core = equal_weight_signal(dates, TICKERS)

    rows = []
    equity = {}
    for cost in [10.0, 20.0]:
        r_ew, _, _ = simulate_self_financing(close, dr, core, TICKERS, cost)
        p_ew = perf(r_ew)
        h_ew = sliced(r_ew, start=args.holdout_start)
        if cost == 10.0:
            equity["full_equal_weight_10bp"] = (1.0 + r_ew).cumprod()

        for top_k in [3, 5, 7]:
            tilt = choose(model, mask, dates, top_k=top_k)
            for core_share in [0.50, 0.75]:
                target = blend_targets(core_share, core, tilt)
                control = cash_matched_equal_weight(target)
                r, w, turnover = simulate_self_financing(close, dr, target, TICKERS, cost)
                rc, wc, tc = simulate_self_financing(close, dr, control, TICKERS, cost)

                full = perf(r); hold = sliced(r, start=args.holdout_start); pre = sliced(r, end=args.holdout_start)
                cf = perf(rc); ch = sliced(rc, start=args.holdout_start)
                eras = era_metrics(r)
                roll = rolling_three_year_relative(r, rc)
                name = f"top{top_k}_core{int(core_share*100):03d}_{cost:g}bp"
                rows.append({
                    "variant": name,
                    "top_k": top_k,
                    "core_share": core_share,
                    "cost_bps": cost,
                    **{f"full_{k}": v for k, v in full.items()},
                    **{f"pre2018_{k}": v for k, v in pre.items()},
                    **{f"holdout_{k}": v for k, v in hold.items()},
                    **eras,
                    "cash_exposure_mean_weight": float(w["CASH"].mean()),
                    "annualized_turnover": float(turnover.mean() * 252),
                    "cashmatched_full_cagr": cf["cagr"],
                    "cashmatched_holdout_cagr": ch["cagr"],
                    "cashmatched_full_mdd": cf["max_drawdown"],
                    "cashmatched_holdout_mdd": ch["max_drawdown"],
                    "cashmatched_full_sharpe": cf["sharpe"],
                    "cashmatched_holdout_sharpe": ch["sharpe"],
                    "selection_full_cagr_contribution": full["cagr"] - cf["cagr"],
                    "selection_holdout_cagr_contribution": hold["cagr"] - ch["cagr"],
                    "selection_full_mdd_contribution": full["max_drawdown"] - cf["max_drawdown"],
                    "selection_holdout_mdd_contribution": hold["max_drawdown"] - ch["max_drawdown"],
                    "selection_full_sharpe_contribution": full["sharpe"] - cf["sharpe"],
                    "selection_holdout_sharpe_contribution": hold["sharpe"] - ch["sharpe"],
                    "full_cagr_alpha_vs_ew": full["cagr"] - p_ew["cagr"],
                    "holdout_cagr_alpha_vs_ew": hold["cagr"] - h_ew["cagr"],
                    "full_mdd_improvement_vs_ew": full["max_drawdown"] - p_ew["max_drawdown"],
                    "holdout_mdd_improvement_vs_ew": hold["max_drawdown"] - h_ew["max_drawdown"],
                    **roll,
                })
                if cost == 10.0:
                    equity[name] = (1.0 + r).cumprod()
                    equity[f"cashmatched_{name}"] = (1.0 + rc).cumprod()

    result = pd.DataFrame(rows)
    era_cagr = ["2009_2013_cagr", "2014_2017_cagr", "2018_2021_cagr", "2022_2026_cagr"]
    result["positive_era_count"] = (result[era_cagr] > 0).sum(axis=1)
    result["worst_era_cagr"] = result[era_cagr].min(axis=1)
    result = result.sort_values(["cost_bps", "core_share", "top_k"])
    result.to_csv(args.out_dir / "grid_results.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    ten = result[result["cost_bps"] == 10.0].copy()
    summary = (
        ten.groupby("top_k", as_index=False)
        .agg(
            median_full_cagr=("full_cagr", "median"),
            median_holdout_cagr=("holdout_cagr", "median"),
            median_full_mdd=("full_max_drawdown", "median"),
            median_full_sharpe=("full_sharpe", "median"),
            median_selection_cagr=("selection_full_cagr_contribution", "median"),
            median_selection_mdd=("selection_full_mdd_contribution", "median"),
            median_selection_sharpe=("selection_full_sharpe_contribution", "median"),
            median_rolling3y_selection_positive=("rolling3y_selection_positive_share", "median"),
            median_turnover=("annualized_turnover", "median"),
            min_positive_era_count=("positive_era_count", "min"),
        )
    )
    summary.to_csv(args.out_dir / "breadth_summary_10bp.csv", index=False)

    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "holdout_cagr", "holdout_max_drawdown", "holdout_sharpe",
        "cash_exposure_mean_weight", "annualized_turnover",
        "selection_full_cagr_contribution", "selection_holdout_cagr_contribution",
        "selection_full_mdd_contribution", "selection_holdout_mdd_contribution",
        "selection_full_sharpe_contribution", "selection_holdout_sharpe_contribution",
        "rolling3y_selection_positive_share", "rolling3y_selection_median_ann", "rolling3y_selection_worst_ann",
        "full_cagr_alpha_vs_ew", "holdout_cagr_alpha_vs_ew",
        "full_mdd_improvement_vs_ew", "holdout_mdd_improvement_vs_ew",
        "positive_era_count", "worst_era_cagr",
    ]
    report = [
        "# TOPIX-17 Phase 9 — Sector Breadth",
        "",
        "Frozen classic momentum + MA200/12m filter; only Top3/Top5/Top7 breadth is varied.",
        "",
        "## Breadth summary at 10bp",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Detailed 10bp results",
        "",
        ten[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision rule",
        "",
        "Prefer broader Top-K only if it preserves the positive sector-selection CAGR contribution while reducing the negative drawdown contribution and turnover. Do not choose based on CAGR alone.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "phase": 9,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "signal_family": "classic 6m + 12-1m; MA200 + positive 12m; monthly",
        "top_k": [3, 5, 7],
        "core_shares": [0.50, 0.75],
        "costs_bps": [10.0, 20.0],
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(ten[cols].to_string(index=False))


if __name__ == "__main__":
    main()
