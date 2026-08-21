#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.1: corrected self-financing TOPIX-17 dual-momentum research.

Corrections vs earlier exploratory studies:
- ETF weights drift naturally between rebalances; no implicit free daily rebalancing.
- Transaction costs are charged on risky ETF notional actually traded.
- Equal-weight benchmark uses the same rebalance frequency and transaction cost.
- Official NEXT FUNDS distribution-reinvested NAV data only.

Signal date -> next NAV date close execution -> new weights earn returns from the
following NAV date. This is conservative and avoids same-close look-ahead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, load_official_prices, build_factors, perf, sliced
from research_dual_momentum import signal_models, absolute_masks, rebalance_dates, choose, era_metrics
from research_portfolio import simulate_self_financing, equal_weight_signal, buy_and_hold_signal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_dual_v2"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    # Use only common official NAV dates across all 17 ETFs. This prevents a
    # missing fund observation from being silently treated as a zero return.
    close = close.dropna(how="any").sort_index()
    dr = close.pct_change(fill_method=None)
    if dr.iloc[1:].isna().any().any():
        raise RuntimeError("unexpected missing common-date returns")

    base = build_factors(close)
    models = signal_models(close, base)
    masks = absolute_masks(close)

    rebalance_modes = ["monthly", "bimonthly", "quarterly"]
    costs = [0.0, 10.0, 20.0]

    benchmark: dict[tuple[str, float], dict] = {}
    benchmark_returns: dict[tuple[str, float], pd.Series] = {}
    rebalance_date_map: dict[str, pd.DatetimeIndex] = {}

    for mode in rebalance_modes:
        dates = rebalance_dates(close.index, mode, args.signal_start)
        rebalance_date_map[mode] = dates
        ew_target = equal_weight_signal(dates, TICKERS)
        for cost in costs:
            r, w, turn = simulate_self_financing(close, dr, ew_target, TICKERS, cost)
            benchmark_returns[(mode, cost)] = r
            benchmark[(mode, cost)] = {
                "full": perf(r),
                "pre2018": sliced(r, end=args.holdout_start),
                "holdout": sliced(r, start=args.holdout_start),
                "eras": era_metrics(r),
                "annualized_turnover": float(turn.mean() * 252),
            }

    first_dates = rebalance_date_map["monthly"]
    bh_signal = buy_and_hold_signal(first_dates[0], TICKERS)
    bh_returns: dict[float, pd.Series] = {}
    buyhold_meta: dict[str, dict] = {}
    for cost in costs:
        r, w, turn = simulate_self_financing(close, dr, bh_signal, TICKERS, cost)
        bh_returns[cost] = r
        buyhold_meta[str(cost)] = {
            "full": perf(r),
            "pre2018": sliced(r, end=args.holdout_start),
            "holdout": sliced(r, start=args.holdout_start),
            "eras": era_metrics(r),
            "annualized_turnover": float(turn.mean() * 252),
        }

    rows: list[dict] = []
    equity: dict[str, pd.Series] = {}

    for mode in rebalance_modes:
        dates = rebalance_date_map[mode]
        for model_name, model in models.items():
            for top_k in [1, 3]:
                for mask_name, mask in masks.items():
                    target = choose(model, mask, dates, top_k)
                    for cost in costs:
                        r, weights, turnover = simulate_self_financing(close, dr, target, TICKERS, cost)
                        full = perf(r)
                        pre = sliced(r, end=args.holdout_start)
                        hold = sliced(r, start=args.holdout_start)
                        eras = era_metrics(r)
                        b = benchmark[(mode, cost)]
                        name = f"{mode}_{model_name}_top{top_k}_{mask_name}_{cost:g}bp"
                        row = {
                            "variant": name,
                            "rebalance": mode,
                            "model": model_name,
                            "top_k": top_k,
                            "absolute_filter": mask_name,
                            "cost_bps": cost,
                            **{f"full_{k}": v for k, v in full.items()},
                            **{f"pre2018_{k}": v for k, v in pre.items()},
                            **{f"holdout_{k}": v for k, v in hold.items()},
                            **eras,
                            "cash_exposure_mean_weight": float(weights["CASH"].mean()),
                            "annualized_turnover": float(turnover.mean() * 252),
                            "signal_count": int(len(target)),
                            "full_cagr_alpha_vs_same_freq_ew": float(full["cagr"] - b["full"]["cagr"]),
                            "holdout_cagr_alpha_vs_same_freq_ew": float(hold["cagr"] - b["holdout"]["cagr"]),
                            "full_mdd_improvement_vs_same_freq_ew": float(full["max_drawdown"] - b["full"]["max_drawdown"]),
                            "holdout_mdd_improvement_vs_same_freq_ew": float(hold["max_drawdown"] - b["holdout"]["max_drawdown"]),
                        }
                        rows.append(row)
                        if cost == 10.0 and top_k == 3 and mode in ("monthly", "bimonthly"):
                            equity[name] = (1.0 + r).cumprod()

    result = pd.DataFrame(rows)
    era_cagr = ["2009_2013_cagr", "2014_2017_cagr", "2018_2021_cagr", "2022_2026_cagr"]
    era_sharpe = ["2009_2013_sharpe", "2014_2017_sharpe", "2018_2021_sharpe", "2022_2026_sharpe"]
    result["positive_era_count"] = (result[era_cagr] > 0).sum(axis=1)
    result["worst_era_cagr"] = result[era_cagr].min(axis=1)
    result["median_era_sharpe"] = result[era_sharpe].median(axis=1)

    result = result.sort_values(
        ["positive_era_count", "median_era_sharpe", "holdout_sharpe", "annualized_turnover"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    result.to_csv(args.out_dir / "grid_results.csv", index=False)

    shortlist = result[
        (result["positive_era_count"] == 4)
        & (result["full_max_drawdown"] > -0.40)
        & (result["holdout_max_drawdown"] > -0.40)
        & (result["pre2018_cagr"] > 0)
        & (result["holdout_cagr"] > 0)
    ].copy()
    shortlist.head(50).to_csv(args.out_dir / "shortlist.csv", index=False)

    # Add executable benchmark equity curves at 10bp for visual/audit comparison.
    for mode in ("monthly", "bimonthly"):
        equity[f"benchmark_{mode}_equal_weight_10bp"] = (1.0 + benchmark_returns[(mode, 10.0)]).cumprod()
    equity["benchmark_buyhold_equal_weight_10bp"] = (1.0 + bh_returns[10.0]).cumprod()
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    serializable_benchmark = {
        f"{mode}_{cost:g}bp": data for (mode, cost), data in benchmark.items()
    }
    meta = {
        "phase": "2.1 corrected self-financing",
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "common_price_start": close.index.min().date().isoformat(),
        "common_price_end": close.index.max().date().isoformat(),
        "common_rows": int(len(close)),
        "signal_start": args.signal_start,
        "holdout_start": args.holdout_start,
        "timing": "signal at close; execute next NAV close; target active following NAV date",
        "cost_method": "bps charged on absolute risky ETF notional traded; cash leg free",
        "source_metadata": source_meta,
        "equal_weight_benchmarks": serializable_benchmark,
        "buy_and_hold_equal_weight": buyhold_meta,
        "variants": int(len(result)),
        "shortlist_count": int(len(shortlist)),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "holdout_cagr", "holdout_max_drawdown", "holdout_sharpe",
        "positive_era_count", "worst_era_cagr", "median_era_sharpe",
        "cash_exposure_mean_weight", "annualized_turnover",
        "full_cagr_alpha_vs_same_freq_ew", "holdout_cagr_alpha_vs_same_freq_ew",
        "full_mdd_improvement_vs_same_freq_ew", "holdout_mdd_improvement_vs_same_freq_ew",
    ]
    best = result.head(35)
    bm10 = benchmark[("monthly", 10.0)]
    bh10 = buyhold_meta["10.0"]
    report = [
        "# TOPIX-17 Phase 2.1 — Corrected Self-Financing Research",
        "",
        "## Correction",
        "",
        "Portfolio weights now drift naturally between rebalances. Transaction costs are charged on ETF notional actually traded. Comparison benchmarks use the same cost and rebalance cadence.",
        "",
        "## 10bp monthly equal-weight benchmark",
        "",
        f"- CAGR: {bm10['full']['cagr']:.2%}",
        f"- Max drawdown: {bm10['full']['max_drawdown']:.2%}",
        f"- Sharpe: {bm10['full']['sharpe']:.3f}",
        f"- Holdout CAGR: {bm10['holdout']['cagr']:.2%}",
        f"- Holdout max drawdown: {bm10['holdout']['max_drawdown']:.2%}",
        "",
        "## 10bp equal-weight buy-and-hold benchmark",
        "",
        f"- CAGR: {bh10['full']['cagr']:.2%}",
        f"- Max drawdown: {bh10['full']['max_drawdown']:.2%}",
        "",
        "## Robustness-ranked variants",
        "",
        best[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Discipline",
        "",
        "Phase 2 hypotheses were informed by earlier results. A candidate selected here must be frozen and prospectively evaluated; repeated tuning against 2008-present data would invalidate an out-of-sample claim.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(best[cols].to_string(index=False))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
