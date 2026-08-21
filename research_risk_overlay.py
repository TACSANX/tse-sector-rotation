#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4: risk overlays on the frozen TOPIX-17 core+tilt candidate family.

Frozen base signal
------------------
- relative momentum: classic 6m + 12-1m
- selected sectors: Top 3
- individual absolute filter: above 200d MA AND 12m return > 0
- rebalance: monthly
- core shares: 25%, 50%, 75% only

Overlays are deliberately simple and long-only. No leverage is permitted.
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


def overlay_scale(base: dict, kind: str) -> pd.Series:
    bench_ret: pd.Series = base["bench_ret"]
    bench: pd.Series = base["bench"]
    vol63 = bench_ret.rolling(63).std(ddof=0) * np.sqrt(252.0)
    trend200 = bench / bench.rolling(200).mean() - 1.0

    one = pd.Series(1.0, index=bench.index)
    vol12 = (0.12 / vol63.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    vol15 = (0.15 / vol63.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    trend50 = pd.Series(np.where(trend200 < 0.0, 0.50, 1.0), index=bench.index)

    if kind == "none":
        return one
    if kind == "vol12":
        return vol12
    if kind == "vol15":
        return vol15
    if kind == "trend200_half":
        return trend50
    if kind == "vol15_trend200_half":
        # Both mechanisms can reduce exposure; never increase it.
        return (vol15 * trend50).clip(0.0, 1.0)
    raise ValueError(kind)


def apply_overlay(target: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    x = target.copy()
    s = scale.reindex(x.index).fillna(1.0).clip(0.0, 1.0)
    x.loc[:, TICKERS] = x[TICKERS].mul(s, axis=0)
    x["CASH"] = 1.0 - x[TICKERS].sum(axis=1)
    if (x < -1e-10).any().any():
        raise RuntimeError("negative target after risk overlay")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_risk_overlay"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    close = close.dropna(how="any").sort_index()
    dr = close.pct_change(fill_method=None)
    base = build_factors(close)
    models = signal_models(close, base)
    masks = absolute_masks(close)

    dates = rebalance_dates(close.index, "monthly", args.signal_start)
    core = equal_weight_signal(dates, TICKERS)
    tilt = choose(models["classic_6_12minus1"], masks["ma200_r12"], dates, top_k=3)

    overlays = ["none", "vol12", "vol15", "trend200_half", "vol15_trend200_half"]
    core_shares = [0.25, 0.50, 0.75]
    costs = [0.0, 10.0, 20.0]

    # Same-frequency broad benchmark.
    benchmarks: dict[float, dict] = {}
    benchmark_returns: dict[float, pd.Series] = {}
    for cost in costs:
        r, w, turn = simulate_self_financing(close, dr, core, TICKERS, cost)
        benchmark_returns[cost] = r
        benchmarks[cost] = {
            "full": perf(r),
            "pre2018": sliced(r, end=args.holdout_start),
            "holdout": sliced(r, start=args.holdout_start),
            "eras": era_metrics(r),
            "annualized_turnover": float(turn.mean() * 252),
        }

    rows: list[dict] = []
    equity: dict[str, pd.Series] = {
        "benchmark_monthly_ew_10bp": (1.0 + benchmark_returns[10.0]).cumprod()
    }

    for core_share in core_shares:
        blended = blend_targets(core_share, core, tilt)
        for ov in overlays:
            target = apply_overlay(blended, overlay_scale(base, ov))
            for cost in costs:
                r, weights, turnover = simulate_self_financing(close, dr, target, TICKERS, cost)
                full = perf(r)
                pre = sliced(r, end=args.holdout_start)
                hold = sliced(r, start=args.holdout_start)
                eras = era_metrics(r)
                b = benchmarks[cost]
                name = f"core{int(core_share*100):03d}_{ov}_{cost:g}bp"
                rows.append({
                    "variant": name,
                    "core_share": core_share,
                    "tilt_share": 1.0 - core_share,
                    "overlay": ov,
                    "cost_bps": cost,
                    **{f"full_{k}": v for k, v in full.items()},
                    **{f"pre2018_{k}": v for k, v in pre.items()},
                    **{f"holdout_{k}": v for k, v in hold.items()},
                    **eras,
                    "cash_exposure_mean_weight": float(weights["CASH"].mean()),
                    "annualized_turnover": float(turnover.mean() * 252),
                    "full_cagr_alpha_vs_ew": float(full["cagr"] - b["full"]["cagr"]),
                    "holdout_cagr_alpha_vs_ew": float(hold["cagr"] - b["holdout"]["cagr"]),
                    "full_mdd_improvement_vs_ew": float(full["max_drawdown"] - b["full"]["max_drawdown"]),
                    "holdout_mdd_improvement_vs_ew": float(hold["max_drawdown"] - b["holdout"]["max_drawdown"]),
                })
                if cost == 10.0:
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
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    robust10 = result[
        (result["cost_bps"] == 10.0)
        & (result["positive_era_count"] == 4)
        & (result["pre2018_cagr"] > 0)
        & (result["holdout_cagr"] > 0)
    ].copy()
    robust10.to_csv(args.out_dir / "shortlist_10bp.csv", index=False)

    # Cost robustness for each frozen parameter combination.
    stability = (
        result.groupby(["core_share", "overlay"], as_index=False)
        .agg(
            min_full_cagr=("full_cagr", "min"),
            min_holdout_cagr=("holdout_cagr", "min"),
            worst_full_mdd=("full_max_drawdown", "min"),
            worst_holdout_mdd=("holdout_max_drawdown", "min"),
            median_full_sharpe=("full_sharpe", "median"),
            median_holdout_sharpe=("holdout_sharpe", "median"),
            max_turnover=("annualized_turnover", "max"),
            min_positive_era_count=("positive_era_count", "min"),
        )
    )
    stability.to_csv(args.out_dir / "cost_stability.csv", index=False)

    bm10 = benchmarks[10.0]
    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "holdout_cagr", "holdout_max_drawdown", "holdout_sharpe",
        "positive_era_count", "worst_era_cagr", "median_era_sharpe",
        "cash_exposure_mean_weight", "annualized_turnover",
        "full_cagr_alpha_vs_ew", "holdout_cagr_alpha_vs_ew",
        "full_mdd_improvement_vs_ew", "holdout_mdd_improvement_vs_ew",
    ]
    report = [
        "# TOPIX-17 Phase 4 — Risk Overlay",
        "",
        "Frozen Phase-3 signal family; official NEXT FUNDS distribution-reinvested NAV; self-financing accounting.",
        "",
        "## 10bp monthly equal-weight benchmark",
        "",
        f"- CAGR: {bm10['full']['cagr']:.2%}",
        f"- Max drawdown: {bm10['full']['max_drawdown']:.2%}",
        f"- Sharpe: {bm10['full']['sharpe']:.3f}",
        f"- Holdout CAGR: {bm10['holdout']['cagr']:.2%}",
        f"- Holdout max drawdown: {bm10['holdout']['max_drawdown']:.2%}",
        "",
        "## 10bp robust variants",
        "",
        robust10.head(30)[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Cost stability (0/10/20bp)",
        "",
        stability.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Selection rule",
        "",
        "A risk overlay is useful only if it improves drawdown/Sharpe across adjacent core shares without destroying pre-2018 or holdout CAGR. No leverage is allowed.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 4,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "signal_family": "classic 6m + 12-1m relative momentum; Top3; MA200 and 12m absolute filter",
        "core_shares": core_shares,
        "overlays": overlays,
        "costs_bps": costs,
        "no_leverage": True,
        "signal_start": args.signal_start,
        "holdout_start": args.holdout_start,
        "source_metadata": source_meta,
        "timing": "signal close -> execute next NAV close -> earn from following NAV date",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(robust10.head(30)[cols].to_string(index=False))
    print("\nCOST STABILITY\n", stability.to_string(index=False))


if __name__ == "__main__":
    main()
