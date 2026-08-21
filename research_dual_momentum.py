#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2: TOPIX-17 dual-momentum research on official NEXT FUNDS history.

This phase is deliberately simpler than the production screener. It asks:
1. Is medium-term relative momentum more stable than the current 1/3/6m blend?
2. Does applying absolute trend filters ETF-by-ETF improve drawdown without
   sacrificing too much return?
3. Does slower rebalancing materially reduce turnover/cost drag?

The data loader and performance functions are reused from research_backtest_official.py.
This is exploratory research; Phase-2 choices were informed by Phase-1 results, so
its results are not a pristine untouched holdout. Any final candidate should be
forward/paper-tested after selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from research_backtest_official import (
    TICKERS,
    load_official_prices,
    build_factors,
    cs_z,
    perf,
    sliced,
)


def rebalance_dates(index: pd.DatetimeIndex, mode: str, start: str) -> pd.DatetimeIndex:
    idx = index[index >= pd.Timestamp(start)]
    holder = pd.Series(idx, index=idx)
    if mode == "monthly":
        key = idx.to_period("M")
    elif mode == "bimonthly":
        serial = idx.year * 12 + idx.month
        key = pd.Index(serial // 2)
    elif mode == "quarterly":
        key = idx.to_period("Q")
    else:
        raise ValueError(mode)
    return pd.DatetimeIndex(holder.groupby(key).last().values)


def signal_models(close: pd.DataFrame, base: dict) -> dict[str, pd.DataFrame]:
    bench: pd.Series = base["bench"]
    r3 = close.pct_change(63)
    r6 = close.pct_change(126)
    r12 = close.pct_change(252)
    # Classic 12-1 momentum: exclude the most recent month.
    r12_1 = close.shift(21) / close.shift(252) - 1.0

    b3 = bench.pct_change(63)
    b6 = bench.pct_change(126)
    b12 = bench.pct_change(252)
    b12_1 = bench.shift(21) / bench.shift(252) - 1.0

    rs3 = r3.sub(b3, axis=0)
    rs6 = r6.sub(b6, axis=0)
    rs12 = r12.sub(b12, axis=0)
    rs12_1 = r12_1.sub(b12_1, axis=0)

    # Current production-style price score from Phase 1.
    current = base["score"]

    # Medium horizon: explicitly de-emphasizes 1-month noise.
    medium = 0.30 * cs_z(rs3) + 0.40 * cs_z(rs6) + 0.30 * cs_z(rs12)

    # Classic momentum family. A small 6m component reduces dependence on a
    # single lookback while preserving the 12-1 literature anchor.
    classic = 0.35 * cs_z(rs6) + 0.65 * cs_z(rs12_1)

    # Simple long-horizon relative momentum; intentionally parsimonious.
    long = 0.50 * cs_z(rs6) + 0.50 * cs_z(rs12)

    return {
        "current": current,
        "medium_3_6_12": medium,
        "classic_6_12minus1": classic,
        "long_6_12": long,
    }


def absolute_masks(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ma200 = close / close.rolling(200).mean() - 1.0
    r6 = close.pct_change(126)
    r12 = close.pct_change(252)
    return {
        "none": pd.DataFrame(True, index=close.index, columns=close.columns),
        "ma200": ma200 > 0.0,
        "ma200_r6": (ma200 > 0.0) & (r6 > 0.0),
        "ma200_r12": (ma200 > 0.0) & (r12 > 0.0),
    }


def choose(
    model: pd.DataFrame,
    mask: pd.DataFrame,
    dates: pd.DatetimeIndex,
    top_k: int,
) -> pd.DataFrame:
    """Rank by relative momentum, then gate each selected ETF individually.

    Each top-k slot has equal notional weight. Failed absolute filters become
    CASH rather than being replaced by a lower-ranked sector. This prevents the
    absolute filter from quietly turning into another relative-ranking rule.
    """
    target = pd.DataFrame(0.0, index=dates, columns=TICKERS + ["CASH"])
    for d in dates:
        ranking = model.loc[d].dropna().sort_values(ascending=False)
        selected = ranking.head(top_k).index.tolist()
        if not selected:
            target.loc[d, "CASH"] = 1.0
            continue
        slot = 1.0 / top_k
        cash = 0.0
        for ticker in selected:
            if bool(mask.loc[d, ticker]):
                target.loc[d, ticker] += slot
            else:
                cash += slot
        # If warm-up leaves fewer than top_k valid names, keep missing slots in cash.
        cash += max(top_k - len(selected), 0) * slot
        target.loc[d, "CASH"] = cash
    return target


def simulate(close: pd.DataFrame, dr: pd.DataFrame, signal_target: pd.DataFrame, cost_bps: float):
    all_dates = close.index
    scheduled: Dict[pd.Timestamp, pd.Series] = {}
    for d, row in signal_target.iterrows():
        future = all_dates[all_dates > d]
        if len(future):
            scheduled[future[0]] = row
    if not scheduled:
        raise RuntimeError("no executable signals")
    start = min(scheduled)
    target = pd.DataFrame(np.nan, index=all_dates[all_dates >= start], columns=signal_target.columns)
    for d, row in scheduled.items():
        target.loc[d] = row
    target = target.ffill().fillna(0.0)
    held = target.shift(1).fillna(0.0)
    rets = dr.reindex(target.index).reindex(columns=TICKERS).fillna(0.0)
    gross = (held[TICKERS] * rets).sum(axis=1)
    prev = target.shift(1).fillna(0.0)
    turnover = (target - prev).abs().sum(axis=1)
    net = gross - turnover * (cost_bps / 10000.0)
    return net, target, turnover


def era_metrics(r: pd.Series) -> dict:
    eras = {
        "2009_2013": ("2009-04-01", "2014-01-01"),
        "2014_2017": ("2014-01-01", "2018-01-01"),
        "2018_2021": ("2018-01-01", "2022-01-01"),
        "2022_2026": ("2022-01-01", None),
    }
    out = {}
    for name, (start, end) in eras.items():
        x = r[r.index >= pd.Timestamp(start)]
        if end:
            x = x[x.index < pd.Timestamp(end)]
        p = perf(x)
        out[f"{name}_cagr"] = p["cagr"]
        out[f"{name}_sharpe"] = p["sharpe"]
        out[f"{name}_max_drawdown"] = p["max_drawdown"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_dual"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    base = build_factors(close)
    models = signal_models(close, base)
    masks = absolute_masks(close)

    rows: list[dict] = []
    equities: dict[str, pd.Series] = {}
    for rebalance in ["monthly", "bimonthly", "quarterly"]:
        dates = rebalance_dates(close.index, rebalance, args.signal_start)
        for model_name, model in models.items():
            for top_k in [1, 3]:
                for mask_name, mask in masks.items():
                    signal_target = choose(model, mask, dates, top_k)
                    for cost_bps in [0.0, 10.0, 20.0]:
                        r, target, turnover = simulate(close, base["dr"], signal_target, cost_bps)
                        full = perf(r)
                        pre = sliced(r, end=args.holdout_start)
                        hold = sliced(r, start=args.holdout_start)
                        eras = era_metrics(r)
                        name = f"{rebalance}_{model_name}_top{top_k}_{mask_name}_{cost_bps:g}bp"
                        row = {
                            "variant": name,
                            "rebalance": rebalance,
                            "model": model_name,
                            "top_k": top_k,
                            "absolute_filter": mask_name,
                            "cost_bps": cost_bps,
                            **{f"full_{k}": v for k, v in full.items()},
                            **{f"pre2018_{k}": v for k, v in pre.items()},
                            **{f"holdout_{k}": v for k, v in hold.items()},
                            **eras,
                            "cash_exposure": float(target["CASH"].mean()),
                            "annualized_turnover": float(turnover.mean() * 252),
                            "signal_count": int(len(signal_target)),
                        }
                        rows.append(row)
                        if cost_bps == 10.0 and top_k == 3 and rebalance == "monthly":
                            equities[name] = (1.0 + r).cumprod()

    result = pd.DataFrame(rows)
    era_cagr_cols = ["2009_2013_cagr", "2014_2017_cagr", "2018_2021_cagr", "2022_2026_cagr"]
    era_sharpe_cols = ["2009_2013_sharpe", "2014_2017_sharpe", "2018_2021_sharpe", "2022_2026_sharpe"]
    result["positive_era_count"] = (result[era_cagr_cols] > 0).sum(axis=1)
    result["worst_era_cagr"] = result[era_cagr_cols].min(axis=1)
    result["median_era_sharpe"] = result[era_sharpe_cols].median(axis=1)

    benchmark = base["bench_ret"]
    benchmark = benchmark[benchmark.index >= pd.Timestamp(args.signal_start)]
    bench_full = perf(benchmark)
    bench_hold = sliced(benchmark, start=args.holdout_start)
    bench_eras = era_metrics(benchmark)
    result["holdout_cagr_alpha_vs_equal_weight"] = result["holdout_cagr"] - bench_hold["cagr"]
    result["holdout_mdd_improvement_vs_equal_weight"] = result["holdout_max_drawdown"] - bench_hold["max_drawdown"]

    # Robustness first: all four eras positive, then median era Sharpe, holdout Sharpe,
    # and lower turnover. This is not an optimizer objective used for trading.
    result = result.sort_values(
        ["positive_era_count", "median_era_sharpe", "holdout_sharpe", "annualized_turnover"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    result.to_csv(args.out_dir / "grid_results.csv", index=False)

    robust = result[
        (result["positive_era_count"] == 4)
        & (result["holdout_max_drawdown"] > -0.40)
        & (result["pre2018_cagr"] > 0)
        & (result["holdout_cagr"] > 0)
    ].copy()
    robust.head(40).to_csv(args.out_dir / "shortlist.csv", index=False)
    pd.DataFrame(equities).to_csv(args.out_dir / "monthly_top3_10bp_equity.csv")

    meta = {
        "phase": 2,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "note": "Phase-2 hypotheses were informed by Phase-1 results; not a pristine untouched holdout.",
        "signal_start": args.signal_start,
        "holdout_start": args.holdout_start,
        "source_metadata": source_meta,
        "benchmark": {
            "full": bench_full,
            "holdout": bench_hold,
            "eras": bench_eras,
        },
        "variants": int(len(result)),
        "shortlist_count": int(len(robust)),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "holdout_cagr", "holdout_max_drawdown", "holdout_sharpe",
        "positive_era_count", "worst_era_cagr", "median_era_sharpe",
        "cash_exposure", "annualized_turnover",
        "holdout_cagr_alpha_vs_equal_weight", "holdout_mdd_improvement_vs_equal_weight",
    ]
    best = result.head(30)
    report = [
        "# TOPIX-17 Phase 2 — Dual Momentum",
        "",
        "Official NEXT FUNDS distribution-reinvested NAV data.",
        "",
        "## Benchmark",
        "",
        f"- Full CAGR: {bench_full['cagr']:.2%}",
        f"- Full max drawdown: {bench_full['max_drawdown']:.2%}",
        f"- Holdout CAGR: {bench_hold['cagr']:.2%}",
        f"- Holdout max drawdown: {bench_hold['max_drawdown']:.2%}",
        "",
        "## Robustness-ranked variants",
        "",
        best[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Research discipline",
        "",
        "Phase 2 is exploratory because its hypotheses were chosen after seeing Phase 1. "
        "A final model should therefore be frozen and evaluated prospectively/paper-traded, "
        "rather than repeatedly tuned against the same 2008-present history.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(best[cols].to_string(index=False))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
