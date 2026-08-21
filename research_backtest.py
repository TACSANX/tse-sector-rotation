#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust execution-layer research for TOPIX-17 sector rotation.

Purpose
-------
Validate whether simple sector rotation has persistent value before adding the
33-industry, macro and fundamental layers.

Design
------
- Universe: 1617.T ... 1633.T only.
- Data: Yahoo Finance adjusted prices via yfinance.
- Known official split repair: 1629.T, 1 -> 500, effective 2026-04-01.
- Signals use only information available at the rebalance close.
- Target portfolio is set on the next trading day; returns use lagged target
  weights, which is deliberately conservative and avoids close-to-close
  look-ahead.
- Grid: monthly/weekly x Top1/Top3 x cash rule x transaction costs.
- Report full sample plus pre-2018 and 2018+ holdout metrics.

Official split source:
https://nextfunds.jp/news/2026/pd_260330a.html

This is research code, not an execution system.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from screener import TOPIX17_ETFS

TICKERS = list(TOPIX17_ETFS)
SPLITS = {
    "1629.T": [(pd.Timestamp("2026-04-01"), 500.0)],
}


def cs_z(x: pd.DataFrame, clip: float = 2.5) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip).fillna(0.0)


def z100(z: pd.DataFrame) -> pd.DataFrame:
    return 100.0 / (1.0 + np.exp(-1.15 * z))


def directional(x: pd.DataFrame | pd.Series, scale: float):
    return (50.0 + 50.0 * np.tanh(x * scale)).clip(0.0, 100.0)


def repair_known_splits(close: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Repair only when the downloaded adjusted series visibly contains the split jump.

    This intentionally does not blindly transform history. If the vendor fixes its
    own historical adjustment later, the repair becomes a no-op.
    """
    out = close.copy()
    repairs: list[dict] = []
    for ticker, events in SPLITS.items():
        if ticker not in out:
            continue
        for effective, factor in events:
            pre = out.loc[out.index < effective, ticker].dropna()
            post = out.loc[out.index >= effective, ticker].dropna()
            if pre.empty or post.empty:
                continue
            before = float(pre.iloc[-1])
            after = float(post.iloc[0])
            if before <= 0 or after <= 0:
                continue
            ratio = after / before
            action = "none"
            # Double-adjusted pre-split history: after/pre ~= factor.
            if 0.70 * factor <= ratio <= 1.30 * factor:
                out.loc[out.index < effective, ticker] *= factor
                action = f"multiply_pre_by_{factor:g}"
            # Unadjusted pre-split history: after/pre ~= 1/factor.
            elif 0.70 / factor <= ratio <= 1.30 / factor:
                out.loc[out.index < effective, ticker] /= factor
                action = f"divide_pre_by_{factor:g}"
            repairs.append({
                "ticker": ticker,
                "effective_date": effective.date().isoformat(),
                "factor": factor,
                "pre_close": before,
                "post_close": after,
                "post_pre_ratio": ratio,
                "action": action,
            })
    return out, repairs


def load_prices(start: str) -> tuple[pd.DataFrame, list[dict], dict]:
    raw = yf.download(
        TICKERS,
        start=start,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=False,
        timeout=60,
    )
    if raw.empty or "Close" not in raw:
        raise RuntimeError("Yahoo Finance returned no usable Close data")
    close = raw["Close"].reindex(columns=TICKERS).dropna(how="all").sort_index()
    close, repairs = repair_known_splits(close)

    ret = close.pct_change(fill_method=None)
    max_abs = ret.abs().max()
    bad = max_abs[max_abs > 0.50]
    if len(bad):
        raise RuntimeError("suspect ETF price jumps remain after repair: " + bad.to_json())

    quality = {
        "start": close.index.min().date().isoformat(),
        "end": close.index.max().date().isoformat(),
        "rows": int(len(close)),
        "max_abs_daily_return_by_ticker": {k: float(v) for k, v in max_abs.items()},
        "missing_share_by_ticker": {k: float(v) for k, v in close.isna().mean().items()},
    }
    return close, repairs, quality


def build_factors(close: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    # Do not forward-fill before returns. Missing observations should remain missing.
    dr = close.pct_change(fill_method=None)
    bench_ret = dr.mean(axis=1, skipna=True).fillna(0.0)
    bench = (1.0 + bench_ret).cumprod()

    r1 = close.pct_change(21)
    r3 = close.pct_change(63)
    r6 = close.pct_change(126)
    b1 = bench.pct_change(21)
    b3 = bench.pct_change(63)
    b6 = bench.pct_change(126)
    rs1 = r1.sub(b1, axis=0)
    rs3 = r3.sub(b3, axis=0)
    rs6 = r6.sub(b6, axis=0)
    ma50 = close / close.rolling(50).mean() - 1.0
    ma200 = close / close.rolling(200).mean() - 1.0
    vol20 = dr.rolling(20).std(ddof=0) * np.sqrt(252)
    peak126 = close.rolling(126, min_periods=63).max()
    dd = close / peak126 - 1.0
    maxdd126 = dd.rolling(126, min_periods=63).min()

    technical = z100(
        0.18 * cs_z(r1)
        + 0.27 * cs_z(r3)
        + 0.15 * cs_z(r6)
        + 0.15 * cs_z(rs3)
        + 0.10 * cs_z(ma50)
        + 0.15 * cs_z(ma200)
    )
    rotation = z100(
        0.25 * cs_z(rs1)
        + 0.40 * cs_z(rs3)
        + 0.20 * cs_z(rs6)
        + 0.15 * cs_z(rs1 - rs3 / 3.0)
    )
    risk = z100(-0.65 * cs_z(vol20) + 0.35 * cs_z(maxdd126))
    absolute = (
        0.25 * directional(r3, 5.0)
        + 0.20 * directional(r6, 3.0)
        + 0.20 * directional(ma50, 9.0)
        + 0.35 * directional(ma200, 7.0)
    )
    relative = 0.56 * technical + 0.31 * rotation + 0.13 * risk
    score = 0.60 * relative + 0.40 * absolute - (ma200 < -0.05) * 8.0

    bm50 = bench / bench.rolling(50).mean() - 1.0
    bm200 = bench / bench.rolling(200).mean() - 1.0
    market_abs = (
        0.25 * directional(b3, 5.0)
        + 0.20 * directional(b6, 3.0)
        + 0.20 * directional(bm50, 9.0)
        + 0.35 * directional(bm200, 7.0)
    )
    breadth50 = (ma50 > 0).mean(axis=1)
    breadth200 = (ma200 > 0).mean(axis=1)
    breadth3 = (r3 > 0).mean(axis=1)
    breadth_score = 100.0 * (0.30 * breadth50 + 0.45 * breadth200 + 0.25 * breadth3)
    risk_on = (0.68 * market_abs + 0.32 * breadth_score).clip(0.0, 100.0)
    top_score = score.max(axis=1)
    current_cash_score = (100.0 - risk_on + (62.0 - top_score).clip(lower=0.0) * 0.80).clip(0.0, 100.0)

    return {
        "dr": dr,
        "bench_ret": bench_ret,
        "bench": bench,
        "score": score,
        "risk_on": risk_on,
        "cash_score": current_cash_score,
        "breadth200": breadth200,
        "breadth50": breadth50,
        "bm200_gap": bm200,
        "bm50_gap": bm50,
    }


def rebalance_dates(index: pd.DatetimeIndex, mode: str, signal_start: str) -> pd.DatetimeIndex:
    idx = index[index >= pd.Timestamp(signal_start)]
    s = pd.Series(idx, index=idx)
    period = idx.to_period("M" if mode == "monthly" else "W-FRI")
    return pd.DatetimeIndex(s.groupby(period).last().values)


def choose_weights(
    factors: dict,
    dates: pd.DatetimeIndex,
    top_k: int,
    cash_rule: str,
) -> pd.DataFrame:
    score: pd.DataFrame = factors["score"]  # type: ignore[assignment]
    target = pd.DataFrame(0.0, index=dates, columns=TICKERS + ["CASH"])
    for d in dates:
        row = score.loc[d].dropna().sort_values(ascending=False)
        if row.empty:
            target.loc[d, "CASH"] = 1.0
            continue
        selected = row.head(top_k).index.tolist()
        use_cash = False
        if cash_rule == "current":
            cash_score = float(factors["cash_score"].loc[d])
            use_cash = cash_score > float(row.iloc[0])
        elif cash_rule == "dual_trend":
            # Broad market below 200d AND fewer than half of sectors above 200d.
            use_cash = (
                float(factors["bm200_gap"].loc[d]) < 0.0
                and float(factors["breadth200"].loc[d]) < 0.50
            )
        elif cash_rule == "risk_off":
            # Slightly more responsive but still requires both weak regime and breadth.
            use_cash = (
                float(factors["risk_on"].loc[d]) < 45.0
                and float(factors["breadth50"].loc[d]) < 0.45
            )
        elif cash_rule != "none":
            raise ValueError(cash_rule)
        if use_cash:
            target.loc[d, "CASH"] = 1.0
        else:
            target.loc[d, selected] = 1.0 / len(selected)
    return target


def simulate(
    close: pd.DataFrame,
    daily_ret: pd.DataFrame,
    signal_target: pd.DataFrame,
    cost_bps: float,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
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
    # Conservative timing: a target set on day t earns returns from t+1 onward.
    held = target.shift(1).fillna(0.0)
    r = daily_ret.reindex(target.index).reindex(columns=TICKERS).fillna(0.0)
    gross = (held[TICKERS] * r).sum(axis=1)

    prev_target = target.shift(1).fillna(0.0)
    turnover = (target - prev_target).abs().sum(axis=1)
    net = gross - turnover * (cost_bps / 10000.0)
    return net, target, turnover


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar"]}
    eq = (1.0 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=0) * np.sqrt(252))
    std = float(r.std(ddof=0))
    sharpe = float(r.mean() / std * np.sqrt(252)) if std > 0 else np.nan
    downside = float(r[r < 0].std(ddof=0)) if (r < 0).any() else np.nan
    sortino = float(r.mean() / downside * np.sqrt(252)) if np.isfinite(downside) and downside > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    maxdd = float(dd.min())
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": maxdd,
        "calmar": float(cagr / abs(maxdd)) if maxdd < 0 else np.nan,
    }


def slice_perf(r: pd.Series, start: str | None = None, end: str | None = None) -> dict:
    x = r
    if start:
        x = x[x.index >= pd.Timestamp(start)]
    if end:
        x = x[x.index < pd.Timestamp(end)]
    return perf(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download-start", default="2008-03-25")
    p.add_argument("--signal-start", default="2009-04-01")
    p.add_argument("--holdout-start", default="2018-01-01")
    p.add_argument("--out-dir", type=Path, default=Path("research"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    close, repairs, quality = load_prices(args.download_start)
    factors = build_factors(close)

    rows = []
    equity_cols: dict[str, pd.Series] = {}
    for rebalance in ["monthly", "weekly"]:
        dates = rebalance_dates(close.index, rebalance, args.signal_start)
        for top_k in [1, 3]:
            for cash_rule in ["none", "current", "dual_trend", "risk_off"]:
                signal_target = choose_weights(factors, dates, top_k, cash_rule)
                for cost_bps in [0.0, 5.0, 10.0, 20.0, 30.0]:
                    r, target, turnover = simulate(close, factors["dr"], signal_target, cost_bps)  # type: ignore[arg-type]
                    full = perf(r)
                    pre = slice_perf(r, end=args.holdout_start)
                    hold = slice_perf(r, start=args.holdout_start)
                    name = f"{rebalance}_top{top_k}_{cash_rule}_{cost_bps:g}bp"
                    rows.append({
                        "variant": name,
                        "rebalance": rebalance,
                        "top_k": top_k,
                        "cash_rule": cash_rule,
                        "cost_bps": cost_bps,
                        **{f"full_{k}": v for k, v in full.items()},
                        **{f"pre2018_{k}": v for k, v in pre.items()},
                        **{f"holdout_{k}": v for k, v in hold.items()},
                        "cash_exposure": float(target["CASH"].mean()),
                        "annualized_turnover": float(turnover.mean() * 252),
                        "signal_count": int(len(signal_target)),
                    })
                    if cost_bps == 10.0 and rebalance == "monthly":
                        equity_cols[name] = (1.0 + r).cumprod()

    # Equal-weight market benchmark. This is daily equal-weight and has no trading-cost deduction;
    # it is used as a broad market yardstick, not as a directly executable portfolio simulation.
    benchmark = factors["bench_ret"]
    bfull = perf(benchmark[benchmark.index >= pd.Timestamp(args.signal_start)])
    bpre = slice_perf(benchmark[benchmark.index >= pd.Timestamp(args.signal_start)], end=args.holdout_start)
    bhold = slice_perf(benchmark, start=args.holdout_start)

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["holdout_sharpe", "holdout_calmar", "full_sharpe"],
        ascending=False,
        na_position="last",
    )
    result.to_csv(args.out_dir / "grid_results.csv", index=False)

    # Robust shortlist: positive holdout CAGR, drawdown no worse than -45%,
    # and positive pre-2018 CAGR. These are screening criteria, not fitted thresholds.
    shortlist = result[
        (result["holdout_cagr"] > 0)
        & (result["pre2018_cagr"] > 0)
        & (result["holdout_max_drawdown"] > -0.45)
    ].copy()
    shortlist.head(25).to_csv(args.out_dir / "shortlist.csv", index=False)

    if equity_cols:
        pd.DataFrame(equity_cols).to_csv(args.out_dir / "monthly_10bp_equity.csv")

    metadata = {
        "download_start": args.download_start,
        "signal_start": args.signal_start,
        "holdout_start": args.holdout_start,
        "price_start": quality["start"],
        "price_end": quality["end"],
        "known_split_repairs": repairs,
        "quality": quality,
        "benchmark": {
            "description": "daily equal-weight return of available TOPIX-17 ETFs",
            "full": bfull,
            "pre2018": bpre,
            "holdout": bhold,
        },
        "variants": int(len(result)),
        "shortlist_count": int(len(shortlist)),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    best = result.head(20).copy()
    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "holdout_cagr", "holdout_max_drawdown", "holdout_sharpe",
        "cash_exposure", "annualized_turnover",
    ]
    report = [
        "# TOPIX-17 Robustness Research",
        "",
        "## Data quality",
        "",
        f"- Period: {quality['start']} to {quality['end']}",
        f"- Known split repairs: `{json.dumps(repairs, ensure_ascii=False)}`",
        f"- Maximum absolute daily return after repair: {max(quality['max_abs_daily_return_by_ticker'].values()):.2%}",
        "",
        "## Equal-weight benchmark",
        "",
        f"- Full CAGR: {bfull['cagr']:.2%}",
        f"- Full max drawdown: {bfull['max_drawdown']:.2%}",
        f"- Full Sharpe: {bfull['sharpe']:.3f}",
        f"- Holdout CAGR: {bhold['cagr']:.2%}",
        f"- Holdout max drawdown: {bhold['max_drawdown']:.2%}",
        f"- Holdout Sharpe: {bhold['sharpe']:.3f}",
        "",
        "## Best variants by holdout Sharpe",
        "",
        best[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Methodology note",
        "",
        "The holdout period starts in 2018. The parameter grid is intentionally small. "
        "A variant is not considered robust merely because it ranks first; pre-2018 and "
        "holdout behavior, drawdown, turnover and transaction-cost sensitivity must agree.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(best[cols].to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
