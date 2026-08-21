#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TOPIX-17 robustness research using official NEXT FUNDS historical data.

Data source
-----------
Nomura Asset Management official historical CSVs:
https://www.nomura-am.co.jp/fund/etf/history/ETF_<code>.csv

For each ETF we use column 7 (zero-based 6): NAV per share with distributions
reinvested. The official 1629 CSV explicitly states that the 2026-04-01
1->500 beneficial-interest split is adjusted in the series.

Research design
---------------
- TOPIX-17 ETFs 1617-1633.
- Official distribution-reinvested NAV series; no Yahoo price adjustments.
- Monthly and weekly rebalance.
- Top 1 and Top 3 sector allocations.
- Four cash rules: none/current score/dual trend/risk-off.
- Cost sensitivity: 0/5/10/20/30 bps per unit of portfolio turnover.
- Signal known at rebalance close; target set next trading day; returns use
  lagged target weights (conservative, no close-to-close look-ahead).
- Pre-2018 and 2018+ holdout metrics reported separately.

This is a research tool, not an order execution system.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests

from screener import TOPIX17_ETFS

TICKERS = list(TOPIX17_ETFS)
BASE_URL = "https://www.nomura-am.co.jp/fund/etf/history/ETF_{code}.csv"


def cs_z(x: pd.DataFrame, clip: float = 2.5) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0).replace(0.0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip).fillna(0.0)


def z100(z: pd.DataFrame) -> pd.DataFrame:
    return 100.0 / (1.0 + np.exp(-1.15 * z))


def directional(x: pd.DataFrame | pd.Series, scale: float):
    return (50.0 + 50.0 * np.tanh(x * scale)).clip(0.0, 100.0)


def fetch_official_series(ticker: str, retries: int = 4) -> tuple[pd.Series, dict]:
    code = ticker.replace(".T", "")
    url = BASE_URL.format(code=code)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(
                url,
                timeout=60,
                headers={"User-Agent": "tse-sector-rotation-research/2.0"},
            )
            r.raise_for_status()
            text = r.content.decode("shift_jis", errors="strict")
            # 0: Japanese title, 1: English title, 2: blank,
            # 3: Japanese header, 4: English header, 5+: data.
            frame = pd.read_csv(io.StringIO(text), skiprows=5, header=None)
            if frame.shape[1] < 7:
                raise RuntimeError(f"{ticker}: unexpected official CSV width {frame.shape[1]}")
            dates = pd.to_datetime(frame.iloc[:, 0].astype(str), format="%Y%m%d", errors="coerce")
            tri_nav = pd.to_numeric(frame.iloc[:, 6], errors="coerce")
            s = pd.Series(tri_nav.values, index=dates, name=ticker).dropna().sort_index()
            s = s[~s.index.duplicated(keep="last")]
            s = s[s > 0]
            if len(s) < 1000:
                raise RuntimeError(f"{ticker}: only {len(s)} official observations")
            dr = s.pct_change(fill_method=None)
            max_abs = float(dr.abs().max())
            if max_abs > 0.50:
                worst = dr.abs().nlargest(5).index
                details = pd.DataFrame({"nav": s, "ret": dr}).reindex(worst).to_dict("index")
                raise RuntimeError(f"{ticker}: suspicious official daily move {max_abs:.2%}: {details}")
            meta = {
                "ticker": ticker,
                "url": url,
                "rows": int(len(s)),
                "start": s.index.min().date().isoformat(),
                "end": s.index.max().date().isoformat(),
                "max_abs_daily_return": max_abs,
            }
            return s, meta
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"official history failed for {ticker}: {last}")


def load_official_prices() -> tuple[pd.DataFrame, list[dict]]:
    series: Dict[str, pd.Series] = {}
    metadata: list[dict] = []
    for ticker in TICKERS:
        s, meta = fetch_official_series(ticker)
        series[ticker] = s
        metadata.append(meta)
        print(f"official {ticker}: {meta['start']} -> {meta['end']} rows={meta['rows']} maxmove={meta['max_abs_daily_return']:.2%}")
    close = pd.concat(series, axis=1).sort_index()
    # Do not forward-fill for return calculation. Each ETF should normally share
    # the same NAV date; missing values remain missing and are ignored cross-sectionally.
    return close, metadata


def build_factors(close: pd.DataFrame) -> dict:
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
    cash_score = (100.0 - risk_on + (62.0 - top_score).clip(lower=0.0) * 0.80).clip(0.0, 100.0)
    return {
        "dr": dr,
        "bench_ret": bench_ret,
        "bench": bench,
        "score": score,
        "risk_on": risk_on,
        "cash_score": cash_score,
        "breadth50": breadth50,
        "breadth200": breadth200,
        "bm50_gap": bm50,
        "bm200_gap": bm200,
    }


def rebalance_dates(index: pd.DatetimeIndex, mode: str, start: str) -> pd.DatetimeIndex:
    idx = index[index >= pd.Timestamp(start)]
    holder = pd.Series(idx, index=idx)
    periods = idx.to_period("M" if mode == "monthly" else "W-FRI")
    return pd.DatetimeIndex(holder.groupby(periods).last().values)


def choose_weights(factors: dict, dates: pd.DatetimeIndex, top_k: int, cash_rule: str) -> pd.DataFrame:
    score: pd.DataFrame = factors["score"]
    target = pd.DataFrame(0.0, index=dates, columns=TICKERS + ["CASH"])
    for d in dates:
        row = score.loc[d].dropna().sort_values(ascending=False)
        if row.empty:
            target.loc[d, "CASH"] = 1.0
            continue
        selected = row.head(top_k).index.tolist()
        use_cash = False
        if cash_rule == "current":
            use_cash = float(factors["cash_score"].loc[d]) > float(row.iloc[0])
        elif cash_rule == "dual_trend":
            use_cash = (
                float(factors["bm200_gap"].loc[d]) < 0.0
                and float(factors["breadth200"].loc[d]) < 0.50
            )
        elif cash_rule == "risk_off":
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
    previous = target.shift(1).fillna(0.0)
    turnover = (target - previous).abs().sum(axis=1)
    net = gross - turnover * (cost_bps / 10000.0)
    return net, target, turnover


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar"]}
    eq = (1.0 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
    std = float(r.std(ddof=0))
    vol = float(std * np.sqrt(252))
    sharpe = float(r.mean() / std * np.sqrt(252)) if std > 0 else np.nan
    neg = r[r < 0]
    downside = float(neg.std(ddof=0)) if len(neg) > 1 else np.nan
    sortino = float(r.mean() / downside * np.sqrt(252)) if np.isfinite(downside) and downside > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
    }


def sliced(r: pd.Series, start: str | None = None, end: str | None = None) -> dict:
    x = r
    if start:
        x = x[x.index >= pd.Timestamp(start)]
    if end:
        x = x[x.index < pd.Timestamp(end)]
    return perf(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--signal-start", default="2009-04-01")
    p.add_argument("--holdout-start", default="2018-01-01")
    p.add_argument("--out-dir", type=Path, default=Path("research_official"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    factors = build_factors(close)
    rows: list[dict] = []
    monthly_equity: dict[str, pd.Series] = {}

    for rebalance in ["monthly", "weekly"]:
        dates = rebalance_dates(close.index, rebalance, args.signal_start)
        for top_k in [1, 3]:
            for cash_rule in ["none", "current", "dual_trend", "risk_off"]:
                signal_target = choose_weights(factors, dates, top_k, cash_rule)
                for cost_bps in [0.0, 5.0, 10.0, 20.0, 30.0]:
                    r, target, turnover = simulate(close, factors["dr"], signal_target, cost_bps)
                    full = perf(r)
                    pre = sliced(r, end=args.holdout_start)
                    hold = sliced(r, start=args.holdout_start)
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
                    if rebalance == "monthly" and cost_bps == 10.0:
                        monthly_equity[name] = (1.0 + r).cumprod()

    results = pd.DataFrame(rows).sort_values(
        ["holdout_sharpe", "holdout_calmar", "full_sharpe"],
        ascending=False,
        na_position="last",
    )
    results.to_csv(args.out_dir / "grid_results.csv", index=False)

    robust = results[
        (results["pre2018_cagr"] > 0)
        & (results["holdout_cagr"] > 0)
        & (results["holdout_max_drawdown"] > -0.45)
    ].copy()
    robust.head(30).to_csv(args.out_dir / "shortlist.csv", index=False)
    pd.DataFrame(monthly_equity).to_csv(args.out_dir / "monthly_10bp_equity.csv")

    bench = factors["bench_ret"]
    bench = bench[bench.index >= pd.Timestamp(args.signal_start)]
    benchmark = {
        "description": "daily equal-weight return of 17 official distribution-reinvested NAV series",
        "full": perf(bench),
        "pre2018": sliced(bench, end=args.holdout_start),
        "holdout": sliced(bench, start=args.holdout_start),
    }
    meta = {
        "signal_start": args.signal_start,
        "holdout_start": args.holdout_start,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "NAV per share with distributions reinvested",
        "source_metadata": source_meta,
        "benchmark": benchmark,
        "variants": int(len(results)),
        "shortlist_count": int(len(robust)),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best = results.head(25)
    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "pre2018_sharpe", "holdout_cagr",
        "holdout_max_drawdown", "holdout_sharpe", "cash_exposure",
        "annualized_turnover",
    ]
    b = benchmark
    report = [
        "# TOPIX-17 Robustness Research — Official NEXT FUNDS Data",
        "",
        "## Data",
        "",
        "Official Nomura Asset Management historical CSVs; distribution-reinvested NAV per share.",
        f"Common research horizon: {close.index.min().date()} to {close.index.max().date()}.",
        "",
        "## Equal-weight benchmark",
        "",
        f"- Full CAGR: {b['full']['cagr']:.2%}",
        f"- Full max drawdown: {b['full']['max_drawdown']:.2%}",
        f"- Full Sharpe: {b['full']['sharpe']:.3f}",
        f"- Holdout CAGR: {b['holdout']['cagr']:.2%}",
        f"- Holdout max drawdown: {b['holdout']['max_drawdown']:.2%}",
        f"- Holdout Sharpe: {b['holdout']['sharpe']:.3f}",
        "",
        "## Best variants by 2018+ holdout Sharpe",
        "",
        best[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation rule",
        "",
        "Do not select the first row mechanically. Prefer parameter neighborhoods where monthly/weekly, "
        "Top1/Top3 and cost assumptions tell a consistent story, and where both pre-2018 and 2018+ "
        "results remain acceptable. A single isolated optimum is treated as overfit.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(best[cols].to_string(index=False))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
