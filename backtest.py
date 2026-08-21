#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from screener import (
    BENCHMARK,
    TOPIX17_ETFS,
    INDUSTRY_TO_TOPIX17,
    TOPIX17_TO_INDUSTRIES,
    SENSITIVITY,
    FRED,
    absolute_score,
    winsor_z,
    z_to_100,
    weights_for,
    load_constituents,
)

VIX_TICKER = "^VIX"


def chunks(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i+n]


def ticker_frame(raw: pd.DataFrame, ticker: str, batch_len: int) -> pd.DataFrame:
    if batch_len == 1 and not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy()
    if ticker in raw.columns.get_level_values(0):
        return raw[ticker].copy()
    if ticker in raw.columns.get_level_values(1):
        return raw.xs(ticker, axis=1, level=1).copy()
    raise KeyError(ticker)


def download_history(tickers: list[str], start: str, batch_size: int = 35) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for batch in chunks(sorted(set(tickers)), batch_size):
        err = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    interval="1d",
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                    timeout=45,
                )
                for t in batch:
                    try:
                        df = ticker_frame(raw, t, len(batch)).dropna(how="all")
                        if "Close" in df and df["Close"].notna().sum() >= 80:
                            out[t] = df
                    except Exception:
                        pass
                err = None
                break
            except Exception as exc:
                err = exc
                time.sleep(2 ** attempt)
        if err is not None:
            print(f"WARNING download failed {batch[:2]}...: {err}")
    return out


def pct_ret(s: pd.Series, n: int, date: pd.Timestamp) -> float:
    x = s.loc[:date].dropna()
    if len(x) <= n:
        return np.nan
    return float(x.iloc[-1] / x.iloc[-n-1] - 1)


def rsi_at(s: pd.Series, date: pd.Timestamp, n: int = 14) -> float:
    x = s.loc[:date].dropna()
    if len(x) < n + 2:
        return np.nan
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return float((100 - 100/(1 + up/(dn + 1e-12))).iloc[-1])


def maxdd_at(s: pd.Series, date: pd.Timestamp, n: int = 126) -> float:
    x = s.loc[:date].dropna().tail(n)
    if len(x) < 20:
        return np.nan
    return float((x / x.cummax() - 1).min())


def moving_gap(s: pd.Series, date: pd.Timestamp, n: int) -> float:
    x = s.loc[:date].dropna()
    if len(x) < n:
        return np.nan
    ma = x.tail(n).mean()
    return float(x.iloc[-1] / ma - 1)


def annualized_vol(s: pd.Series, date: pd.Timestamp, n: int = 20) -> float:
    x = s.loc[:date].dropna().pct_change().dropna().tail(n)
    return float(x.std(ddof=0) * np.sqrt(252)) if len(x) >= 10 else np.nan


def turnover_at(df: pd.DataFrame, date: pd.Timestamp, n: int = 20) -> float:
    if "Volume" not in df or "Close" not in df:
        return np.nan
    x = (df["Close"] * df["Volume"]).loc[:date].dropna().tail(n)
    return float(x.mean()) if len(x) else np.nan


def make_proxy(group: pd.DataFrame, prices: Dict[str, pd.DataFrame]) -> tuple[pd.Series, Dict[str, pd.Series], pd.Series]:
    members = {}
    weights = {}
    for row in group.itertuples(index=False):
        if row.ticker in prices:
            members[row.ticker] = prices[row.ticker]["Close"].dropna()
            weights[row.ticker] = float(row.weight)
    if len(members) < 2:
        return pd.Series(dtype=float), {}, pd.Series(dtype=float)
    returns = pd.concat({t: s.pct_change() for t, s in members.items()}, axis=1).sort_index()
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    valid_w = returns.notna().mul(w, axis=1)
    wr = returns.fillna(0).mul(w, axis=1).sum(axis=1)
    denom = valid_w.sum(axis=1).replace(0, np.nan)
    proxy = (1 + (wr / denom).dropna()).cumprod()
    return proxy, members, w


def industry_metrics(
    industry: str,
    group: pd.DataFrame,
    proxy: pd.Series,
    members: Dict[str, pd.Series],
    bench: pd.Series,
    date: pd.Timestamp,
) -> dict | None:
    p = proxy.loc[:date].dropna()
    if len(p) < 200:
        return None
    b = bench.reindex(p.index).ffill().dropna()
    p = p.reindex(b.index).dropna()
    b = b.reindex(p.index)
    if len(p) < 200:
        return None

    ret1, ret3, ret6 = [pct_ret(p, n, date) for n in (21, 63, 126)]
    b1, b3, b6 = [pct_ret(b, n, date) for n in (21, 63, 126)]
    ma50, ma200 = moving_gap(p, date, 50), moving_gap(p, date, 200)

    eligible = []
    for t, s in members.items():
        x = s.loc[:date].dropna()
        if len(x) < 126:
            continue
        m50 = moving_gap(s, date, 50)
        m200 = moving_gap(s, date, 200)
        m3 = pct_ret(s, 63, date)
        bm3 = pct_ret(bench.reindex(s.index).ffill(), 63, date)
        eligible.append({
            "above50": float(m50 > 0) if np.isfinite(m50) else np.nan,
            "above200": float(m200 > 0) if np.isfinite(m200) else np.nan,
            "positive3m": float(m3 > 0) if np.isfinite(m3) else np.nan,
            "beat3m": float(m3 > bm3) if np.isfinite(m3) and np.isfinite(bm3) else np.nan,
        })
    if len(eligible) < 2:
        return None
    ms = pd.DataFrame(eligible)
    configured = len(group)
    coverage = len(eligible) / max(configured, 1)

    return {
        "industry": industry,
        "group": INDUSTRY_TO_TOPIX17[industry],
        "coverage": coverage,
        "ret_1m": ret1,
        "ret_3m": ret3,
        "ret_6m": ret6,
        "rs_1m": ret1 - b1,
        "rs_3m": ret3 - b3,
        "rs_6m": ret6 - b6,
        "rotation_accel": (ret1 - b1) - (ret3 - b3)/3.0,
        "ma50_gap": ma50,
        "ma200_gap": ma200,
        "rsi14": rsi_at(p, date),
        "vol20": annualized_vol(p, date),
        "maxdd_6m": maxdd_at(p, date),
        "breadth_50d": float(ms["above50"].mean()),
        "breadth_200d": float(ms["above200"].mean()),
        "breadth_positive_3m": float(ms["positive3m"].mean()),
        "breadth_beat_topix_3m": float(ms["beat3m"].mean()),
        "absolute_score": absolute_score(ret3, ret6, ma50, ma200),
    }


def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30, headers={"User-Agent": "tse-sector-rotation-backtest/1.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return df.dropna().set_index(df.columns[0])[df.columns[1]].sort_index()


def change_signal(s: pd.Series, short: int, long: int, scale: float = 5.0) -> float:
    s = s.dropna()
    if len(s) <= long:
        return 0.0
    a, b = s.iloc[-short-1], s.iloc[-long-1]
    if a == 0 or b == 0:
        return 0.0
    sc = s.iloc[-1] / a - 1
    lc = s.iloc[-1] / b - 1
    return float(np.tanh(scale * (0.65*sc + 0.35*lc)))


def level_change_signal(s: pd.Series, short: int, long: int, scale: float = 1.0) -> float:
    s = s.dropna()
    if len(s) <= long:
        return 0.0
    sc = s.iloc[-1] - s.iloc[-short-1]
    lc = s.iloc[-1] - s.iloc[-long-1]
    return float(np.tanh(scale * (0.65*sc + 0.35*lc)))


def macro_at(fred: Dict[str, pd.Series], date: pd.Timestamp) -> dict:
    monthly_cut = date - pd.Timedelta(days=45)
    daily_cut = date - pd.Timedelta(days=1)
    jgb = fred["jgb10y"].loc[:monthly_cut]
    ip = fred["industrial_prod"].loc[:monthly_cut]
    cpi = fred["cpi"].loc[:monthly_cut]
    oil = fred["brent"].loc[:daily_cut]
    fx = fred["usdjpy"].loc[:daily_cut]
    vix = fred["vix"].loc[:daily_cut]
    rates = level_change_signal(jgb, 3, 12, 1.3)
    growth = change_signal(ip, 3, 12, 6.0)
    inflation = change_signal(cpi, 3, 12, 8.0)
    oil_s = change_signal(oil, 21, 126, 3.0)
    fx_s = change_signal(fx, 21, 126, 4.0)
    risk = float(np.tanh((vix.iloc[-1] / vix.tail(63).median() - 1)*2.0)) if len(vix) >= 63 else 0.0
    if growth >= .15 and inflation < .15:
        regime = "Goldilocks / expansion"
    elif growth >= .15 and inflation >= .15:
        regime = "Reflation / late-cycle"
    elif growth < -.15 and inflation >= .15:
        regime = "Stagflation risk"
    elif growth < -.15 and inflation < .15:
        regime = "Disinflation / slowdown"
    else:
        regime = "Mixed / transition"
    return {"rates": rates, "growth": growth, "inflation": inflation, "oil": oil_s, "usdjpy": fx_s, "risk_off": risk, "regime": regime}


def macro_scores(industries: pd.Index, macro: dict) -> pd.Series:
    raw = {}
    for industry in industries:
        group = INDUSTRY_TO_TOPIX17[industry]
        sens = SENSITIVITY[group]
        v = sum(sens[k] * macro[k] for k in ("rates", "growth", "inflation", "oil", "usdjpy"))
        v -= 0.30 * max(macro["risk_off"], 0) * abs(sens["growth"])
        raw[industry] = v
    return z_to_100(winsor_z(pd.Series(raw)))


def etf_metrics(prices: Dict[str, pd.DataFrame], bench: pd.Series, date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for ticker, group in TOPIX17_ETFS.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        c = df["Close"].loc[:date].dropna()
        if len(c) < 200:
            continue
        ret1, ret3, ret6 = [pct_ret(c, n, date) for n in (21, 63, 126)]
        b1, b3, b6 = [pct_ret(bench.reindex(c.index).ffill(), n, date) for n in (21, 63, 126)]
        m50, m200 = moving_gap(c, date, 50), moving_gap(c, date, 200)
        rows.append({
            "ticker": ticker, "group": group,
            "ret_1m": ret1, "ret_3m": ret3, "ret_6m": ret6,
            "rs_1m": ret1-b1, "rs_3m": ret3-b3, "rs_6m": ret6-b6,
            "ma50_gap": m50, "ma200_gap": m200, "rsi14": rsi_at(c, date),
            "vol20": annualized_vol(c, date), "maxdd_6m": maxdd_at(c, date),
            "avg_turnover_yen": turnover_at(df, date),
            "absolute_score": absolute_score(ret3, ret6, m50, m200),
        })
    x = pd.DataFrame(rows).set_index("ticker")
    if len(x) < 14:
        return pd.DataFrame()
    z = (
        .18*winsor_z(x["ret_1m"]) + .27*winsor_z(x["ret_3m"]) + .15*winsor_z(x["ret_6m"])
        + .15*winsor_z(x["rs_3m"]) + .10*winsor_z(x["ma50_gap"]) + .15*winsor_z(x["ma200_gap"])
    )
    x["etf_technical_score"] = z_to_100(z)
    x["etf_risk_score"] = z_to_100(-.65*winsor_z(x["vol20"]) + .35*winsor_z(x["maxdd_6m"]))
    x["liquidity_score"] = z_to_100(winsor_z(np.log1p(x["avg_turnover_yen"].clip(lower=0))))
    return x


def score_date(
    date: pd.Timestamp,
    constituents: pd.DataFrame,
    proxies: dict,
    prices: Dict[str, pd.DataFrame],
    bench: pd.Series,
    fred: Dict[str, pd.Series],
) -> tuple[str, str, dict] | None:
    im = []
    for industry, group in constituents.groupby("industry", sort=False):
        proxy, members, _ = proxies[industry]
        m = industry_metrics(industry, group, proxy, members, bench, date)
        if m is not None:
            im.append(m)
    ind = pd.DataFrame(im).set_index("industry")
    if len(ind) < 28:
        return None

    ind["technical_score"] = z_to_100(
        .18*winsor_z(ind["ret_1m"]) + .27*winsor_z(ind["ret_3m"]) + .20*winsor_z(ind["ret_6m"])
        + .15*winsor_z(ind["ma50_gap"]) + .20*winsor_z(ind["ma200_gap"])
    )
    ind["rotation_score"] = z_to_100(
        .18*winsor_z(ind["rs_1m"]) + .24*winsor_z(ind["rs_3m"]) + .13*winsor_z(ind["rs_6m"])
        + .15*winsor_z(ind["rotation_accel"]) + .10*winsor_z(ind["breadth_50d"])
        + .10*winsor_z(ind["breadth_200d"]) + .10*winsor_z(ind["breadth_beat_topix_3m"])
    )
    ind["risk_score"] = z_to_100(-.60*winsor_z(ind["vol20"]) + .40*winsor_z(ind["maxdd_6m"]))

    macro = macro_at(fred, date)
    ind["macro_score"] = macro_scores(ind.index, macro)
    w = weights_for(macro["regime"], False)
    ind["relative_score"] = (
        w["technical"]*ind["technical_score"] + w["rotation"]*ind["rotation_score"]
        + w["macro"]*ind["macro_score"] + w["risk"]*ind["risk_score"]
        + w["absolute"]*ind["absolute_score"]
    )
    ind["penalty"] = 0.0
    ind.loc[ind["ma200_gap"] < -.05, "penalty"] += 7
    ind.loc[ind["breadth_200d"] < .30, "penalty"] += 5
    ind.loc[ind["coverage"] < .70, "penalty"] += 4
    ind.loc[ind["rsi14"] > 82, "penalty"] += 4
    if macro["risk_off"] > .35:
        ind.loc[ind["vol20"] > ind["vol20"].quantile(.75), "penalty"] += 4
    ind["final_score"] = (0.60*ind["relative_score"] + 0.40*ind["absolute_score"] - ind["penalty"]).clip(0,100)

    etf = etf_metrics(prices, bench, date)
    if etf.empty:
        return None
    underlying = {}
    for ticker, row in etf.iterrows():
        s = ind[ind["group"] == row["group"]]
        underlying[ticker] = 50.0 if s.empty else float(.75*s["final_score"].mean() + .25*s["final_score"].max())
    etf["underlying_score"] = pd.Series(underlying)
    etf["relative_score"] = (
        .54*etf["underlying_score"] + .24*etf["etf_technical_score"]
        + .12*etf["etf_risk_score"] + .10*etf["liquidity_score"]
    )
    etf["penalty"] = 0.0
    etf.loc[etf["ma200_gap"] < -.05, "penalty"] += 8
    etf.loc[etf["rsi14"] > 82, "penalty"] += 5
    etf.loc[etf["avg_turnover_yen"] < 20_000_000, "penalty"] += 5
    etf["final_score"] = (0.60*etf["relative_score"] + 0.40*etf["absolute_score"] - etf["penalty"]).clip(0,100)

    bc = bench.loc[:date].dropna()
    br3, br6 = pct_ret(bc,63,date), pct_ret(bc,126,date)
    bg50, bg200 = moving_gap(bc,date,50), moving_gap(bc,date,200)
    market_abs = absolute_score(br3, br6, bg50, bg200)
    breadth50 = float((etf["ma50_gap"] > 0).mean())
    breadth200 = float((etf["ma200_gap"] > 0).mean())
    breadth3 = float((etf["ret_3m"] > 0).mean())
    breadth_score = 100*(.30*breadth50 + .45*breadth200 + .25*breadth3)
    risk_on = float(np.clip(.68*market_abs + .32*breadth_score - 12*max(macro["risk_off"],0),0,100))
    cash_base = float(np.clip(100-risk_on + 12*max(macro["risk_off"],0),0,100))
    top_etf_score = float(etf["final_score"].max())
    cash_score = float(np.clip(cash_base + max(0,62-top_etf_score)*.80,0,100))

    top_etf = str(etf["final_score"].idxmax())
    full = "CASH" if cash_score > top_etf_score else top_etf
    return full, top_etf, {
        "date": date.date().isoformat(),
        "selected": full,
        "top_etf": top_etf,
        "top_etf_score": top_etf_score,
        "cash_score": cash_score,
        "risk_on_score": risk_on,
        "regime": macro["regime"],
        "industries_usable": int(len(ind)),
        "avg_coverage": float(ind["coverage"].mean()),
    }


def signal_dates(bench: pd.Series, start: str) -> list[pd.Timestamp]:
    idx = bench.loc[pd.Timestamp(start):].dropna().index
    if idx.empty:
        return []
    s = pd.Series(idx, index=idx)
    last = s.groupby(idx.to_period("W-FRI")).last()
    return list(pd.DatetimeIndex(last.values))


def simulate(
    signals: pd.Series,
    prices: Dict[str, pd.DataFrame],
    bench: pd.Series,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = bench.index
    signals = signals.sort_index()
    exec_map = {}
    for d, asset in signals.items():
        future = dates[dates > d]
        if len(future):
            exec_map[future[0]] = asset
    if len(exec_map) < 2:
        raise RuntimeError("not enough executable signals")
    exec_series = pd.Series(exec_map).sort_index()

    start, end = exec_series.index[0], dates[-1]
    daily_idx = dates[(dates >= start) & (dates <= end)]
    pos = pd.Series(index=daily_idx, dtype="object")
    for d, a in exec_series.items():
        if d in pos.index:
            pos.loc[d] = a
    pos = pos.ffill().fillna("CASH")

    close = pd.DataFrame({t: df["Close"] for t, df in prices.items() if t in TOPIX17_ETFS}).reindex(daily_idx).ffill()
    asset_ret = close.pct_change().fillna(0)
    held = pos.shift(1).fillna("CASH")
    strat_ret = pd.Series(0.0, index=daily_idx)
    for t in TOPIX17_ETFS:
        mask = held == t
        strat_ret.loc[mask] = asset_ret.loc[mask, t].fillna(0)

    changes = pos.ne(pos.shift(1))
    costs = pd.Series(0.0, index=daily_idx)
    prev = pos.shift(1).fillna("CASH")
    side_cost = cost_bps / 10000.0
    for d in daily_idx[changes]:
        p, n = prev.loc[d], pos.loc[d]
        sides = int(p != "CASH") + int(n != "CASH")
        costs.loc[d] = sides * side_cost
    strat_ret = strat_ret - costs

    bench_ret = bench.reindex(daily_idx).ffill().pct_change().fillna(0)
    eq = pd.DataFrame({
        "strategy_return": strat_ret,
        "benchmark_return": bench_ret,
        "position": pos,
        "transaction_cost": costs,
    })
    eq["strategy_equity"] = (1 + eq["strategy_return"]).cumprod()
    eq["benchmark_equity"] = (1 + eq["benchmark_return"]).cumprod()

    trades = pd.DataFrame({
        "execution_date": exec_series.index,
        "asset": exec_series.values,
    })
    trades["previous"] = trades["asset"].shift(1).fillna("CASH")
    trades = trades[trades["asset"] != trades["previous"]].reset_index(drop=True)
    return eq, trades


def perf(r: pd.Series, equity: pd.Series) -> dict:
    r = r.dropna()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1/365.25)
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1/years) - 1)
    vol = float(r.std(ddof=0) * np.sqrt(252))
    sharpe = float(r.mean()/r.std(ddof=0)*np.sqrt(252)) if r.std(ddof=0) > 0 else np.nan
    downside = r[r < 0].std(ddof=0)
    sortino = float(r.mean()/downside*np.sqrt(252)) if np.isfinite(downside) and downside > 0 else np.nan
    dd = equity/equity.cummax()-1
    mdd = float(dd.min())
    calmar = float(cagr/abs(mdd)) if mdd < 0 else np.nan
    return {"total_return": total, "cagr": cagr, "volatility": vol, "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd, "calmar": calmar}


def annual_returns(r: pd.Series) -> pd.Series:
    return (1+r).groupby(r.index.year).prod()-1


def write_report(summary: pd.DataFrame, meta: dict, annual: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Backtest report",
        "",
        f"- Period: {meta['start']} to {meta['end']}",
        "- Signal frequency: weekly; signal at week-end close, effective next trading day close",
        f"- Transaction cost: {meta['cost_bps']:.1f} bps per side",
        "- Prices: yfinance adjusted prices (dividend/split adjusted)",
        "- Fundamentals: excluded (live template is empty)",
        "- Macro: FRED historical observations with conservative 45-day monthly / 1-day daily lag",
        "- Important: 33-industry proxy uses today's configured constituents, so its historical test has survivorship/classification bias.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Annual returns",
        "",
        annual.to_markdown(),
        "",
        "## Interpretation",
        "",
        "Treat this as model validation, not proof of future profitability. The 33-industry historical layer is less reliable until point-in-time membership data is available.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-start", default="2008-03-25")
    ap.add_argument("--signal-start", default="2012-01-01")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, default=Path("backtest"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    constituents = load_constituents(Path("config/industry_constituents.csv"))
    tickers = constituents["ticker"].tolist() + list(TOPIX17_ETFS) + [BENCHMARK]
    prices = download_history(tickers, args.download_start)
    if BENCHMARK not in prices:
        raise RuntimeError("benchmark unavailable")
    missing_etfs = [t for t in TOPIX17_ETFS if t not in prices]
    if missing_etfs:
        raise RuntimeError(f"missing ETFs: {missing_etfs}")

    bench = prices[BENCHMARK]["Close"].dropna().sort_index()
    proxies = {}
    for industry, group in constituents.groupby("industry", sort=False):
        proxies[industry] = make_proxy(group, prices)

    fred = {}
    for k, sid in FRED.items():
        fred[k] = fred_series(sid)

    rows = []
    for i, d in enumerate(signal_dates(bench, args.signal_start), 1):
        try:
            out = score_date(d, constituents, proxies, prices, bench, fred)
            if out:
                full, no_cash, meta = out
                meta["selected_no_cash"] = no_cash
                rows.append(meta)
        except Exception as exc:
            print(f"WARNING {d.date()}: {exc}")
        if i % 100 == 0:
            print(f"processed {i} signal dates")
    sig = pd.DataFrame(rows)
    if len(sig) < 100:
        raise RuntimeError(f"only {len(sig)} valid signal dates")
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.set_index("date").sort_index()

    full_eq, full_trades = simulate(sig["selected"], prices, bench, args.cost_bps)
    nocash_eq, nocash_trades = simulate(sig["selected_no_cash"], prices, bench, args.cost_bps)
    bret = full_eq["benchmark_return"]
    beq = full_eq["benchmark_equity"]

    full_perf = perf(full_eq["strategy_return"], full_eq["strategy_equity"])
    nocash_perf = perf(nocash_eq["strategy_return"].reindex(full_eq.index).fillna(0), nocash_eq["strategy_equity"].reindex(full_eq.index).ffill())
    bench_perf = perf(bret, beq)

    def row(name, p, eq, trades, signals_col=None):
        exposure = float((eq["position"] != "CASH").mean()) if "position" in eq else 1.0
        d = {"strategy": name, **p, "etf_exposure": exposure, "switches": len(trades)}
        if signals_col is not None:
            d["cash_signal_share"] = float((sig[signals_col] == "CASH").mean())
        else:
            d["cash_signal_share"] = 0.0
        return d

    summary = pd.DataFrame([
        row("33-industry + TOPIX-17 + CASH", full_perf, full_eq, full_trades, "selected"),
        row("33-industry + TOPIX-17 (always invested)", nocash_perf, nocash_eq, nocash_trades, "selected_no_cash"),
        {"strategy": "TOPIX 1306 buy&hold", **bench_perf, "etf_exposure": 1.0, "switches": 0, "cash_signal_share": 0.0},
    ])

    annual = pd.DataFrame({
        "strategy_cash": annual_returns(full_eq["strategy_return"]),
        "strategy_no_cash": annual_returns(nocash_eq["strategy_return"].reindex(full_eq.index).fillna(0)),
        "topix_1306": annual_returns(full_eq["benchmark_return"]),
    })

    equity = pd.DataFrame(index=full_eq.index)
    equity["strategy_cash"] = full_eq["strategy_equity"]
    equity["strategy_no_cash"] = nocash_eq["strategy_equity"].reindex(full_eq.index).ffill()
    equity["topix_1306"] = full_eq["benchmark_equity"]
    equity["position"] = full_eq["position"]

    meta = {
        "start": str(full_eq.index[0].date()),
        "end": str(full_eq.index[-1].date()),
        "cost_bps": args.cost_bps,
        "signals": len(sig),
        "avg_industry_coverage": float(sig["avg_coverage"].mean()),
        "min_industries_usable": int(sig["industries_usable"].min()),
        "cash_signal_share": float((sig["selected"] == "CASH").mean()),
        "latest_signal": str(sig.index[-1].date()),
        "latest_selected": str(sig["selected"].iloc[-1]),
    }

    summary.to_csv(args.out_dir/"summary.csv", index=False)
    annual.to_csv(args.out_dir/"annual_returns.csv")
    equity.to_csv(args.out_dir/"equity.csv")
    sig.to_csv(args.out_dir/"signals.csv")
    full_trades.to_csv(args.out_dir/"trades.csv", index=False)
    (args.out_dir/"results.json").write_text(
        json.dumps({"meta": meta, "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary, meta, annual, args.out_dir/"REPORT.md")
    print(summary.to_string(index=False))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
