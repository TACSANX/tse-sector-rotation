#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSE 33-industry rotation screener -> TOPIX-17 ETF execution universe + CASH.

Research architecture
---------------------
1. Build 33 industry proxy baskets from config/industry_constituents.csv.
2. Score each industry on technical trend, relative strength / breadth,
   macro fit, optional fundamentals and risk.
3. Map the 33 industry scores to the 17 tradable TOPIX-17 ETFs.
4. Blend each ETF's underlying-industry score with the ETF's own market data.
5. Add CASH as an 18th candidate. CASH wins when absolute market trend,
   breadth and risk conditions are poor enough.

Free data implementation
------------------------
- Prices / volume: Yahoo Finance via yfinance
- Macro: FRED CSV endpoints
- Fundamentals: optional config/fundamentals.csv (never guessed)

Outputs
-------
- data/industry_latest.csv
- data/industry_history.csv
- data/etf_latest.csv
- data/etf_history.csv
- data/macro.json

The 33-industry series are independent proxy baskets, not official JPX index values.
This is a research/screening tool, not an order execution system.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


BENCHMARK = "1306.T"

TOPIX17_ETFS = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}

INDUSTRY_TO_TOPIX17 = {
    "水産・農林業": "食品",
    "鉱業": "エネルギー資源",
    "建設業": "建設・資材",
    "食料品": "食品",
    "繊維製品": "素材・化学",
    "パルプ・紙": "素材・化学",
    "化学": "素材・化学",
    "医薬品": "医薬品",
    "石油・石炭製品": "エネルギー資源",
    "ゴム製品": "自動車・輸送機",
    "ガラス・土石製品": "建設・資材",
    "鉄鋼": "鉄鋼・非鉄",
    "非鉄金属": "鉄鋼・非鉄",
    "金属製品": "鉄鋼・非鉄",
    "機械": "機械",
    "電気機器": "電機・精密",
    "輸送用機器": "自動車・輸送機",
    "精密機器": "電機・精密",
    "その他製品": "情報通信・サービスその他",
    "電気・ガス業": "電力・ガス",
    "陸運業": "運輸・物流",
    "海運業": "運輸・物流",
    "空運業": "運輸・物流",
    "倉庫・運輸関連業": "運輸・物流",
    "情報・通信業": "情報通信・サービスその他",
    "卸売業": "商社・卸売",
    "小売業": "小売",
    "銀行業": "銀行",
    "証券、商品先物取引業": "金融（除く銀行）",
    "保険業": "金融（除く銀行）",
    "その他金融業": "金融（除く銀行）",
    "不動産業": "不動産",
    "サービス業": "情報通信・サービスその他",
}

TOPIX17_TO_ETF = {group: ticker for ticker, group in TOPIX17_ETFS.items()}
TOPIX17_TO_INDUSTRIES = {
    group: [industry for industry, mapped in INDUSTRY_TO_TOPIX17.items() if mapped == group]
    for group in TOPIX17_TO_ETF
}

SENSITIVITY = {
    "食品": {"rates": -0.2, "growth": -0.2, "inflation": -0.4, "oil": -0.4, "usdjpy": -0.3},
    "エネルギー資源": {"rates": 0.1, "growth": 0.5, "inflation": 0.7, "oil": 1.0, "usdjpy": 0.2},
    "建設・資材": {"rates": -0.3, "growth": 0.7, "inflation": 0.2, "oil": -0.2, "usdjpy": -0.1},
    "素材・化学": {"rates": -0.1, "growth": 0.8, "inflation": 0.3, "oil": -0.4, "usdjpy": 0.2},
    "医薬品": {"rates": -0.3, "growth": -0.3, "inflation": -0.1, "oil": -0.1, "usdjpy": 0.2},
    "自動車・輸送機": {"rates": -0.2, "growth": 0.9, "inflation": -0.2, "oil": -0.4, "usdjpy": 0.8},
    "鉄鋼・非鉄": {"rates": 0.1, "growth": 1.0, "inflation": 0.5, "oil": -0.1, "usdjpy": 0.3},
    "機械": {"rates": -0.2, "growth": 1.0, "inflation": 0.0, "oil": -0.1, "usdjpy": 0.6},
    "電機・精密": {"rates": -0.4, "growth": 0.9, "inflation": -0.1, "oil": -0.1, "usdjpy": 0.5},
    "情報通信・サービスその他": {"rates": -0.5, "growth": 0.6, "inflation": -0.2, "oil": -0.1, "usdjpy": 0.0},
    "電力・ガス": {"rates": -0.5, "growth": -0.2, "inflation": 0.2, "oil": -0.7, "usdjpy": -0.3},
    "運輸・物流": {"rates": -0.3, "growth": 0.5, "inflation": -0.3, "oil": -0.9, "usdjpy": -0.3},
    "商社・卸売": {"rates": 0.1, "growth": 0.8, "inflation": 0.5, "oil": 0.6, "usdjpy": 0.4},
    "小売": {"rates": -0.3, "growth": 0.4, "inflation": -0.7, "oil": -0.4, "usdjpy": -0.5},
    "銀行": {"rates": 1.0, "growth": 0.5, "inflation": 0.2, "oil": -0.1, "usdjpy": 0.0},
    "金融（除く銀行）": {"rates": 0.4, "growth": 0.6, "inflation": 0.1, "oil": 0.0, "usdjpy": 0.0},
    "不動産": {"rates": -1.0, "growth": 0.5, "inflation": 0.2, "oil": -0.1, "usdjpy": -0.1},
}

FRED = {
    "jgb10y": "IRLTLT01JPM156N",
    "cpi": "JPNCPIALLMINMEI",
    "industrial_prod": "JPNPROINDMISMEI",
    "usdjpy": "DEXJPUS",
    "brent": "DCOILBRENTEU",
    "vix": "VIXCLS",
}

FUNDAMENTAL_COLUMNS = ["pe", "pb", "dividend_yield", "roe", "earnings_growth", "revision_3m"]


@dataclass
class MacroState:
    rates: float
    growth: float
    inflation: float
    oil: float
    usdjpy: float
    risk_off: float
    regime: str


@dataclass
class MarketState:
    ret_1m: float
    ret_3m: float
    ret_6m: float
    ma50_gap: float
    ma200_gap: float
    breadth_50d: float
    breadth_200d: float
    breadth_positive_3m: float
    risk_on_score: float
    cash_score: float


def winsor_z(s: pd.Series, clip: float = 2.5) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).clip(-clip, clip)


def z_to_100(z: pd.Series) -> pd.Series:
    return 100.0 / (1.0 + np.exp(-1.15 * z))


def directional_score(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 50.0
    return float(np.clip(50.0 + 50.0 * np.tanh(value * scale), 0.0, 100.0))


def trailing_return(close: pd.Series, n: int) -> float:
    close = close.dropna()
    if len(close) <= n:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-n - 1] - 1)


def max_drawdown(close: pd.Series, n: int = 126) -> float:
    x = close.dropna().tail(n)
    if x.empty:
        return np.nan
    return float((x / x.cummax() - 1).min())


def rsi(close: pd.Series, n: int = 14) -> float:
    close = close.dropna()
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    value = 100 - 100 / (1 + up / (down + 1e-12))
    return float(value.iloc[-1])


def absolute_score(ret_3m: float, ret_6m: float, ma50_gap: float, ma200_gap: float) -> float:
    parts = [
        0.25 * directional_score(ret_3m, 5.0),
        0.20 * directional_score(ret_6m, 3.0),
        0.20 * directional_score(ma50_gap, 9.0),
        0.35 * directional_score(ma200_gap, 7.0),
    ]
    return float(np.clip(sum(parts), 0.0, 100.0))


def _ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            return raw[ticker].copy()
        if ticker in raw.columns.get_level_values(1):
            return raw.xs(ticker, axis=1, level=1).copy()
    if len(raw.columns) and ticker == str(getattr(raw, "name", "")):
        return raw.copy()
    raise KeyError(ticker)


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def download_prices(tickers: list[str], period: str = "2y", batch_size: int = 45, retries: int = 3) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for batch in chunks(sorted(set(tickers)), batch_size):
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    tickers=batch,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    timeout=35,
                )
                for ticker in batch:
                    try:
                        if len(batch) == 1 and not isinstance(raw.columns, pd.MultiIndex):
                            df = raw.copy()
                        else:
                            df = _ticker_frame(raw, ticker)
                        df = df.dropna(how="all")
                        if "Close" in df and len(df["Close"].dropna()) >= 80:
                            out[ticker] = df
                    except Exception:
                        continue
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        if last_error is not None:
            print(f"WARNING: batch price download failed: {last_error}")
    return out


def load_constituents(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"industry", "code", "name", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"constituent file missing columns: {sorted(missing)}")
    frame["industry"] = frame["industry"].astype(str)
    frame["code"] = frame["code"].astype(str).str.replace(r"\.0$", "", regex=True)
    frame["ticker"] = frame["code"].where(frame["code"].str.endswith(".T"), frame["code"] + ".T")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(1.0).clip(lower=0.0)
    unknown = sorted(set(frame["industry"]) - set(INDUSTRY_TO_TOPIX17))
    if unknown:
        raise ValueError(f"unknown TSE industries in config: {unknown}")
    return frame


def build_industry_proxy(group: pd.DataFrame, prices: Dict[str, pd.DataFrame], benchmark: pd.Series) -> Optional[dict]:
    members = []
    for row in group.itertuples(index=False):
        if row.ticker not in prices:
            continue
        close = prices[row.ticker]["Close"].dropna()
        if len(close) < 126:
            continue
        members.append((row.ticker, close, float(row.weight)))
    if len(members) < 2:
        return None

    returns = pd.concat({ticker: close.pct_change() for ticker, close, _ in members}, axis=1)
    config_weights = pd.Series({ticker: weight for ticker, _, weight in members}, dtype=float)
    config_weights = config_weights / config_weights.sum()
    valid_weight = returns.notna().mul(config_weights, axis=1)
    weighted_ret = returns.fillna(0).mul(config_weights, axis=1).sum(axis=1)
    daily_weight_sum = valid_weight.sum(axis=1).replace(0, np.nan)
    proxy_ret = (weighted_ret / daily_weight_sum).dropna()
    proxy = (1.0 + proxy_ret).cumprod()
    if len(proxy) < 126:
        return None

    aligned_benchmark = benchmark.reindex(proxy.index).ffill().dropna()
    proxy = proxy.reindex(aligned_benchmark.index).dropna()
    aligned_benchmark = aligned_benchmark.reindex(proxy.index)

    ret_1m = trailing_return(proxy, 21)
    ret_3m = trailing_return(proxy, 63)
    ret_6m = trailing_return(proxy, 126)
    b1 = trailing_return(aligned_benchmark, 21)
    b3 = trailing_return(aligned_benchmark, 63)
    b6 = trailing_return(aligned_benchmark, 126)
    ma50 = proxy.rolling(50).mean().iloc[-1]
    ma200 = proxy.rolling(200).mean().iloc[-1] if len(proxy) >= 200 else np.nan
    ma50_gap = float(proxy.iloc[-1] / ma50 - 1)
    ma200_gap = float(proxy.iloc[-1] / ma200 - 1) if pd.notna(ma200) else np.nan

    member_stats = []
    for _, close, _ in members:
        m50 = close.rolling(50).mean().iloc[-1]
        m200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        m3 = trailing_return(close, 63)
        bm3 = trailing_return(benchmark.reindex(close.index).ffill().dropna(), 63)
        member_stats.append({
            "above50": float(close.iloc[-1] > m50) if pd.notna(m50) else np.nan,
            "above200": float(close.iloc[-1] > m200) if pd.notna(m200) else np.nan,
            "positive3m": float(m3 > 0) if pd.notna(m3) else np.nan,
            "beat_topix3m": float(m3 > bm3) if pd.notna(m3) and pd.notna(bm3) else np.nan,
        })
    mstats = pd.DataFrame(member_stats)

    return {
        "price_date": proxy.index[-1].date().isoformat(),
        "member_count": len(members),
        "configured_count": len(group),
        "coverage": len(members) / max(len(group), 1),
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "rs_1m": ret_1m - b1,
        "rs_3m": ret_3m - b3,
        "rs_6m": ret_6m - b6,
        "rotation_accel": (ret_1m - b1) - (ret_3m - b3) / 3.0,
        "ma50_gap": ma50_gap,
        "ma200_gap": ma200_gap,
        "rsi14": rsi(proxy),
        "vol20": float(proxy.pct_change().tail(20).std(ddof=0) * np.sqrt(252)),
        "maxdd_6m": max_drawdown(proxy),
        "breadth_50d": float(mstats["above50"].mean()),
        "breadth_200d": float(mstats["above200"].mean()),
        "breadth_positive_3m": float(mstats["positive3m"].mean()),
        "breadth_beat_topix_3m": float(mstats["beat_topix3m"].mean()),
        "absolute_score": absolute_score(ret_3m, ret_6m, ma50_gap, ma200_gap),
    }


def calc_industry_market_factors(constituents: pd.DataFrame, prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if BENCHMARK not in prices:
        raise RuntimeError(f"benchmark {BENCHMARK} price data unavailable")
    benchmark = prices[BENCHMARK]["Close"].dropna()
    rows = []
    for industry, group in constituents.groupby("industry", sort=False):
        metrics = build_industry_proxy(group, prices, benchmark)
        if metrics is None:
            print(f"WARNING: insufficient data for industry {industry}")
            continue
        metrics["industry"] = industry
        metrics["topix17_group"] = INDUSTRY_TO_TOPIX17[industry]
        metrics["execution_etf_code"] = TOPIX17_TO_ETF[metrics["topix17_group"]].replace(".T", "")
        n_group = len(TOPIX17_TO_INDUSTRIES[metrics["topix17_group"]])
        metrics["execution_purity"] = 100.0 / math.sqrt(n_group)
        rows.append(metrics)

    x = pd.DataFrame(rows).set_index("industry")
    if len(x) < 28:
        raise RuntimeError(f"only {len(x)} / 33 industries had usable proxy data")

    technical_z = (
        0.18 * winsor_z(x["ret_1m"])
        + 0.27 * winsor_z(x["ret_3m"])
        + 0.20 * winsor_z(x["ret_6m"])
        + 0.15 * winsor_z(x["ma50_gap"])
        + 0.20 * winsor_z(x["ma200_gap"])
    )
    x["technical_score"] = z_to_100(technical_z)

    rotation_z = (
        0.18 * winsor_z(x["rs_1m"])
        + 0.24 * winsor_z(x["rs_3m"])
        + 0.13 * winsor_z(x["rs_6m"])
        + 0.15 * winsor_z(x["rotation_accel"])
        + 0.10 * winsor_z(x["breadth_50d"])
        + 0.10 * winsor_z(x["breadth_200d"])
        + 0.10 * winsor_z(x["breadth_beat_topix_3m"])
    )
    x["rotation_score"] = z_to_100(rotation_z)
    risk_z = -0.60 * winsor_z(x["vol20"]) + 0.40 * winsor_z(x["maxdd_6m"])
    x["risk_score"] = z_to_100(risk_z)
    return x


def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=25, headers={"User-Agent": "tse-sector-rotation/2.0"})
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    frame.iloc[:, 0] = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    frame.iloc[:, 1] = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return frame.dropna().set_index(frame.columns[0])[frame.columns[1]]


def change_signal(s: pd.Series, short: int, long: int, scale: float = 5.0) -> float:
    s = s.dropna()
    if len(s) <= long:
        return 0.0
    base_short = s.iloc[-short - 1]
    base_long = s.iloc[-long - 1]
    if base_short == 0 or base_long == 0:
        return 0.0
    short_change = s.iloc[-1] / base_short - 1
    long_change = s.iloc[-1] / base_long - 1
    return float(np.tanh(scale * (0.65 * short_change + 0.35 * long_change)))


def level_change_signal(s: pd.Series, short: int, long: int, scale: float = 1.0) -> float:
    s = s.dropna()
    if len(s) <= long:
        return 0.0
    short_change = s.iloc[-1] - s.iloc[-short - 1]
    long_change = s.iloc[-1] - s.iloc[-long - 1]
    return float(np.tanh(scale * (0.65 * short_change + 0.35 * long_change)))


def calc_macro_state() -> MacroState:
    data: Dict[str, pd.Series] = {}
    for key, series_id in FRED.items():
        try:
            data[key] = fred_series(series_id)
        except Exception as exc:
            print(f"WARNING: FRED {key} unavailable: {exc}")
            data[key] = pd.Series(dtype=float)

    rates = level_change_signal(data["jgb10y"], 3, 12, scale=1.3)
    growth = change_signal(data["industrial_prod"], 3, 12, scale=6.0)
    inflation = change_signal(data["cpi"], 3, 12, scale=8.0)
    oil = change_signal(data["brent"], 21, 126, scale=3.0)
    usdjpy = change_signal(data["usdjpy"], 21, 126, scale=4.0)
    vix = data["vix"].dropna()
    risk_off = float(np.tanh((vix.iloc[-1] / vix.tail(63).median() - 1) * 2.0)) if len(vix) >= 63 else 0.0

    if growth >= 0.15 and inflation < 0.15:
        regime = "Goldilocks / expansion"
    elif growth >= 0.15 and inflation >= 0.15:
        regime = "Reflation / late-cycle"
    elif growth < -0.15 and inflation >= 0.15:
        regime = "Stagflation risk"
    elif growth < -0.15 and inflation < 0.15:
        regime = "Disinflation / slowdown"
    else:
        regime = "Mixed / transition"
    return MacroState(rates, growth, inflation, oil, usdjpy, risk_off, regime)


def calc_macro_fit(industry_index: pd.Index, macro: MacroState) -> pd.Series:
    factors = {
        "rates": macro.rates,
        "growth": macro.growth,
        "inflation": macro.inflation,
        "oil": macro.oil,
        "usdjpy": macro.usdjpy,
    }
    raw = {}
    for industry in industry_index:
        group = INDUSTRY_TO_TOPIX17[industry]
        sensitivity = SENSITIVITY[group]
        value = sum(sensitivity[k] * factors[k] for k in factors)
        value -= 0.30 * max(macro.risk_off, 0) * abs(sensitivity["growth"])
        raw[industry] = value
    return z_to_100(winsor_z(pd.Series(raw)))


def load_fundamentals(path: Optional[Path], index: pd.Index) -> pd.DataFrame:
    neutral = pd.DataFrame(index=index, data={"fundamental_score": 50.0, "fundamental_available": False})
    if path is None or not path.exists():
        return neutral
    frame = pd.read_csv(path, comment="#")
    if "industry" not in frame.columns:
        return neutral
    frame["industry"] = frame["industry"].astype(str)
    frame = frame.set_index("industry").reindex(index)
    for col in FUNDAMENTAL_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    score_z = (
        -0.22 * winsor_z(frame["pe"])
        - 0.12 * winsor_z(frame["pb"])
        + 0.12 * winsor_z(frame["dividend_yield"])
        + 0.18 * winsor_z(frame["roe"])
        + 0.18 * winsor_z(frame["earnings_growth"])
        + 0.18 * winsor_z(frame["revision_3m"])
    )
    available = frame[FUNDAMENTAL_COLUMNS].notna().sum(axis=1) >= 3
    result = pd.DataFrame(index=index)
    result["fundamental_score"] = z_to_100(score_z)
    result["fundamental_available"] = available
    result.loc[~available, "fundamental_score"] = 50.0
    return result


def weights_for(regime: str, fundamentals_available: bool) -> Dict[str, float]:
    if regime == "Stagflation risk":
        w = {"technical": .20, "rotation": .20, "macro": .30, "fundamental": .12, "risk": .08, "absolute": .10}
    elif regime == "Disinflation / slowdown":
        w = {"technical": .19, "rotation": .18, "macro": .25, "fundamental": .15, "risk": .10, "absolute": .13}
    else:
        w = {"technical": .22, "rotation": .22, "macro": .22, "fundamental": .13, "risk": .08, "absolute": .13}
    if not fundamentals_available:
        missing = w["fundamental"]
        w["fundamental"] = 0.0
        live_total = sum(w.values())
        for key in w:
            if key != "fundamental":
                w[key] += missing * w[key] / live_total
    return w


def label_industry(row: pd.Series) -> str:
    if row["final_score"] >= 72 and row["absolute_score"] >= 55 and row["breadth_200d"] >= 0.50:
        return "LEADING"
    if row["final_score"] >= 62:
        return "IMPROVING"
    if row["final_score"] >= 48:
        return "NEUTRAL"
    return "LAGGING"


def score_industries(market: pd.DataFrame, macro: MacroState, fundamentals_path: Optional[Path]) -> tuple[pd.DataFrame, Dict[str, float]]:
    x = market.copy()
    x["macro_score"] = calc_macro_fit(x.index, macro)
    x = x.join(load_fundamentals(fundamentals_path, x.index))
    has_fundamentals = bool(x["fundamental_available"].any())
    weights = weights_for(macro.regime, has_fundamentals)

    x["relative_score"] = (
        weights["technical"] * x["technical_score"]
        + weights["rotation"] * x["rotation_score"]
        + weights["macro"] * x["macro_score"]
        + weights["fundamental"] * x["fundamental_score"]
        + weights["risk"] * x["risk_score"]
        + weights["absolute"] * x["absolute_score"]
    )

    x["penalty"] = 0.0
    x.loc[x["ma200_gap"] < -0.05, "penalty"] += 7.0
    x.loc[x["breadth_200d"] < 0.30, "penalty"] += 5.0
    x.loc[x["coverage"] < 0.70, "penalty"] += 4.0
    x.loc[x["rsi14"] > 82, "penalty"] += 4.0
    if macro.risk_off > 0.35:
        x.loc[x["vol20"] > x["vol20"].quantile(0.75), "penalty"] += 4.0

    x["final_score"] = (0.60 * x["relative_score"] + 0.40 * x["absolute_score"] - x["penalty"]).clip(0, 100)
    x["signal"] = x.apply(label_industry, axis=1)
    x["rank"] = x["final_score"].rank(ascending=False, method="min").astype(int)

    ordered = [
        "rank", "topix17_group", "execution_etf_code", "execution_purity", "price_date",
        "member_count", "configured_count", "coverage", "final_score", "signal", "relative_score",
        "absolute_score", "technical_score", "rotation_score", "macro_score", "fundamental_score",
        "fundamental_available", "risk_score", "penalty", "ret_1m", "ret_3m", "ret_6m",
        "rs_1m", "rs_3m", "rs_6m", "rotation_accel", "rsi14", "ma50_gap", "ma200_gap",
        "breadth_50d", "breadth_200d", "breadth_positive_3m", "breadth_beat_topix_3m",
        "vol20", "maxdd_6m",
    ]
    return x.sort_values("final_score", ascending=False)[ordered], weights


def calc_etf_market_factors(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    benchmark = prices[BENCHMARK]["Close"].dropna()
    rows = []
    for ticker, group in TOPIX17_ETFS.items():
        if ticker not in prices:
            continue
        df = prices[ticker].dropna(how="all")
        close = df["Close"].dropna()
        volume = df["Volume"].reindex(close.index).fillna(0) if "Volume" in df else pd.Series(0, index=close.index)
        ret1 = trailing_return(close, 21)
        ret3 = trailing_return(close, 63)
        ret6 = trailing_return(close, 126)
        b1 = trailing_return(benchmark.reindex(close.index).ffill().dropna(), 21)
        b3 = trailing_return(benchmark.reindex(close.index).ffill().dropna(), 63)
        b6 = trailing_return(benchmark.reindex(close.index).ffill().dropna(), 126)
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        ma50_gap = float(close.iloc[-1] / ma50 - 1)
        ma200_gap = float(close.iloc[-1] / ma200 - 1) if pd.notna(ma200) else np.nan
        rows.append({
            "ticker": ticker,
            "code": ticker.replace(".T", ""),
            "etf_group": group,
            "price_date": close.index[-1].date().isoformat(),
            "last": float(close.iloc[-1]),
            "ret_1m": ret1,
            "ret_3m": ret3,
            "ret_6m": ret6,
            "rs_1m": ret1 - b1,
            "rs_3m": ret3 - b3,
            "rs_6m": ret6 - b6,
            "ma50_gap": ma50_gap,
            "ma200_gap": ma200_gap,
            "rsi14": rsi(close),
            "vol20": float(close.pct_change().tail(20).std(ddof=0) * np.sqrt(252)),
            "maxdd_6m": max_drawdown(close),
            "avg_turnover_yen": float((close * volume).tail(20).mean()),
            "absolute_score": absolute_score(ret3, ret6, ma50_gap, ma200_gap),
        })
    x = pd.DataFrame(rows).set_index("ticker")
    if len(x) < 14:
        raise RuntimeError(f"only {len(x)} / 17 TOPIX-17 ETFs had usable data")
    tech_z = (
        0.18 * winsor_z(x["ret_1m"])
        + 0.27 * winsor_z(x["ret_3m"])
        + 0.15 * winsor_z(x["ret_6m"])
        + 0.15 * winsor_z(x["rs_3m"])
        + 0.10 * winsor_z(x["ma50_gap"])
        + 0.15 * winsor_z(x["ma200_gap"])
    )
    x["etf_technical_score"] = z_to_100(tech_z)
    risk_z = -0.65 * winsor_z(x["vol20"]) + 0.35 * winsor_z(x["maxdd_6m"])
    x["etf_risk_score"] = z_to_100(risk_z)
    x["liquidity_score"] = z_to_100(winsor_z(np.log1p(x["avg_turnover_yen"])))
    return x


def calc_market_state(prices: Dict[str, pd.DataFrame], etf_market: pd.DataFrame, macro: MacroState) -> MarketState:
    close = prices[BENCHMARK]["Close"].dropna()
    ret1 = trailing_return(close, 21)
    ret3 = trailing_return(close, 63)
    ret6 = trailing_return(close, 126)
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    gap50 = float(close.iloc[-1] / ma50 - 1)
    gap200 = float(close.iloc[-1] / ma200 - 1) if pd.notna(ma200) else np.nan
    breadth50 = float((etf_market["ma50_gap"] > 0).mean())
    breadth200 = float((etf_market["ma200_gap"] > 0).mean())
    breadth3 = float((etf_market["ret_3m"] > 0).mean())

    market_absolute = absolute_score(ret3, ret6, gap50, gap200)
    breadth_score = 100.0 * (0.30 * breadth50 + 0.45 * breadth200 + 0.25 * breadth3)
    risk_on = float(np.clip(0.68 * market_absolute + 0.32 * breadth_score - 12.0 * max(macro.risk_off, 0), 0, 100))
    cash = float(np.clip(100.0 - risk_on + 12.0 * max(macro.risk_off, 0), 0, 100))
    return MarketState(ret1, ret3, ret6, gap50, gap200, breadth50, breadth200, breadth3, risk_on, cash)


def score_etfs(industry: pd.DataFrame, etf_market: pd.DataFrame, market_state: MarketState) -> pd.DataFrame:
    x = etf_market.copy()
    underlying_scores = {}
    purity_scores = {}
    for ticker, row in x.iterrows():
        group = row["etf_group"]
        subset = industry[industry["topix17_group"] == group]
        if subset.empty:
            underlying_scores[ticker] = 50.0
        else:
            underlying_scores[ticker] = float(0.75 * subset["final_score"].mean() + 0.25 * subset["final_score"].max())
        purity_scores[ticker] = 100.0 / math.sqrt(max(len(TOPIX17_TO_INDUSTRIES[group]), 1))
    x["underlying_score"] = pd.Series(underlying_scores)
    x["purity_score"] = pd.Series(purity_scores)
    x["relative_score"] = (
        0.54 * x["underlying_score"]
        + 0.24 * x["etf_technical_score"]
        + 0.12 * x["etf_risk_score"]
        + 0.10 * x["liquidity_score"]
    )
    x["penalty"] = 0.0
    x.loc[x["ma200_gap"] < -0.05, "penalty"] += 8.0
    x.loc[x["rsi14"] > 82, "penalty"] += 5.0
    x.loc[x["avg_turnover_yen"] < 20_000_000, "penalty"] += 5.0
    x["final_score"] = (0.60 * x["relative_score"] + 0.40 * x["absolute_score"] - x["penalty"]).clip(0, 100)

    top_etf_score = float(x["final_score"].max()) if len(x) else 0.0
    weakness_boost = max(0.0, 62.0 - top_etf_score) * 0.80
    cash_score = float(np.clip(market_state.cash_score + weakness_boost, 0, 100))
    asof = max(x["price_date"].astype(str))
    cash_row = pd.DataFrame(index=["CASH"], data={
        "code": "CASH",
        "etf_group": "現金 / NO TRADE",
        "price_date": asof,
        "last": 1.0,
        "ret_1m": 0.0,
        "ret_3m": 0.0,
        "ret_6m": 0.0,
        "rs_1m": np.nan,
        "rs_3m": np.nan,
        "rs_6m": np.nan,
        "ma50_gap": 0.0,
        "ma200_gap": 0.0,
        "rsi14": 50.0,
        "vol20": 0.0,
        "maxdd_6m": 0.0,
        "avg_turnover_yen": np.nan,
        "absolute_score": 100.0 - market_state.risk_on_score,
        "etf_technical_score": 100.0 - market_state.risk_on_score,
        "etf_risk_score": 100.0,
        "liquidity_score": 100.0,
        "underlying_score": 100.0 - market_state.risk_on_score,
        "purity_score": 100.0,
        "relative_score": cash_score,
        "penalty": 0.0,
        "final_score": cash_score,
    })
    x = pd.concat([x, cash_row], axis=0, sort=False)
    x["rank"] = x["final_score"].rank(ascending=False, method="min").astype(int)

    def signal(row: pd.Series) -> str:
        if row.name == "CASH":
            return "NO TRADE / CASH"
        if row["final_score"] >= 72 and row["absolute_score"] >= 58 and row["ma200_gap"] >= 0:
            return "BUY"
        if row["final_score"] >= 62 and row["absolute_score"] >= 50:
            return "WATCH"
        if row["final_score"] >= 48:
            return "NEUTRAL"
        return "AVOID"

    x["signal"] = x.apply(signal, axis=1)
    ordered = [
        "rank", "code", "etf_group", "price_date", "last", "final_score", "signal", "relative_score",
        "absolute_score", "underlying_score", "purity_score", "etf_technical_score", "etf_risk_score",
        "liquidity_score", "penalty", "ret_1m", "ret_3m", "ret_6m", "rs_1m", "rs_3m", "rs_6m",
        "rsi14", "ma50_gap", "ma200_gap", "vol20", "maxdd_6m", "avg_turnover_yen",
    ]
    return x.sort_values("final_score", ascending=False)[ordered]


def append_history(latest: pd.DataFrame, path: Path, asof: str, index_name: str) -> None:
    snapshot = latest.reset_index().rename(columns={"index": index_name})
    snapshot.insert(0, "asof", asof)
    if path.exists():
        old = pd.read_csv(path)
        if "asof" in old.columns:
            old = old[old["asof"].astype(str) != asof]
        combined = pd.concat([old, snapshot], ignore_index=True, sort=False)
    else:
        combined = snapshot
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def save_outputs(
    industry: pd.DataFrame,
    etf: pd.DataFrame,
    macro: MacroState,
    market: MarketState,
    weights: Dict[str, float],
    data_dir: Path,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    industry_latest = data_dir / "industry_latest.csv"
    industry_history = data_dir / "industry_history.csv"
    etf_latest = data_dir / "etf_latest.csv"
    etf_history = data_dir / "etf_history.csv"
    macro_path = data_dir / "macro.json"

    industry.to_csv(industry_latest, encoding="utf-8-sig")
    etf.to_csv(etf_latest, encoding="utf-8-sig")
    asof = str(max(industry["price_date"].max(), etf["price_date"].max()))
    append_history(industry, industry_history, asof, "industry")
    append_history(etf, etf_history, asof, "ticker")

    top = etf.iloc[0]
    allocation = {
        "code": str(top["code"]),
        "name": str(top["etf_group"]),
        "signal": str(top["signal"]),
        "score": round(float(top["final_score"]), 3),
        "is_cash": bool(top.name == "CASH"),
    }
    payload = {
        "asof": asof,
        "macro": asdict(macro),
        "market": asdict(market),
        "industry_weights": weights,
        "fundamentals_active": bool(industry["fundamental_available"].any()),
        "allocation_decision": allocation,
        "method": {
            "industry_series": "independent equal/config-weight proxy baskets; not official JPX index values",
            "etf_universe": "17 TOPIX-17 ETFs + CASH",
            "etf_final_score": "60% relative/underlying composite + 40% absolute trend - penalties",
            "cash_rule": "market absolute trend + ETF breadth + macro risk-off + weak-top-ETF boost",
        },
    }
    macro_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_screen(constituents_path: Path, fundamentals_path: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame, MacroState, MarketState, Dict[str, float]]:
    constituents = load_constituents(constituents_path)
    tickers = constituents["ticker"].tolist() + list(TOPIX17_ETFS) + [BENCHMARK]
    prices = download_prices(tickers)
    if BENCHMARK not in prices:
        raise RuntimeError(f"benchmark {BENCHMARK} unavailable")

    industry_market = calc_industry_market_factors(constituents, prices)
    macro = calc_macro_state()
    industry, weights = score_industries(industry_market, macro, fundamentals_path)
    etf_market = calc_etf_market_factors(prices)
    market_state = calc_market_state(prices, etf_market, macro)
    etf = score_etfs(industry, etf_market, market_state)
    return industry, etf, macro, market_state, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constituents", type=Path, default=Path("config/industry_constituents.csv"))
    parser.add_argument("--fundamentals", type=Path, default=Path("config/fundamentals.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    industry, etf, macro, market, weights = run_screen(args.constituents, args.fundamentals)
    save_outputs(industry, etf, macro, market, weights, args.data_dir)

    print(f"Macro regime: {macro.regime}")
    print(f"Market risk-on score: {market.risk_on_score:.1f}")
    print(f"CASH score: {etf.loc['CASH', 'final_score']:.1f}")
    print("\n=== Top 33-industry proxies ===")
    print(industry[["rank", "topix17_group", "final_score", "signal", "absolute_score", "rs_3m", "breadth_200d"]].head(12).round(3).to_string())
    print("\n=== TOPIX-17 ETFs + CASH ===")
    print(etf[["rank", "code", "etf_group", "final_score", "signal", "relative_score", "absolute_score", "underlying_score", "ret_3m", "ma200_gap"]].round(3).to_string())


if __name__ == "__main__":
    main()
