#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOPIX-17 sector rotation screener.

Free-data implementation:
- Prices/volume: Yahoo Finance via yfinance
- Macro: FRED CSV endpoints
- Fundamentals: optional config/fundamentals.csv

Outputs:
- data/latest.csv
- data/history.csv
- data/macro.json

This is a research/screening tool, not an order execution system.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


ETF_NAMES = {
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

BENCHMARK = "1306.T"

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


@dataclass
class MacroState:
    rates: float
    growth: float
    inflation: float
    oil: float
    usdjpy: float
    risk_off: float
    regime: str


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


def _ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            return raw[ticker].copy()
        if ticker in raw.columns.get_level_values(1):
            return raw.xs(ticker, axis=1, level=1).copy()
    raise KeyError(ticker)


def download_prices(period: str = "2y", retries: int = 3) -> Dict[str, pd.DataFrame]:
    tickers = list(ETF_NAMES) + [BENCHMARK]
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=30,
            )
            out: Dict[str, pd.DataFrame] = {}
            for ticker in tickers:
                try:
                    df = _ticker_frame(raw, ticker).dropna(how="all")
                    if "Close" in df and len(df["Close"].dropna()) >= 80:
                        out[ticker] = df
                except Exception:
                    continue
            if BENCHMARK in out and len(out) >= 10:
                return out
            raise RuntimeError(f"insufficient price data: {len(out)} tickers")
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"price download failed after {retries} attempts: {last_error}")


def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    value = 100 - 100 / (1 + up / (down + 1e-12))
    return float(value.iloc[-1])


def max_drawdown(close: pd.Series, n: int = 126) -> float:
    x = close.tail(n)
    return float((x / x.cummax() - 1).min())


def calc_market_factors(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    benchmark = prices[BENCHMARK]["Close"].dropna()

    def trailing_return(close: pd.Series, n: int) -> float:
        if len(close) <= n:
            return np.nan
        return float(close.iloc[-1] / close.iloc[-n - 1] - 1)

    bret = {n: trailing_return(benchmark, n) for n in (21, 63, 126)}
    rows = []
    for ticker, sector in ETF_NAMES.items():
        if ticker not in prices:
            continue
        df = prices[ticker].dropna(how="all")
        close = df["Close"].dropna()
        volume = df["Volume"].reindex(close.index).fillna(0)
        rets = {n: trailing_return(close, n) for n in (21, 63, 126)}
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

        rows.append({
            "ticker": ticker,
            "sector": sector,
            "last": float(close.iloc[-1]),
            "price_date": close.index[-1].date().isoformat(),
            "ret_1m": rets[21],
            "ret_3m": rets[63],
            "ret_6m": rets[126],
            "rs_1m": rets[21] - bret[21],
            "rs_3m": rets[63] - bret[63],
            "rs_6m": rets[126] - bret[126],
            "rotation_accel": (rets[21] - bret[21]) - (rets[63] - bret[63]) / 3.0,
            "ma50_gap": float(close.iloc[-1] / ma50 - 1),
            "ma200_gap": float(close.iloc[-1] / ma200 - 1) if pd.notna(ma200) else np.nan,
            "rsi14": rsi(close),
            "vol20": float(close.pct_change().tail(20).std(ddof=0) * np.sqrt(252)),
            "maxdd_6m": max_drawdown(close),
            "avg_turnover_yen": float((close * volume).tail(20).mean()),
        })

    x = pd.DataFrame(rows).set_index("ticker")
    if len(x) < 10:
        raise RuntimeError(f"only {len(x)} sector ETFs had usable price data")

    momentum_z = (
        0.18 * winsor_z(x["ret_1m"])
        + 0.27 * winsor_z(x["ret_3m"])
        + 0.20 * winsor_z(x["ret_6m"])
        + 0.15 * winsor_z(x["ma50_gap"])
        + 0.20 * winsor_z(x["ma200_gap"])
    )
    x["technical_score"] = z_to_100(momentum_z)

    rotation_z = (
        0.25 * winsor_z(x["rs_1m"])
        + 0.35 * winsor_z(x["rs_3m"])
        + 0.20 * winsor_z(x["rs_6m"])
        + 0.20 * winsor_z(x["rotation_accel"])
    )
    x["rotation_score"] = z_to_100(rotation_z)

    risk_z = -0.60 * winsor_z(x["vol20"]) + 0.40 * winsor_z(x["maxdd_6m"])
    x["risk_score"] = z_to_100(risk_z)
    x["liquidity_score"] = z_to_100(winsor_z(np.log1p(x["avg_turnover_yen"])))
    return x


def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=25, headers={"User-Agent": "tse-sector-rotation/1.0"})
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
        except Exception:
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


def calc_macro_fit(index: pd.Index, macro: MacroState) -> pd.Series:
    factors = {"rates": macro.rates, "growth": macro.growth, "inflation": macro.inflation, "oil": macro.oil, "usdjpy": macro.usdjpy}
    raw = {}
    for ticker in index:
        sensitivity = SENSITIVITY[ETF_NAMES[ticker]]
        value = sum(sensitivity[k] * factors[k] for k in factors)
        value -= 0.30 * max(macro.risk_off, 0) * abs(sensitivity["growth"])
        raw[ticker] = value
    return z_to_100(winsor_z(pd.Series(raw)))


FUNDAMENTAL_COLUMNS = ["pe", "pb", "dividend_yield", "roe", "earnings_growth", "revision_3m"]


def load_fundamentals(path: Optional[Path], index: pd.Index) -> pd.DataFrame:
    neutral = pd.DataFrame(index=index, data={"fundamental_score": 50.0, "fundamental_available": False})
    if path is None or not path.exists():
        return neutral
    f = pd.read_csv(path, comment="#")
    if "ticker" not in f.columns:
        return neutral
    f["ticker"] = f["ticker"].astype(str).str.replace(r"\.0$", "", regex=True)
    f["ticker"] = f["ticker"].where(f["ticker"].str.endswith(".T"), f["ticker"] + ".T")
    f = f.set_index("ticker").reindex(index)
    for col in FUNDAMENTAL_COLUMNS:
        if col not in f.columns:
            f[col] = np.nan
        f[col] = pd.to_numeric(f[col], errors="coerce")
    score_z = (
        -0.22 * winsor_z(f["pe"]) - 0.12 * winsor_z(f["pb"])
        + 0.12 * winsor_z(f["dividend_yield"]) + 0.18 * winsor_z(f["roe"])
        + 0.18 * winsor_z(f["earnings_growth"]) + 0.18 * winsor_z(f["revision_3m"])
    )
    available = f[FUNDAMENTAL_COLUMNS].notna().sum(axis=1) >= 3
    result = pd.DataFrame(index=index)
    result["fundamental_score"] = z_to_100(score_z)
    result["fundamental_available"] = available
    result.loc[~available, "fundamental_score"] = 50.0
    return result


def weights_for(regime: str, fundamentals_available: bool) -> Dict[str, float]:
    if regime == "Stagflation risk":
        w = {"technical": .22, "rotation": .18, "macro": .32, "fundamental": .13, "risk": .10, "liquidity": .05}
    elif regime == "Disinflation / slowdown":
        w = {"technical": .23, "rotation": .17, "macro": .28, "fundamental": .17, "risk": .10, "liquidity": .05}
    else:
        w = {"technical": .25, "rotation": .20, "macro": .25, "fundamental": .15, "risk": .10, "liquidity": .05}
    if not fundamentals_available:
        missing = w["fundamental"]
        w["fundamental"] = 0.0
        live_total = sum(w.values())
        for key in w:
            if key != "fundamental":
                w[key] += missing * w[key] / live_total
    return w


def label_signal(row: pd.Series) -> str:
    score = row["final_score"]
    if score >= 72 and row["ma200_gap"] >= 0:
        return "BUY"
    if score >= 62:
        return "WATCH"
    if score >= 50:
        return "NEUTRAL"
    return "AVOID"


def run_screen(fundamentals_path: Optional[Path]) -> tuple[pd.DataFrame, MacroState, Dict[str, float]]:
    prices = download_prices()
    x = calc_market_factors(prices)
    macro = calc_macro_state()
    x["macro_score"] = calc_macro_fit(x.index, macro)
    x = x.join(load_fundamentals(fundamentals_path, x.index))
    has_fundamentals = bool(x["fundamental_available"].any())
    weights = weights_for(macro.regime, has_fundamentals)
    x["raw_score"] = (
        weights["technical"] * x["technical_score"] + weights["rotation"] * x["rotation_score"]
        + weights["macro"] * x["macro_score"] + weights["fundamental"] * x["fundamental_score"]
        + weights["risk"] * x["risk_score"] + weights["liquidity"] * x["liquidity_score"]
    )
    x["penalty"] = 0.0
    x.loc[x["ma200_gap"] < -0.05, "penalty"] += 8.0
    x.loc[x["rsi14"] > 82, "penalty"] += 6.0
    x.loc[x["avg_turnover_yen"] < 20_000_000, "penalty"] += 5.0
    if macro.risk_off > 0.35:
        x.loc[x["vol20"] > x["vol20"].quantile(0.75), "penalty"] += 5.0
    x["final_score"] = (x["raw_score"] - x["penalty"]).clip(0, 100)
    x["signal"] = x.apply(label_signal, axis=1)
    x["rank"] = x["final_score"].rank(ascending=False, method="min").astype(int)
    ordered = [
        "rank", "sector", "price_date", "last", "final_score", "signal", "technical_score",
        "rotation_score", "macro_score", "fundamental_score", "fundamental_available", "risk_score",
        "liquidity_score", "penalty", "ret_1m", "ret_3m", "ret_6m", "rs_1m", "rs_3m", "rs_6m",
        "rotation_accel", "rsi14", "ma50_gap", "ma200_gap", "vol20", "maxdd_6m", "avg_turnover_yen",
    ]
    return x.sort_values("final_score", ascending=False)[ordered], macro, weights


def save_outputs(result: pd.DataFrame, macro: MacroState, weights: Dict[str, float], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    latest = data_dir / "latest.csv"
    history = data_dir / "history.csv"
    macro_path = data_dir / "macro.json"
    result.to_csv(latest, encoding="utf-8-sig")
    snapshot = result.reset_index().rename(columns={"index": "ticker"})
    asof = str(snapshot["price_date"].max())
    snapshot.insert(0, "asof", asof)
    if history.exists():
        old = pd.read_csv(history)
        if "asof" in old.columns:
            old = old[old["asof"].astype(str) != asof]
        combined = pd.concat([old, snapshot], ignore_index=True, sort=False)
    else:
        combined = snapshot
    combined.to_csv(history, index=False, encoding="utf-8-sig")
    payload = {"asof": asof, "macro": asdict(macro), "weights": weights, "fundamentals_active": bool(result["fundamental_available"].any())}
    macro_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fundamentals", type=Path, default=Path("config/fundamentals.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    result, macro, weights = run_screen(args.fundamentals)
    save_outputs(result, macro, weights, args.data_dir)
    show = result[["rank", "sector", "final_score", "signal", "technical_score", "rotation_score", "macro_score", "risk_score", "ret_1m", "ret_3m", "rsi14", "ma200_gap"]].copy()
    print(f"Regime: {macro.regime}")
    print(f"Weights: {weights}")
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
