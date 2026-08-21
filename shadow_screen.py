#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audited 33-industry shadow screener.

This runner is intentionally separate from the frozen primary model and from
`screener.py`'s production outputs. It adds data-quality diagnostics needed for
prospective evaluation of the 33-industry + macro layer:

- audited constituent exclusions (e.g. confirmed delistings),
- missing price tickers and industry coverage,
- concurrent FRED retrieval with per-series status/latest observation,
- explicit partial/unavailable macro labels instead of silently calling missing
  macro data a real "Mixed / transition" regime,
- separate `shadow/` outputs so the layer cannot alter the frozen v1 model.
"""
from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests

import screener as s


def load_exclusions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "industry", "name", "effective_date", "reason", "source"])
    x = pd.read_csv(path, dtype=str).fillna("")
    required = {"ticker", "industry", "name", "effective_date", "reason", "source"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"exclusion file missing columns: {sorted(missing)}")
    return x[list(required)].copy()


def audited_constituents(path: Path, exclusions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    configured = s.load_constituents(path)
    exclusions = load_exclusions(exclusions_path)
    excluded = set(exclusions["ticker"].astype(str))
    active = configured[~configured["ticker"].isin(excluded)].copy()
    if active.empty:
        raise RuntimeError("all constituents excluded")
    return active, exclusions


def fetch_fred_with_meta(key: str, series_id: str, timeout: float) -> tuple[str, pd.Series, dict]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    started = datetime.now(timezone.utc)
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "tse-sector-rotation-shadow/1.0"},
        )
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if frame.shape[1] < 2:
            raise RuntimeError(f"unexpected FRED CSV width {frame.shape[1]}")
        dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
        values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
        series = pd.Series(values.to_numpy(), index=dates, name=key).dropna().sort_index()
        if series.empty:
            raise RuntimeError("empty FRED series")
        meta = {
            "key": key,
            "series_id": series_id,
            "status": "ok",
            "rows": int(len(series)),
            "latest_observation": series.index[-1].date().isoformat(),
            "latest_value": float(series.iloc[-1]),
            "error": None,
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        }
        return key, series, meta
    except Exception as exc:
        meta = {
            "key": key,
            "series_id": series_id,
            "status": "unavailable",
            "rows": 0,
            "latest_observation": None,
            "latest_value": None,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        }
        return key, pd.Series(dtype=float, name=key), meta


def audited_macro(timeout: float = 8.0) -> tuple[s.MacroState, dict]:
    data: Dict[str, pd.Series] = {}
    metadata: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(s.FRED)) as pool:
        futures = {
            pool.submit(fetch_fred_with_meta, key, series_id, timeout): key
            for key, series_id in s.FRED.items()
        }
        for future in as_completed(futures):
            key, series, meta = future.result()
            data[key] = series
            metadata[key] = meta

    for key in s.FRED:
        data.setdefault(key, pd.Series(dtype=float, name=key))
        metadata.setdefault(key, {
            "key": key,
            "series_id": s.FRED[key],
            "status": "unavailable",
            "rows": 0,
            "latest_observation": None,
            "latest_value": None,
            "error": "missing future result",
            "elapsed_seconds": None,
        })

    rates = s.level_change_signal(data["jgb10y"], 3, 12, scale=1.3)
    growth = s.change_signal(data["industrial_prod"], 3, 12, scale=6.0)
    inflation = s.change_signal(data["cpi"], 3, 12, scale=8.0)
    oil = s.change_signal(data["brent"], 21, 126, scale=3.0)
    usdjpy = s.change_signal(data["usdjpy"], 21, 126, scale=4.0)
    vix = data["vix"].dropna()
    risk_off = float(np.tanh((vix.iloc[-1] / vix.tail(63).median() - 1) * 2.0)) if len(vix) >= 63 else 0.0

    ok = [k for k, m in metadata.items() if m["status"] == "ok"]
    unavailable = [k for k in s.FRED if k not in ok]
    critical = {"jgb10y", "cpi", "industrial_prod"}
    critical_ok = critical.issubset(ok)

    if len(ok) == len(s.FRED):
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
        macro_status = "complete"
    elif len(ok) == 0:
        regime = "UNAVAILABLE / neutral fallback"
        macro_status = "unavailable"
    elif not critical_ok:
        regime = f"PARTIAL {len(ok)}/{len(s.FRED)} / critical-data fallback"
        macro_status = "partial_critical_missing"
    else:
        regime = f"PARTIAL {len(ok)}/{len(s.FRED)}"
        macro_status = "partial"

    macro = s.MacroState(rates, growth, inflation, oil, usdjpy, risk_off, regime)
    diagnostics = {
        "status": macro_status,
        "available_count": len(ok),
        "total_count": len(s.FRED),
        "available_keys": sorted(ok),
        "unavailable_keys": sorted(unavailable),
        "critical_keys": sorted(critical),
        "critical_complete": critical_ok,
        "neutral_fallback_used_for_missing_series": bool(unavailable),
        "series": {key: metadata[key] for key in sorted(metadata)},
    }
    return macro, diagnostics


def coverage_diagnostics(constituents: pd.DataFrame, prices: Dict[str, pd.DataFrame]) -> dict:
    expected = sorted(set(constituents["ticker"]))
    missing = sorted(t for t in expected if t not in prices)
    by_industry = []
    for industry, group in constituents.groupby("industry", sort=False):
        configured = sorted(set(group["ticker"]))
        usable = [t for t in configured if t in prices and len(prices[t]["Close"].dropna()) >= 126]
        absent = sorted(set(configured) - set(usable))
        by_industry.append({
            "industry": industry,
            "configured": len(configured),
            "usable": len(usable),
            "coverage": len(usable) / max(len(configured), 1),
            "missing_or_short": absent,
        })
    return {
        "configured_constituent_count": len(expected),
        "downloaded_constituent_count": len(expected) - len(missing),
        "missing_price_tickers": missing,
        "industry_coverage": by_industry,
        "min_industry_coverage": min((row["coverage"] for row in by_industry), default=0.0),
    }


def append_history(latest: pd.DataFrame, path: Path, asof: str, index_name: str) -> None:
    snapshot = latest.reset_index().rename(columns={"index": index_name})
    snapshot.insert(0, "asof", asof)
    if path.exists():
        old = pd.read_csv(path)
        if "asof" in old.columns:
            old = old[old["asof"].astype(str) != asof]
        snapshot = pd.concat([old, snapshot], ignore_index=True, sort=False)
    snapshot.to_csv(path, index=False, encoding="utf-8-sig")


def save_shadow(
    out_dir: Path,
    industry: pd.DataFrame,
    etf: pd.DataFrame,
    macro: s.MacroState,
    market: s.MarketState,
    weights: Dict[str, float],
    macro_diag: dict,
    price_diag: dict,
    exclusions: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    industry.to_csv(out_dir / "industry_latest.csv", encoding="utf-8-sig")
    etf.to_csv(out_dir / "etf_latest.csv", encoding="utf-8-sig")
    asof = str(max(industry["price_date"].max(), etf["price_date"].max()))
    append_history(industry, out_dir / "industry_history.csv", asof, "industry")
    append_history(etf, out_dir / "etf_history.csv", asof, "ticker")

    top = etf.iloc[0]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "asof": asof,
        "role": "experimental_shadow_only",
        "primary_model_affected": False,
        "macro": asdict(macro),
        "macro_diagnostics": macro_diag,
        "market": asdict(market),
        "industry_weights": weights,
        "price_diagnostics": price_diag,
        "exclusions": exclusions.to_dict("records"),
        "top_shadow_etf": {
            "ticker": str(top.name),
            "code": str(top["code"]),
            "name": str(top["etf_group"]),
            "signal": str(top["signal"]),
            "score": float(top["final_score"]),
        },
        "caveat": "Shadow scores must not alter frozen model topix17_1306_core_tilt_v1. Missing macro inputs are neutral fallbacks and are explicitly flagged.",
    }
    (out_dir / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constituents", type=Path, default=Path("config/industry_constituents.csv"))
    ap.add_argument("--exclusions", type=Path, default=Path("config/constituent_exclusions.csv"))
    ap.add_argument("--fundamentals", type=Path, default=Path("config/fundamentals.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("shadow"))
    ap.add_argument("--fred-timeout", type=float, default=8.0)
    args = ap.parse_args()

    constituents, exclusions = audited_constituents(args.constituents, args.exclusions)
    tickers = constituents["ticker"].tolist() + list(s.TOPIX17_ETFS) + [s.BENCHMARK]
    prices = s.download_prices(tickers)
    if s.BENCHMARK not in prices:
        raise RuntimeError(f"benchmark {s.BENCHMARK} unavailable")

    price_diag = coverage_diagnostics(constituents, prices)
    industry_market = s.calc_industry_market_factors(constituents, prices)
    macro, macro_diag = audited_macro(args.fred_timeout)
    industry, weights = s.score_industries(industry_market, macro, args.fundamentals)
    etf_market = s.calc_etf_market_factors(prices)
    market = s.calc_market_state(prices, etf_market, macro)
    etf = s.score_etfs(industry, etf_market, market)
    save_shadow(args.out_dir, industry, etf, macro, market, weights, macro_diag, price_diag, exclusions)

    print(f"Shadow asof: {max(industry['price_date'].max(), etf['price_date'].max())}")
    print(f"Macro status: {macro_diag['status']} ({macro_diag['available_count']}/{macro_diag['total_count']})")
    print(f"Missing constituent prices: {len(price_diag['missing_price_tickers'])}")
    print(f"Minimum industry coverage: {price_diag['min_industry_coverage']:.1%}")
    print("\nTop shadow industries:")
    print(industry[["rank", "topix17_group", "final_score", "signal", "absolute_score", "rs_3m", "coverage"]].head(10).round(3).to_string())
    print("\nShadow TOPIX-17 + CASH:")
    print(etf[["rank", "code", "etf_group", "final_score", "signal", "absolute_score", "underlying_score"]].head(8).round(3).to_string())


if __name__ == "__main__":
    main()
