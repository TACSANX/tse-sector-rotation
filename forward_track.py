#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prospective tracker for frozen TOPIX-17 core+tilt model v1.

This script deliberately separates prospective evidence from historical research.
Once a month-end signal is captured, its target weights are appended to
forward/signals.csv and are never recomputed or overwritten. Portfolio returns
are reconstructed only from those recorded targets and official NEXT FUNDS
NAV series.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, build_factors, fetch_official_series, load_official_prices, perf
from research_broad_core import BROAD, broad_core_target
from research_dual_momentum import absolute_masks, choose, signal_models
from research_portfolio import simulate_self_financing

SIGNAL_COLUMNS = [
    "model_id", "model_version", "signal_month", "signal_date", "execution_date",
    "captured_at_utc", "data_through_date", "late_capture_nav_days",
    "selected_1", "selected_1_score", "selected_1_pass",
    "selected_2", "selected_2_score", "selected_2_pass",
    "selected_3", "selected_3_score", "selected_3_pass",
    "core_weight", "tilt_cash_weight", "portfolio_cash_weight", "target_weights_json",
]


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ["model_id", "version", "freeze_date", "portfolio", "relative_momentum", "absolute_filter", "rebalance"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise RuntimeError(f"frozen spec missing keys: {missing}")
    if spec["status"] != "prospective_tracking":
        raise RuntimeError(f"unexpected frozen model status: {spec['status']}")
    return spec


def load_official_panel() -> tuple[pd.DataFrame, dict]:
    sectors, sector_meta = load_official_prices()
    broad, broad_meta = fetch_official_series(BROAD)
    close = pd.concat([sectors, broad.rename(BROAD)], axis=1).dropna(how="any").sort_index()
    if close.empty:
        raise RuntimeError("official panel is empty")
    return close, {"sector_metadata": sector_meta, "broad_metadata": broad_meta}


def completed_month_signal_dates(index: pd.DatetimeIndex, freeze_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Return month-end NAV observations known to be complete without calendar guessing.

    A month is treated as complete only after the official panel contains at
    least one observation in a later calendar month. This means an August
    signal is first capturable once September NAV data exists.
    """
    if len(index) < 2:
        return pd.DatetimeIndex([])
    latest_period = index[-1].to_period("M")
    eligible = index[index.to_period("M") < latest_period]
    if len(eligible) == 0:
        return pd.DatetimeIndex([])
    holder = pd.Series(eligible, index=eligible)
    dates = pd.DatetimeIndex(holder.groupby(eligible.to_period("M")).last().values)
    return dates[dates > freeze_date]


def next_nav_date(index: pd.DatetimeIndex, signal_date: pd.Timestamp) -> pd.Timestamp | None:
    pos = int(index.searchsorted(signal_date, side="right"))
    return index[pos] if pos < len(index) else None


def read_signal_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    df = pd.read_csv(path, dtype={"signal_month": str})
    for c in SIGNAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[SIGNAL_COLUMNS]


def signal_target_details(
    signal_date: pd.Timestamp,
    model: pd.DataFrame,
    mask: pd.DataFrame,
    core_weight: float,
    top_k: int,
) -> tuple[pd.Series, list[dict]]:
    ranking = model.loc[signal_date, TICKERS].dropna().sort_values(ascending=False)
    selected = ranking.head(top_k).index.tolist()
    tilt = choose(model, mask, pd.DatetimeIndex([signal_date]), top_k=top_k)
    target = broad_core_target(pd.DatetimeIndex([signal_date]), tilt, core_weight).iloc[0]
    details = []
    for ticker in selected:
        details.append({
            "ticker": ticker,
            "score": float(ranking.loc[ticker]),
            "pass": bool(mask.loc[signal_date, ticker]),
        })
    while len(details) < top_k:
        details.append({"ticker": "", "score": np.nan, "pass": False})
    return target, details


def append_new_signals(
    existing: pd.DataFrame,
    close: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    sector_close = close[TICKERS]
    base = build_factors(sector_close)
    model_name = spec["relative_momentum"]["model_name"]
    model = signal_models(sector_close, base)[model_name]
    mask = absolute_masks(sector_close)["ma200_r12"]

    freeze = pd.Timestamp(spec["freeze_date"])
    eligible_dates = completed_month_signal_dates(close.index, freeze)
    seen_months = set(existing["signal_month"].dropna().astype(str))
    latest_date = close.index[-1]
    captured_at = datetime.now(timezone.utc).isoformat()
    core_weight = float(spec["portfolio"]["core_weight"])
    top_k = int(spec["portfolio"]["top_k"])
    new_rows: list[dict] = []

    for signal_date in eligible_dates:
        signal_month = str(signal_date.to_period("M"))
        if signal_month in seen_months:
            continue
        execution_date = next_nav_date(close.index, signal_date)
        if execution_date is None:
            continue
        target, details = signal_target_details(signal_date, model, mask, core_weight, top_k)
        exec_pos = int(close.index.get_loc(execution_date))
        latest_pos = len(close.index) - 1
        late_nav_days = max(latest_pos - exec_pos, 0)
        weights = {k: float(v) for k, v in target.items() if abs(float(v)) > 1e-14}
        row: Dict[str, object] = {
            "model_id": spec["model_id"],
            "model_version": spec["version"],
            "signal_month": signal_month,
            "signal_date": signal_date.date().isoformat(),
            "execution_date": execution_date.date().isoformat(),
            "captured_at_utc": captured_at,
            "data_through_date": latest_date.date().isoformat(),
            "late_capture_nav_days": int(late_nav_days),
            "core_weight": core_weight,
            "tilt_cash_weight": float(target["CASH"] / max(1.0 - core_weight, 1e-12)),
            "portfolio_cash_weight": float(target["CASH"]),
            "target_weights_json": json.dumps(weights, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        }
        for i in range(3):
            d = details[i] if i < len(details) else {"ticker": "", "score": np.nan, "pass": False}
            row[f"selected_{i+1}"] = d["ticker"]
            row[f"selected_{i+1}_score"] = d["score"]
            row[f"selected_{i+1}_pass"] = d["pass"]
        new_rows.append(row)
        seen_months.add(signal_month)

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined = combined.sort_values(["signal_date", "captured_at_utc"]).drop_duplicates("signal_month", keep="first")
    else:
        combined = existing.copy()
    return combined[SIGNAL_COLUMNS], new_rows


def targets_from_recorded_signals(signals: pd.DataFrame) -> pd.DataFrame:
    cols = TICKERS + [BROAD, "CASH"]
    rows = []
    idx = []
    for _, rec in signals.sort_values("signal_date").iterrows():
        weights = json.loads(str(rec["target_weights_json"]))
        row = {c: float(weights.get(c, 0.0)) for c in cols}
        total = sum(row.values())
        if abs(total - 1.0) > 1e-8:
            raise RuntimeError(f"recorded target does not sum to 1 for {rec['signal_date']}: {total}")
        rows.append(row)
        idx.append(pd.Timestamp(rec["signal_date"]))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=cols)


def pure_1306_target(dates: pd.DatetimeIndex) -> pd.DataFrame:
    cols = TICKERS + [BROAD, "CASH"]
    x = pd.DataFrame(0.0, index=dates, columns=cols)
    x[BROAD] = 1.0
    return x


def sector_ew_no_cash_target(dates: pd.DatetimeIndex, core_weight: float) -> pd.DataFrame:
    cols = TICKERS + [BROAD, "CASH"]
    x = pd.DataFrame(0.0, index=dates, columns=cols)
    x[BROAD] = core_weight
    for t in TICKERS:
        x[t] = (1.0 - core_weight) / len(TICKERS)
    return x


def cashmatched_sector_ew_target(candidate: pd.DataFrame, core_weight: float) -> pd.DataFrame:
    cols = TICKERS + [BROAD, "CASH"]
    x = pd.DataFrame(0.0, index=candidate.index, columns=cols)
    x[BROAD] = core_weight
    cash = candidate["CASH"].astype(float).clip(0.0, 1.0 - core_weight)
    risky_tilt = (1.0 - core_weight) - cash
    for t in TICKERS:
        x[t] = risky_tilt / len(TICKERS)
    x["CASH"] = cash
    return x


def prospective_performance(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, dict]:
    if signals.empty:
        return pd.DataFrame(), {}
    candidate = targets_from_recorded_signals(signals)
    dates = candidate.index
    core_weight = float(spec["portfolio"]["core_weight"])
    cost = float(spec["research_cost_assumption"]["basis_points_per_risky_side"])
    controls = {
        "candidate": candidate,
        "benchmark_100pct_1306": pure_1306_target(dates),
        "control_75pct_1306_plus_25pct_sector_ew_no_cash": sector_ew_no_cash_target(dates, core_weight),
        "control_cashmatched_sector_ew": cashmatched_sector_ew_target(candidate, core_weight),
    }
    dr = close.pct_change(fill_method=None)
    risky = TICKERS + [BROAD]
    daily = {}
    summary = {}
    for name, target in controls.items():
        r, weights, turnover = simulate_self_financing(close, dr, target, risky, cost)
        daily[f"{name}_return"] = r
        daily[f"{name}_equity"] = (1.0 + r).cumprod()
        p = perf(r)
        summary[name] = {
            **{k: (None if not np.isfinite(v) else float(v)) for k, v in p.items()},
            "mean_cash_weight": float(weights["CASH"].mean()),
            "annualized_turnover": float(turnover.mean() * 252.0),
        }
    return pd.DataFrame(daily), summary


def make_status(
    spec: dict,
    close: pd.DataFrame,
    signals: pd.DataFrame,
    new_rows: list[dict],
    summary: dict,
    source_meta: dict,
) -> dict:
    latest_signal = signals.iloc[-1].to_dict() if not signals.empty else None
    return {
        "model_id": spec["model_id"],
        "model_version": spec["version"],
        "freeze_date": spec["freeze_date"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_data_through": close.index[-1].date().isoformat(),
        "prospective_signal_count": int(len(signals)),
        "new_signals_this_run": int(len(new_rows)),
        "latest_signal": latest_signal,
        "performance": summary,
        "source_metadata": source_meta,
        "interpretation": "Prospective results are primary evidence after freeze. Historical parameters must not be changed under this model_id."
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, default=Path("config/frozen_model_v1.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("forward"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_spec(args.spec)
    close, source_meta = load_official_panel()
    signals_path = args.out_dir / "signals.csv"
    existing = read_signal_log(signals_path)
    signals, new_rows = append_new_signals(existing, close, spec)
    signals.to_csv(signals_path, index=False)

    daily, summary = prospective_performance(signals, close, spec)
    daily.to_csv(args.out_dir / "performance.csv")
    status = make_status(spec, close, signals, new_rows, summary, source_meta)
    (args.out_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps({
        "model_id": spec["model_id"],
        "data_through": close.index[-1].date().isoformat(),
        "signal_count": len(signals),
        "new_signal_count": len(new_rows),
        "latest_signal_month": None if signals.empty else signals.iloc[-1]["signal_month"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
