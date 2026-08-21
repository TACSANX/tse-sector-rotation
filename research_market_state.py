#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 15: market-trend and breadth interaction with TOPIX-17 selection.

Exploratory follow-up.  The frozen candidate is unchanged.  When a state gate
is false, only sector selection is disabled: the strategy keeps the exact same
CASH target and allocates risky capital equally across all 17 ETFs.  This
isolates whether market trend/breadth identifies months where ranking adds value.
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
from research_attribution import cash_matched_equal_weight
from research_portfolio import simulate_self_financing, equal_weight_signal


def first_after(index: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    pos = int(index.searchsorted(d, side="right"))
    return index[pos] if pos < len(index) else None


def gated_target(
    baseline: pd.DataFrame,
    control: pd.DataFrame,
    gate: pd.Series,
) -> pd.DataFrame:
    out = control.copy()
    for d in baseline.index:
        out.loc[d] = baseline.loc[d] if bool(gate.loc[d]) else control.loc[d]
    return out


def metrics(r: pd.Series, turn: pd.Series, holdout_start: str) -> dict:
    p = perf(r)
    pre = sliced(r, end=holdout_start)
    h = sliced(r, start=holdout_start)
    return {
        "full_cagr": p["cagr"],
        "full_max_drawdown": p["max_drawdown"],
        "full_sharpe": p["sharpe"],
        "pre2018_cagr": pre["cagr"],
        "holdout_cagr": h["cagr"],
        "holdout_max_drawdown": h["max_drawdown"],
        "holdout_sharpe": h["sharpe"],
        "annualized_turnover": float(turn.mean() * 252.0),
        **era_metrics(r),
    }


def state_signal_summary(sig: pd.DataFrame, state_col: str, period: str, start: str | None, end: str | None) -> list[dict]:
    x = sig.copy()
    if start:
        x = x[x["signal_date"] >= pd.Timestamp(start)]
    if end:
        x = x[x["signal_date"] < pd.Timestamp(end)]
    rows = []
    for flag in [True, False]:
        y = x[x[state_col] == flag]
        rows.append({
            "state_variable": state_col,
            "period": period,
            "state": "on" if flag else "off",
            "months": int(len(y)),
            "share": float(len(y) / len(x)) if len(x) else np.nan,
            "mean_ic": float(y["ic"].mean()) if len(y) else np.nan,
            "ic_positive_share": float((y["ic"] > 0).mean()) if len(y) else np.nan,
            "mean_top3_minus_ew": float(y["top3_minus_ew"].mean()) if len(y) else np.nan,
            "top3_minus_ew_positive_share": float((y["top3_minus_ew"] > 0).mean()) if len(y) else np.nan,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_market_state"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    close = close.dropna(how="any").sort_index()
    dr = close.pct_change(fill_method=None)
    base = build_factors(close)
    model = signal_models(close, base)["classic_6_12minus1"]
    mask = absolute_masks(close)["ma200_r12"]
    dates = rebalance_dates(close.index, "monthly", args.signal_start)
    core = equal_weight_signal(dates, TICKERS)
    tilt = choose(model, mask, dates, top_k=3)
    baseline = blend_targets(0.75, core, tilt)
    control = cash_matched_equal_weight(baseline)

    market_above200 = (base["bm200_gap"].reindex(dates) > 0.0).fillna(False)
    breadth_high = (base["breadth200"].reindex(dates) >= 0.50).fillna(False)
    risk_on = market_above200 & breadth_high

    targets = {
        "baseline_selection_all_months": baseline,
        "cashmatched_no_selection": control,
        "selection_market_above200_only": gated_target(baseline, control, market_above200),
        "selection_breadth_ge50_only": gated_target(baseline, control, breadth_high),
        "selection_risk_on_only": gated_target(baseline, control, risk_on),
    }

    result_rows = []
    equity = {}
    for cost in [0.0, 10.0, 20.0]:
        for name, target in targets.items():
            r, _, turn = simulate_self_financing(close, dr, target, TICKERS, cost)
            result_rows.append({"variant": name, "cost_bps": cost, **metrics(r, turn, args.holdout_start)})
            if cost == 10.0:
                equity[name] = (1.0 + r).cumprod()
    results = pd.DataFrame(result_rows)
    results.to_csv(args.out_dir / "strategy_results.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    # Direct next-month cross-sectional signal validity by state.
    signal_rows = []
    for i in range(len(dates) - 1):
        d = dates[i]
        entry = first_after(close.index, d)
        exit_ = first_after(close.index, dates[i + 1])
        if entry is None or exit_ is None or exit_ <= entry:
            continue
        s = model.loc[d, TICKERS].dropna()
        if len(s) != len(TICKERS):
            continue
        fwd = close.loc[exit_, TICKERS] / close.loc[entry, TICKERS] - 1.0
        j = pd.DataFrame({"score": s, "fwd": fwd}).dropna().sort_values("score", ascending=False)
        if len(j) != len(TICKERS):
            continue
        ic = float(j["score"].rank().corr(j["fwd"].rank()))
        top3 = float(j.head(3)["fwd"].mean())
        ew = float(j["fwd"].mean())
        signal_rows.append({
            "signal_date": d,
            "market_above200": bool(market_above200.loc[d]),
            "breadth_high": bool(breadth_high.loc[d]),
            "risk_on": bool(risk_on.loc[d]),
            "ic": ic,
            "top3_minus_ew": top3 - ew,
        })
    sig = pd.DataFrame(signal_rows)
    sig["signal_date"] = pd.to_datetime(sig["signal_date"])
    sig.to_csv(args.out_dir / "signal_by_month.csv", index=False)

    state_rows = []
    periods = [
        ("full", None, None),
        ("pre2018", None, args.holdout_start),
        ("holdout2018plus", args.holdout_start, None),
    ]
    for variable in ["market_above200", "breadth_high", "risk_on"]:
        for period, start, end in periods:
            state_rows.extend(state_signal_summary(sig, variable, period, start, end))
    states = pd.DataFrame(state_rows)
    states.to_csv(args.out_dir / "state_summary.csv", index=False)

    ten = results[results["cost_bps"] == 10.0].copy()
    report = [
        "# TOPIX-17 Phase 15 — Market Trend / Breadth State",
        "",
        "When a gate is off, the exact CASH schedule is preserved and risky capital becomes equal-weight. Thus the comparison targets the incremental value of sector selection only.",
        "",
        "## Signal validity by state",
        "",
        states.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 10bp strategy comparison",
        "",
        ten.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Discipline",
        "",
        "This is exploratory and follows prior observations. A gate is worth keeping only if it improves full/pre-2018/2018+ behavior without a fragile trade-off or materially higher turnover.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 15,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "candidate": "75% equal-weight core + 25% classic 6m/12-1m Top3 slot-gate, MA200+positive12m",
        "gates": {
            "market_above200": "equal-weight TOPIX-17 proxy above its 200-NAV-day average",
            "breadth_high": ">=50% of 17 ETFs above own 200-NAV-day average",
            "risk_on": "both conditions true",
        },
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(states.to_string(index=False))
    print(ten.to_string(index=False))


if __name__ == "__main__":
    main()
