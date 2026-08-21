#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14: does cross-sector dispersion identify when momentum ranking works?

Exploratory regime study using only information observable at each signal date.
The regime variable is the cross-sectional standard deviation of 6-month ETF
returns.  "High dispersion" means current dispersion is above the trailing
252-NAV-day median, shifted by one day so the threshold uses only prior data.

The frozen 75% core + 25% Top3 slot-gate strategy is compared with a
cash-matched equal-weight risky control.  A conditional strategy uses sector
selection only in high-dispersion months; in low-dispersion months it keeps the
same CASH target but replaces sector selection with equal-weight risky exposure.
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


def regime_target(
    baseline: pd.DataFrame,
    cashmatched: pd.DataFrame,
    high_dispersion: pd.Series,
    select_when_high: bool,
) -> pd.DataFrame:
    out = cashmatched.copy() if select_when_high else baseline.copy()
    for d in baseline.index:
        high = bool(high_dispersion.loc[d])
        use_selection = high if select_when_high else not high
        out.loc[d] = baseline.loc[d] if use_selection else cashmatched.loc[d]
    return out


def strategy_metrics(r: pd.Series, turn: pd.Series, holdout_start: str) -> dict:
    p = perf(r)
    h = sliced(r, start=holdout_start)
    pre = sliced(r, end=holdout_start)
    eras = era_metrics(r)
    return {
        "full_cagr": p["cagr"],
        "full_max_drawdown": p["max_drawdown"],
        "full_sharpe": p["sharpe"],
        "pre2018_cagr": pre["cagr"],
        "holdout_cagr": h["cagr"],
        "holdout_max_drawdown": h["max_drawdown"],
        "holdout_sharpe": h["sharpe"],
        "annualized_turnover": float(turn.mean() * 252.0),
        **eras,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_dispersion_regime"))
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
    cashmatched = cash_matched_equal_weight(baseline)

    r6 = close.pct_change(126)
    dispersion = r6.std(axis=1, ddof=0)
    threshold = dispersion.rolling(252, min_periods=126).median().shift(1)
    high_daily = (dispersion > threshold) & threshold.notna()
    high = high_daily.reindex(dates).fillna(False)

    high_only = regime_target(baseline, cashmatched, high, select_when_high=True)
    low_only = regime_target(baseline, cashmatched, high, select_when_high=False)

    targets = {
        "baseline_selection_all_months": baseline,
        "cashmatched_no_selection": cashmatched,
        "selection_high_dispersion_only": high_only,
        "selection_low_dispersion_only": low_only,
    }

    rows = []
    equity = {}
    for cost in [0.0, 10.0, 20.0]:
        for name, target in targets.items():
            r, _, turn = simulate_self_financing(close, dr, target, TICKERS, cost)
            rows.append({"variant": name, "cost_bps": cost, **strategy_metrics(r, turn, args.holdout_start)})
            if cost == 10.0:
                equity[name] = (1.0 + r).cumprod()
    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "strategy_results.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    # Direct cross-sectional signal validity by ex-ante dispersion state.
    signal_rows = []
    for i in range(len(dates) - 1):
        d = dates[i]
        next_d = dates[i + 1]
        entry = first_after(close.index, d)
        exit_ = first_after(close.index, next_d)
        if entry is None or exit_ is None or exit_ <= entry:
            continue
        s = model.loc[d, TICKERS].dropna()
        if len(s) != len(TICKERS):
            continue
        fwd = close.loc[exit_, TICKERS] / close.loc[entry, TICKERS] - 1.0
        j = pd.DataFrame({"score": s, "fwd": fwd}).dropna().sort_values("score", ascending=False)
        if len(j) != len(TICKERS):
            continue
        sr = j["score"].rank(method="average")
        fr = j["fwd"].rank(method="average")
        ic = float(sr.corr(fr))
        top3 = float(j.head(3)["fwd"].mean())
        ew = float(j["fwd"].mean())
        signal_rows.append({
            "signal_date": d,
            "high_dispersion": bool(high.loc[d]),
            "dispersion_6m": float(dispersion.loc[d]),
            "dispersion_threshold": float(threshold.loc[d]) if pd.notna(threshold.loc[d]) else np.nan,
            "ic": ic,
            "top3_minus_ew": top3 - ew,
        })
    sig = pd.DataFrame(signal_rows)
    sig.to_csv(args.out_dir / "signal_by_month.csv", index=False)

    state_rows = []
    for period, start, end in [
        ("full", None, None),
        ("pre2018", None, args.holdout_start),
        ("holdout2018plus", args.holdout_start, None),
    ]:
        x = sig.copy()
        x["signal_date"] = pd.to_datetime(x["signal_date"])
        if start:
            x = x[x["signal_date"] >= pd.Timestamp(start)]
        if end:
            x = x[x["signal_date"] < pd.Timestamp(end)]
        for state, flag in [("high", True), ("low", False)]:
            y = x[x["high_dispersion"] == flag]
            state_rows.append({
                "period": period,
                "state": state,
                "months": int(len(y)),
                "share_of_period": float(len(y) / len(x)) if len(x) else np.nan,
                "mean_ic": float(y["ic"].mean()) if len(y) else np.nan,
                "median_ic": float(y["ic"].median()) if len(y) else np.nan,
                "ic_positive_share": float((y["ic"] > 0).mean()) if len(y) else np.nan,
                "mean_top3_minus_ew": float(y["top3_minus_ew"].mean()) if len(y) else np.nan,
                "top3_minus_ew_positive_share": float((y["top3_minus_ew"] > 0).mean()) if len(y) else np.nan,
            })
    states = pd.DataFrame(state_rows)
    states.to_csv(args.out_dir / "state_summary.csv", index=False)

    ten = result[result["cost_bps"] == 10.0].copy()
    report = [
        "# TOPIX-17 Phase 14 — Cross-Sector Dispersion Regime",
        "",
        "High dispersion is defined ex ante: current cross-sectional dispersion of 6m ETF returns above the prior 252-NAV-day median.",
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
        "This is exploratory follow-up motivated by the observed post-2018 strengthening of the momentum signal. Any dispersion gate that looks attractive here must be frozen and forward-tested; it is not an untouched out-of-sample discovery.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 14,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "candidate": "75% equal-weight core + 25% classic 6m/12-1m Top3 slot-gate, MA200+positive12m",
        "dispersion": "cross-sectional std of 126-NAV-day ETF returns",
        "threshold": "rolling 252-NAV-day median shifted one day; min 126 observations",
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(states.to_string(index=False))
    print(ten.to_string(index=False))


if __name__ == "__main__":
    main()
