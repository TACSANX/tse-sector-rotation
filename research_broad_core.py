#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16: replace the 75% equal-weight TOPIX-17 core with NEXT FUNDS 1306.

No signal parameters are changed.  The frozen 25% sector tilt remains classic
6m + 12-1m relative momentum, Top3 slot-gate, MA200 + positive 12m.  This study
only tests whether a single broad TOPIX core (1306) is a more practical
substitute for continuously holding all 17 sector ETFs as the 75% core.

All series use Nomura Asset Management official distribution-reinvested NAV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research_backtest_official import (
    TICKERS,
    fetch_official_series,
    load_official_prices,
    build_factors,
    perf,
    sliced,
)
from research_dual_momentum import signal_models, absolute_masks, rebalance_dates, choose, era_metrics
from research_core_tilt import blend_targets
from research_portfolio import simulate_self_financing, equal_weight_signal

BROAD = "1306.T"


def broad_core_target(dates: pd.DatetimeIndex, tilt: pd.DataFrame, core_share: float = 0.75) -> pd.DataFrame:
    cols = TICKERS + [BROAD, "CASH"]
    out = pd.DataFrame(0.0, index=dates, columns=cols)
    out[BROAD] = core_share
    tilt_share = 1.0 - core_share
    for t in TICKERS:
        out[t] = tilt_share * tilt[t]
    out["CASH"] = tilt_share * tilt["CASH"]
    err = (out.sum(axis=1) - 1.0).abs().max()
    if float(err) > 1e-8:
        raise RuntimeError(f"broad-core target sum error {err}")
    return out


def ew17_core_target(dates: pd.DatetimeIndex, tilt: pd.DataFrame, core_share: float = 0.75) -> pd.DataFrame:
    core = equal_weight_signal(dates, TICKERS)
    x = blend_targets(core_share, core, tilt)
    x[BROAD] = 0.0
    return x.reindex(columns=TICKERS + [BROAD, "CASH"], fill_value=0.0)


def pure_target(dates: pd.DatetimeIndex, risky: list[str], weights: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=dates, columns=risky + ["CASH"])
    for k, v in weights.items():
        out[k] = v
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_broad_core"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sectors, source_meta = load_official_prices()
    broad, broad_meta = fetch_official_series(BROAD)
    close = pd.concat([sectors, broad.rename(BROAD)], axis=1).dropna(how="any").sort_index()
    sector_close = close[TICKERS]
    dr = close.pct_change(fill_method=None)

    base = build_factors(sector_close)
    model = signal_models(sector_close, base)["classic_6_12minus1"]
    mask = absolute_masks(sector_close)["ma200_r12"]
    dates = rebalance_dates(close.index, "monthly", args.signal_start)
    tilt = choose(model, mask, dates, top_k=3)

    risky = TICKERS + [BROAD]
    targets = {
        "frozen_75pct_ew17_core": ew17_core_target(dates, tilt, 0.75),
        "practical_75pct_topix1306_core": broad_core_target(dates, tilt, 0.75),
        "benchmark_100pct_ew17": pure_target(dates, risky, {t: 1.0 / len(TICKERS) for t in TICKERS}),
        "benchmark_100pct_topix1306": pure_target(dates, risky, {BROAD: 1.0}),
    }

    rows = []
    equity = {}
    for cost in [0.0, 10.0, 20.0]:
        for name, target in targets.items():
            r, _, turn = simulate_self_financing(close, dr, target, risky, cost)
            rows.append({"variant": name, "cost_bps": cost, **metrics(r, turn, args.holdout_start)})
            if cost == 10.0:
                equity[name] = (1.0 + r).cumprod()
    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "results.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    ten = result[result["cost_bps"] == 10.0].copy()
    frozen = ten[ten["variant"] == "frozen_75pct_ew17_core"].iloc[0]
    broadrow = ten[ten["variant"] == "practical_75pct_topix1306_core"].iloc[0]
    delta = {
        "delta_cagr_broad_minus_ew17": float(broadrow["full_cagr"] - frozen["full_cagr"]),
        "delta_mdd_broad_minus_ew17": float(broadrow["full_max_drawdown"] - frozen["full_max_drawdown"]),
        "delta_sharpe_broad_minus_ew17": float(broadrow["full_sharpe"] - frozen["full_sharpe"]),
        "delta_holdout_cagr_broad_minus_ew17": float(broadrow["holdout_cagr"] - frozen["holdout_cagr"]),
        "delta_turnover_broad_minus_ew17": float(broadrow["annualized_turnover"] - frozen["annualized_turnover"]),
    }
    pd.DataFrame([delta]).to_csv(args.out_dir / "core_substitution_delta_10bp.csv", index=False)

    report = [
        "# TOPIX-17 Phase 16 — Broad TOPIX Core Substitution",
        "",
        "Only the 75% core implementation changes. Sector signal, Top3 rule, absolute filter, monthly timing, and cost assumptions are frozen.",
        "",
        "## 10bp results",
        "",
        ten.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 1306 core minus equal-weight-17 core",
        "",
        pd.DataFrame([delta]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "If performance is comparable, the 1306 core is operationally preferable because the core requires one ETF instead of 17 sector positions.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 16,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "broad_core": BROAD,
        "sector_source_metadata": source_meta,
        "broad_source_metadata": broad_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ten.to_string(index=False))
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()
