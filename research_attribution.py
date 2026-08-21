#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 8: attribute TOPIX-17 candidate performance to CASH timing vs sector selection.

For each frozen core+tilt candidate, construct a control portfolio with exactly
the same target CASH weight at every rebalance, but allocate all risky capital
equally across the 17 ETFs. This isolates sector-selection contribution from the
risk reduction caused by holding CASH.
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
from research_portfolio import simulate_self_financing, equal_weight_signal


def cash_matched_equal_weight(strategy_target: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(0.0, index=strategy_target.index, columns=TICKERS + ["CASH"])
    cash = strategy_target["CASH"].clip(0.0, 1.0)
    risky = 1.0 - cash
    x.loc[:, TICKERS] = np.asarray(risky)[:, None] / len(TICKERS)
    x["CASH"] = cash
    return x


def rolling_three_year_relative(strategy: pd.Series, control: pd.Series, window: int = 756) -> dict:
    x = pd.concat([strategy.rename("s"), control.rename("c")], axis=1).dropna()
    log_rel = np.log1p(x["s"]) - np.log1p(x["c"])
    ann = np.expm1(log_rel.rolling(window).sum() * (252.0 / window)).dropna()
    return {
        "rolling3y_selection_positive_share": float((ann > 0).mean()) if len(ann) else np.nan,
        "rolling3y_selection_median_ann": float(ann.median()) if len(ann) else np.nan,
        "rolling3y_selection_worst_ann": float(ann.min()) if len(ann) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_attribution"))
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

    rows = []
    equity = {}
    for cost in [10.0, 20.0]:
        r_ew, _, t_ew = simulate_self_financing(close, dr, core, TICKERS, cost)
        p_ew = perf(r_ew)
        h_ew = sliced(r_ew, start=args.holdout_start)
        if cost == 10.0:
            equity["full_equal_weight_10bp"] = (1.0 + r_ew).cumprod()

        for core_share in [0.25, 0.50, 0.75]:
            st = blend_targets(core_share, core, tilt)
            ct = cash_matched_equal_weight(st)
            r_s, w_s, t_s = simulate_self_financing(close, dr, st, TICKERS, cost)
            r_c, w_c, t_c = simulate_self_financing(close, dr, ct, TICKERS, cost)

            ps = perf(r_s); pc = perf(r_c)
            hs = sliced(r_s, start=args.holdout_start); hc = sliced(r_c, start=args.holdout_start)
            pres = sliced(r_s, end=args.holdout_start); prec = sliced(r_c, end=args.holdout_start)
            eras_s = era_metrics(r_s); eras_c = era_metrics(r_c)
            roll = rolling_three_year_relative(r_s, r_c)

            rows.append({
                "core_share": core_share,
                "cost_bps": cost,
                "mean_cash_weight_strategy": float(w_s["CASH"].mean()),
                "mean_cash_weight_control": float(w_c["CASH"].mean()),
                "strategy_full_cagr": ps["cagr"],
                "cashmatched_full_cagr": pc["cagr"],
                "full_ew_cagr": p_ew["cagr"],
                "strategy_holdout_cagr": hs["cagr"],
                "cashmatched_holdout_cagr": hc["cagr"],
                "full_ew_holdout_cagr": h_ew["cagr"],
                "selection_full_cagr_contribution": ps["cagr"] - pc["cagr"],
                "cash_timing_full_cagr_contribution": pc["cagr"] - p_ew["cagr"],
                "selection_holdout_cagr_contribution": hs["cagr"] - hc["cagr"],
                "cash_timing_holdout_cagr_contribution": hc["cagr"] - h_ew["cagr"],
                "strategy_full_mdd": ps["max_drawdown"],
                "cashmatched_full_mdd": pc["max_drawdown"],
                "full_ew_mdd": p_ew["max_drawdown"],
                "strategy_holdout_mdd": hs["max_drawdown"],
                "cashmatched_holdout_mdd": hc["max_drawdown"],
                "full_ew_holdout_mdd": h_ew["max_drawdown"],
                "selection_full_mdd_contribution": ps["max_drawdown"] - pc["max_drawdown"],
                "cash_timing_full_mdd_contribution": pc["max_drawdown"] - p_ew["max_drawdown"],
                "selection_holdout_mdd_contribution": hs["max_drawdown"] - hc["max_drawdown"],
                "cash_timing_holdout_mdd_contribution": hc["max_drawdown"] - h_ew["max_drawdown"],
                "strategy_full_sharpe": ps["sharpe"],
                "cashmatched_full_sharpe": pc["sharpe"],
                "strategy_holdout_sharpe": hs["sharpe"],
                "cashmatched_holdout_sharpe": hc["sharpe"],
                "selection_full_sharpe_contribution": ps["sharpe"] - pc["sharpe"],
                "selection_holdout_sharpe_contribution": hs["sharpe"] - hc["sharpe"],
                "strategy_pre2018_cagr": pres["cagr"],
                "cashmatched_pre2018_cagr": prec["cagr"],
                "strategy_annualized_turnover": float(t_s.mean() * 252),
                "cashmatched_annualized_turnover": float(t_c.mean() * 252),
                **{f"strategy_{k}": v for k, v in eras_s.items()},
                **{f"cashmatched_{k}": v for k, v in eras_c.items()},
                **roll,
            })
            if cost == 10.0:
                equity[f"strategy_core{int(core_share*100):03d}_10bp"] = (1.0 + r_s).cumprod()
                equity[f"cashmatched_core{int(core_share*100):03d}_10bp"] = (1.0 + r_c).cumprod()

    result = pd.DataFrame(rows).sort_values(["cost_bps", "core_share"])
    result.to_csv(args.out_dir / "attribution.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    ten = result[result["cost_bps"] == 10.0].copy()
    cols = [
        "core_share", "mean_cash_weight_strategy",
        "strategy_full_cagr", "cashmatched_full_cagr", "full_ew_cagr",
        "selection_full_cagr_contribution", "cash_timing_full_cagr_contribution",
        "strategy_holdout_cagr", "cashmatched_holdout_cagr", "full_ew_holdout_cagr",
        "selection_holdout_cagr_contribution", "cash_timing_holdout_cagr_contribution",
        "strategy_full_mdd", "cashmatched_full_mdd", "full_ew_mdd",
        "selection_full_mdd_contribution", "cash_timing_full_mdd_contribution",
        "strategy_full_sharpe", "cashmatched_full_sharpe", "selection_full_sharpe_contribution",
        "rolling3y_selection_positive_share", "rolling3y_selection_median_ann", "rolling3y_selection_worst_ann",
        "strategy_annualized_turnover", "cashmatched_annualized_turnover",
    ]
    report = [
        "# TOPIX-17 Phase 8 — Attribution: CASH vs Sector Selection",
        "",
        "Control portfolios use the exact same target CASH weight as the strategy but allocate risky capital equally across all 17 ETFs.",
        "",
        "## 10bp attribution",
        "",
        ten[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Reading the decomposition",
        "",
        "- `cash_timing_*_contribution`: effect of the strategy's dynamic CASH schedule versus full equal-weight exposure.",
        "- `selection_*_contribution`: incremental effect of sector selection versus an equal-weight risky portfolio with identical CASH exposure.",
        "- Positive MDD contribution means a shallower drawdown.",
        "",
        "If most drawdown improvement comes from the cash-matched control, sector rotation is not the primary source of risk reduction.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "phase": 8,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "signal_family": "classic 6m + 12-1m; Top3; MA200 + positive 12m; monthly",
        "core_shares": [0.25, 0.50, 0.75],
        "costs_bps": [10.0, 20.0],
        "control": "same target CASH weight, equal-weight risky allocation",
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ten[cols].to_string(index=False))


if __name__ == "__main__":
    main()
