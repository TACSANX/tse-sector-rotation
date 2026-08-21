#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 10: absolute-filter backfill policy for frozen TOPIX-17 momentum.

Current policy (slot-gating): rank all sectors, take Top-K, and turn failed
absolute-trend slots into CASH.

Alternative (eligible-backfill): first require MA200 > 0 and 12m return > 0,
then take the highest-ranked eligible sectors. CASH is used only if fewer than K
eligible sectors exist.

This tests whether current CASH exposure is unnecessarily high without opening a
large parameter search.
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
from research_attribution import cash_matched_equal_weight, rolling_three_year_relative
from research_portfolio import simulate_self_financing, equal_weight_signal


def choose_backfill(model: pd.DataFrame, mask: pd.DataFrame, dates: pd.DatetimeIndex, top_k: int) -> pd.DataFrame:
    target = pd.DataFrame(0.0, index=dates, columns=TICKERS + ["CASH"])
    slot = 1.0 / top_k
    for d in dates:
        ranking = model.loc[d].dropna().sort_values(ascending=False)
        eligible = [t for t in ranking.index if bool(mask.loc[d, t])]
        selected = eligible[:top_k]
        for t in selected:
            target.loc[d, t] = slot
        target.loc[d, "CASH"] = 1.0 - len(selected) * slot
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_backfill"))
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

    rows = []
    equity = {}
    for cost in [10.0, 20.0]:
        rb, _, _ = simulate_self_financing(close, dr, core, TICKERS, cost)
        p_b = perf(rb); h_b = sliced(rb, start=args.holdout_start)
        if cost == 10.0:
            equity["equal_weight_10bp"] = (1.0 + rb).cumprod()

        for top_k in [3, 5]:
            policy_targets = {
                "slot_gate": choose(model, mask, dates, top_k),
                "eligible_backfill": choose_backfill(model, mask, dates, top_k),
            }
            for policy, tilt in policy_targets.items():
                for core_share in [0.50, 0.75]:
                    target = blend_targets(core_share, core, tilt)
                    control = cash_matched_equal_weight(target)
                    r, w, turn = simulate_self_financing(close, dr, target, TICKERS, cost)
                    rc, wc, tc = simulate_self_financing(close, dr, control, TICKERS, cost)
                    full = perf(r); hold = sliced(r, start=args.holdout_start); pre = sliced(r, end=args.holdout_start)
                    cf = perf(rc); ch = sliced(rc, start=args.holdout_start)
                    eras = era_metrics(r)
                    roll = rolling_three_year_relative(r, rc)
                    rows.append({
                        "policy": policy,
                        "top_k": top_k,
                        "core_share": core_share,
                        "cost_bps": cost,
                        **{f"full_{k}": v for k, v in full.items()},
                        **{f"pre2018_{k}": v for k, v in pre.items()},
                        **{f"holdout_{k}": v for k, v in hold.items()},
                        **eras,
                        "cash_exposure_mean_weight": float(w["CASH"].mean()),
                        "annualized_turnover": float(turn.mean() * 252),
                        "cashmatched_full_cagr": cf["cagr"],
                        "cashmatched_holdout_cagr": ch["cagr"],
                        "cashmatched_full_mdd": cf["max_drawdown"],
                        "selection_full_cagr_contribution": full["cagr"] - cf["cagr"],
                        "selection_holdout_cagr_contribution": hold["cagr"] - ch["cagr"],
                        "selection_full_mdd_contribution": full["max_drawdown"] - cf["max_drawdown"],
                        "selection_full_sharpe_contribution": full["sharpe"] - cf["sharpe"],
                        "full_cagr_alpha_vs_ew": full["cagr"] - p_b["cagr"],
                        "holdout_cagr_alpha_vs_ew": hold["cagr"] - h_b["cagr"],
                        "full_mdd_improvement_vs_ew": full["max_drawdown"] - p_b["max_drawdown"],
                        "holdout_mdd_improvement_vs_ew": hold["max_drawdown"] - h_b["max_drawdown"],
                        **roll,
                    })
                    if cost == 10.0:
                        name = f"{policy}_top{top_k}_core{int(core_share*100):03d}"
                        equity[name] = (1.0 + r).cumprod()

    result = pd.DataFrame(rows)
    era_cagr = ["2009_2013_cagr", "2014_2017_cagr", "2018_2021_cagr", "2022_2026_cagr"]
    result["positive_era_count"] = (result[era_cagr] > 0).sum(axis=1)
    result["worst_era_cagr"] = result[era_cagr].min(axis=1)
    result = result.sort_values(["cost_bps", "top_k", "core_share", "policy"])
    result.to_csv(args.out_dir / "grid_results.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    ten = result[result["cost_bps"] == 10.0].copy()
    paired = []
    keys = ["top_k", "core_share"]
    for key, g in ten.groupby(keys):
        if set(g.policy) != {"slot_gate", "eligible_backfill"}:
            continue
        s = g.set_index("policy")
        paired.append({
            "top_k": key[0],
            "core_share": key[1],
            "delta_backfill_minus_gate_full_cagr": s.loc["eligible_backfill","full_cagr"] - s.loc["slot_gate","full_cagr"],
            "delta_holdout_cagr": s.loc["eligible_backfill","holdout_cagr"] - s.loc["slot_gate","holdout_cagr"],
            "delta_full_mdd": s.loc["eligible_backfill","full_max_drawdown"] - s.loc["slot_gate","full_max_drawdown"],
            "delta_full_sharpe": s.loc["eligible_backfill","full_sharpe"] - s.loc["slot_gate","full_sharpe"],
            "delta_cash_weight": s.loc["eligible_backfill","cash_exposure_mean_weight"] - s.loc["slot_gate","cash_exposure_mean_weight"],
            "delta_turnover": s.loc["eligible_backfill","annualized_turnover"] - s.loc["slot_gate","annualized_turnover"],
            "delta_selection_cagr": s.loc["eligible_backfill","selection_full_cagr_contribution"] - s.loc["slot_gate","selection_full_cagr_contribution"],
            "delta_selection_mdd": s.loc["eligible_backfill","selection_full_mdd_contribution"] - s.loc["slot_gate","selection_full_mdd_contribution"],
            "delta_rolling3y_selection_positive": s.loc["eligible_backfill","rolling3y_selection_positive_share"] - s.loc["slot_gate","rolling3y_selection_positive_share"],
        })
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(args.out_dir / "paired_policy_10bp.csv", index=False)

    cols = [
        "policy","top_k","core_share","full_cagr","full_max_drawdown","full_sharpe",
        "pre2018_cagr","holdout_cagr","holdout_max_drawdown","holdout_sharpe",
        "cash_exposure_mean_weight","annualized_turnover",
        "selection_full_cagr_contribution","selection_holdout_cagr_contribution",
        "selection_full_mdd_contribution","selection_full_sharpe_contribution",
        "rolling3y_selection_positive_share","rolling3y_selection_median_ann","rolling3y_selection_worst_ann",
        "full_cagr_alpha_vs_ew","holdout_cagr_alpha_vs_ew","full_mdd_improvement_vs_ew","positive_era_count","worst_era_cagr"
    ]
    report = [
        "# TOPIX-17 Phase 10 — Absolute-Filter Backfill Policy",
        "",
        "Compare current slot-gating with eligible-sector backfill. Everything else is frozen.",
        "",
        "## Detailed 10bp results",
        "",
        ten[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Backfill minus slot-gate deltas at 10bp",
        "",
        paired_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision rule",
        "",
        "Adopt backfill only if reduced CASH improves CAGR without materially surrendering drawdown improvement or cross-era stability. A small return lift with large DD deterioration is not enough.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "phase": 10,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "signal_family": "classic 6m + 12-1m; MA200 + positive 12m; monthly",
        "policies": ["slot_gate", "eligible_backfill"],
        "top_k": [3, 5],
        "core_shares": [0.50, 0.75],
        "costs_bps": [10.0, 20.0],
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ten[cols].to_string(index=False))
    print("\nPAIRED DELTAS\n", paired_df.to_string(index=False))


if __name__ == "__main__":
    main()
