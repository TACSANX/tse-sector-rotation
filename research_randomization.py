#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 11: randomization test for TOPIX-17 sector-ranking value.

The candidate is frozen:
- monthly signal
- classic 6m + 12-1m relative momentum
- Top3 slot-gate
- MA200 + positive 12m absolute filter
- 75% equal-weight core + 25% sector tilt

Null controls preserve the candidate's exact CASH schedule and the exact number
of active tilt slots at every rebalance.  The active tilt names are sampled
uniformly from ETFs that pass the same absolute filter.  This asks whether the
momentum ranking adds value beyond the absolute-trend/CASH mechanism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, load_official_prices, build_factors, perf, sliced
from research_dual_momentum import signal_models, absolute_masks, rebalance_dates, choose
from research_core_tilt import blend_targets
from research_attribution import cash_matched_equal_weight
from research_portfolio import simulate_self_financing, equal_weight_signal


def random_tilt_like_actual(
    actual_tilt: pd.DataFrame,
    eligible_mask: pd.DataFrame,
    rng: np.random.Generator,
    top_k: int = 3,
) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=actual_tilt.index, columns=TICKERS + ["CASH"])
    slot = 1.0 / top_k
    for d in actual_tilt.index:
        cash = float(actual_tilt.loc[d, "CASH"])
        n_active = int(round((1.0 - cash) * top_k))
        eligible = [t for t in TICKERS if bool(eligible_mask.loc[d, t])]
        if n_active > len(eligible):
            raise RuntimeError(
                f"{d.date()}: need {n_active} random eligible ETFs but only {len(eligible)} pass"
            )
        if n_active:
            selected = rng.choice(np.asarray(eligible, dtype=object), size=n_active, replace=False)
            for t in selected.tolist():
                out.loc[d, t] = slot
        out.loc[d, "CASH"] = cash
        total = float(out.loc[d].sum())
        if abs(total - 1.0) > 1e-8:
            raise RuntimeError(f"random target does not sum to one at {d.date()}: {total}")
    return out


def metrics(r: pd.Series, turnover: pd.Series, holdout_start: str) -> dict:
    full = perf(r)
    hold = sliced(r, start=holdout_start)
    return {
        "cagr": full["cagr"],
        "max_drawdown": full["max_drawdown"],
        "sharpe": full["sharpe"],
        "holdout_cagr": hold["cagr"],
        "holdout_max_drawdown": hold["max_drawdown"],
        "holdout_sharpe": hold["sharpe"],
        "annualized_turnover": float(turnover.mean() * 252.0),
    }


def summarize(actual: dict, random_df: pd.DataFrame, cost: float) -> list[dict]:
    rows: list[dict] = []
    higher_is_better = {
        "cagr": True,
        "max_drawdown": True,
        "sharpe": True,
        "holdout_cagr": True,
        "holdout_max_drawdown": True,
        "holdout_sharpe": True,
        "annualized_turnover": False,
    }
    x = random_df[random_df["cost_bps"] == cost]
    for metric, hib in higher_is_better.items():
        vals = x[metric].dropna().to_numpy(float)
        a = float(actual[metric])
        if hib:
            percentile = float((vals <= a).mean())
            p_worse_or_equal = float((vals >= a).mean())
        else:
            percentile = float((vals >= a).mean())
            p_worse_or_equal = float((vals <= a).mean())
        rows.append({
            "cost_bps": cost,
            "metric": metric,
            "actual": a,
            "random_mean": float(np.mean(vals)),
            "random_median": float(np.median(vals)),
            "random_q025": float(np.quantile(vals, 0.025)),
            "random_q975": float(np.quantile(vals, 0.975)),
            "actual_better_percentile": percentile,
            "random_as_good_or_better_share": p_worse_or_equal,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out-dir", type=Path, default=Path("research_randomization"))
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
    actual_tilt = choose(model, mask, dates, top_k=3)
    actual_target = blend_targets(0.75, core, actual_tilt)
    cash_control_target = cash_matched_equal_weight(actual_target)

    rng = np.random.default_rng(args.seed)
    random_rows: list[dict] = []
    actual_by_cost: dict[float, dict] = {}
    cash_control_by_cost: dict[float, dict] = {}
    equity: dict[str, pd.Series] = {}

    for cost in [0.0, 10.0]:
        r_a, _, t_a = simulate_self_financing(close, dr, actual_target, TICKERS, cost)
        r_c, _, t_c = simulate_self_financing(close, dr, cash_control_target, TICKERS, cost)
        actual_by_cost[cost] = metrics(r_a, t_a, args.holdout_start)
        cash_control_by_cost[cost] = metrics(r_c, t_c, args.holdout_start)
        if cost == 10.0:
            equity["actual_75core_top3_10bp"] = (1.0 + r_a).cumprod()
            equity["cash_matched_equal_weight_10bp"] = (1.0 + r_c).cumprod()

    # Generate each null ranking once, then evaluate it at both cost assumptions.
    for rep in range(args.reps):
        tilt = random_tilt_like_actual(actual_tilt, mask, rng, top_k=3)
        target = blend_targets(0.75, core, tilt)
        for cost in [0.0, 10.0]:
            r, _, turn = simulate_self_financing(close, dr, target, TICKERS, cost)
            row = {"rep": rep, "cost_bps": cost, **metrics(r, turn, args.holdout_start)}
            random_rows.append(row)
        if (rep + 1) % 250 == 0:
            print(f"completed randomization {rep + 1}/{args.reps}")

    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(args.out_dir / "random_results.csv", index=False)

    summary_rows: list[dict] = []
    for cost in [0.0, 10.0]:
        summary_rows.extend(summarize(actual_by_cost[cost], random_df, cost))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "random_summary.csv", index=False)

    actual_table = []
    for cost in [0.0, 10.0]:
        a = actual_by_cost[cost]
        c = cash_control_by_cost[cost]
        actual_table.append({
            "cost_bps": cost,
            **{f"actual_{k}": v for k, v in a.items()},
            **{f"cashmatched_{k}": v for k, v in c.items()},
            "selection_cagr_contribution": a["cagr"] - c["cagr"],
            "selection_holdout_cagr_contribution": a["holdout_cagr"] - c["holdout_cagr"],
            "selection_mdd_contribution": a["max_drawdown"] - c["max_drawdown"],
            "selection_sharpe_contribution": a["sharpe"] - c["sharpe"],
        })
    actual_df = pd.DataFrame(actual_table)
    actual_df.to_csv(args.out_dir / "actual_vs_cashmatched.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    s10 = summary[summary["cost_bps"] == 10.0]
    report = [
        "# TOPIX-17 Phase 11 — Randomized Ranking Test",
        "",
        "Null portfolios preserve the frozen candidate's exact CASH schedule and active tilt-slot count at every monthly rebalance. Active sector names are chosen uniformly at random from ETFs passing the same MA200 + positive-12m filter.",
        "",
        f"Random repetitions: {args.reps:,}",
        f"Seed: {args.seed}",
        "",
        "## Frozen candidate vs cash-matched control",
        "",
        actual_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 10bp randomization distribution",
        "",
        s10.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "`actual_better_percentile` is the share of randomized strategies the frozen momentum ranking beats on that metric (for turnover, lower is treated as better).",
        "`random_as_good_or_better_share` is a one-sided randomization-style tail probability. This is a model-checking statistic, not a pristine p-value because the candidate was selected after earlier exploratory work.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 11,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "candidate": "75% equal-weight core + 25% classic 6m/12-1m Top3 slot-gate, MA200+positive12m",
        "null": "same CASH schedule and active slot count; random eligible sector names",
        "reps": args.reps,
        "seed": args.seed,
        "costs_bps": [0.0, 10.0],
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(actual_df.to_string(index=False))
    print(s10.to_string(index=False))


if __name__ == "__main__":
    main()
