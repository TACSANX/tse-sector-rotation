#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 13: fixed-label permutation placebo for the frozen TOPIX-17 candidate.

The exact monthly target path of the frozen 75% core + 25% Top3 slot-gate
strategy is retained, including its CASH schedule and persistence.  For each
placebo, the 17 ETF labels are permuted once and that mapping is held fixed for
the entire history.  This preserves the strategy's target-weight dynamics far
better than independent random monthly picks and tests whether the actual
sector identities selected by the momentum ranking matter.
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
from research_portfolio import simulate_self_financing, equal_weight_signal


def permuted_target(actual: pd.DataFrame, perm: np.ndarray) -> pd.DataFrame:
    out = actual.copy()
    values = actual[TICKERS].to_numpy(float)
    out.loc[:, TICKERS] = values[:, perm]
    out["CASH"] = actual["CASH"].to_numpy(float)
    err = (out.sum(axis=1) - 1.0).abs().max()
    if float(err) > 1e-8:
        raise RuntimeError(f"permuted target sum error: {err}")
    return out


def metrics(r: pd.Series, turnover: pd.Series, holdout_start: str) -> dict:
    p = perf(r)
    h = sliced(r, start=holdout_start)
    return {
        "cagr": p["cagr"],
        "max_drawdown": p["max_drawdown"],
        "sharpe": p["sharpe"],
        "holdout_cagr": h["cagr"],
        "holdout_max_drawdown": h["max_drawdown"],
        "holdout_sharpe": h["sharpe"],
        "annualized_turnover": float(turnover.mean() * 252.0),
    }


def summarize(actual: dict, df: pd.DataFrame, cost: float) -> list[dict]:
    x = df[df["cost_bps"] == cost]
    rows = []
    higher = {
        "cagr": True,
        "max_drawdown": True,
        "sharpe": True,
        "holdout_cagr": True,
        "holdout_max_drawdown": True,
        "holdout_sharpe": True,
        "annualized_turnover": False,
    }
    for m, hib in higher.items():
        vals = x[m].dropna().to_numpy(float)
        a = float(actual[m])
        better_pct = float((vals <= a).mean()) if hib else float((vals >= a).mean())
        as_good = float((vals >= a).mean()) if hib else float((vals <= a).mean())
        rows.append({
            "cost_bps": cost,
            "metric": m,
            "actual": a,
            "placebo_mean": float(vals.mean()),
            "placebo_median": float(np.median(vals)),
            "placebo_q025": float(np.quantile(vals, 0.025)),
            "placebo_q975": float(np.quantile(vals, 0.975)),
            "actual_better_percentile": better_pct,
            "placebo_as_good_or_better_share": as_good,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out-dir", type=Path, default=Path("research_label_permutation"))
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
    actual_target = blend_targets(0.75, core, tilt)

    actual_by_cost = {}
    for cost in [0.0, 10.0]:
        r, _, turn = simulate_self_financing(close, dr, actual_target, TICKERS, cost)
        actual_by_cost[cost] = metrics(r, turn, args.holdout_start)

    rng = np.random.default_rng(args.seed)
    rows = []
    identity = np.arange(len(TICKERS))
    for rep in range(args.reps):
        perm = rng.permutation(len(TICKERS))
        if np.array_equal(perm, identity):
            perm = np.roll(perm, 1)
        target = permuted_target(actual_target, perm)
        for cost in [0.0, 10.0]:
            r, _, turn = simulate_self_financing(close, dr, target, TICKERS, cost)
            rows.append({
                "rep": rep,
                "cost_bps": cost,
                "permutation": ",".join(str(int(v)) for v in perm),
                **metrics(r, turn, args.holdout_start),
            })
        if (rep + 1) % 250 == 0:
            print(f"completed permutation {rep + 1}/{args.reps}")

    placebo = pd.DataFrame(rows)
    placebo.to_csv(args.out_dir / "placebo_results.csv", index=False)

    summary_rows = []
    for cost in [0.0, 10.0]:
        summary_rows.extend(summarize(actual_by_cost[cost], placebo, cost))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "placebo_summary.csv", index=False)

    actual_df = pd.DataFrame([
        {"cost_bps": c, **m} for c, m in actual_by_cost.items()
    ])
    actual_df.to_csv(args.out_dir / "actual_metrics.csv", index=False)

    s0 = summary[summary["cost_bps"] == 0.0]
    s10 = summary[summary["cost_bps"] == 10.0]
    report = [
        "# TOPIX-17 Phase 13 — Fixed-Label Permutation Placebo",
        "",
        "Each placebo applies one random permutation of the 17 ETF labels to the frozen strategy's entire target path. CASH and target persistence are unchanged; only sector identity is reassigned.",
        "",
        f"Placebo repetitions: {args.reps:,}",
        f"Seed: {args.seed}",
        "",
        "## Actual candidate",
        "",
        actual_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 0bp placebo distribution",
        "",
        s0.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 10bp placebo distribution",
        "",
        s10.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "The 0bp comparison is the cleanest test of sector-identity information because transaction-cost differences cannot explain it. The 10bp comparison adds implementation effects from return-driven weight drift. This is still exploratory because the frozen candidate was selected after prior research.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 13,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "candidate": "75% equal-weight core + 25% classic 6m/12-1m Top3 slot-gate, MA200+positive12m",
        "placebo": "single fixed permutation of 17 ETF labels over entire target history",
        "reps": args.reps,
        "seed": args.seed,
        "costs_bps": [0.0, 10.0],
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(actual_df.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
