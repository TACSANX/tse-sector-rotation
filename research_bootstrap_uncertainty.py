#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 7: paired block-bootstrap uncertainty for frozen TOPIX-17 candidates.

This is NOT a selection-adjusted significance test. The candidate family was
chosen after prior exploratory research, so conventional confidence intervals
remain optimistic with respect to data mining. The purpose is narrower: quantify
how uncertain the realized excess CAGR, Sharpe difference and drawdown
improvement are under serially dependent resampling of observed monthly returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, load_official_prices, build_factors
from research_dual_momentum import signal_models, absolute_masks, rebalance_dates, choose
from research_core_tilt import blend_targets
from research_execution_delay import delay_signal_index
from research_portfolio import simulate_self_financing, equal_weight_signal


def metrics(r: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(r, dtype=float)
    n = len(r)
    total = float(np.prod(1.0 + r))
    cagr = total ** (12.0 / n) - 1.0
    sd = float(np.std(r, ddof=0))
    sharpe = float(np.mean(r) / sd * np.sqrt(12.0)) if sd > 0 else np.nan
    eq = np.cumprod(1.0 + r)
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
    return cagr, sharpe, mdd


def month_returns(equity_ret: pd.Series, end_complete_month: pd.Timestamp) -> pd.Series:
    eq = (1.0 + equity_ret).cumprod()
    m = eq.resample("ME").last().pct_change().dropna()
    return m[m.index <= end_complete_month]


def circular_block_bootstrap(
    strategy: pd.Series,
    benchmark: pd.Series,
    block: int,
    reps: int,
    seed: int,
) -> dict:
    d = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    a = d["s"].to_numpy(float)
    b = d["b"].to_numpy(float)
    n = len(d)
    if n < 60:
        raise RuntimeError("too few monthly observations")

    rng = np.random.default_rng(seed)
    starts = np.arange(n)
    samples = np.empty((reps, 3), dtype=float)
    for i in range(reps):
        inds: list[int] = []
        while len(inds) < n:
            s = int(rng.choice(starts))
            inds.extend((s + j) % n for j in range(block))
        ix = np.asarray(inds[:n], dtype=int)
        ca, cs, cm = metrics(a[ix])
        ba, bs, bm = metrics(b[ix])
        # MDD is negative; positive candidate-minus-benchmark means shallower DD.
        samples[i] = [ca - ba, cs - bs, cm - bm]

    actual_s = np.asarray(metrics(a))
    actual_b = np.asarray(metrics(b))
    actual = actual_s - actual_b
    qs = np.quantile(samples, [0.025, 0.50, 0.975], axis=0)
    probs = (samples > 0.0).mean(axis=0)
    names = ["cagr_alpha", "sharpe_diff", "mdd_improvement"]
    out = {"months": n, "block_months": block, "reps": reps}
    for j, name in enumerate(names):
        out[f"actual_{name}"] = float(actual[j])
        out[f"ci025_{name}"] = float(qs[0, j])
        out[f"median_{name}"] = float(qs[1, j])
        out[f"ci975_{name}"] = float(qs[2, j])
        out[f"prob_{name}_positive"] = float(probs[j])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_bootstrap"))
    ap.add_argument("--reps", type=int, default=5000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    close = close.dropna(how="any").sort_index()
    dr = close.pct_change(fill_method=None)
    base = build_factors(close)
    model = signal_models(close, base)["classic_6_12minus1"]
    mask = absolute_masks(close)["ma200_r12"]
    dates = rebalance_dates(close.index, "monthly", args.signal_start)
    core0 = equal_weight_signal(dates, TICKERS)
    tilt0 = choose(model, mask, dates, top_k=3)

    # Last fully completed calendar month in the official series.
    max_date = close.index.max()
    month_start = max_date.to_period("M").start_time
    end_complete = month_start - pd.Timedelta(days=1)

    rows = []
    costs = 10.0
    for delay in [0, 1]:
        core = delay_signal_index(core0, close.index, delay)
        rb, _, _ = simulate_self_financing(close, dr, core, TICKERS, costs)
        bm = month_returns(rb, end_complete)
        for core_share in [0.50, 0.75]:
            raw = blend_targets(core_share, core0, tilt0)
            target = delay_signal_index(raw, close.index, delay)
            r, _, _ = simulate_self_financing(close, dr, target, TICKERS, costs)
            sm = month_returns(r, end_complete)
            for block in [6, 12, 24]:
                row = circular_block_bootstrap(
                    sm, bm, block=block, reps=args.reps,
                    seed=20260821 + delay * 100 + int(core_share * 100) + block,
                )
                row.update({
                    "candidate": f"delay{delay}_core{int(core_share*100):03d}",
                    "extra_execution_delay_nav_days": delay,
                    "core_share": core_share,
                    "cost_bps": costs,
                })
                rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "bootstrap_results.csv", index=False)

    summary = (
        result.groupby(["candidate", "core_share", "extra_execution_delay_nav_days"], as_index=False)
        .agg(
            min_prob_alpha_positive=("prob_cagr_alpha_positive", "min"),
            median_prob_alpha_positive=("prob_cagr_alpha_positive", "median"),
            min_prob_sharpe_positive=("prob_sharpe_diff_positive", "min"),
            median_prob_sharpe_positive=("prob_sharpe_diff_positive", "median"),
            min_prob_mdd_improvement=("prob_mdd_improvement_positive", "min"),
            median_prob_mdd_improvement=("prob_mdd_improvement_positive", "median"),
            worst_ci025_alpha=("ci025_cagr_alpha", "min"),
            best_ci975_alpha=("ci975_cagr_alpha", "max"),
            worst_ci025_mdd=("ci025_mdd_improvement", "min"),
            best_ci975_mdd=("ci975_mdd_improvement", "max"),
        )
    )
    summary.to_csv(args.out_dir / "bootstrap_summary.csv", index=False)

    report = [
        "# TOPIX-17 Phase 7 — Block Bootstrap Uncertainty",
        "",
        "Frozen candidates only. Paired circular block bootstrap of monthly strategy and same-delay equal-weight benchmark returns.",
        "",
        "**Important:** these intervals are not corrected for the fact that the strategy family was selected after earlier experiments. They must not be read as proof of alpha.",
        "",
        "## Cross-block summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Detailed 6/12/24-month block results",
        "",
        result.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "Persistent probability of shallower drawdown with a confidence interval crossing zero is evidence of risk-shaping, not statistical proof. Excess CAGR must be treated as unproven unless it survives prospective data and selection-adjusted testing.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "phase": 7,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "reps": args.reps,
        "block_months": [6, 12, 24],
        "candidates": ["core50", "core75"],
        "execution_delays": [0, 1],
        "cost_bps": costs,
        "complete_month_end": end_complete.date().isoformat(),
        "selection_adjusted": False,
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
