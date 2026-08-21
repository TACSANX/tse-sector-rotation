#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 18: characterize the frozen 1306-core overlay as tail insurance.

No parameters are optimized here. The Phase-17 frozen candidate is evaluated
conditional on contemporaneous 1306 monthly returns to determine whether the
overlay's historical benefit is concentrated in down/tail months and what it
costs in up months.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, build_factors, fetch_official_series, load_official_prices
from research_broad_core import BROAD, broad_core_target
from research_broad_core_attribution import (
    cashmatched_broad_control,
    broad_plus_sector_ew_target,
    pure_1306_target,
    monthly_returns,
)
from research_dual_momentum import absolute_masks, choose, rebalance_dates, signal_models
from research_portfolio import simulate_self_financing


def conditional_rows(monthly: pd.DataFrame, control: str) -> list[dict]:
    bench = monthly["benchmark_100pct_1306"]
    candidate = monthly["candidate"]
    delta = candidate - monthly[control]
    q10 = float(bench.quantile(0.10))
    q20 = float(bench.quantile(0.20))
    q80 = float(bench.quantile(0.80))
    states = {
        "worst_10pct_1306_months": bench <= q10,
        "worst_20pct_1306_months": bench <= q20,
        "all_1306_down_months": bench < 0.0,
        "all_1306_up_months": bench >= 0.0,
        "best_20pct_1306_months": bench >= q80,
    }
    rows = []
    for name, mask in states.items():
        d = delta[mask].dropna()
        b = bench[mask].dropna()
        c = candidate[mask].dropna()
        rows.append({
            "control": control,
            "state": name,
            "months": int(len(d)),
            "benchmark_1306_mean": float(b.mean()),
            "candidate_mean": float(c.mean()),
            "candidate_minus_control_mean": float(d.mean()),
            "candidate_beats_control_share": float((d > 0).mean()),
            "candidate_minus_control_median": float(d.median()),
        })
    return rows


def slope(y: pd.Series, x: pd.Series) -> float:
    z = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(z) < 3 or float(z["x"].var(ddof=0)) <= 0:
        return np.nan
    return float(np.cov(z["x"], z["y"], ddof=0)[0, 1] / z["x"].var(ddof=0))


def capture_summary(monthly: pd.DataFrame) -> dict:
    b = monthly["benchmark_100pct_1306"]
    c = monthly["candidate"]
    down = b < 0
    up = b >= 0
    b_down = float(b[down].mean())
    c_down = float(c[down].mean())
    b_up = float(b[up].mean())
    c_up = float(c[up].mean())
    return {
        "down_months": int(down.sum()),
        "up_months": int(up.sum()),
        "benchmark_mean_down": b_down,
        "candidate_mean_down": c_down,
        "downside_capture_ratio": float(c_down / b_down) if b_down != 0 else np.nan,
        "benchmark_mean_up": b_up,
        "candidate_mean_up": c_up,
        "upside_capture_ratio": float(c_up / b_up) if b_up != 0 else np.nan,
        "down_month_beta": slope(c[down], b[down]),
        "up_month_beta": slope(c[up], b[up]),
    }


def bootstrap_tail(
    monthly: pd.DataFrame,
    control: str,
    block: int,
    reps: int,
    rng: np.random.Generator,
) -> list[dict]:
    cols = ["candidate", "benchmark_100pct_1306", control]
    arr = monthly[cols].dropna().to_numpy(float)
    n = len(arr)
    starts = np.arange(0, n - block + 1)
    nblocks = int(np.ceil(n / block))
    state_values = {
        "worst_10pct_1306_months": [],
        "worst_20pct_1306_months": [],
        "all_1306_down_months": [],
        "all_1306_up_months": [],
        "best_20pct_1306_months": [],
    }
    for _ in range(reps):
        chosen = rng.choice(starts, size=nblocks, replace=True)
        sample = np.concatenate([arr[s:s + block] for s in chosen], axis=0)[:n]
        cand = sample[:, 0]
        bench = sample[:, 1]
        ctl = sample[:, 2]
        delta = cand - ctl
        q10 = np.quantile(bench, 0.10)
        q20 = np.quantile(bench, 0.20)
        q80 = np.quantile(bench, 0.80)
        masks = {
            "worst_10pct_1306_months": bench <= q10,
            "worst_20pct_1306_months": bench <= q20,
            "all_1306_down_months": bench < 0.0,
            "all_1306_up_months": bench >= 0.0,
            "best_20pct_1306_months": bench >= q80,
        }
        for state, mask in masks.items():
            state_values[state].append(float(delta[mask].mean()))
    rows = []
    for state, vals in state_values.items():
        a = np.asarray(vals, dtype=float)
        rows.append({
            "control": control,
            "block_months": block,
            "state": state,
            "median_delta": float(np.median(a)),
            "q025": float(np.quantile(a, 0.025)),
            "q975": float(np.quantile(a, 0.975)),
            "prob_positive": float((a > 0).mean()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out-dir", type=Path, default=Path("research_tail_profile"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sectors, sector_meta = load_official_prices()
    broad, broad_meta = fetch_official_series(BROAD)
    close = pd.concat([sectors, broad.rename(BROAD)], axis=1, sort=False).dropna(how="any").sort_index()
    sector_close = close[TICKERS]
    base = build_factors(sector_close)
    model = signal_models(sector_close, base)["classic_6_12minus1"]
    mask = absolute_masks(sector_close)["ma200_r12"]
    dates = rebalance_dates(close.index, "monthly", args.signal_start)
    tilt = choose(model, mask, dates, top_k=3)
    candidate_target = broad_core_target(dates, tilt, 0.75)

    targets = {
        "candidate": candidate_target,
        "benchmark_100pct_1306": pure_1306_target(dates),
        "control_75pct_1306_plus_25pct_sector_ew_no_cash": broad_plus_sector_ew_target(dates, 0.75),
        "control_cashmatched_sector_ew": cashmatched_broad_control(candidate_target, 0.75),
    }
    dr = close.pct_change(fill_method=None)
    risky = TICKERS + [BROAD]
    monthly_series = {}
    for name, target in targets.items():
        r, _, _ = simulate_self_financing(close, dr, target, risky, args.cost_bps)
        monthly_series[name] = monthly_returns(r)
    monthly = pd.DataFrame(monthly_series).dropna()
    monthly.to_csv(args.out_dir / "monthly_returns.csv")

    controls = [
        "benchmark_100pct_1306",
        "control_75pct_1306_plus_25pct_sector_ew_no_cash",
        "control_cashmatched_sector_ew",
    ]
    cond = pd.DataFrame([row for ctl in controls for row in conditional_rows(monthly, ctl)])
    cond.to_csv(args.out_dir / "conditional_profile.csv", index=False)

    capture = capture_summary(monthly)
    pd.DataFrame([capture]).to_csv(args.out_dir / "capture_summary.csv", index=False)

    rng = np.random.default_rng(args.seed)
    boot_rows = []
    for ctl in controls:
        for block in [6, 12, 24]:
            boot_rows.extend(bootstrap_tail(monthly, ctl, block, args.reps, rng))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(args.out_dir / "bootstrap_summary.csv", index=False)

    primary = cond[cond["control"] == "benchmark_100pct_1306"].copy()
    primary_boot = boot[boot["control"] == "benchmark_100pct_1306"].copy()
    report = [
        "# TOPIX-17 Phase 18 — Tail / Insurance Profile",
        "",
        "Frozen candidate: 75% 1306 + 25% Top3 classic 6m/12-1m sector tilt, MA200+positive12m slot-gate, failed slots to CASH.",
        "",
        "This is diagnostic, not parameter optimization. Market states are defined from contemporaneous 1306 monthly returns.",
        "",
        "## Candidate versus 100% 1306",
        "",
        primary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Capture / beta summary",
        "",
        pd.DataFrame([capture]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Moving-block bootstrap: candidate minus 100% 1306",
        "",
        primary_boot.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Full conditional attribution",
        "",
        cond.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Interpretation: a useful insurance-like overlay should exhibit positive candidate-minus-benchmark returns in down/tail states, while any negative difference in up states is the historical opportunity cost of that protection.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "phase": 18,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "candidate": "frozen topix17_1306_core_tilt_v1",
        "cost_bps": args.cost_bps,
        "bootstrap_reps": args.reps,
        "bootstrap_blocks_months": [6, 12, 24],
        "seed": args.seed,
        "sector_source_metadata": sector_meta,
        "broad_source_metadata": broad_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(primary.to_string(index=False))
    print(pd.DataFrame([capture]).to_string(index=False))
    print(primary_boot.to_string(index=False))


if __name__ == "__main__":
    main()
