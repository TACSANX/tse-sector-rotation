#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3: core + sector tilt research for TOPIX-17.

Hypothesis
----------
Pure winner rotation gives up too much broad-market return in some eras. Keep a
broad equal-weight TOPIX-17 core and use only part of the portfolio for a
relative/absolute-momentum sector tilt.

Research discipline
-------------------
This phase deliberately uses only the two slower momentum families that survived
Phase 2.1 reasonably well. It is not a broad optimizer.

Data/accounting
---------------
- Official NEXT FUNDS distribution-reinvested NAV history only.
- Signals at rebalance close; target executed at next NAV close; target earns
  returns from the following NAV date.
- Self-financing weights drift naturally between rebalances.
- Transaction costs are charged on actual risky ETF notional traded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, load_official_prices, build_factors, perf, sliced
from research_dual_momentum import signal_models, absolute_masks, rebalance_dates, choose, era_metrics
from research_portfolio import simulate_self_financing, equal_weight_signal


def blend_targets(core_share: float, core: pd.DataFrame, tilt: pd.DataFrame) -> pd.DataFrame:
    """Blend broad risky core and sector tilt; tilt's failed slots remain CASH."""
    if not 0.0 <= core_share <= 1.0:
        raise ValueError(core_share)
    x = core_share * core + (1.0 - core_share) * tilt
    # Numerical hygiene.
    x = x.clip(lower=0.0)
    x = x.div(x.sum(axis=1), axis=0)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--out-dir", type=Path, default=Path("research_core_tilt"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    close = close.dropna(how="any").sort_index()
    dr = close.pct_change(fill_method=None)
    if dr.iloc[1:].isna().any().any():
        raise RuntimeError("unexpected missing common-date returns")

    base = build_factors(close)
    models_all = signal_models(close, base)
    masks_all = absolute_masks(close)

    # Freeze the research family based on Phase 2.1, rather than searching every
    # model/filter combination again.
    model_names = ["classic_6_12minus1", "long_6_12"]
    mask_names = ["none", "ma200", "ma200_r12"]
    core_shares = [0.0, 0.25, 0.50, 0.75, 1.0]
    rebalance_modes = ["monthly", "bimonthly"]
    costs = [0.0, 10.0, 20.0]

    rows: list[dict] = []
    equity: dict[str, pd.Series] = {}
    benchmark: dict[tuple[str, float], dict] = {}

    for mode in rebalance_modes:
        dates = rebalance_dates(close.index, mode, args.signal_start)
        core = equal_weight_signal(dates, TICKERS)

        for cost in costs:
            r_bm, w_bm, t_bm = simulate_self_financing(close, dr, core, TICKERS, cost)
            benchmark[(mode, cost)] = {
                "full": perf(r_bm),
                "pre2018": sliced(r_bm, end=args.holdout_start),
                "holdout": sliced(r_bm, start=args.holdout_start),
                "eras": era_metrics(r_bm),
                "annualized_turnover": float(t_bm.mean() * 252),
            }
            if cost == 10.0:
                equity[f"benchmark_{mode}_ew_10bp"] = (1.0 + r_bm).cumprod()

        for model_name in model_names:
            model = models_all[model_name]
            for mask_name in mask_names:
                mask = masks_all[mask_name]
                tilt = choose(model, mask, dates, top_k=3)
                for core_share in core_shares:
                    target = blend_targets(core_share, core, tilt)
                    for cost in costs:
                        r, weights, turnover = simulate_self_financing(close, dr, target, TICKERS, cost)
                        full = perf(r)
                        pre = sliced(r, end=args.holdout_start)
                        hold = sliced(r, start=args.holdout_start)
                        eras = era_metrics(r)
                        b = benchmark[(mode, cost)]
                        name = (
                            f"{mode}_{model_name}_top3_{mask_name}_"
                            f"core{int(core_share*100):03d}_{cost:g}bp"
                        )
                        rows.append({
                            "variant": name,
                            "rebalance": mode,
                            "model": model_name,
                            "absolute_filter": mask_name,
                            "core_share": core_share,
                            "tilt_share": 1.0 - core_share,
                            "cost_bps": cost,
                            **{f"full_{k}": v for k, v in full.items()},
                            **{f"pre2018_{k}": v for k, v in pre.items()},
                            **{f"holdout_{k}": v for k, v in hold.items()},
                            **eras,
                            "cash_exposure_mean_weight": float(weights["CASH"].mean()),
                            "annualized_turnover": float(turnover.mean() * 252),
                            "full_cagr_alpha_vs_same_freq_ew": float(full["cagr"] - b["full"]["cagr"]),
                            "holdout_cagr_alpha_vs_same_freq_ew": float(hold["cagr"] - b["holdout"]["cagr"]),
                            "full_mdd_improvement_vs_same_freq_ew": float(full["max_drawdown"] - b["full"]["max_drawdown"]),
                            "holdout_mdd_improvement_vs_same_freq_ew": float(hold["max_drawdown"] - b["holdout"]["max_drawdown"]),
                        })
                        if cost == 10.0 and mode == "monthly" and core_share in (0.0, 0.5, 0.75, 1.0):
                            equity[name] = (1.0 + r).cumprod()

    result = pd.DataFrame(rows)
    era_cagr = ["2009_2013_cagr", "2014_2017_cagr", "2018_2021_cagr", "2022_2026_cagr"]
    era_sharpe = ["2009_2013_sharpe", "2014_2017_sharpe", "2018_2021_sharpe", "2022_2026_sharpe"]
    result["positive_era_count"] = (result[era_cagr] > 0).sum(axis=1)
    result["worst_era_cagr"] = result[era_cagr].min(axis=1)
    result["median_era_sharpe"] = result[era_sharpe].median(axis=1)

    # Robustness score is descriptive only, not a trading signal. Reward return,
    # drawdown improvement and cross-era stability; penalize excessive turnover.
    result["robustness_score"] = (
        2.0 * result["positive_era_count"]
        + 4.0 * result["median_era_sharpe"].clip(-1, 2)
        + 8.0 * result["holdout_cagr_alpha_vs_same_freq_ew"]
        + 4.0 * result["holdout_mdd_improvement_vs_same_freq_ew"]
        - 0.03 * result["annualized_turnover"]
    )
    result = result.sort_values(
        ["positive_era_count", "median_era_sharpe", "holdout_sharpe", "annualized_turnover"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    result.to_csv(args.out_dir / "grid_results.csv", index=False)

    robust = result[
        (result["cost_bps"] == 10.0)
        & (result["positive_era_count"] == 4)
        & (result["pre2018_cagr"] > 0)
        & (result["holdout_cagr"] > 0)
        & (result["full_max_drawdown"] > -0.40)
        & (result["holdout_max_drawdown"] > -0.40)
    ].copy()
    robust.to_csv(args.out_dir / "shortlist_10bp.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity_10bp.csv")

    # Parameter-neighborhood summary: a good core percentage should not depend on
    # one exact model/filter choice.
    neigh = (
        result[result["cost_bps"] == 10.0]
        .groupby(["rebalance", "core_share"], as_index=False)
        .agg(
            median_full_cagr=("full_cagr", "median"),
            median_holdout_cagr=("holdout_cagr", "median"),
            median_full_sharpe=("full_sharpe", "median"),
            median_holdout_sharpe=("holdout_sharpe", "median"),
            median_full_mdd=("full_max_drawdown", "median"),
            median_holdout_mdd=("holdout_max_drawdown", "median"),
            median_turnover=("annualized_turnover", "median"),
            min_positive_era_count=("positive_era_count", "min"),
        )
    )
    neigh.to_csv(args.out_dir / "core_share_neighborhood_10bp.csv", index=False)

    bm10 = benchmark[("monthly", 10.0)]
    cols = [
        "variant", "full_cagr", "full_max_drawdown", "full_sharpe",
        "pre2018_cagr", "holdout_cagr", "holdout_max_drawdown", "holdout_sharpe",
        "positive_era_count", "worst_era_cagr", "median_era_sharpe",
        "cash_exposure_mean_weight", "annualized_turnover",
        "full_cagr_alpha_vs_same_freq_ew", "holdout_cagr_alpha_vs_same_freq_ew",
        "full_mdd_improvement_vs_same_freq_ew", "holdout_mdd_improvement_vs_same_freq_ew",
    ]
    report = [
        "# TOPIX-17 Phase 3 — Core + Sector Tilt",
        "",
        "Official NEXT FUNDS distribution-reinvested NAV; corrected self-financing accounting.",
        "",
        "## 10bp monthly equal-weight benchmark",
        "",
        f"- CAGR: {bm10['full']['cagr']:.2%}",
        f"- Max drawdown: {bm10['full']['max_drawdown']:.2%}",
        f"- Sharpe: {bm10['full']['sharpe']:.3f}",
        f"- Holdout CAGR: {bm10['holdout']['cagr']:.2%}",
        f"- Holdout max drawdown: {bm10['holdout']['max_drawdown']:.2%}",
        "",
        "## Best 10bp robust variants",
        "",
        robust.head(35)[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Core-share neighborhood (10bp)",
        "",
        neigh.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation rule",
        "",
        "Prefer a broad plateau across adjacent core shares and across both surviving signal families. Do not select an isolated peak.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 3,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "common_price_start": close.index.min().date().isoformat(),
        "common_price_end": close.index.max().date().isoformat(),
        "signal_start": args.signal_start,
        "holdout_start": args.holdout_start,
        "models": model_names,
        "filters": mask_names,
        "core_shares": core_shares,
        "rebalance_modes": rebalance_modes,
        "costs_bps": costs,
        "variants": int(len(result)),
        "source_metadata": source_meta,
        "timing": "signal close -> execute next NAV close -> earn from following NAV date",
        "accounting": "self-financing drift; costs on actual risky ETF notional traded",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(robust.head(35)[cols].to_string(index=False))
    print("\nCORE SHARE NEIGHBORHOOD\n", neigh.to_string(index=False))


if __name__ == "__main__":
    main()
