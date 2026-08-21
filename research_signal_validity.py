#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 12: cross-sectional validity of the frozen TOPIX-17 momentum ranking.

This test removes portfolio construction, CASH allocation, and core/tilt choices.
At each month-end signal date, it asks whether the frozen classic 6m + 12-1m
relative-momentum score predicts the next executable monthly holding-period
return across the 17 ETFs.

Primary diagnostics:
- monthly cross-sectional Spearman information coefficient (IC)
- Top3 minus Bottom3 forward-return spread
- Top3 minus all-17 equal-weight forward-return spread
- rank-bucket monotonicity
- pre-2018 and 2018+ splits
- HAC t-statistics and moving-block bootstrap uncertainty
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, load_official_prices, build_factors
from research_dual_momentum import signal_models, rebalance_dates


def first_after(index: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    pos = int(index.searchsorted(d, side="right"))
    return index[pos] if pos < len(index) else None


def hac_mean_t(x: pd.Series, lag: int = 6) -> float:
    a = x.dropna().to_numpy(float)
    n = len(a)
    if n < 12:
        return np.nan
    u = a - a.mean()
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(u[k:], u[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * weight * gamma
    var_mean = lrv / n
    if var_mean <= 0 or not np.isfinite(var_mean):
        return np.nan
    return float(a.mean() / np.sqrt(var_mean))


def moving_block_bootstrap_mean(
    x: pd.Series,
    block: int,
    reps: int,
    rng: np.random.Generator,
) -> dict:
    a = x.dropna().to_numpy(float)
    n = len(a)
    if n < block * 2:
        return {"mean": np.nan, "q025": np.nan, "q975": np.nan, "p_gt_0": np.nan}
    starts = np.arange(0, n - block + 1)
    nblocks = int(np.ceil(n / block))
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        chosen = rng.choice(starts, size=nblocks, replace=True)
        sample = np.concatenate([a[s:s + block] for s in chosen])[:n]
        means[i] = float(sample.mean())
    return {
        "mean": float(a.mean()),
        "q025": float(np.quantile(means, 0.025)),
        "q975": float(np.quantile(means, 0.975)),
        "p_gt_0": float((means > 0).mean()),
    }


def period_summary(df: pd.DataFrame, start: str | None, end: str | None) -> dict:
    x = df.copy()
    if start:
        x = x[x.index >= pd.Timestamp(start)]
    if end:
        x = x[x.index < pd.Timestamp(end)]
    out = {"months": int(len(x))}
    for col in ["ic", "top3_minus_bottom3", "top3_minus_ew"]:
        s = x[col].dropna()
        out[f"{col}_mean"] = float(s.mean()) if len(s) else np.nan
        out[f"{col}_median"] = float(s.median()) if len(s) else np.nan
        out[f"{col}_positive_share"] = float((s > 0).mean()) if len(s) else np.nan
        out[f"{col}_hac_t6"] = hac_mean_t(s, 6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out-dir", type=Path, default=Path("research_signal_validity"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    close, source_meta = load_official_prices()
    close = close.dropna(how="any").sort_index()
    base = build_factors(close)
    score = signal_models(close, base)["classic_6_12minus1"]
    signal_dates = rebalance_dates(close.index, "monthly", args.signal_start)

    rows: list[dict] = []
    bucket_rows: list[dict] = []
    for i in range(len(signal_dates) - 1):
        d = signal_dates[i]
        next_signal = signal_dates[i + 1]
        entry = first_after(close.index, d)
        exit_ = first_after(close.index, next_signal)
        if entry is None or exit_ is None or exit_ <= entry:
            continue
        s = score.loc[d, TICKERS].dropna()
        if len(s) < len(TICKERS):
            continue
        fwd = close.loc[exit_, TICKERS] / close.loc[entry, TICKERS] - 1.0
        joined = pd.DataFrame({"score": s, "fwd": fwd}).dropna()
        if len(joined) != len(TICKERS):
            continue
        ranked = joined.sort_values("score", ascending=False)
        top3 = ranked.head(3)["fwd"]
        bottom3 = ranked.tail(3)["fwd"]
        ew = float(ranked["fwd"].mean())
        ic = float(ranked["score"].corr(ranked["fwd"], method="spearman"))
        rows.append({
            "signal_date": d,
            "entry_date": entry,
            "exit_date": exit_,
            "ic": ic,
            "top3_return": float(top3.mean()),
            "bottom3_return": float(bottom3.mean()),
            "equal_weight_return": ew,
            "top3_minus_bottom3": float(top3.mean() - bottom3.mean()),
            "top3_minus_ew": float(top3.mean() - ew),
        })

        # Fixed rank buckets chosen before looking at results.
        buckets = {
            "rank_1_3": ranked.iloc[0:3],
            "rank_4_6": ranked.iloc[3:6],
            "rank_7_11": ranked.iloc[6:11],
            "rank_12_14": ranked.iloc[11:14],
            "rank_15_17": ranked.iloc[14:17],
        }
        for name, b in buckets.items():
            bucket_rows.append({
                "signal_date": d,
                "bucket": name,
                "forward_return": float(b["fwd"].mean()),
                "forward_excess_vs_ew": float(b["fwd"].mean() - ew),
            })

    monthly = pd.DataFrame(rows).set_index("signal_date").sort_index()
    monthly.to_csv(args.out_dir / "monthly_signal_test.csv")
    buckets = pd.DataFrame(bucket_rows)
    buckets.to_csv(args.out_dir / "rank_bucket_monthly.csv", index=False)

    period_rows = []
    periods = {
        "full": (None, None),
        "pre2018": (None, args.holdout_start),
        "holdout2018plus": (args.holdout_start, None),
        "2009_2013": ("2009-04-01", "2014-01-01"),
        "2014_2017": ("2014-01-01", "2018-01-01"),
        "2018_2021": ("2018-01-01", "2022-01-01"),
        "2022_2026": ("2022-01-01", None),
    }
    for name, (start, end) in periods.items():
        period_rows.append({"period": name, **period_summary(monthly, start, end)})
    period_df = pd.DataFrame(period_rows)
    period_df.to_csv(args.out_dir / "period_summary.csv", index=False)

    bucket_summary = (
        buckets.groupby("bucket", as_index=False)
        .agg(
            mean_forward_return=("forward_return", "mean"),
            median_forward_return=("forward_return", "median"),
            mean_excess_vs_ew=("forward_excess_vs_ew", "mean"),
            positive_excess_share=("forward_excess_vs_ew", lambda x: float((x > 0).mean())),
            months=("forward_return", "size"),
        )
    )
    order = ["rank_1_3", "rank_4_6", "rank_7_11", "rank_12_14", "rank_15_17"]
    bucket_summary["bucket"] = pd.Categorical(bucket_summary["bucket"], categories=order, ordered=True)
    bucket_summary = bucket_summary.sort_values("bucket")
    bucket_summary.to_csv(args.out_dir / "rank_bucket_summary.csv", index=False)

    rng = np.random.default_rng(args.seed)
    boot_rows = []
    for period_name, (start, end) in {"full": (None, None), "holdout2018plus": (args.holdout_start, None)}.items():
        x = monthly.copy()
        if start:
            x = x[x.index >= pd.Timestamp(start)]
        if end:
            x = x[x.index < pd.Timestamp(end)]
        for metric in ["ic", "top3_minus_bottom3", "top3_minus_ew"]:
            for block in [6, 12]:
                b = moving_block_bootstrap_mean(x[metric], block, args.bootstrap_reps, rng)
                boot_rows.append({
                    "period": period_name,
                    "metric": metric,
                    "block_months": block,
                    "reps": args.bootstrap_reps,
                    **b,
                })
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(args.out_dir / "bootstrap_summary.csv", index=False)

    full = period_df[period_df["period"] == "full"].iloc[0]
    hold = period_df[period_df["period"] == "holdout2018plus"].iloc[0]
    report = [
        "# TOPIX-17 Phase 12 — Cross-Sectional Signal Validity",
        "",
        "The frozen classic 6m + 12-1m ranking is tested without CASH/core/tilt portfolio construction.",
        "",
        "## Period summary",
        "",
        period_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Rank-bucket monotonicity",
        "",
        bucket_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Moving-block bootstrap",
        "",
        boot.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation discipline",
        "",
        "A robust ranking signal should show positive average IC, a positive Top3-Bottom3 spread, and broadly monotonic rank buckets in both pre-2018 and 2018+ samples. Because this signal family was selected after earlier exploration, bootstrap intervals quantify stability rather than constitute a pristine confirmatory test.",
        "",
        f"Full mean IC: {full['ic_mean']:.4f}; holdout mean IC: {hold['ic_mean']:.4f}.",
        f"Full Top3-Bottom3 mean: {full['top3_minus_bottom3_mean']:.2%}; holdout: {hold['top3_minus_bottom3_mean']:.2%}.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    meta = {
        "phase": 12,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "signal": "classic 6m + 12-1m relative momentum",
        "timing": "month-end signal; enter next NAV close; exit next month-end signal's following NAV close",
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "source_metadata": source_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(period_df.to_string(index=False))
    print(bucket_summary.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
