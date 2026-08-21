#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 19: material 1306 drawdown episodes for the frozen candidate.

Every completed 100% 1306 drawdown of at least 10% is detected mechanically.
The frozen candidate is then evaluated over exactly the same benchmark
peak-to-recovery window. No episode is hand-picked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, build_factors, fetch_official_series, load_official_prices
from research_broad_core import BROAD, broad_core_target
from research_broad_core_attribution import pure_1306_target
from research_dual_momentum import absolute_masks, choose, rebalance_dates, signal_models
from research_portfolio import simulate_self_financing


def benchmark_drawdown_episodes(equity: pd.Series) -> list[dict]:
    x = equity.dropna()
    if x.empty:
        return []
    peak_value = float(x.iloc[0])
    peak_date = x.index[0]
    in_drawdown = False
    trough_value = peak_value
    trough_date = peak_date
    episodes = []
    for dt, value_obj in x.iloc[1:].items():
        value = float(value_obj)
        if value >= peak_value:
            if in_drawdown:
                episodes.append({
                    "benchmark_peak_date": peak_date,
                    "benchmark_trough_date": trough_date,
                    "benchmark_recovery_date": dt,
                    "benchmark_peak_value": peak_value,
                    "benchmark_trough_value": trough_value,
                    "benchmark_mdd": trough_value / peak_value - 1.0,
                })
                in_drawdown = False
            peak_value = value
            peak_date = dt
            trough_value = value
            trough_date = dt
        else:
            if not in_drawdown:
                in_drawdown = True
                trough_value = value
                trough_date = dt
            elif value < trough_value:
                trough_value = value
                trough_date = dt
    if in_drawdown:
        episodes.append({
            "benchmark_peak_date": peak_date,
            "benchmark_trough_date": trough_date,
            "benchmark_recovery_date": pd.NaT,
            "benchmark_peak_value": peak_value,
            "benchmark_trough_value": trough_value,
            "benchmark_mdd": trough_value / peak_value - 1.0,
        })
    return episodes


def first_recovery(equity: pd.Series, start: pd.Timestamp, level: float) -> pd.Timestamp | pd.NaT:
    after = equity[equity.index > start]
    hit = after[after >= level]
    return hit.index[0] if len(hit) else pd.NaT


def episode_comparison(benchmark_eq: pd.Series, candidate_eq: pd.Series, threshold: float) -> pd.DataFrame:
    rows = []
    for ep in benchmark_drawdown_episodes(benchmark_eq):
        if float(ep["benchmark_mdd"]) > -abs(threshold):
            continue
        peak = pd.Timestamp(ep["benchmark_peak_date"])
        trough = pd.Timestamp(ep["benchmark_trough_date"])
        bench_recovery = ep["benchmark_recovery_date"]
        end = pd.Timestamp(bench_recovery) if pd.notna(bench_recovery) else benchmark_eq.index[-1]
        c0 = float(candidate_eq.loc[peak])
        cwin = candidate_eq.loc[peak:end] / c0
        cand_mdd = float(cwin.min() - 1.0)
        cand_trough_date = cwin.idxmin()
        cand_at_bench_trough = float(candidate_eq.loc[trough] / c0 - 1.0)
        candidate_recovery = first_recovery(candidate_eq, peak, c0)
        recovery_delta_days = np.nan
        if pd.notna(bench_recovery) and pd.notna(candidate_recovery):
            recovery_delta_days = int((pd.Timestamp(candidate_recovery) - pd.Timestamp(bench_recovery)).days)
        rows.append({
            "benchmark_peak_date": peak.date().isoformat(),
            "benchmark_trough_date": trough.date().isoformat(),
            "benchmark_recovery_date": None if pd.isna(bench_recovery) else pd.Timestamp(bench_recovery).date().isoformat(),
            "benchmark_mdd": float(ep["benchmark_mdd"]),
            "candidate_mdd_same_window": cand_mdd,
            "candidate_trough_date": cand_trough_date.date().isoformat(),
            "candidate_at_benchmark_trough": cand_at_bench_trough,
            "trough_improvement": cand_at_bench_trough - float(ep["benchmark_mdd"]),
            "candidate_recovery_date": None if pd.isna(candidate_recovery) else pd.Timestamp(candidate_recovery).date().isoformat(),
            "candidate_recovery_minus_benchmark_days": recovery_delta_days,
            "candidate_return_at_benchmark_recovery": float(candidate_eq.loc[end] / c0 - 1.0),
        })
    return pd.DataFrame(rows).sort_values("benchmark_mdd") if rows else pd.DataFrame()


def summary_for(df: pd.DataFrame, cutoff: float) -> dict:
    x = df[df["benchmark_mdd"] <= -abs(cutoff)].copy()
    if x.empty:
        return {"cutoff": cutoff, "episodes": 0}
    rec = x["candidate_recovery_minus_benchmark_days"].dropna()
    return {
        "cutoff": cutoff,
        "episodes": int(len(x)),
        "candidate_shallower_at_benchmark_trough_share": float((x["trough_improvement"] > 0).mean()),
        "mean_trough_improvement": float(x["trough_improvement"].mean()),
        "median_trough_improvement": float(x["trough_improvement"].median()),
        "candidate_same_window_mdd_shallower_share": float((x["candidate_mdd_same_window"] > x["benchmark_mdd"]).mean()),
        "candidate_recovers_no_later_share": float((rec <= 0).mean()) if len(rec) else np.nan,
        "median_recovery_days_delta": float(rec.median()) if len(rec) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--out-dir", type=Path, default=Path("research_drawdown_episodes"))
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
    benchmark_target = pure_1306_target(dates)

    dr = close.pct_change(fill_method=None)
    risky = TICKERS + [BROAD]
    candidate_r, _, _ = simulate_self_financing(close, dr, candidate_target, risky, args.cost_bps)
    benchmark_r, _, _ = simulate_self_financing(close, dr, benchmark_target, risky, args.cost_bps)
    candidate_eq = (1.0 + candidate_r).cumprod()
    benchmark_eq = (1.0 + benchmark_r).cumprod()

    episodes = episode_comparison(benchmark_eq, candidate_eq, args.threshold)
    episodes.to_csv(args.out_dir / "episodes.csv", index=False)
    summaries = pd.DataFrame([summary_for(episodes, c) for c in [0.10, 0.15, 0.20]])
    summaries.to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame({"candidate": candidate_eq, "benchmark_100pct_1306": benchmark_eq}).to_csv(args.out_dir / "equity.csv")

    report = [
        "# TOPIX-17 Phase 19 — Material Drawdown Episodes",
        "",
        "All completed 100% 1306 drawdowns of at least 10% are detected mechanically; no crash episode is hand-picked.",
        "",
        "## Episode-level comparison",
        "",
        episodes.to_markdown(index=False, floatfmt=".4f") if len(episodes) else "No qualifying episodes.",
        "",
        "## Summary by benchmark drawdown depth",
        "",
        summaries.to_markdown(index=False, floatfmt=".4f"),
        "",
        "A positive `trough_improvement` means the candidate lost less than 1306 at the benchmark trough. A positive recovery-day delta means the candidate took longer to recover its benchmark-peak-date value, exposing the possible upside/recovery cost of the overlay.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "phase": 19,
        "candidate": "frozen topix17_1306_core_tilt_v1",
        "threshold": args.threshold,
        "cost_bps": args.cost_bps,
        "data_source": "Nomura Asset Management official NEXT FUNDS distribution-reinvested NAV",
        "sector_source_metadata": sector_meta,
        "broad_source_metadata": broad_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(episodes.to_string(index=False))
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
