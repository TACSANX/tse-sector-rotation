#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 17: attribution and paired block-bootstrap for the 1306-core candidate.

Frozen candidate:
- 75% NEXT FUNDS TOPIX 1306 core
- 25% TOPIX-17 sector tilt
- classic 6m + 12-1m relative momentum
- Top3 slot-gate
- each selected ETF must be above MA200 and have positive 12m return
- failed tilt slots become CASH
- monthly signal, next official NAV close execution

Controls decompose:
A. 100% 1306
B. 75% 1306 + 25% equal-weight TOPIX-17, no CASH
C. Cash-matched control: 75% 1306 plus the candidate's exact tilt CASH weight;
   remaining tilt risk is equal-weight across all 17 sector ETFs.

Paired moving-block bootstrap uses monthly compounded realized returns and
resamples the candidate/control pair together. This preserves contemporaneous
market shocks within each sampled block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_backtest_official import TICKERS, fetch_official_series, load_official_prices, build_factors, perf, sliced
from research_dual_momentum import signal_models, absolute_masks, rebalance_dates, choose, era_metrics
from research_portfolio import simulate_self_financing
from research_broad_core import BROAD, broad_core_target


def pure_1306_target(dates: pd.DatetimeIndex) -> pd.DataFrame:
    cols = TICKERS + [BROAD, "CASH"]
    x = pd.DataFrame(0.0, index=dates, columns=cols)
    x[BROAD] = 1.0
    return x


def broad_plus_sector_ew_target(dates: pd.DatetimeIndex, core_share: float = 0.75) -> pd.DataFrame:
    cols = TICKERS + [BROAD, "CASH"]
    x = pd.DataFrame(0.0, index=dates, columns=cols)
    x[BROAD] = core_share
    for t in TICKERS:
        x[t] = (1.0 - core_share) / len(TICKERS)
    return x


def cashmatched_broad_control(candidate: pd.DataFrame, core_share: float = 0.75) -> pd.DataFrame:
    cols = TICKERS + [BROAD, "CASH"]
    out = pd.DataFrame(0.0, index=candidate.index, columns=cols)
    out[BROAD] = core_share
    tilt_cash = candidate["CASH"].clip(lower=0.0, upper=1.0 - core_share)
    risky_tilt = (1.0 - core_share) - tilt_cash
    for t in TICKERS:
        out[t] = risky_tilt / len(TICKERS)
    out["CASH"] = tilt_cash
    err = (out.sum(axis=1) - 1.0).abs().max()
    if float(err) > 1e-8:
        raise RuntimeError(f"cashmatched target sum error {err}")
    return out


def realized_metrics(r: pd.Series, turn: pd.Series, holdout_start: str) -> dict:
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


def monthly_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0


def path_metrics_monthly(r: np.ndarray) -> tuple[float, float, float]:
    if len(r) == 0:
        return np.nan, np.nan, np.nan
    wealth = np.cumprod(1.0 + r)
    years = len(r) / 12.0
    cagr = float(wealth[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth[-1] > 0 else np.nan
    peak = np.maximum.accumulate(wealth)
    mdd = float(np.min(wealth / peak - 1.0))
    sd = float(np.std(r, ddof=0))
    sharpe = float(np.mean(r) / sd * np.sqrt(12.0)) if sd > 0 else np.nan
    return cagr, mdd, sharpe


def paired_block_bootstrap(
    candidate: pd.Series,
    control: pd.Series,
    block_months: int,
    reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    x = pd.concat([candidate.rename("candidate"), control.rename("control")], axis=1).dropna()
    arr = x.to_numpy(float)
    n = len(arr)
    starts = np.arange(0, n - block_months + 1)
    nblocks = int(np.ceil(n / block_months))
    rows = []
    for rep in range(reps):
        chosen = rng.choice(starts, size=nblocks, replace=True)
        sample = np.concatenate([arr[s:s + block_months] for s in chosen], axis=0)[:n]
        c_cagr, c_mdd, c_sh = path_metrics_monthly(sample[:, 0])
        b_cagr, b_mdd, b_sh = path_metrics_monthly(sample[:, 1])
        rows.append({
            "rep": rep,
            "candidate_cagr": c_cagr,
            "control_cagr": b_cagr,
            "delta_cagr": c_cagr - b_cagr,
            "candidate_mdd": c_mdd,
            "control_mdd": b_mdd,
            "delta_mdd": c_mdd - b_mdd,
            "candidate_sharpe": c_sh,
            "control_sharpe": b_sh,
            "delta_sharpe": c_sh - b_sh,
        })
    return pd.DataFrame(rows)


def summarize_boot(df: pd.DataFrame, comparison: str, block: int) -> list[dict]:
    rows = []
    for metric in ["delta_cagr", "delta_mdd", "delta_sharpe"]:
        a = df[metric].dropna().to_numpy(float)
        rows.append({
            "comparison": comparison,
            "block_months": block,
            "metric": metric,
            "median": float(np.median(a)),
            "q025": float(np.quantile(a, 0.025)),
            "q975": float(np.quantile(a, 0.975)),
            "prob_positive": float((a > 0).mean()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-start", default="2009-04-01")
    ap.add_argument("--holdout-start", default="2018-01-01")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out-dir", type=Path, default=Path("research_broad_core_attribution"))
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
    candidate = broad_core_target(dates, tilt, 0.75)
    controls = {
        "cashmatched_same_cash_sector_ew": cashmatched_broad_control(candidate, 0.75),
        "75pct_1306_plus_25pct_sector_ew_no_cash": broad_plus_sector_ew_target(dates, 0.75),
        "100pct_1306": pure_1306_target(dates),
    }
    targets = {"candidate_75pct_1306_plus_25pct_top3": candidate, **controls}

    returns = {}
    realized_rows = []
    equity = {}
    for name, target in targets.items():
        r, weights, turn = simulate_self_financing(close, dr, target, risky, args.cost_bps)
        returns[name] = r
        realized_rows.append({
            "variant": name,
            "mean_cash_weight": float(weights["CASH"].mean()),
            **realized_metrics(r, turn, args.holdout_start),
        })
        equity[name] = (1.0 + r).cumprod()
    realized = pd.DataFrame(realized_rows)
    realized.to_csv(args.out_dir / "realized_results.csv", index=False)
    pd.DataFrame(equity).to_csv(args.out_dir / "equity.csv")

    c = realized[realized["variant"] == "candidate_75pct_1306_plus_25pct_top3"].iloc[0]
    attribution_rows = []
    for control_name in controls:
        b = realized[realized["variant"] == control_name].iloc[0]
        attribution_rows.append({
            "control": control_name,
            "delta_full_cagr": float(c["full_cagr"] - b["full_cagr"]),
            "delta_full_mdd": float(c["full_max_drawdown"] - b["full_max_drawdown"]),
            "delta_full_sharpe": float(c["full_sharpe"] - b["full_sharpe"]),
            "delta_pre2018_cagr": float(c["pre2018_cagr"] - b["pre2018_cagr"]),
            "delta_holdout_cagr": float(c["holdout_cagr"] - b["holdout_cagr"]),
            "delta_holdout_mdd": float(c["holdout_max_drawdown"] - b["holdout_max_drawdown"]),
            "delta_turnover": float(c["annualized_turnover"] - b["annualized_turnover"]),
        })
    attribution = pd.DataFrame(attribution_rows)
    attribution.to_csv(args.out_dir / "attribution.csv", index=False)

    rng = np.random.default_rng(args.seed)
    boot_summary_rows = []
    boot_detail_frames = []
    candidate_monthly = monthly_returns(returns["candidate_75pct_1306_plus_25pct_top3"])
    for control_name in controls:
        control_monthly = monthly_returns(returns[control_name])
        for block in [6, 12, 24]:
            b = paired_block_bootstrap(candidate_monthly, control_monthly, block, args.reps, rng)
            b.insert(0, "control", control_name)
            b.insert(1, "block_months", block)
            boot_detail_frames.append(b)
            boot_summary_rows.extend(summarize_boot(b, control_name, block))
    boot_detail = pd.concat(boot_detail_frames, ignore_index=True)
    boot_detail.to_csv(args.out_dir / "bootstrap_results.csv", index=False)
    boot_summary = pd.DataFrame(boot_summary_rows)
    boot_summary.to_csv(args.out_dir / "bootstrap_summary.csv", index=False)

    report = [
        "# TOPIX-17 Phase 17 — 1306-Core Attribution and Uncertainty",
        "",
        "Frozen candidate: 75% 1306 + 25% Top3 classic 6m/12-1m tilt, MA200+positive12m slot-gate, failed tilt slots to CASH.",
        "",
        f"Transaction cost: {args.cost_bps:g} bp per ETF side; bootstrap reps: {args.reps:,}.",
        "",
        "## Realized results",
        "",
        realized.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Realized candidate minus controls",
        "",
        attribution.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired moving-block bootstrap",
        "",
        boot_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- Candidate minus cash-matched sector-EW isolates sector-selection identity/ranking while preserving the candidate's tilt CASH schedule.",
        "- Candidate minus 75%1306+25%sector-EW-no-CASH includes both the slot-gate/CASH effect and sector selection.",
        "- Candidate minus 100%1306 asks whether the complete overlay improves the broad-core portfolio at all.",
        "- Positive `delta_mdd` means the candidate has a shallower maximum drawdown.",
        "",
        "This phase evaluates a pre-specified implementation simplification discovered after earlier research; it is not a pristine confirmatory trial. If uncertainty remains large, the correct next step is prospective tracking rather than more historical threshold tuning.",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "phase": 17,
        "data_source": "Nomura Asset Management official NEXT FUNDS historical CSV",
        "series": "distribution-reinvested NAV per share",
        "candidate": "75% 1306 + 25% classic 6m/12-1m Top3 slot-gate, MA200+positive12m",
        "cost_bps": args.cost_bps,
        "bootstrap": {"reps": args.reps, "blocks_months": [6, 12, 24], "seed": args.seed},
        "sector_source_metadata": source_meta,
        "broad_source_metadata": broad_meta,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(realized.to_string(index=False))
    print(attribution.to_string(index=False))
    print(boot_summary.to_string(index=False))


if __name__ == "__main__":
    main()
