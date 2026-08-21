#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared self-financing portfolio simulation for TOPIX-17 research.

Signals are observed at a day's close. A target is scheduled for the next NAV
observation and executed at that day's close, so the new target earns returns
from the following observation onward. This is deliberately conservative.

Between rebalances, weights drift with asset returns. No hidden daily
rebalancing is assumed.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def schedule_targets(
    trading_dates: pd.DatetimeIndex,
    signal_target: pd.DataFrame,
) -> Dict[pd.Timestamp, pd.Series]:
    scheduled: Dict[pd.Timestamp, pd.Series] = {}
    for signal_date, row in signal_target.iterrows():
        future = trading_dates[trading_dates > signal_date]
        if len(future):
            scheduled[future[0]] = row.astype(float)
    return scheduled


def simulate_self_financing(
    close: pd.DataFrame,
    daily_ret: pd.DataFrame,
    signal_target: pd.DataFrame,
    risky_assets: Iterable[str],
    cost_bps_per_side: float,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    risky = list(risky_assets)
    columns = risky + ["CASH"]
    if list(signal_target.columns) != columns:
        signal_target = signal_target.reindex(columns=columns, fill_value=0.0)

    all_dates = close.index
    scheduled = schedule_targets(all_dates, signal_target)
    if not scheduled:
        raise RuntimeError("no executable signals")

    dates = all_dates[all_dates >= min(scheduled)]
    weights = pd.Series(0.0, index=columns)
    weights["CASH"] = 1.0

    net_returns = pd.Series(0.0, index=dates, dtype=float)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    weight_history = pd.DataFrame(0.0, index=dates, columns=columns)
    cost_rate = float(cost_bps_per_side) / 10000.0

    for date in dates:
        asset_ret = daily_ret.reindex(index=[date], columns=risky).iloc[0]
        if asset_ret.isna().any():
            missing = asset_ret.index[asset_ret.isna()].tolist()
            held_missing = [x for x in missing if weights[x] > 1e-10]
            if held_missing:
                raise RuntimeError(f"missing return for held assets on {date.date()}: {held_missing}")
            asset_ret = asset_ret.fillna(0.0)

        gross_ret = float((weights[risky] * asset_ret).sum())
        wealth_factor = 1.0 + gross_ret
        if not np.isfinite(wealth_factor) or wealth_factor <= 0:
            raise RuntimeError(f"invalid portfolio wealth factor on {date.date()}: {wealth_factor}")

        # Drift end-of-day weights before any rebalance trade.
        drifted = pd.Series(0.0, index=columns)
        drifted[risky] = weights[risky] * (1.0 + asset_ret) / wealth_factor
        drifted["CASH"] = weights["CASH"] / wealth_factor

        trade_cost_fraction = 0.0
        if date in scheduled:
            target = scheduled[date].reindex(columns).fillna(0.0).astype(float)
            if (target < -1e-10).any():
                raise RuntimeError(f"negative target weights on {date.date()}")
            total = float(target.sum())
            if abs(total - 1.0) > 1e-8:
                raise RuntimeError(f"target weights do not sum to 1 on {date.date()}: {total}")

            # Cost only ETF trades. CASH itself has no bid/ask commission.
            # A->B produces 200% gross risky turnover; A->CASH produces 100%.
            risky_turnover = float((target[risky] - drifted[risky]).abs().sum())
            turnover.loc[date] = risky_turnover
            trade_cost_fraction = risky_turnover * cost_rate
            if trade_cost_fraction >= 1.0:
                raise RuntimeError(f"transaction cost consumes portfolio on {date.date()}")
            weights = target
        else:
            weights = drifted

        net_factor = wealth_factor * (1.0 - trade_cost_fraction)
        net_returns.loc[date] = net_factor - 1.0
        weight_history.loc[date] = weights

    return net_returns, weight_history, turnover


def equal_weight_signal(
    dates: pd.DatetimeIndex,
    risky_assets: Iterable[str],
) -> pd.DataFrame:
    risky = list(risky_assets)
    target = pd.DataFrame(0.0, index=dates, columns=risky + ["CASH"])
    target.loc[:, risky] = 1.0 / len(risky)
    return target


def buy_and_hold_signal(
    first_signal_date: pd.Timestamp,
    risky_assets: Iterable[str],
) -> pd.DataFrame:
    risky = list(risky_assets)
    target = pd.DataFrame(0.0, index=[first_signal_date], columns=risky + ["CASH"])
    target.loc[:, risky] = 1.0 / len(risky)
    return target
