#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared self-financing portfolio simulation for TOPIX-17 research.

Signals are observed at a day's close. A target is scheduled for the next NAV
observation and executed at that day's close, so the new target earns returns
from the following observation onward. This is deliberately conservative.

Between rebalances, weights drift with asset returns. No hidden daily
rebalancing is assumed. The inner loop uses NumPy arrays for speed while
preserving exactly the same self-financing accounting.
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
        pos = int(trading_dates.searchsorted(signal_date, side="right"))
        if pos < len(trading_dates):
            scheduled[trading_dates[pos]] = row.astype(float)
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
    signal_target = signal_target.reindex(columns=columns, fill_value=0.0).astype(float)

    all_dates = close.index
    if not all_dates.is_monotonic_increasing:
        raise RuntimeError("close index must be sorted")

    # Map signal dates to execution positions: first NAV observation strictly
    # after the signal date. Targets are validated once before the fast loop.
    schedule_by_pos: dict[int, np.ndarray] = {}
    for signal_date, row in signal_target.iterrows():
        pos = int(all_dates.searchsorted(signal_date, side="right"))
        if pos >= len(all_dates):
            continue
        target = row.to_numpy(dtype=float)
        if np.any(target < -1e-10):
            raise RuntimeError(f"negative target weights for signal {signal_date.date()}")
        total = float(target.sum())
        if abs(total - 1.0) > 1e-8:
            raise RuntimeError(f"target weights do not sum to 1 for signal {signal_date.date()}: {total}")
        schedule_by_pos[pos] = target
    if not schedule_by_pos:
        raise RuntimeError("no executable signals")

    start_pos = min(schedule_by_pos)
    dates = all_dates[start_pos:]
    ret_matrix = daily_ret.reindex(index=all_dates, columns=risky).to_numpy(dtype=float)
    n_risky = len(risky)
    n_days = len(dates)

    current_risky = np.zeros(n_risky, dtype=float)
    current_cash = 1.0
    net_arr = np.zeros(n_days, dtype=float)
    turn_arr = np.zeros(n_days, dtype=float)
    weight_arr = np.zeros((n_days, n_risky + 1), dtype=float)
    cost_rate = float(cost_bps_per_side) / 10000.0

    for local_i, global_i in enumerate(range(start_pos, len(all_dates))):
        asset_ret = ret_matrix[global_i]
        if np.isnan(asset_ret).any():
            held_missing = np.isnan(asset_ret) & (current_risky > 1e-10)
            if held_missing.any():
                names = [risky[i] for i in np.flatnonzero(held_missing)]
                raise RuntimeError(f"missing return for held assets on {all_dates[global_i].date()}: {names}")
            asset_ret = np.nan_to_num(asset_ret, nan=0.0)

        gross_ret = float(np.dot(current_risky, asset_ret))
        wealth_factor = 1.0 + gross_ret
        if not np.isfinite(wealth_factor) or wealth_factor <= 0.0:
            raise RuntimeError(f"invalid portfolio wealth factor on {all_dates[global_i].date()}: {wealth_factor}")

        drifted_risky = current_risky * (1.0 + asset_ret) / wealth_factor
        drifted_cash = current_cash / wealth_factor
        trade_cost_fraction = 0.0

        target = schedule_by_pos.get(global_i)
        if target is not None:
            target_risky = target[:n_risky]
            risky_turnover = float(np.abs(target_risky - drifted_risky).sum())
            turn_arr[local_i] = risky_turnover
            trade_cost_fraction = risky_turnover * cost_rate
            if trade_cost_fraction >= 1.0:
                raise RuntimeError(f"transaction cost consumes portfolio on {all_dates[global_i].date()}")
            current_risky = target_risky.copy()
            current_cash = float(target[-1])
        else:
            current_risky = drifted_risky
            current_cash = float(drifted_cash)

        net_arr[local_i] = wealth_factor * (1.0 - trade_cost_fraction) - 1.0
        weight_arr[local_i, :n_risky] = current_risky
        weight_arr[local_i, -1] = current_cash

    net_returns = pd.Series(net_arr, index=dates, dtype=float)
    turnover = pd.Series(turn_arr, index=dates, dtype=float)
    weight_history = pd.DataFrame(weight_arr, index=dates, columns=columns)
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
