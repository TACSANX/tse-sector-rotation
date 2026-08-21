#!/usr/bin/env python3
"""Monthly-rebalance price-core backtest."""
from __future__ import annotations

import pandas as pd
import backtest_price_core  # noqa: F401 - applies neutral macro patches
import backtest


def monthly_signal_dates(bench: pd.Series, start: str) -> list[pd.Timestamp]:
    idx = bench.loc[pd.Timestamp(start):].dropna().index
    if idx.empty:
        return []
    s = pd.Series(idx, index=idx)
    last = s.groupby(idx.to_period("M")).last()
    return list(pd.DatetimeIndex(last.values))


backtest.signal_dates = monthly_signal_dates

if __name__ == "__main__":
    backtest.main()
