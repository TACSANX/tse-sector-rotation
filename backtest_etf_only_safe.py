#!/usr/bin/env python3
"""Compatibility wrapper for backtest_etf_only.py.

- Disable yfinance threads to avoid SQLite cache lock on GitHub-hosted runners.
- Make row-wise idxmax tolerate warm-up rows where every factor is NaN.
- Use Yahoo Japan's TOPIX code 998405.T as the historical benchmark.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

_orig_download = yf.download
_orig_idxmax = pd.DataFrame.idxmax


def serial_download(*args, **kwargs):
    kwargs["threads"] = False
    return _orig_download(*args, **kwargs)


def safe_idxmax(self, axis=0, skipna=True, numeric_only=False):
    if axis in (1, "columns"):
        out = pd.Series(index=self.index, dtype=object)
        valid = self.notna().any(axis=1)
        if valid.any():
            out.loc[valid] = _orig_idxmax(
                self.loc[valid], axis=1, skipna=skipna, numeric_only=numeric_only
            )
        return out
    return _orig_idxmax(self, axis=axis, skipna=skipna, numeric_only=numeric_only)


yf.download = serial_download
pd.DataFrame.idxmax = safe_idxmax

import backtest_etf_only
backtest_etf_only.BENCHMARK = "998405.T"

if __name__ == "__main__":
    backtest_etf_only.main()
