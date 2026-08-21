#!/usr/bin/env python3
"""Run the historical backtest with macro inputs neutralized.

This isolates the price/rotation/breadth/risk/CASH decision core from FRED
availability and historical-vintage issues. The production screener is not changed.
"""
from __future__ import annotations

import pandas as pd
import backtest


def neutral_fred_series(_: str) -> pd.Series:
    return pd.Series(dtype=float)


def neutral_macro_at(_fred, _date):
    return {
        "rates": 0.0,
        "growth": 0.0,
        "inflation": 0.0,
        "oil": 0.0,
        "usdjpy": 0.0,
        "risk_off": 0.0,
        "regime": "Mixed / transition",
    }


_original_report = backtest.write_report


def price_core_report(summary, meta, annual, out):
    _original_report(summary, meta, annual, out)
    text = out.read_text(encoding="utf-8")
    text = text.replace(
        "- Macro: FRED historical observations with conservative 45-day monthly / 1-day daily lag",
        "- Macro: neutralized in this run; this report isolates price/rotation/breadth/risk/CASH logic",
    )
    out.write_text(text, encoding="utf-8")


backtest.fred_series = neutral_fred_series
backtest.macro_at = neutral_macro_at
backtest.write_report = price_core_report

if __name__ == "__main__":
    backtest.main()
