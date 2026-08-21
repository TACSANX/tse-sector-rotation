#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

START = "<!-- RANKING_START -->"
END = "<!-- RANKING_END -->"


def pct(v) -> str:
    try:
        return f"{float(v) * 100:+.1f}%"
    except Exception:
        return "-"


def num(v, digits=1) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "-"


def build_block(latest_path: Path, macro_path: Path) -> str:
    df = pd.read_csv(latest_path, index_col=0)
    macro = json.loads(macro_path.read_text(encoding="utf-8"))
    rows = [
        f"**データ基準日:** {macro.get('asof', '-')}",
        f"**マクロ・レジーム:** {macro.get('macro', {}).get('regime', '-')}",
        f"**ファンダメンタル層:** {'有効' if macro.get('fundamentals_active') else '未入力のため重みを他因子へ再配分'}",
        "",
        "| Rank | Code | Sector | Score | Signal | 1M | 3M | Rotation | Macro | RSI |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for ticker, row in df.iterrows():
        code = str(ticker).replace(".T", "")
        rows.append(
            f"| {int(row['rank'])} | {code} | {row['sector']} | "
            f"{num(row['final_score'])} | {row['signal']} | "
            f"{pct(row['ret_1m'])} | {pct(row['ret_3m'])} | "
            f"{num(row['rotation_score'])} | {num(row['macro_score'])} | {num(row['rsi14'])} |"
        )
    return "\n".join(rows)


def main() -> None:
    readme = Path("README.md")
    latest = Path("data/latest.csv")
    macro = Path("data/macro.json")
    if not (readme.exists() and latest.exists() and macro.exists()):
        return
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise RuntimeError("README ranking markers are missing")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(before + START + "\n" + build_block(latest, macro) + "\n" + END + after, encoding="utf-8")


if __name__ == "__main__":
    main()
