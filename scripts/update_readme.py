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


def build_block(industry_path: Path, etf_path: Path, macro_path: Path) -> str:
    ind = pd.read_csv(industry_path, index_col=0)
    etf = pd.read_csv(etf_path, index_col=0)
    meta = json.loads(macro_path.read_text(encoding="utf-8"))
    rows = [
        f"**データ基準日:** {meta.get('asof', '-')}",
        f"**マクロ・レジーム:** {meta.get('macro', {}).get('regime', '-')}",
        f"**ファンダメンタル層:** {'有効' if meta.get('fundamentals_active') else '未入力のため重みを他因子へ再配分'}",
        "",
        "### 東証33業種ローテーション（代理指数）",
        "",
        "| Rank | Industry | Score | Signal | 1M | 3M RS | Breadth 200D | ETF | Purity |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for industry, row in ind.head(15).iterrows():
        rows.append(
            f"| {int(row['rank'])} | {industry} | {num(row['final_score'])} | {row['signal']} | "
            f"{pct(row['ret_1m'])} | {pct(row['rs_3m'])} | {pct(row['breadth_200d'])} | "
            f"{str(row['execution_etf_code'])} | {num(row['execution_purity'], 0)} |"
        )
    rows += [
        "",
        "### 売買候補ETFランキング",
        "",
        "| Rank | Code | ETF group | Score | Signal | Underlying | Purity | 1M | 3M |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in etf.head(12).iterrows():
        rows.append(
            f"| {int(row['rank'])} | {row['code']} | {row['etf_group']} | {num(row['final_score'])} | "
            f"{row['signal']} | {num(row['underlying_score'])} | {num(row['purity_score'], 0)} | "
            f"{pct(row['ret_1m'])} | {pct(row['ret_3m'])} |"
        )
    return "\n".join(rows)


def main() -> None:
    readme = Path("README.md")
    industry = Path("data/industry_latest.csv")
    etf = Path("data/etf_latest.csv")
    macro = Path("data/macro.json")
    if not all(p.exists() for p in (readme, industry, etf, macro)):
        return
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise RuntimeError("README ranking markers are missing")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    block = build_block(industry, etf, macro)
    readme.write_text(before + START + "\n" + block + "\n" + END + after, encoding="utf-8")


if __name__ == "__main__":
    main()
