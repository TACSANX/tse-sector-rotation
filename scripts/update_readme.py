#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

START = "<!-- RANKING_START -->"
END = "<!-- RANKING_END -->"


def pct(v) -> str:
    try:
        if pd.isna(v):
            return "-"
        return f"{float(v) * 100:+.1f}%"
    except Exception:
        return "-"


def num(v, digits=1) -> str:
    try:
        if pd.isna(v):
            return "-"
        return f"{float(v):.{digits}f}"
    except Exception:
        return "-"


def build_block(industry_path: Path, etf_path: Path, macro_path: Path) -> str:
    ind = pd.read_csv(industry_path, index_col=0)
    etf = pd.read_csv(etf_path, index_col=0)
    meta = json.loads(macro_path.read_text(encoding="utf-8"))
    allocation = meta.get("allocation_decision", {})
    market = meta.get("market", {})
    decision = "CASH / NO TRADE" if allocation.get("is_cash") else f"{allocation.get('code', '-')} {allocation.get('name', '')}"

    rows = [
        f"**データ基準日:** {meta.get('asof', '-')}",
        f"**マクロ・レジーム:** {meta.get('macro', {}).get('regime', '-')}",
        f"**市場Risk-on score:** {num(market.get('risk_on_score'))}",
        f"**現在の最上位候補:** **{decision}**（score {num(allocation.get('score'))}）",
        f"**ファンダメンタル層:** {'有効' if meta.get('fundamentals_active') else '未入力のため重みを他因子へ再配分'}",
        "",
        "### 東証33業種ローテーション（独自代理指数）",
        "",
        "| Rank | Industry | Score | Signal | Abs | 1M | 3M RS | Breadth 200D | ETF | Purity |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for industry, row in ind.head(15).iterrows():
        rows.append(
            f"| {int(row['rank'])} | {industry} | {num(row['final_score'])} | {row['signal']} | "
            f"{num(row['absolute_score'])} | {pct(row['ret_1m'])} | {pct(row['rs_3m'])} | {pct(row['breadth_200d'])} | "
            f"{str(row['execution_etf_code'])} | {num(row['execution_purity'], 0)} |"
        )

    rows += [
        "",
        "### 売買候補：TOPIX-17 ETF + CASH",
        "",
        "CASHは18番目の候補です。全ETFの絶対トレンドや市場Breadthが悪化するとCASH scoreが上昇し、1位なら `NO TRADE / CASH` と判定します。",
        "",
        "| Rank | Code | ETF group | Score | Signal | Abs | Underlying | 1M | 3M | 200D gap |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    show = etf.head(12).copy()
    if "CASH" in etf.index and "CASH" not in show.index:
        show = pd.concat([show, etf.loc[["CASH"]]])
    for _, row in show.iterrows():
        rows.append(
            f"| {int(row['rank'])} | {row['code']} | {row['etf_group']} | {num(row['final_score'])} | "
            f"{row['signal']} | {num(row['absolute_score'])} | {num(row['underlying_score'])} | "
            f"{pct(row['ret_1m'])} | {pct(row['ret_3m'])} | {pct(row['ma200_gap'])} |"
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
