# tse-sector-rotation

東証上場の **NEXT FUNDS TOPIX-17 業種別ETF（1617–1633）** を、テクニカル・相対強度・マクロ環境・リスク・流動性・任意のファンダメンタル入力で毎営業日スクリーニングするリポジトリです。

GitHub Actions が平日 **18:30 JST** に自動実行し、ランキングを `data/` とこのREADMEへ保存します。`workflow_dispatch` から手動実行もできます。

## Latest ranking

<!-- RANKING_START -->
初回のGitHub Actions実行後にランキングがここへ表示されます。
<!-- RANKING_END -->

## Scoring

最終スコアは 0–100 点です。通常局面では以下を基準にし、景気減速・スタグフレーション局面ではマクロの比重を自動的に引き上げます。

| Layer | 主な入力 |
|---|---|
| Technical | 1/3/6か月リターン、50/200日線 |
| Sector rotation | TOPIX対比の1/3/6か月相対強度、相対強度の加速 |
| Macro | 日本長期金利、鉱工業生産、CPI、原油、USD/JPY、VIX |
| Fundamental | PER、PBR、配当利回り、ROE、利益成長、業績修正 |
| Risk | 20日ボラティリティ、6か月最大ドローダウン |
| Liquidity | 20日平均売買代金 |

### Signals

- `BUY`: 高スコアかつ200日線を維持
- `WATCH`: 条件は良いが押し目・確認待ち
- `NEUTRAL`: 優位性が弱い
- `AVOID`: 相対的に低スコア

200日線を5%以上割る、RSIが極端に過熱する、流動性が低い、強いrisk-off局面で高ボラティリティ、といった条件にはペナルティを加えます。

## Fundamental data

`config/fundamentals.csv` は任意です。最低3項目が入ったセクターだけファンダメンタルスコアを有効化します。

無料データだけで最新のTOPIX-17構成銘柄ファンダメンタルを正確に自動取得するのは信頼性に問題があるため、**未入力値を推測・補完しません**。ファンダメンタルが一件も有効でない場合、そのウェイトはテクニカル・ローテーション・マクロ・リスク・流動性へ比例再配分されます。

率の入力は小数です（例: 2.5% = `0.025`）。

## Data sources

- ETF価格・出来高: Yahoo Finance（`yfinance`）
- マクロ系列: Federal Reserve Economic Data (FRED)
- ベンチマーク: 1306.T（TOPIX連動ETF）を価格ベースの相対強度計算に使用

無料データソースは取得遅延、欠損、仕様変更があり得ます。売買執行には使わず、ランキング出力を必ず検証してください。

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python screener.py
python scripts/update_readme.py
```

出力:

```text
data/latest.csv
data/history.csv
data/macro.json
```

## Automation

`.github/workflows/screen.yml`

- 平日 18:30 JST
- 手動実行対応
- `data/latest.csv` を更新
- 日次スナップショットを `data/history.csv` に蓄積
- マクロ状態を `data/macro.json` に保存
- READMEランキングを自動更新
- 変更がある場合のみ `main` へ自動コミット

## Disclaimer

本リポジトリは調査・スクリーニング用途です。投資助言、利益保証、発注システムではありません。
