# tse-sector-rotation

東証の **33業種ローテーションを独自の代表銘柄バスケットで分析**し、その結果を実際に売買できる **NEXT FUNDS TOPIX-17 ETF（1617–1633）+ CASH** に落とし込むスクリーナーです。

GitHub Actions が平日 **18:30 JST** に自動実行し、33業種ランキング・ETFランキング・市場レジーム・CASH判定を `data/` とこのREADMEへ保存します。`workflow_dispatch` から手動実行もできます。

> `config/industry_constituents.csv` から生成する33業種系列は、JPX公式指数値を取得・複製したものではなく、構成銘柄の市場価格から作る独自の代理指数です。

## Latest ranking

<!-- RANKING_START -->
初回のGitHub Actions実行後にランキングがここへ表示されます。
<!-- RANKING_END -->

## Architecture

```text
代表銘柄バスケット
        ↓
東証33業種 proxy ranking
        ↓
33業種 → TOPIX-17 マッピング
        ↓
TOPIX-17 ETF自身のテクニカル確認
        ↓
17 ETF + CASH を同一ランキング
        ↓
BUY / WATCH / NEUTRAL / AVOID
または NO TRADE / CASH
```

## Scoring

### 33業種レイヤー

| Layer | 主な入力 |
|---|---|
| Technical | 1/3/6か月リターン、50/200日線 |
| Sector rotation | TOPIX対比1/3/6か月相対強度、相対強度の加速 |
| Breadth | 50日線上、200日線上、3か月プラス、TOPIX超過の銘柄比率 |
| Macro | 日本長期金利、鉱工業生産、CPI、原油、USD/JPY、VIX |
| Fundamental | PER、PBR、配当利回り、ROE、利益成長、業績修正（任意） |
| Risk | 20日ボラティリティ、6か月最大ドローダウン |
| Absolute | 3/6か月リターン、50/200日線の絶対トレンド |

相対ランキングだけでなく絶対トレンドを明示的に混ぜることで、**「全部下がっている中の1位」を機械的に買う問題**を抑えます。

### ETFレイヤー

各TOPIX-17 ETFは、対応する33業種のスコアとETF自身の価格データを組み合わせます。

- 対応33業種の総合スコア
- ETF自身のモメンタム / TOPIX相対強度
- ETF自身の50/200日線
- ボラティリティ / 最大ドローダウン
- 売買代金
- 絶対トレンド

最終ETFスコアは **相対・業種複合評価60% + 絶対トレンド40% - ペナルティ** を基本とします。

## CASH — 18番目の候補

CASHを「スコア0の待機先」ではなく、ETFと競合する正式な18番目の候補として扱います。

CASH scoreは主に以下で上昇します。

- TOPIX proxy（1306.T）が50日・200日線を下回る
- TOPIXの3か月・6か月リターンが悪化する
- 17 ETFのうち50日・200日線を上回る銘柄比率が低下する
- 3か月プラスのETF比率が低下する
- VIXベースのrisk-offシグナルが強まる
- 最上位ETFですら十分なスコアを得られない

CASHがランキング1位になった場合、システムの最上位判断は **`NO TRADE / CASH`** です。

これは「常に何かを買う」ことを避けるための仕組みであり、下落相場で利益を保証するものではありません。

## Signals

### Industry

- `LEADING`: 相対・絶対の両面で強くBreadthも十分
- `IMPROVING`: 上位へ改善中
- `NEUTRAL`: 明確な優位性なし
- `LAGGING`: 相対的に弱い

### ETF

- `BUY`: 高スコアかつ絶対トレンドも良好
- `WATCH`: 条件は良いが押し目・確認待ち
- `NEUTRAL`: 優位性が弱い
- `AVOID`: 低スコア
- `NO TRADE / CASH`: 現金待機が全ETFより上位

## Fundamental data

`config/fundamentals.csv` は任意です。最低3項目が入った業種だけファンダメンタルスコアを有効化します。

無料データだけで最新ファンダメンタルを正確に自動取得することを前提にせず、**未入力値を推測・補完しません**。ファンダメンタルが一件も有効でない場合、そのウェイトは有効な他因子へ再配分されます。

率の入力は小数です（例: 2.5% = `0.025`）。

## Data sources

- 個別株・ETF価格 / 出来高: Yahoo Finance（`yfinance`）
- マクロ系列: Federal Reserve Economic Data (FRED)
- ベンチマーク: 1306.T（TOPIX連動ETF）
- 33業種分類・代表銘柄設定: `config/industry_constituents.csv`

無料データソースには取得遅延、欠損、仕様変更があります。売買執行には使わず、ランキング出力を必ず検証してください。

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
data/industry_latest.csv
data/industry_history.csv
data/etf_latest.csv
data/etf_history.csv
data/macro.json
```

## Automation

`.github/workflows/screen.yml`

- 平日 18:30 JST
- 手動実行対応
- 33業種 proxyランキング更新
- TOPIX-17 ETF + CASHランキング更新
- 日次履歴をCSVに蓄積
- READMEランキングを自動更新
- 変更がある場合のみ `main` へ自動コミット

## Disclaimer

本リポジトリは調査・スクリーニング用途です。投資助言、利益保証、発注システムではありません。
