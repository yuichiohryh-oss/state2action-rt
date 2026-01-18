# PROJECT_STATE.md

## 目的

本プロジェクトは、リアルタイム対戦ゲームのプレイ画面（動画）から
**「次に取るべき行動を提案する AI」** を構築することを目的とする。

* 自動操作ではなく **人間への提案（Top-k）** が主目的
* 教師あり学習を基本とし、段階的に高度化する
* repo 内部では実名カード名を使用し、公開記事では一般名に一般化する

---

## 全体構成（俯瞰）

```
動画 / スクリーン
   ↓
[前処理]
  - 盤面 ROI 切り出し
  - フレーム整列（current / past / diff）
  - 手札特徴抽出（hand_available / hand_card_ids / hand_scores）
  - エリクサー特徴抽出（elixir / elixir_frac）
   ↓
入力テンソル [9,256,256] + 数値特徴
   ↓
CNN（将来 Transformer へ拡張可能）
   ↓
2ヘッド出力
  - action head
  - grid head
   ↓
Top-k 行動提案（hand_* / elixir による action mask を適用）
```

---

## 入力設計（確定）

### 盤面画像入力

* 盤面 ROI を 256×256 に正規化
* 入力テンソル shape: **[9, 256, 256]**

| チャンネル | 内容                         |
| ----- | -------------------------- |
| 0–2   | current frame（RGB）         |
| 3–5   | past frame（RGB）            |
| 6–8   | diff = current − past（RGB） |

* 2フレーム入力 + 差分チャンネル
* 入力 IF は将来も固定（CNN / Transformer 差し替え前提）

---

## 盤面 ROI（確定ベースライン）

* 対象：720p 縦長動画（YouTube / scrcpy）
* 実機確認済みパラメータ

```
--roi-y1 110
--roi-y2-mode fixed
--roi-y2-fixed 570
--gw 6 --gh 9
--lead-sec 0.8
```

* 上下 UI を除外し、盤面のみを安定して取得
* 動画レイアウトが変わる場合は inspect_sample.py で必ず目視確認

---

## 出力設計（確定）

### 2ヘッド構成

1. **action head**

   * 行動の種類を分類
   * カード / スキル / NOOP

2. **grid head**

   * 盤面上の配置位置（6×9）

---

## 行動実行フロー（重要・確定）

### カード行動（2段階）

1. **手札選択**

   * 画面下部の手札 UI からカードを選択
   * 手札は常に 4 枠表示（順序はランダム）

2. **盤面配置**

   * 選択後、盤面上の grid を指定

推論時の要件：

* 推論された card action が **hand_card_ids に存在**すること
* 対応する **slot_index（0–3）** を特定
* 存在しない／available=0 の場合は **実行しない**（安全側）
* 候補が空なら **WAIT（NOOP）にフォールバック**

### スキル行動（1段階）

* スキルボタンを押すだけで即時発動
* ボタンがグレー／存在しない場合は使用不可

### WAIT / NOOP（0段階）

* 何も操作しない
* エリクサー不足や手札制約により行動不能な場合の **有効な提案**

---

## 行動制約と手札特徴（最優先）

### 背景

盤面が同じでも、

* 手札状態
* エリクサー状況

によって取れる行動は異なる。

→ **hand_* / elixir を使った action mask が必須**

### 手札特徴（hand_features）

* dataset / inspect / 推論で **同一 API を使用**
* UI のカラー／グレー表示を直接利用

#### ROI 仕様

* Default：画面下端 **90–97%**
* scrcpy 330x752：固定 pixel ROI

  * x1=75, x2=318, y1=630, y2=680
* その他解像度：ratio fallback
* CLI override（--hand-roi-x1/x2/y1/y2）が最優先

#### 判定仕様

* 横方向に 4 分割（4 スロット）
* slot **inner-crop** で mean_S を計算（UI混入回避）
* HSV 彩度 S の平均値を使用

| 判定   | 条件           |
| ---- | ------------ |
| 出せる  | mean_S > 約30 |
| 出せない | mean_S ≤ 約30 |

#### データ表現

```json
"hand_available": [0, 0, 1, 1],
"hand_card_ids":  [-1, -1, 3, 7],
"hand_scores":    [0.0, 0.0, 0.86, 0.91]
```

* 推論時：hand_available + hand_card_ids で card action をマスク
* 欠損時：**カード行動を除外し WAIT のみ許可**

---

## エリクサー特徴（導入済み・優先度：高）

### 方針

* OCR は用いず、**エリクサーバーの塗り割合を画像処理で推定**する
* dataset / inspect / 推論で **同一ロジック（elixir_features）を使用**
* 推定に失敗した場合は **安全側（-1）** として扱う

### ROI 仕様

* scrcpy 330x752：固定 pixel ROI（デフォルト）

  * x1=40, x2=320, y1=712, y2=745
* 実動画によりズレるため、inspect_elixir.py で目視確認・調整する
* CLI override（--elixir-roi-x1/x2/y1/y2）を用意

### 指標

* `elixir_frac` : 0.0–10.0（バー塗りの推定値。fill_ratio * 10）
* `elixir` : 0–10 の整数値（`round(elixir_frac)` を clamp）
* 欠損時：`elixir = -1`, `elixir_frac = -1.0`

### データ表現

```json
"elixir": 7,
"elixir_frac": 7.2
```

* 学習時：数値特徴量として使用（hand_* より優先度は低い）
* 推論時：

  * 出せないカードをマスク（--enable-elixir-mask）
  * 全候補が落ちた場合は **WAIT にフォールバック**

---

## 学習方針

* 教師あり学習
* 出力は Top-k 提案
* 人間が最終判断

### NOOP / WAIT の扱い

* 実データでは NOOP / WAIT が多くなりがち
* augment_noop.py で補完
* loss weight / mask で過多を防止

---

## 設計思想

* **入出力 IF を固定**し、内部実装を差し替え可能に
* 盤面理解（CNN）と制約理解（hand / elixir）を分離
* 完全自動化より、人間との協調を重視

---

## 手札テンプレ（hand_card_id）

```
templates/hand_cards/
  0.png  # cannon
  1.png  # skeletons
  2.png  # fireball
  3.png  # hog
  4.png  # icegolem
  5.png  # ice_spirit
  6.png  # log
  7.png  # musketeer
```

Note:

* 進化カード（evo）は現時点では unknown として扱い、テンプレには含めない
* 進化判定は将来の独立モジュールとして検討する
