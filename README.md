# state2action-rt

## 目的

本リポジトリは、リアルタイム対戦ゲームのプレイ動画（画面録画）から、
**模倣学習（Imitation Learning）用の軽量なデータセットを構築し、次に取るべき行動を提案するモデル**を作成することを目的としています。

* 盤面（アリーナ）ROI を切り出して 256x256 に正規化
* 行動座標を 6x9 グリッド ID に変換
* 推論時は **人間への提案（Top-k）** を目的とし、自動操作は行いません

本リポジトリでは、特定ゲームや IP に依存しないよう、
コード・ドキュメント内では「game / arena / unit / spell / action」などの**一般名称のみ**を使用します。

---

## 動作環境

* Python 3.10+
* opencv-python, numpy
* torch, torchvision, tqdm（学習・推論用）

---

## 入力ファイル：events.jsonl

`events.jsonl` は 1 行 1 レコードの JSONL 形式です。スキーマは柔軟ですが、以下のキーは必須です。

* `t` : float（秒）
* `x` : float（画面座標・左上原点）
* `y` : float
* `action_id` : string

例：

```jsonl
{"t": 12.34, "x": 512.0, "y": 640.0, "action_id": "spell_01"}
{"t": 13.10, "x": 420.0, "y": 700.0, "action_id": "unit_07", "note": "extra fields ok"}
{"t": 15.42, "x": 300.0, "y": 500.0, "action_id": "action_generic"}
```

### ラベラーの起動

```bash
python tools/label_events.py \
  --video /path/to/video.mp4 \
  --out /path/to/events.jsonl
```

---

## Windows / PowerShell 利用時の注意

PowerShell では Linux / macOS で一般的な `\\` による改行は使用できません。
**必ずバッククォート（`）を使用してください。**

```powershell
python tools/inspect_hand.py `
  --video videos/example.mp4 `
  --out-dir out/debug
```

---

## データセット生成

```bash
python tools/build_dataset.py \
  --video /path/to/video.mp4 \
  --events /path/to/events.jsonl \
  --out-dir /path/to/out \
  --lead-sec 0.8 \
  --gw 6 --gh 9
```

生成される `dataset.jsonl` の各レコードには、以下の情報が含まれます（一部抜粋）：

* `action_id` : 行動ラベル（文字列）
* `grid_id` : 盤面グリッド ID（0..53）
* `state_path` : 256x256 盤面画像へのパス
* `hand_available` : [0/1, 0/1, 0/1, 0/1]
* `hand_card_ids` : [card_id または -1, ...]
* `hand_scores` : [score または 0.0, ...]

`hand_*` フィールドは **常に出力されます**。
テンプレートが存在しない場合でも、
`hand_card_ids = -1`、`hand_scores = 0.0` が設定されます。

---

## 手札特徴の確認（inspect_hand）

```bash
python tools/inspect_hand.py \
  --video /path/to/video.mp4 \
  --out-dir out/hand_debug \
  --stride 15
```

### Hand ROI の既定値

* **330x752（scrcpy）** : 固定ピクセル

  * `x1=75, x2=318, y1=630, y2=680`
* その他の解像度 : ratio fallback（`y1=0.90, y2=0.97`）
* CLI override : `--hand-roi-x1/x2/y1/y2`（4 つすべて指定時のみ有効）

`out/hand_debug/summary.csv` は **1 スロット 1 行**形式で、以下の列を持ちます。

```
frame_idx,time_sec,slot,mean_s,available,card_id,score
```

`--debug-roi-overlay` を指定すると、
`frame_*_full_roi_overlay.png` が出力され、ROI の目視確認が可能です。

---

## 学習

学習時には、CNN で抽出した画像特徴量に
`hand_available`（長さ 4）を結合して action / grid を予測します。

```bash
python tools/train_policy.py \
  --data-dir /path/to/out \
  --out-dir /path/to/policy_out \
  --epochs 5 \
  --batch-size 32 \
  --gw 6 --gh 9
```

---

## 推論（Prediction）

```bash
python tools/predict_policy.py \
  --checkpoint policy_out/checkpoints/best.pt \
  --data-dir /path/to/out \
  --idx 0 \
  --topk 5
```

### 推論時の挙動

* dataset に含まれる `hand_available / hand_card_ids` を **最優先で使用**
* 手札に存在しない、または available=0 のカード行動は **候補から除外**
* 有効なカード行動が 1 つも無い場合は **NOOP にフォールバック**
* カード行動が選択された場合、対応する **slot_index（0–3）** を出力

---

## 出力構成

```
out/
  dataset.jsonl
  state_frames/
policy_out/
  checkpoints/
```

---

## ポリシー

* コード・コメント・ドキュメントに **ゲーム固有の固有名詞を含めない**
* 詳細な設計思想・優先度は `PROJECT_STATE.md` を参照してください
