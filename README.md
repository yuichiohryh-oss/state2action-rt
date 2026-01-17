# state2action-rt

## Purpose
Build a lightweight dataset for imitation learning by cropping a playable arena ROI from screen recordings, resizing it to 256x256, and converting action coordinates into a 6x9 grid ID.

This repository intentionally avoids game- or IP-specific naming. Use only general terms like "game", "arena", "unit", "spell", or "action".

## Requirements
- Python 3.10+
- opencv-python, numpy
- torch, torchvision, tqdm (for learning/prediction)

## Input events.jsonl
Each line is a JSON object. The schema is flexible, but the following keys are required:
- `t`: float (seconds)
- `x`: float (screen coordinate, origin at top-left)
- `y`: float
- `action_id`: string

Example:
```jsonl
{"t": 12.34, "x": 512.0, "y": 640.0, "action_id": "spell_01"}
{"t": 13.10, "x": 420.0, "y": 700.0, "action_id": "unit_07", "note": "extra fields ok"}
{"t": 15.42, "x": 300.0, "y": 500.0, "action_id": "action_generic"}
```

Create events.jsonl with the interactive labeler:
```bash
python tools/label_events.py \
  --video /path/to/video.mp4 \
  --out /path/to/events.jsonl
```

ROI-limited labeling (example values):
```bash
python tools/label_events.py \
  --video /path/to/video.mp4 \
  --out /path/to/events.jsonl \
  --roi-y1 110 \
  --roi-y2-mode fixed \
  --roi-y2-fixed 570 \
  --roi-x1 0 \
  --roi-x2 1280
```

Labeler keys:
- Space: play/pause
- A / D: -1s / +1s
- J / L: -0.1s / +0.1s
- Q / ESC: quit
- Click: select point
- 0-9: save action (action_0 .. action_9)
- N: save "__NOOP__" without a click
- U: undo last event
- H: print help to console

## Windows / PowerShell 利用時の注意

### ❗ コマンド改行に関する注意

本リポジトリのコマンド例では、Linux / macOS で一般的な `\`（バックスラッシュ）による改行を使用することがあります。

**PowerShell では、この書き方は使用できません。**

PowerShell で `\` を使うと、次のエラーが発生します。

```text
単項演算子 '--' の後に式が存在しません
```

### 原因

これは **PowerShell 固有の構文仕様**です。
エラーは Python や本プロジェクトの不具合ではありません。

### 対処方法（PowerShell）

#### 方法1: 1行で実行（最も安全）

```powershell
python tools/inspect_hand.py --video videos/example.mp4 --out-dir out/debug
```

#### 方法2: バッククォート（`）による改行

```powershell
python tools/inspect_hand.py `
  --video videos/example.mp4 `
  --out-dir out/debug
```

※ 行末に空白を入れないよう注意してください。

### 補足

* `\` による改行は Linux / macOS 専用です
* PowerShell では必ずバッククォート（`）を使用してください

## Usage
Install dependencies:
```bash
python -m pip install -e .[dev]
```

Build the dataset:
```bash
python tools/build_dataset.py \
  --video /path/to/video.mp4 \
  --events /path/to/events.jsonl \
  --out-dir /path/to/out \
  --lead-sec 0.8 \
  --gw 6 --gh 9 \
  --roi-y1 70 --roi-y2-mode auto
```

Inspect a sample with a debug overlay:
```bash
python tools/inspect_sample.py --out-dir /path/to/out --idx 0
```

Inspect hand slot availability (HSV S-mean):
```bash
python tools/inspect_hand.py \
  --video /path/to/video.mp4 \
  --out-dir out/hand_debug \
  --stride 15 \
  --start-sec 0 \
  --end-sec -1
```
`out/hand_debug/summary.csv` columns:
`t_sec,frame_idx,mean_s0,mean_s1,mean_s2,mean_s3,avail0,avail1,avail2,avail3`

## No-op augmentation
To reduce unnecessary actions, you can augment the dataset with explicit "__NOOP__" samples. This adds records where the model should output "do nothing" for a given state.

Generate augmented data:
```bash
python tools/augment_noop.py \
  --data-dir /path/to/out \
  --out-dir /path/to/out/noop_aug \
  --noop-per-action 1 \
  --min-gap-sec 0.6 \
  --span-sec 1.5 \
  --seed 7
```

The tool writes `dataset_with_noop.jsonl` and keeps the original `dataset.jsonl` untouched.

## Learning model
This repository includes a minimal imitation learning model that predicts the next action as two heads:
state image (256x256) -> CNN embedding -> action_id (card) + grid_id.

Input modes:
- single: [3,256,256]
- two-frame: [6,256,256]
- two+diff: [9,256,256] where diff = t - (t - delta)

Two-frame mode concatenates the current frame (t) and a past frame (t - delta) along channels, producing a
[6,256,256] tensor (current 3 channels, then past 3 channels). Two+diff mode appends a diff tensor
[9,256,256] where diff = current - past. The past frame is loaded from the previous filename in the same
directory (index - delta_frames); if it does not exist, the current frame is reused. When `--delta-sec`
is provided and `meta.fps_effective` exists, the offset is converted to frames.
During training, `hand_available` (length 4) is concatenated to the image embedding before the action/grid heads.

Train the policy:
```bash
python tools/train_policy.py \
  --data-dir /path/to/out \
  --out-dir /path/to/policy_out \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 7 \
  --val-ratio 0.1 \
  --gw 6 --gh 9 \
  --device auto
```

Train with two-frame input:
```bash
python tools/train_policy.py \
  --data-dir /path/to/out \
  --out-dir /path/to/policy_out \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 7 \
  --val-ratio 0.1 \
  --two-frame \
  --delta-frames 1 \
  --gw 6 --gh 9 \
  --device auto
```

Train with two+diff input:
```bash
python tools/train_policy.py \
  --data-dir /path/to/out \
  --out-dir /path/to/policy_out \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 7 \
  --val-ratio 0.1 \
  --two-frame \
  --diff-channels \
  --delta-frames 1 \
  --gw 6 --gh 9 \
  --device auto
```
`--diff-channels` requires `--two-frame`.

Train with a no-op augmented dataset:
```bash
python tools/train_policy.py \
  --data-dir /path/to/out/noop_aug \
  --dataset-path /path/to/out/noop_aug/dataset_with_noop.jsonl \
  --out-dir /path/to/policy_out \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 7 \
  --val-ratio 0.1 \
  --gw 6 --gh 9 \
  --device auto
```

When `action_id="__NOOP__"` is present, the grid loss is masked for those samples (`grid_id=-1`).

CUDA training example:
```bash
python tools/train_policy.py \
  --data-dir /path/to/out/noop_aug \
  --dataset-path /path/to/out/noop_aug/dataset_with_noop.jsonl \
  --out-dir /path/to/policy_out \
  --epochs 1 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 7 \
  --val-ratio 0.1 \
  --gw 6 --gh 9 \
  --device cuda
```
Confirm `torch.cuda.is_available()` returns `True` before training.

Predict top-k candidates (inference can also use GPU via `--device`):
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out \
  --idx 0 \
  --topk 5 \
  --device auto
```
Prediction computes `hand_available` from the current frame and applies a NOOP logit penalty when any slot is
available. When hand templates are available, it also estimates `hand_card_ids` and masks card logits so
unavailable cards are excluded from top-k results.

Hand card templates (fixed 8-card deck):
- Place card icon templates in `templates/hand_cards/` (e.g. `0.png` .. `7.png`).
- Filenames must contain a numeric card_id; these ids must match the numeric card ids embedded in `action_id`.
- Use `--hand-templates-dir` and `--hand-card-min-score` to control the matcher.
- Set `--disable-hand-card-mask` to debug without masking.
- Matching quality depends on template resolution and UI theme consistency.

Predict with a specific dataset jsonl:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out/noop_aug \
  --dataset-path /path/to/out/noop_aug/dataset_with_noop.jsonl \
  --idx 0 \
  --topk 5 \
  --device auto
```

Override to use two+diff input:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out \
  --idx 0 \
  --topk 5 \
  --two-frame \
  --diff-channels \
  --device auto
```

Optional overlay for the top-1 grid cell:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out \
  --idx 0 \
  --topk 5 \
  --render-overlay --overlay-out out.png \
  --device auto
```

Debug options for prediction ranking:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out/noop_aug \
  --dataset-path /path/to/out/noop_aug/dataset_with_noop.jsonl \
  --idx 0 \
  --topk 5 \
  --exclude-noop \
  --score-mode logadd \
  --device auto

CUDA prediction example:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out \
  --idx 0 \
  --topk 5 \
  --device cuda
```
Confirm `torch.cuda.is_available()` returns `True` before prediction.
```

## Output structure
```
out/
  dataset.jsonl
  state_frames/
    000000.png
    000001.png
templates/
  hand_cards/
    0.png
    1.png
```

Learning outputs:
```
policy_out/
  checkpoints/
    best.pt
    last.pt
  metrics.json
```
```
`out/vocab.json` stores the action_id vocabulary used for card labels.

Checkpoint policy:
- `last.pt` is always saved at the end of each epoch.
- `best.pt` is saved when validation loss improves (when validation is enabled).

Each line in `dataset.jsonl` contains:
- `idx`: int
- `t_action`: float
- `t_state`: float
- `action_id`: string
- `grid_id`: int (0..53 for 6x9)
- `x`: float (original screen coordinate)
- `y`: float
- `x_rel`: float (normalized within ROI, 0..1)
- `y_rel`: float
- `roi`: [x1, y1, x2, y2]
- `state_path`: relative path to 256x256 state image
- `hand_available`: [0/1, 0/1, 0/1, 0/1] (per-slot availability via HSV mean S; bottom 90-97% ROI, 4 slots, threshold 30)
- `hand_card_ids`: [card_id, card_id, card_id, card_id] or -1 for unknown (optional; present when templates are provided)
- `meta`: object (example: gw/gh/lead_sec/fps_effective)

## Parameters
- `lead-sec`: state timestamp is `t_action - lead_sec`
- `gw`, `gh`: grid width/height used for `grid_id`
- `roi-y1`: fixed top offset in pixels
- `roi-y2-mode`: `auto` detects the bottom UI bar via row intensity change; `fixed` uses `roi-y2-fixed`
- `roi-y2-fixed`: explicit y2 when `roi-y2-mode=fixed`
- `roi-x1`, `roi-x2`: optional horizontal bounds
- `fps`: override/fallback FPS (required when `--video` is an image directory)

## Notes
When `--video` points to an image directory, files are treated as a time series sorted by filename.

## Policy
Do not include any game-specific proper nouns in code, comments, or documentation.

## データセット生成（YouTube バッチ例）

このリポジトリでは、動画ファイルと `events.jsonl` から、学習用の `dataset.jsonl` と `state_frames/*.png` を生成します。

### 盤面 ROI（推奨ベースライン）

720p の縦長動画に対して、以下の ROI パラメータが `videos/batch1/*` で実機確認されており、現在のベースライン設定です。

```powershell
python tools\\build_dataset.py `
  --video  /path/to/videos/batch1/hog_yt_2026-01-11_164307.mp4 `
  --events /path/to/videos/batch1/hog_yt_2026-01-11_164307.jsonl `
  --out-dir /path/to/data/batch1 `
  --lead-sec 0.8 `
  --gw 6 --gh 9 `
  --roi-y1 110 `
  --roi-y2-mode fixed `
  --roi-y2-fixed 570
```

**注意点**

* `roi-y1 / roi-y2-fixed` は、上部・下部の UI を除外し、盤面のみが安定して切り出されるように調整されています。
* これらの値は **720p 縦長動画を前提**としています。
* 動画レイアウトが異なる場合は、学習前に必ず `tools/inspect_sample.py` で ROI を目視確認してください。

---

## 予定：手札 availability（行動制約）の導入

盤面画像のみを入力とした場合、
「リソース不足により実行できない行動（例：エリクサー不足）」を区別できず、
**同一盤面に対して異なる教師行動が付く**という学習上の矛盾が発生します。

これを避けるため、本プロジェクトでは **手札の availability（出せる／出せない）** を特徴量として追加する予定です。

### 設計方針

* 手札 UI の ROI：画面下端 **90〜97%**
* 横方向に **4 スロット**へ分割
* 各スロットを HSV 色空間に変換
* 彩度（S）の平均値を使用

  * `mean_S > 約30`  → 出せる（カラー表示）
  * `mean_S <= 約30` → 出せない（グレー表示）

### データセット形式（予定）

各レコードに以下を追加します。

```json
"hand_available": [0, 0, 1, 1]
```

この情報は、

* 学習時には追加の数値特徴量として使用し
* 推論時には「不可能な行動を抑制するマスク」として利用します。

この方法は、エリクサー量を直接数値認識するよりも実装が簡単かつ頑健であり、
行動制約を表現するうえで本質的な情報を捉えられます。

## Status

詳細は PROJECT_STATE.md を参照。
