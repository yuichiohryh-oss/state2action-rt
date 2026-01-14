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

Predict top-k candidates:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out \
  --idx 0 \
  --topk 5
```

Predict with a specific dataset jsonl:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out/noop_aug \
  --dataset-path /path/to/out/noop_aug/dataset_with_noop.jsonl \
  --idx 0 \
  --topk 5
```

Override to use two+diff input:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out \
  --idx 0 \
  --topk 5 \
  --two-frame \
  --diff-channels
```

Optional overlay for the top-1 grid cell:
```bash
python tools/predict_policy.py \
  --checkpoint /path/to/policy_out/checkpoints/best.pt \
  --data-dir /path/to/out \
  --idx 0 \
  --topk 5 \
  --render-overlay --overlay-out out.png
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
  --score-mode logadd
```

## Output structure
```
out/
  dataset.jsonl
  state_frames/
    000000.png
    000001.png
```

Learning outputs:
```
policy_out/
  checkpoints/
    best.pt
    last.pt
  metrics.json
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
