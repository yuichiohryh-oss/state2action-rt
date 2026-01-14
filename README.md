# state2action-rt

## Purpose
Build a lightweight dataset for imitation learning by cropping a playable arena ROI from screen recordings, resizing it to 256x256, and converting action coordinates into a 6x9 grid ID.

This repository intentionally avoids game- or IP-specific naming. Use only general terms like "game", "arena", "unit", "spell", or "action".

## Requirements
- Python 3.10+
- opencv-python, numpy

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

## Output structure
```
out/
  dataset.jsonl
  state_frames/
    000000.png
    000001.png
```

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
