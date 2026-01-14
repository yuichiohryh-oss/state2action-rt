from __future__ import annotations

import argparse
import json
import os
import random
import sys
from bisect import bisect_left
from typing import Dict, List, Tuple


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def load_records(dataset_path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def find_available_out_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        return base_dir
    for i in range(1, 1000):
        candidate = f"{base_dir}_{i}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError("failed to find available out-dir")


def to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def is_gap_safe(action_times: List[float], t_noop: float, min_gap_sec: float) -> bool:
    if not action_times:
        return True
    idx = bisect_left(action_times, t_noop)
    if idx > 0 and abs(action_times[idx - 1] - t_noop) <= min_gap_sec:
        return False
    if idx < len(action_times) and abs(action_times[idx] - t_noop) <= min_gap_sec:
        return False
    return True


def build_noop_record(
    record: Dict,
    t_noop: float,
    idx: int,
    state_path: str,
) -> Dict:
    noop_record = dict(record)
    noop_record["idx"] = idx
    noop_record["t_action"] = t_noop
    noop_record["t_state"] = t_noop
    noop_record["action_id"] = "__NOOP__"
    noop_record["grid_id"] = -1
    noop_record["x"] = None
    noop_record["y"] = None
    noop_record["x_rel"] = None
    noop_record["y_rel"] = None
    noop_record["state_path"] = state_path
    meta = dict(record.get("meta", {}))
    meta["noop"] = True
    if "idx" in record:
        meta["noop_source_idx"] = record["idx"]
    noop_record["meta"] = meta
    return noop_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment dataset.jsonl with __NOOP__ samples.")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing dataset.jsonl")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: data-dir/noop_aug)")
    parser.add_argument("--noop-per-action", type=int, default=1)
    parser.add_argument("--min-gap-sec", type=float, default=0.6)
    parser.add_argument("--span-sec", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="Only print counts without writing outputs")
    args = parser.parse_args()

    dataset_path = os.path.join(args.data_dir, "dataset.jsonl")
    if not os.path.exists(dataset_path):
        warn(f"dataset.jsonl not found in {args.data_dir}")
        return 1

    records = load_records(dataset_path)
    if not records:
        warn("dataset.jsonl is empty")
        return 1

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = find_available_out_dir(os.path.join(args.data_dir, "noop_aug"))

    action_times = sorted(
        t for t in (to_float(record.get("t_action")) for record in records) if t is not None
    )

    rng = random.Random(args.seed)
    noop_records: List[Dict] = []
    rejected_gap = 0
    rejected_negative = 0
    rejected_missing_time = 0

    existing_idxs = [int(record.get("idx")) for record in records if str(record.get("idx", "")).isdigit()]
    next_idx = (max(existing_idxs) + 1) if existing_idxs else len(records)

    for record in records:
        t_state = to_float(record.get("t_state"))
        if t_state is None:
            rejected_missing_time += args.noop_per_action
            continue
        for _ in range(args.noop_per_action):
            dt = rng.uniform(-args.span_sec, args.span_sec)
            t_noop = t_state + dt
            if t_noop < 0:
                rejected_negative += 1
                continue
            if not is_gap_safe(action_times, t_noop, args.min_gap_sec):
                rejected_gap += 1
                continue
            state_path = record.get("state_path")
            if state_path is None:
                warn("record missing state_path, skipping noop")
                continue
            if os.path.normpath(os.path.abspath(args.data_dir)) != os.path.normpath(os.path.abspath(out_dir)):
                state_path = os.path.abspath(os.path.join(args.data_dir, state_path))
            noop_records.append(build_noop_record(record, t_noop, next_idx, state_path))
            next_idx += 1

    total_created = len(noop_records)
    total_rejected = rejected_gap + rejected_negative + rejected_missing_time
    print(
        "noop_summary "
        f"created={total_created} rejected_gap={rejected_gap} "
        f"rejected_negative={rejected_negative} rejected_missing_time={rejected_missing_time}"
    )

    if args.dry_run:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_with_noop.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            if os.path.normpath(os.path.abspath(args.data_dir)) != os.path.normpath(os.path.abspath(out_dir)):
                record = dict(record)
                state_path = record.get("state_path")
                if state_path is not None:
                    record["state_path"] = os.path.abspath(os.path.join(args.data_dir, state_path))
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        for record in noop_records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
