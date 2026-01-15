import json

from state2action_rt.label_events_utils import (
    append_event,
    apply_roi_offset,
    count_events,
    undo_last_event,
)


def test_append_and_undo_events(tmp_path):
    path = tmp_path / "events.jsonl"
    event_a = {"t": 1.2, "x": 10, "y": 20, "action_id": "action_0"}
    event_b = {"t": 2.3, "x": 30, "y": 40, "action_id": "action_1"}

    append_event(path, event_a)
    append_event(path, event_b)

    assert count_events(path) == 2
    removed = undo_last_event(path)
    assert removed == event_b
    assert count_events(path) == 1

    remaining = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert remaining == [event_a]


def test_apply_roi_offset():
    roi = (100, 200, 400, 500)
    assert apply_roi_offset(5, 7, roi) == (105, 207)
    assert apply_roi_offset(5, 7, None) == (5, 7)
