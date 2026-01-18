import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from state2action_rt.learning.dataset import ActionVocab
from tools import predict_policy


def test_predict_exclude_noop_selects_next_candidate() -> None:
    vocab = ActionVocab(["__NOOP__", "action_spell"])
    card_probs = torch.tensor([0.9, 0.1])
    grid_probs = torch.tensor([0.8, 0.2])

    candidates = predict_policy.build_candidates(
        card_probs,
        grid_probs,
        vocab,
        topk=3,
        score_mode="mul",
        exclude_noop=True,
    )

    assert candidates
    top = candidates[0]
    assert vocab.id_to_action[top[1]] == "action_spell"
    assert top[3] >= 0


def test_predict_score_modes_do_not_error() -> None:
    vocab = ActionVocab(["__NOOP__", "action_spell"])
    card_probs = torch.tensor([0.2, 0.8])
    grid_probs = torch.tensor([0.6, 0.4])

    for mode in ("mul", "add", "logadd"):
        candidates = predict_policy.build_candidates(
            card_probs,
            grid_probs,
            vocab,
            topk=2,
            score_mode=mode,
            exclude_noop=False,
        )
        assert candidates
        assert all(math.isfinite(candidate[0]) for candidate in candidates)


def test_predict_parser_accepts_device() -> None:
    parser = predict_policy.build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "best.pt",
            "--data-dir",
            "data",
            "--idx",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert args.device == "cpu"


def test_predict_select_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = predict_policy.select_device("auto")
    assert device.type == "cuda"


def test_predict_select_device_uses_mps(monkeypatch) -> None:
    class DummyMps:
        @staticmethod
        def is_available() -> bool:
            return True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", DummyMps(), raising=False)
    device = predict_policy.select_device("auto")
    assert device.type == "mps"


def test_apply_noop_penalty_when_hand_available() -> None:
    logits = torch.tensor([0.1, 0.2, 0.3])
    adjusted = predict_policy.apply_noop_penalty(logits, [0, 1, 0, 0], noop_idx=0, penalty=1.5)
    assert torch.isclose(adjusted[0], torch.tensor(-1.4))
    assert torch.allclose(adjusted[1:], logits[1:])


def test_apply_noop_penalty_when_hand_empty() -> None:
    logits = torch.tensor([0.1, 0.2, 0.3])
    adjusted = predict_policy.apply_noop_penalty(logits, [0, 0, 0, 0], noop_idx=0, penalty=1.5)
    assert torch.allclose(adjusted, logits)


def test_apply_action_mask_keeps_allowed_and_exempt() -> None:
    logits = torch.zeros(10)
    card_id_to_action_idx = {i: i for i in range(8)}
    masked = predict_policy.apply_action_mask(
        logits,
        allowed_card_ids={2, 5},
        noop_idx=8,
        skill_idx=9,
        card_id_to_action_idx_map=card_id_to_action_idx,
    )
    for idx in range(8):
        if idx in (2, 5):
            assert torch.isclose(masked[idx], torch.tensor(0.0))
        else:
            assert torch.isclose(masked[idx], torch.tensor(-1e9))
    assert torch.isclose(masked[8], torch.tensor(0.0))
    assert torch.isclose(masked[9], torch.tensor(0.0))


def test_build_candidates_respects_valid_action_indices() -> None:
    vocab = ActionVocab(["__NOOP__", "action_1", "action_2"])
    card_probs = torch.tensor([0.1, 0.9, 0.8])
    grid_probs = torch.tensor([0.6, 0.4])

    candidates = predict_policy.build_candidates(
        card_probs,
        grid_probs,
        vocab,
        topk=3,
        score_mode="mul",
        exclude_noop=True,
        valid_action_indices={2},
    )

    assert candidates
    assert all(candidate[1] == 2 for candidate in candidates)


def test_hand_mask_drops_unavailable_card_id() -> None:
    vocab = ActionVocab(["__NOOP__", "card_2", "card_3", "card_5", "card_7"])
    hand_available = [1, 1, 1, 1]
    hand_card_ids = [3, 7, -1, 2]
    allowed = predict_policy.build_allowed_card_ids(hand_available, hand_card_ids)
    mapping = predict_policy.build_card_id_to_action_idx_map(vocab)

    logits = torch.zeros(len(vocab.id_to_action))
    masked = predict_policy.apply_action_mask(
        logits,
        allowed_card_ids=allowed,
        noop_idx=vocab.action_to_id["__NOOP__"],
        skill_idx=None,
        card_id_to_action_idx_map=mapping,
    )

    assert torch.isclose(masked[mapping[5]], torch.tensor(-1e9))
    for card_id in (2, 3, 7):
        assert torch.isclose(masked[mapping[card_id]], torch.tensor(0.0))


def test_fallback_to_noop_when_candidates_empty() -> None:
    vocab = ActionVocab(["__NOOP__", "card_5"])
    card_probs = torch.tensor([0.7, 0.3])
    noop_idx = vocab.action_to_id["__NOOP__"]
    candidates = predict_policy.ensure_candidates_with_noop(
        [],
        card_probs,
        noop_idx,
        disable_hand_card_mask=False,
    )

    assert candidates
    score, action_idx, card_prob, grid_idx, grid_prob = candidates[0]
    assert action_idx == noop_idx
    assert grid_idx == -1
    assert grid_prob == 0.0
    assert math.isclose(score, float(card_probs[noop_idx].item()))
    assert math.isclose(card_prob, float(card_probs[noop_idx].item()))


def test_compute_min_hand_cost_returns_lowest() -> None:
    allowed = {2, 5, 7}
    min_cost = predict_policy.compute_min_hand_cost(allowed, predict_policy.CARD_COSTS)
    assert min_cost == 1


def test_build_affordable_fields_handles_card_and_noop() -> None:
    affordable, cost, required_cost = predict_policy.build_affordable_fields(
        "card_7",
        elixir=3,
        card_costs=predict_policy.CARD_COSTS,
    )
    assert affordable == 0
    assert cost == 4
    assert required_cost == 4
    affordable, cost, required_cost = predict_policy.build_affordable_fields(
        "__NOOP__",
        elixir=0,
        card_costs=predict_policy.CARD_COSTS,
    )
    assert affordable == 1
    assert cost == 0
    assert required_cost == 0


def test_predict_help_includes_new_args() -> None:
    script_path = Path(__file__).resolve().parents[1] / "tools" / "predict_policy.py"
    root_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    python_path = os.pathsep.join(
        [str(root_dir / "src"), str(root_dir), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["PYTHONPATH"] = python_path
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert "--dedup-topk-by-slot" in stdout
    assert "--no-dedup-topk-by-slot" in stdout
    assert "--enable-elixir-mask" in stdout
