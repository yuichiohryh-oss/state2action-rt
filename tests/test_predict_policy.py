import math

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
