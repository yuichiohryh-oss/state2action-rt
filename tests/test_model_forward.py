import torch

from state2action_rt.learning.model import PolicyNet


def test_model_forward_shapes() -> None:
    model = PolicyNet(num_actions=4, num_grids=6)
    batch = torch.zeros((2, 3, 256, 256))
    card_logits, grid_logits = model(batch)
    assert card_logits.shape == (2, 4)
    assert grid_logits.shape == (2, 6)
