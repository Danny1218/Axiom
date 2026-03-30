import torch

from axiom.primitives.liquid_tensor import LiquidStateTensor, stack_liquid_states


def test_liquid_tau_positive():
    s = LiquidStateTensor(5, tau_init=2.0)
    assert s.tau.item() > 0


def test_assign_from():
    s = LiquidStateTensor(3)
    s.assign_from(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(s.data, torch.tensor([1.0, 2.0, 3.0]))


def test_stack_liquid_states():
    a = LiquidStateTensor(4)
    b = LiquidStateTensor(4)
    with torch.no_grad():
        a.data.fill_(1.0)
        b.data.fill_(2.0)
    vals, taus = stack_liquid_states([a, b])
    assert vals.shape == (2, 4) and taus.shape == (2,)


def test_liquid_forward_returns_data():
    s = LiquidStateTensor(2)
    assert s.forward().shape == (2,)
