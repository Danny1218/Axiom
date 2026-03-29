import torch
import torch.nn as nn

from engine.router import SinkhornRouter


def test_uniform_routing_triggers_mutation_at_low_threshold():
    r = SinkhornRouter(4, 2, num_iters=48, epsilon=1.0, mutation_entropy_norm_threshold=0.5)
    nn.init.zeros_(r.proj.weight)
    nn.init.zeros_(r.proj.bias)
    x = torch.randn(10, 4)
    w = r(x)
    assert w.shape == (10, 2)
    sig = r.last_mutation_signal
    assert sig is not None and sig.triggered and sig.num_active_experts == 2
    assert sig.normalized_entropy >= 0.5


def test_empty_expert_mask_sets_no_mutation():
    r = SinkhornRouter(3, 2)
    x = torch.randn(2, 3)
    r(x, expert_mask=torch.zeros(2, dtype=torch.bool))
    assert r.last_mutation_signal is not None and not r.last_mutation_signal.triggered
