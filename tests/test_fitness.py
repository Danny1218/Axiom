import torch

from engine.fitness import (
    ShadowFitnessEvaluator,
    apply_shadow_verdict,
    localized_adapter_loss,
    run_shadow_training_epochs,
)
from engine.supernet import LatentSupernet, TTLoRAAdapter


def test_shadow_fitness_evaluator_verdict():
    ev = ShadowFitnessEvaluator("e", epochs=3)
    ev.record_epoch_loss(1.0)
    ev.record_epoch_loss(0.9)
    ev.record_epoch_loss(0.5)
    assert ev.verdict() == "integrate"


def test_shadow_fitness_prune_when_flat():
    ev = ShadowFitnessEvaluator("e", epochs=2)
    ev.record_epoch_loss(1.0)
    ev.record_epoch_loss(1.0)
    assert ev.verdict() == "prune"


def test_apply_verdict_integrate_and_prune():
    sn = LatentSupernet(3, ("p",), rank=2)
    sn.set_adapter_mask("p", 1.0)
    sn.is_shadow[0] = True
    apply_shadow_verdict(sn, "p", "integrate")
    assert sn.is_shadow[0].item() is False
    sn.is_shadow[0] = True
    apply_shadow_verdict(sn, "p", "prune")
    assert sn.adapter_mask[0] < 0.5 and sn.is_shadow[0].item() is False


def test_localized_adapter_loss_scalar():
    a = TTLoRAAdapter(5, rank=2)
    h = torch.randn(4, 5)
    t = torch.randn(4, 5)
    loss = localized_adapter_loss(a, h, t)
    assert loss.ndim == 0 and loss.item() >= 0


def test_run_shadow_training_improves_on_easy_task():
    torch.manual_seed(0)
    dim = 6
    adapter = TTLoRAAdapter(dim, rank=3)
    h = torch.randn(32, dim)
    tgt = torch.zeros_like(h)
    batches = [(h, tgt)]
    losses, verdict = run_shadow_training_epochs(adapter, batches, epochs=5, lr=0.2)
    assert len(losses) == 5
    assert losses[-1] < losses[0]
    assert verdict == "integrate"
