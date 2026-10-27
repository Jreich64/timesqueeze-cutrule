"""Shared training pieces: micro-batch splitting and best-validation model selection."""
import torch
from torch import nn

from timesqueeze.training.common import BestOnValidation, split_batch


def test_split_batch_covers_every_row_once_in_order():
  b = {"y": torch.arange(10).view(5, 2), "n_valid": torch.arange(5), "patch_id": torch.arange(15).view(5, 3)}
  parts = split_batch(b, 2)
  assert [len(p["y"]) for p in parts] == [2, 2, 1]
  for k in b:
    assert torch.equal(torch.cat([p[k] for p in parts]), b[k])


def test_best_on_validation_keeps_the_lowest_epoch_and_restores_it(artifacts):
  model = nn.Linear(2, 1)
  best = BestOnValidation()
  history = []
  for epoch, val in enumerate([0.50, 0.45, 0.47, 0.60]):              # epoch 0 = before fine-tuning
    with torch.no_grad():
      model.weight.fill_(float(epoch))
    history.append(best.update(val, epoch, model, extra=f"tau{epoch}"))
  assert history == [True, True, False, False] and best.epoch == 1
  assert best.restore(model) == "tau1" and torch.all(model.weight == 1.0)
  again = BestOnValidation.from_dict(best.to_dict())                  # survives a checkpoint round trip
  assert again.epoch == 1 and torch.equal(again.state["weight"], best.state["weight"])
  artifacts.json("selection.json", {"val": [0.50, 0.45, 0.47, 0.60], "selected_epoch": best.epoch})
