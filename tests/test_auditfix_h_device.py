"""Any-device support: resolution fallbacks and backbone placement."""

import torch
from suites._backbones import _MLP

from harness.devices import current_device, resolve_device, set_default_device


def test_resolve_auto_without_gpu_falls_back_to_cpu() -> None:
    resolved = resolve_device("auto")
    if not torch.cuda.is_available():
        assert resolved.type == "cpu"


def test_resolve_unavailable_backend_falls_back() -> None:
    if torch.cuda.is_available():
        return  # nothing to prove on a GPU host
    assert resolve_device("cuda").type == "cpu"
    assert resolve_device("cuda:0").type == "cpu"


def test_explicit_cpu_round_trips() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_set_default_device_places_new_backbones(monkeypatch) -> None:
    set_default_device("cpu")
    model = _MLP(in_dim=4, hidden=8, out_dim=1)
    assert next(model.parameters()).device == current_device()

    # Simulate a GPU session default even without hardware: placement must
    # follow whatever the process default is.
    monkeypatch.setattr("harness.devices._DEFAULT", torch.device("meta"), raising=True)
    meta_model = _MLP(in_dim=4, hidden=8, out_dim=1)
    assert next(meta_model.parameters()).device.type == "meta"
    set_default_device("cpu")
