"""Tests for InterventionRequest dataclasses and builder API. Pure Python, no CUDA."""

from __future__ import annotations

import torch
from minisgl.intervention.request import (
    ConditionalWriteOp,
    InterventionRequest,
    ObserveOp,
    WriteOp,
)


class TestObserveOp:
    def test_field_storage(self):
        op = ObserveOp(layer=3)
        assert op.layer == 3


class TestWriteOp:
    def test_ablate_fields(self):
        op = WriteOp(layer=2, kind="ablate")
        assert op.layer == 2
        assert op.kind == "ablate"
        assert op.vector is None
        assert op.alpha == 1.0

    def test_steer_fields(self):
        v = torch.randn(16)
        op = WriteOp(layer=1, kind="steer", vector=v, alpha=0.5)
        assert op.kind == "steer"
        assert op.vector is v
        assert op.alpha == 0.5

    def test_patch_fields(self):
        v = torch.randn(4, 16)
        op = WriteOp(layer=0, kind="patch", vector=v)
        assert op.kind == "patch"
        assert op.vector is v


class TestConditionalWriteOp:
    def test_fields_and_fn(self):
        fn = lambda x, r: x + r
        op = ConditionalWriteOp(read_layer=1, write_layer=3, fn=fn)
        assert op.read_layer == 1
        assert op.write_layer == 3
        assert op.fn is fn


class TestInterventionRequest:
    def test_empty_default(self):
        req = InterventionRequest()
        assert req.observations == []
        assert req.writes == []
        assert req.conditional_writes == []

    def test_observe(self):
        req = InterventionRequest().observe(2)
        assert len(req.observations) == 1
        assert req.observations[0].layer == 2

    def test_ablate(self):
        req = InterventionRequest().ablate(3)
        assert len(req.writes) == 1
        assert req.writes[0].kind == "ablate"
        assert req.writes[0].layer == 3

    def test_steer(self):
        v = torch.randn(16)
        req = InterventionRequest().steer(1, v, alpha=0.5)
        assert len(req.writes) == 1
        assert req.writes[0].kind == "steer"
        assert req.writes[0].vector is v
        assert req.writes[0].alpha == 0.5

    def test_patch(self):
        v = torch.randn(4, 16)
        req = InterventionRequest().patch(0, v)
        assert len(req.writes) == 1
        assert req.writes[0].kind == "patch"
        assert req.writes[0].vector is v

    def test_chaining(self):
        v = torch.randn(16)
        req = InterventionRequest().observe(0).steer(1, v).ablate(2)
        assert len(req.observations) == 1
        assert len(req.writes) == 2
        assert req.observations[0].layer == 0
        assert req.writes[0].kind == "steer"
        assert req.writes[1].kind == "ablate"

    def test_conditional_write_auto_observes(self):
        fn = lambda x, r: x
        req = InterventionRequest().conditional_write(1, 3, fn)
        # Should auto-add ObserveOp for read_layer=1
        assert len(req.observations) == 1
        assert req.observations[0].layer == 1
        assert len(req.conditional_writes) == 1
        assert req.conditional_writes[0].read_layer == 1
        assert req.conditional_writes[0].write_layer == 3

    def test_conditional_write_no_duplicate_observe(self):
        fn = lambda x, r: x
        req = InterventionRequest().observe(1).conditional_write(1, 3, fn)
        # Should NOT double-add ObserveOp for layer 1
        assert len(req.observations) == 1
        assert req.observations[0].layer == 1

    def test_steer_default_alpha(self):
        v = torch.randn(16)
        req = InterventionRequest().steer(1, v)
        assert req.writes[0].alpha == 1.0

    def test_conditional_write_once_default(self):
        fn = lambda x, r: x
        req = InterventionRequest().conditional_write(1, 3, fn)
        assert req.conditional_writes[0].once is True

    def test_conditional_write_once_false(self):
        fn = lambda x, r: x
        req = InterventionRequest().conditional_write(1, 3, fn, once=False)
        assert req.conditional_writes[0].once is False


class TestWriteOpValidation:
    def test_valid_kinds_accepted(self):
        for kind in ("ablate", "steer", "patch"):
            WriteOp(layer=0, kind=kind)  # should not raise

    def test_invalid_kind_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown write kind"):
            WriteOp(layer=0, kind="abblate")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
