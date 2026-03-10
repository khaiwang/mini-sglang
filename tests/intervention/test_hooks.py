"""Tests for intervention hooks: wrap_layers / unwrap_layers."""

from __future__ import annotations

from typing import Tuple

import minisgl.core as core_module
import pytest
import torch
from minisgl.core import Batch, Context, set_global_ctx
from minisgl.intervention.buffers import MaskBuffer, ObservationBuffer
from minisgl.intervention.context import InterventionContext, clear_intervention_ctx
from minisgl.intervention.hooks import unwrap_layers, wrap_layers

# Small test sizes
NUM_LAYERS = 4
HIDDEN_DIM = 16
MAX_RUNNING_REQ = 8
MAX_TOKENS = 8

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ─── Mock layer ──────────────────────────────────────────────────────────────


class MockLayer:
    """Mimics a decoder layer: forward(x, residual=None) -> (x, residual)."""

    def __init__(self, bias: float = 1.0):
        self._bias = bias

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = torch.zeros_like(x)
        out = x + self._bias
        return out, residual + out


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Reset global singletons before and after each test."""
    clear_intervention_ctx()
    yield
    core_module._GLOBAL_CTX = None
    clear_intervention_ctx()


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.fixture
def ictx(device):
    """Create a minimal InterventionContext with dual obs buffers."""
    x_obs_buf = ObservationBuffer(
        num_layers=NUM_LAYERS,
        max_tokens_per_slot=MAX_TOKENS,
        hidden_dim=HIDDEN_DIM,
        device=device,
    )
    res_obs_buf = ObservationBuffer(
        num_layers=NUM_LAYERS,
        max_tokens_per_slot=MAX_TOKENS,
        hidden_dim=HIDDEN_DIM,
        device=device,
    )
    mb = MaskBuffer(
        num_layers=NUM_LAYERS,
        max_running_req=MAX_RUNNING_REQ,
        hidden_dim=HIDDEN_DIM,
        device=device,
    )
    obs_mask = torch.zeros(
        NUM_LAYERS, MAX_RUNNING_REQ + 1, dtype=torch.float32, device=device
    )
    return InterventionContext(
        x_obs_buffer=x_obs_buf, residual_obs_buffer=res_obs_buf,
        mask_buffer=mb, obs_mask=obs_mask,
    )


def _setup_global_ctx(device, req_map_data, page_size=1):
    """Set up global Context with a mock batch carrying the given req_map."""
    ctx = Context(page_size=page_size)
    ctx.page_table = torch.zeros(1, dtype=torch.int64, device=device)
    set_global_ctx(ctx)

    batch = Batch.__new__(Batch)
    batch.req_map = torch.tensor(req_map_data, dtype=torch.long, device=device)
    batch.phase = "decode"
    batch.reqs = []
    batch.padded_reqs = []
    ctx._batch = batch
    return ctx


# ─── TestWrapUnwrap ──────────────────────────────────────────────────────────


class TestWrapUnwrap:
    def test_wrap_replaces_forward(self):
        layer = MockLayer()
        ictx_dummy = InterventionContext.__new__(InterventionContext)
        ictx_dummy.x_obs_buffer = None
        ictx_dummy.residual_obs_buffer = None
        ictx_dummy.mask_buffer = None
        ictx_dummy.obs_mask = None
        # Before wrapping, forward is a bound method from the class
        assert "forward" not in layer.__dict__
        wrap_layers([layer], ictx_dummy)
        # After wrapping, forward is an instance attribute (closure)
        assert "forward" in layer.__dict__
        assert hasattr(layer, "_original_forward")

    def test_unwrap_restores_forward(self):
        layer = MockLayer()
        ictx_dummy = InterventionContext.__new__(InterventionContext)
        ictx_dummy.x_obs_buffer = None
        ictx_dummy.residual_obs_buffer = None
        ictx_dummy.mask_buffer = None
        ictx_dummy.obs_mask = None
        wrap_layers([layer], ictx_dummy)
        assert "forward" in layer.__dict__
        unwrap_layers([layer])
        # After unwrap, instance attribute is removed; class method resolves again
        assert "forward" not in layer.__dict__
        assert not hasattr(layer, "_original_forward")

    def test_unwrap_noop_on_unwrapped(self):
        layer = MockLayer()
        unwrap_layers([layer])  # should not crash
        assert not hasattr(layer, "_original_forward")


# ─── TestWrappedForward ──────────────────────────────────────────────────────


@requires_cuda
class TestWrappedForward:
    def test_identity_preserves_both(self, device, ictx):
        """With identity masks (scale=1, add=0, obs_mask=0), both x and residual unchanged."""
        layer = MockLayer()
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map_data = [0, 0, 0, 0]
        _setup_global_ctx(device, req_map_data)

        # Unwrapped reference
        x_ref, res_ref = layer.forward(x.clone())

        wrap_layers([layer], ictx)
        x_wrapped, res_wrapped = layer.forward(x.clone())

        assert torch.allclose(x_wrapped, x_ref, atol=1e-6)
        assert torch.allclose(res_wrapped, res_ref, atol=1e-6)

    def test_observe_writes_to_both_buffers(self, device, ictx):
        """With obs_mask enabled, both x and residual obs buffers get non-zero data."""
        layer_idx = 0
        table_idx = 0
        ictx.obs_mask[layer_idx, table_idx] = 1.0

        layer = MockLayer()
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map_data = [table_idx] * 4
        _setup_global_ctx(device, req_map_data)

        # Both buffers should start at zero
        assert torch.all(ictx.x_obs_buffer._buf == 0)
        assert torch.all(ictx.residual_obs_buffer._buf == 0)

        wrap_layers([layer], ictx)
        layer.forward(x.clone())

        # Both buffer regions should now have data
        start = layer_idx * MAX_TOKENS
        x_observed = ictx.x_obs_buffer._buf[start : start + 4]
        res_observed = ictx.residual_obs_buffer._buf[start : start + 4]
        assert not torch.all(x_observed == 0)
        assert not torch.all(res_observed == 0)

    def test_blend_ablation_zeros_both(self, device, ictx):
        """Ablation (scale=0, add=0) zeros both x and residual."""
        layer_idx = 0
        table_idx = 0
        ictx.mask_buffer.set_ablate(layer_idx, table_idx)

        layer = MockLayer()
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map_data = [table_idx] * 4
        _setup_global_ctx(device, req_map_data)

        wrap_layers([layer], ictx)
        x_ablated, res_ablated = layer.forward(x.clone())

        # Both should be zero
        assert torch.all(x_ablated == 0)
        assert torch.all(res_ablated == 0)

    def test_blend_steering_shifts_x_only(self, device, ictx):
        """Steering (scale=1, add=vector) shifts x by the vector, residual unchanged."""
        layer_idx = 0
        table_idx = 0
        steer_v = torch.randn(HIDDEN_DIM, device=device)
        ictx.mask_buffer.set_steer(layer_idx, table_idx, steer_v, alpha=1.0)

        layer = MockLayer()
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map_data = [table_idx] * 4
        _setup_global_ctx(device, req_map_data)

        # Unwrapped reference
        x_ref, res_ref = MockLayer().forward(x.clone())

        wrap_layers([layer], ictx)
        x_steered, res_steered = layer.forward(x.clone())

        expected_x = x_ref + steer_v.unsqueeze(0)
        assert torch.allclose(x_steered, expected_x, atol=1e-6)
        # residual unchanged (scale=1, no add applied to residual)
        assert torch.allclose(res_steered, res_ref, atol=1e-6)

    def test_multi_layer_wrap(self, device, ictx):
        """Wrapping multiple layers: each writes to correct buffer region."""
        layers = [MockLayer(bias=float(i)) for i in range(NUM_LAYERS)]
        # Enable observation for all layers on req 0
        ictx.obs_mask[:, 0] = 1.0

        req_map_data = [0, 0]
        _setup_global_ctx(device, req_map_data)

        wrap_layers(layers, ictx)

        x = torch.randn(2, HIDDEN_DIM, device=device)
        current = x.clone()
        residual = None
        for layer in layers:
            current, residual = layer.forward(current, residual)

        # Check each layer's x observation region has non-zero data
        for layer_idx in range(NUM_LAYERS):
            start = layer_idx * MAX_TOKENS
            x_observed = ictx.x_obs_buffer._buf[start : start + 2]
            assert not torch.all(x_observed == 0), f"Layer {layer_idx} x obs buffer is all zeros"
            res_observed = ictx.residual_obs_buffer._buf[start : start + 2]
            assert not torch.all(
                res_observed == 0
            ), f"Layer {layer_idx} residual obs buffer is all zeros"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
