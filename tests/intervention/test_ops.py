"""Tests for intervention ops: observe, mask_blend."""

from __future__ import annotations

import pytest
import torch
from minisgl.intervention.ops import mask_blend, observe

# Small test sizes
NUM_LAYERS = 4
HIDDEN_DIM = 16
MAX_RUNNING_REQ = 8
MAX_TOKENS = 8

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    return torch.device("cuda:0")


# ─── observe ─────────────────────────────────────────────────────────────────


@requires_cuda
class TestObserve:
    def _make_flat_buf(self, device):
        total = NUM_LAYERS * MAX_TOKENS
        flat_buf = torch.zeros(total, HIDDEN_DIM, device=device)
        offsets = torch.arange(0, total, MAX_TOKENS, dtype=torch.int64, device=device)
        base_indices = torch.arange(MAX_TOKENS, dtype=torch.int64, device=device)
        obs_mask = torch.zeros(NUM_LAYERS, MAX_RUNNING_REQ, dtype=torch.float32, device=device)
        return flat_buf, offsets, base_indices, obs_mask

    def test_all_masked_out(self, device):
        flat_buf, offsets, base_indices, obs_mask = self._make_flat_buf(device)
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        observe(
            x,
            layer_idx=0,
            flat_buf=flat_buf,
            obs_mask=obs_mask,
            req_map=req_map,
            base_indices=base_indices,
            offsets=offsets,
        )
        assert torch.all(flat_buf == 0)

    def test_single_request(self, device):
        flat_buf, offsets, base_indices, obs_mask = self._make_flat_buf(device)
        obs_mask[0, 0] = 1.0
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        observe(
            x,
            layer_idx=0,
            flat_buf=flat_buf,
            obs_mask=obs_mask,
            req_map=req_map,
            base_indices=base_indices,
            offsets=offsets,
        )
        assert torch.allclose(flat_buf[:4], x, atol=1e-6)
        assert torch.all(flat_buf[4:] == 0)

    def test_single_layer_at_offset(self, device):
        flat_buf, offsets, base_indices, obs_mask = self._make_flat_buf(device)
        obs_mask[1, 0] = 1.0
        bs = 3
        x = torch.randn(bs, HIDDEN_DIM, device=device)
        req_map = torch.zeros(bs, dtype=torch.long, device=device)

        observe(
            x,
            layer_idx=1,
            flat_buf=flat_buf,
            obs_mask=obs_mask,
            req_map=req_map,
            base_indices=base_indices,
            offsets=offsets,
        )

        # Layer 1 region starts at MAX_TOKENS
        start = MAX_TOKENS
        assert torch.allclose(flat_buf[start : start + bs], x, atol=1e-6)
        # Layer 0 region should be untouched
        assert torch.all(flat_buf[:MAX_TOKENS] == 0)

    def test_multi_layer(self, device):
        flat_buf, offsets, base_indices, obs_mask = self._make_flat_buf(device)
        obs_mask[:, 0] = 1.0  # enable all layers for req 0
        bs = 2
        req_map = torch.zeros(bs, dtype=torch.long, device=device)

        xs = []
        for layer_idx in range(NUM_LAYERS):
            x = torch.randn(bs, HIDDEN_DIM, device=device)
            xs.append(x)
            observe(
                x,
                layer_idx=layer_idx,
                flat_buf=flat_buf,
                obs_mask=obs_mask,
                req_map=req_map,
                base_indices=base_indices,
                offsets=offsets,
            )

        for layer_idx in range(NUM_LAYERS):
            start = layer_idx * MAX_TOKENS
            assert torch.allclose(flat_buf[start : start + bs], xs[layer_idx], atol=1e-6)

    def test_multi_request_isolation(self, device):
        flat_buf, offsets, base_indices, obs_mask = self._make_flat_buf(device)
        obs_mask[0, 0] = 1.0  # enable req 0
        obs_mask[0, 1] = 0.0  # disable req 1

        # Prefill-style: multiple tokens per request
        x = torch.randn(6, HIDDEN_DIM, device=device)
        req_map = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)

        observe(
            x,
            layer_idx=0,
            flat_buf=flat_buf,
            obs_mask=obs_mask,
            req_map=req_map,
            base_indices=base_indices,
            offsets=offsets,
        )

        # Req 0 tokens observed, req 1 tokens zeroed
        assert torch.allclose(flat_buf[:3], x[:3], atol=1e-6)
        assert torch.all(flat_buf[3:6] == 0)

    def test_decode_style_one_token_per_request(self, device):
        flat_buf, offsets, base_indices, obs_mask = self._make_flat_buf(device)
        obs_mask[0, 0] = 1.0  # enable req 0
        obs_mask[0, 1] = 0.0  # disable req 1
        bs = 2
        x = torch.randn(bs, HIDDEN_DIM, device=device)
        req_map = torch.tensor([0, 1], dtype=torch.long, device=device)

        observe(
            x,
            layer_idx=0,
            flat_buf=flat_buf,
            obs_mask=obs_mask,
            req_map=req_map,
            base_indices=base_indices,
            offsets=offsets,
        )

        # Token 0 (req 0) observed, token 1 (req 1) zeroed
        assert torch.allclose(flat_buf[0], x[0], atol=1e-6)
        assert torch.all(flat_buf[1] == 0)

    def test_does_not_modify_x(self, device):
        flat_buf, offsets, base_indices, obs_mask = self._make_flat_buf(device)
        obs_mask[0, 0] = 1.0
        x = torch.randn(4, HIDDEN_DIM, device=device)
        x_copy = x.clone()
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        observe(
            x,
            layer_idx=0,
            flat_buf=flat_buf,
            obs_mask=obs_mask,
            req_map=req_map,
            base_indices=base_indices,
            offsets=offsets,
        )
        assert torch.equal(x, x_copy)


# ─── mask_blend ──────────────────────────────────────────────────────────────


@requires_cuda
class TestMaskBlend:
    def _make_masks(self, device):
        scale = torch.ones(NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM, device=device)
        add = torch.zeros(NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM, device=device)
        return scale, add

    def test_identity(self, device):
        scale, add = self._make_masks(device)
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)
        assert torch.allclose(out, x, atol=1e-6)

    def test_ablation(self, device):
        scale, add = self._make_masks(device)
        scale[0, 0] = 0.0
        add[0, 0] = 0.0
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)
        assert torch.all(out == 0)

    def test_steer(self, device):
        scale, add = self._make_masks(device)
        v = torch.randn(HIDDEN_DIM, device=device)
        add[0, 0] = v
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)
        expected = x + v.unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_patch(self, device):
        scale, add = self._make_masks(device)
        act = torch.randn(HIDDEN_DIM, device=device)
        scale[0, 0] = 0.0
        add[0, 0] = act
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)
        expected = act.unsqueeze(0).expand(4, -1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_multi_request_isolation(self, device):
        scale, add = self._make_masks(device)
        # Req 0: identity, Req 1: ablation
        scale[0, 1] = 0.0
        add[0, 1] = 0.0
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)
        # Req 0 tokens: identity
        assert torch.allclose(out[:2], x[:2], atol=1e-6)
        # Req 1 tokens: ablated
        assert torch.all(out[2:] == 0)

    def test_different_interventions_per_request(self, device):
        scale, add = self._make_masks(device)
        steer_v = torch.randn(HIDDEN_DIM, device=device)
        patch_act = torch.randn(HIDDEN_DIM, device=device)

        # Req 0: steer (scale=1, add=v)
        add[0, 0] = steer_v
        # Req 1: patch (scale=0, add=act)
        scale[0, 1] = 0.0
        add[0, 1] = patch_act
        # Req 2: identity (default)

        x = torch.randn(6, HIDDEN_DIM, device=device)
        req_map = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)

        # Req 0: x + steer_v
        assert torch.allclose(out[:2], x[:2] + steer_v.unsqueeze(0), atol=1e-6)
        # Req 1: patch_act
        assert torch.allclose(out[2:4], patch_act.unsqueeze(0).expand(2, -1), atol=1e-6)
        # Req 2: identity
        assert torch.allclose(out[4:6], x[4:6], atol=1e-6)

    def test_per_layer_different_masks(self, device):
        scale, add = self._make_masks(device)
        v0 = torch.randn(HIDDEN_DIM, device=device)
        v1 = torch.randn(HIDDEN_DIM, device=device)
        add[0, 0] = v0  # layer 0: steer with v0
        add[1, 0] = v1  # layer 1: steer with v1

        x = torch.randn(2, HIDDEN_DIM, device=device)
        req_map = torch.zeros(2, dtype=torch.long, device=device)

        out0 = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)
        out1 = mask_blend(x, layer_idx=1, scale=scale, add=add, req_map=req_map)

        assert torch.allclose(out0, x + v0.unsqueeze(0), atol=1e-6)
        assert torch.allclose(out1, x + v1.unsqueeze(0), atol=1e-6)

    def test_returns_new_tensor(self, device):
        scale, add = self._make_masks(device)
        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)
        assert out is not x
        assert out.data_ptr() != x.data_ptr()

    def test_decode_style_req_map(self, device):
        """One token per request (decode pattern)."""
        scale, add = self._make_masks(device)
        v = torch.randn(HIDDEN_DIM, device=device)
        add[0, 2] = v  # steer req at table_idx=2

        x = torch.randn(3, HIDDEN_DIM, device=device)
        req_map = torch.tensor([0, 2, 5], dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)

        # Token 0 (req 0): identity
        assert torch.allclose(out[0], x[0], atol=1e-6)
        # Token 1 (req 2): steered
        assert torch.allclose(out[1], x[1] + v, atol=1e-6)
        # Token 2 (req 5): identity
        assert torch.allclose(out[2], x[2], atol=1e-6)

    def test_prefill_style_req_map(self, device):
        """Multiple tokens per request (prefill pattern)."""
        scale, add = self._make_masks(device)
        scale[0, 1] = 0.0  # ablate req 1

        x = torch.randn(8, HIDDEN_DIM, device=device)
        # 3 tokens for req 0, 5 tokens for req 1
        req_map = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long, device=device)

        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)

        assert torch.allclose(out[:3], x[:3], atol=1e-6)
        assert torch.all(out[3:] == 0)


# ─── End-to-End ──────────────────────────────────────────────────────────────


@requires_cuda
class TestEndToEnd:
    def _make_flat_buf(self, device):
        total = NUM_LAYERS * MAX_TOKENS
        flat_buf = torch.zeros(total, HIDDEN_DIM, device=device)
        offsets = torch.arange(0, total, MAX_TOKENS, dtype=torch.int64, device=device)
        base_indices = torch.arange(MAX_TOKENS, dtype=torch.int64, device=device)
        return flat_buf, offsets, base_indices

    def test_observe_then_blend(self, device):
        """Observe and blend in sequence, like a real layer."""
        flat_buf, offsets, base_indices = self._make_flat_buf(device)
        obs_mask = torch.zeros(NUM_LAYERS, MAX_RUNNING_REQ, dtype=torch.float32, device=device)
        obs_mask[0, 0] = 1.0

        scale = torch.ones(NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM, device=device)
        add = torch.zeros(NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM, device=device)
        steer_v = torch.randn(HIDDEN_DIM, device=device)
        add[0, 0] = steer_v

        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        # Observe first (records x before blend)
        observe(
            x,
            layer_idx=0,
            flat_buf=flat_buf,
            obs_mask=obs_mask,
            req_map=req_map,
            base_indices=base_indices,
            offsets=offsets,
        )
        # Then blend
        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)

        # Buffer has original x
        assert torch.allclose(flat_buf[:4], x, atol=1e-6)
        # Output is steered
        assert torch.allclose(out, x + steer_v.unsqueeze(0), atol=1e-6)

    def test_identity_is_noop(self, device):
        """All masks identity/zero — output equals input exactly."""
        flat_buf, offsets, base_indices = self._make_flat_buf(device)
        obs_mask = torch.zeros(NUM_LAYERS, MAX_RUNNING_REQ, dtype=torch.float32, device=device)
        scale = torch.ones(NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM, device=device)
        add = torch.zeros(NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM, device=device)

        x = torch.randn(4, HIDDEN_DIM, device=device)
        req_map = torch.zeros(4, dtype=torch.long, device=device)

        observe(
            x,
            layer_idx=0,
            flat_buf=flat_buf,
            obs_mask=obs_mask,
            req_map=req_map,
            base_indices=base_indices,
            offsets=offsets,
        )
        out = mask_blend(x, layer_idx=0, scale=scale, add=add, req_map=req_map)

        # Buffer should be zero (obs_mask=0)
        assert torch.all(flat_buf[:4] == 0)
        # Output should equal input (scale=1, add=0)
        assert torch.allclose(out, x, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
