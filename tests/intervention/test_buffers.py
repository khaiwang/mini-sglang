"""Tests for intervention buffer classes: ObservationRingBuffer, DecodeObservationBuffer, MaskBuffer."""

from __future__ import annotations

import pytest
import torch
from minisgl.intervention.buffers import (
    DecodeObservationBuffer,
    MaskBuffer,
    ObservationRingBuffer,
)

# Small test sizes
RING_SIZE = 64
NUM_LAYERS = 4
HIDDEN_DIM = 16
MAX_RUNNING_REQ = 8
MAX_DECODE_BS = 8

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    return torch.device("cuda:0")


# ─── ObservationRingBuffer ───────────────────────────────────────────────────


@requires_cuda
class TestObservationRingBuffer:
    def _make_buf(self, device):
        return ObservationRingBuffer(
            ring_size=RING_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            max_running_req=MAX_RUNNING_REQ,
            device=device,
        )

    def test_init_shapes(self, device):
        buf = self._make_buf(device)
        assert buf._buf.shape == (RING_SIZE, HIDDEN_DIM)
        assert buf._obs_mask.shape == (NUM_LAYERS, MAX_RUNNING_REQ)
        assert buf._cpu_staging.shape == (RING_SIZE, HIDDEN_DIM)
        assert buf._buf.device.type == "cuda"
        assert buf._cpu_staging.is_pinned()

    def test_write_single_layer(self, device):
        buf = self._make_buf(device)
        n_tokens = 4
        x = torch.randn(n_tokens, HIDDEN_DIM, device=device)
        req_map = torch.zeros(n_tokens, dtype=torch.long, device=device)  # all req 0
        buf._obs_mask[0, 0] = 1.0  # enable layer 0, req 0

        buf.write(x, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map)
        torch.cuda.synchronize()

        # Data should be at offset 0
        written = buf._buf[:n_tokens]
        assert torch.allclose(written, x, atol=1e-6)

    def test_write_advances_ptr(self, device):
        buf = self._make_buf(device)
        n_tokens = 10
        x = torch.randn(n_tokens, HIDDEN_DIM, device=device)
        req_map = torch.zeros(n_tokens, dtype=torch.long, device=device)
        buf._obs_mask[0, 0] = 1.0

        assert buf._write_ptr == 0
        buf.write(x, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map)
        assert buf._write_ptr == n_tokens

    def test_write_wraps_around(self, device):
        buf = self._make_buf(device)
        req_map = torch.zeros(1, dtype=torch.long, device=device)
        buf._obs_mask[0, 0] = 1.0

        # Write enough to get near the end
        n_tokens = RING_SIZE - 2
        x1 = torch.randn(n_tokens, HIDDEN_DIM, device=device)
        buf.write(x1, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map[:1].expand(n_tokens))
        assert buf._write_ptr == n_tokens

        # Write more to wrap around
        x2 = torch.randn(4, HIDDEN_DIM, device=device)
        req_map4 = torch.zeros(4, dtype=torch.long, device=device)
        buf.write(x2, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map4)
        # (RING_SIZE - 2 + 4) % RING_SIZE = 2
        assert buf._write_ptr == (n_tokens + 4) % RING_SIZE

    def test_flush_returns_all(self, device):
        buf = self._make_buf(device)
        n_tokens = 4
        x = torch.randn(n_tokens, HIDDEN_DIM, device=device)
        req_map = torch.zeros(n_tokens, dtype=torch.long, device=device)
        buf._obs_mask[:, 0] = 1.0  # enable all layers for req 0

        buf.write(x, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map)
        buf.write(x, layer_idx=1, obs_mask=buf._obs_mask, req_map=req_map)

        results = buf.flush()
        assert len(results) == 2
        assert results[0][0] == 0  # layer_idx
        assert results[1][0] == 1

    def test_flush_data_correctness(self, device):
        buf = self._make_buf(device)
        n_tokens = 4
        x = torch.randn(n_tokens, HIDDEN_DIM, device=device)
        req_map = torch.zeros(n_tokens, dtype=torch.long, device=device)
        buf._obs_mask[0, 0] = 1.0

        buf.write(x, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map)
        results = buf.flush()

        assert len(results) == 1
        layer_idx, cpu_data = results[0]
        assert layer_idx == 0
        assert cpu_data.device.type == "cpu"
        assert torch.allclose(cpu_data, x.cpu(), atol=1e-6)

    def test_poll_completed(self, device):
        buf = self._make_buf(device)
        n_tokens = 4
        x = torch.randn(n_tokens, HIDDEN_DIM, device=device)
        req_map = torch.zeros(n_tokens, dtype=torch.long, device=device)
        buf._obs_mask[0, 0] = 1.0

        buf.write(x, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map)
        # Synchronize to ensure copy completes
        torch.cuda.synchronize()
        buf._copy_stream.synchronize()

        results = buf.poll()
        assert len(results) == 1
        assert results[0][0] == 0

    def test_poll_does_not_block(self, device):
        buf = self._make_buf(device)
        # With no writes, poll should return empty immediately
        results = buf.poll()
        assert results == []

    def test_reset_clears_state(self, device):
        buf = self._make_buf(device)
        n_tokens = 4
        x = torch.randn(n_tokens, HIDDEN_DIM, device=device)
        req_map = torch.zeros(n_tokens, dtype=torch.long, device=device)
        buf._obs_mask[0, 0] = 1.0

        buf.write(x, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map)
        buf._copy_stream.synchronize()
        buf.reset()

        assert buf._write_ptr == 0
        assert len(buf._pending) == 0
        assert torch.all(buf._buf == 0)

    def test_masked_write(self, device):
        buf = self._make_buf(device)
        n_tokens = 6
        x = torch.randn(n_tokens, HIDDEN_DIM, device=device)
        # req_map: tokens 0-2 → req 0, tokens 3-5 → req 1
        req_map = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)
        # Only observe req 0 at layer 0
        buf._obs_mask[0, 0] = 1.0
        buf._obs_mask[0, 1] = 0.0  # req 1 disabled

        buf.write(x, layer_idx=0, obs_mask=buf._obs_mask, req_map=req_map)
        torch.cuda.synchronize()

        written = buf._buf[:n_tokens]
        # Tokens 0-2 should match x, tokens 3-5 should be zero
        assert torch.allclose(written[:3], x[:3], atol=1e-6)
        assert torch.all(written[3:] == 0)

    def test_multi_layer_workflow(self, device):
        buf = self._make_buf(device)
        n_tokens = 4
        req_map = torch.zeros(n_tokens, dtype=torch.long, device=device)
        buf._obs_mask[:, 0] = 1.0  # enable all layers for req 0

        xs = []
        for layer_idx in range(NUM_LAYERS):
            x = torch.randn(n_tokens, HIDDEN_DIM, device=device)
            xs.append(x.cpu())
            buf.write(x, layer_idx=layer_idx, obs_mask=buf._obs_mask, req_map=req_map)

        results = buf.flush()
        assert len(results) == NUM_LAYERS
        for i, (layer_idx, cpu_data) in enumerate(results):
            assert layer_idx == i
            assert torch.allclose(cpu_data, xs[i], atol=1e-6)


# ─── DecodeObservationBuffer ────────────────────────────────────────────────


@requires_cuda
class TestDecodeObservationBuffer:
    def _make_buf(self, device):
        return DecodeObservationBuffer(
            num_layers=NUM_LAYERS,
            max_decode_bs=MAX_DECODE_BS,
            hidden_dim=HIDDEN_DIM,
            max_running_req=MAX_RUNNING_REQ,
            device=device,
        )

    def test_init_shapes(self, device):
        buf = self._make_buf(device)
        total_rows = NUM_LAYERS * MAX_DECODE_BS
        assert buf._buf.shape == (total_rows, HIDDEN_DIM)
        assert buf._offsets.shape == (NUM_LAYERS,)
        assert buf._base_indices.shape == (MAX_DECODE_BS,)
        assert buf._cpu_buf.shape == (total_rows, HIDDEN_DIM)

    def test_offsets_correct(self, device):
        buf = self._make_buf(device)
        expected = torch.arange(0, NUM_LAYERS * MAX_DECODE_BS, MAX_DECODE_BS, dtype=torch.int64)
        assert torch.equal(buf._offsets.cpu(), expected)

    def test_get_write_args(self, device):
        buf = self._make_buf(device)
        bs = 4
        for layer_idx in range(NUM_LAYERS):
            flat_buf, indices = buf.get_write_args(layer_idx, bs)
            assert flat_buf is buf._buf
            expected_indices = torch.arange(bs, dtype=torch.int64, device=device) + (
                layer_idx * MAX_DECODE_BS
            )
            assert torch.equal(indices, expected_indices)

    def test_reset(self, device):
        buf = self._make_buf(device)
        buf._buf.fill_(42.0)
        buf.reset()
        assert torch.all(buf._buf == 0)

    def test_copy_to_cpu(self, device):
        buf = self._make_buf(device)
        # Write some known data
        data = torch.randn_like(buf._buf)
        buf._buf.copy_(data)

        cpu_result = buf.copy_to_cpu()
        torch.cuda.synchronize()

        assert cpu_result.device.type == "cpu"
        assert torch.allclose(cpu_result, data.cpu(), atol=1e-6)


# ─── MaskBuffer ──────────────────────────────────────────────────────────────


@requires_cuda
class TestMaskBuffer:
    def _make_buf(self, device):
        return MaskBuffer(
            num_layers=NUM_LAYERS,
            max_running_req=MAX_RUNNING_REQ,
            hidden_dim=HIDDEN_DIM,
            device=device,
        )

    def test_init_identity(self, device):
        buf = self._make_buf(device)
        assert torch.all(buf._scale == 1.0)
        assert torch.all(buf._add == 0.0)

    def test_init_shapes(self, device):
        buf = self._make_buf(device)
        assert buf._scale.shape == (NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM)
        assert buf._add.shape == (NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM)

    def test_reset(self, device):
        buf = self._make_buf(device)
        buf._scale.fill_(0.5)
        buf._add.fill_(0.5)
        buf.reset()
        assert torch.all(buf._scale == 1.0)
        assert torch.all(buf._add == 0.0)

    def test_set_ablate(self, device):
        buf = self._make_buf(device)
        buf.set_ablate(layer=1, table_idx=3)
        assert torch.all(buf._scale[1, 3] == 0.0)
        assert torch.all(buf._add[1, 3] == 0.0)
        # Others unchanged
        assert torch.all(buf._scale[0, 0] == 1.0)
        assert torch.all(buf._add[0, 0] == 0.0)

    def test_set_steer(self, device):
        buf = self._make_buf(device)
        vec = torch.randn(HIDDEN_DIM, device=device)
        alpha = 2.5
        buf.set_steer(layer=2, table_idx=1, vector=vec, alpha=alpha)
        assert torch.all(buf._scale[2, 1] == 1.0)
        assert torch.allclose(buf._add[2, 1], alpha * vec, atol=1e-6)

    def test_set_steer_cpu_vector(self, device):
        buf = self._make_buf(device)
        vec_cpu = torch.randn(HIDDEN_DIM)  # CPU tensor
        buf.set_steer(layer=0, table_idx=0, vector=vec_cpu, alpha=1.0)
        assert torch.allclose(buf._add[0, 0], vec_cpu.to(device), atol=1e-6)

    def test_set_patch(self, device):
        buf = self._make_buf(device)
        activation = torch.randn(HIDDEN_DIM, device=device)
        buf.set_patch(layer=3, table_idx=5, activation=activation)
        assert torch.all(buf._scale[3, 5] == 0.0)
        assert torch.allclose(buf._add[3, 5], activation, atol=1e-6)

    def test_multiple_interventions(self, device):
        buf = self._make_buf(device)
        vec = torch.randn(HIDDEN_DIM, device=device)
        act = torch.randn(HIDDEN_DIM, device=device)

        buf.set_ablate(layer=0, table_idx=0)
        buf.set_steer(layer=1, table_idx=1, vector=vec, alpha=1.0)
        buf.set_patch(layer=2, table_idx=2, activation=act)

        # Ablation
        assert torch.all(buf._scale[0, 0] == 0.0)
        assert torch.all(buf._add[0, 0] == 0.0)
        # Steering
        assert torch.all(buf._scale[1, 1] == 1.0)
        assert torch.allclose(buf._add[1, 1], vec, atol=1e-6)
        # Patching
        assert torch.all(buf._scale[2, 2] == 0.0)
        assert torch.allclose(buf._add[2, 2], act, atol=1e-6)
        # Untouched slot
        assert torch.all(buf._scale[3, 3] == 1.0)
        assert torch.all(buf._add[3, 3] == 0.0)

    def test_overwrite(self, device):
        buf = self._make_buf(device)
        vec = torch.randn(HIDDEN_DIM, device=device)

        buf.set_ablate(layer=0, table_idx=0)
        assert torch.all(buf._scale[0, 0] == 0.0)

        buf.set_steer(layer=0, table_idx=0, vector=vec, alpha=1.0)
        assert torch.all(buf._scale[0, 0] == 1.0)
        assert torch.allclose(buf._add[0, 0], vec, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
