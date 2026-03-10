"""Tests for intervention buffer classes: ObservationBuffer, MaskBuffer."""

from __future__ import annotations

import pytest
import torch
from minisgl.intervention.buffers import (
    MaskBuffer,
    ObservationBuffer,
)

# Small test sizes
NUM_LAYERS = 4
HIDDEN_DIM = 16
MAX_RUNNING_REQ = 8
MAX_TOKENS = 8

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    return torch.device("cuda:0")


# ─── ObservationBuffer ─────────────────────────────────────────────────────


@requires_cuda
class TestObservationBuffer:
    def _make_buf(self, device, max_tokens=MAX_TOKENS, dtype=torch.float32):
        return ObservationBuffer(
            num_layers=NUM_LAYERS,
            max_tokens_per_slot=max_tokens,
            hidden_dim=HIDDEN_DIM,
            device=device,
            dtype=dtype,
        )

    def test_init_shapes(self, device):
        buf = self._make_buf(device)
        total_rows = NUM_LAYERS * MAX_TOKENS
        assert buf._buf.shape == (total_rows, HIDDEN_DIM)
        assert buf._offsets.shape == (NUM_LAYERS,)
        assert buf._base_indices.shape == (MAX_TOKENS,)
        assert len(buf._cpu_bufs) == 2
        for cb in buf._cpu_bufs:
            assert cb.shape == (total_rows, HIDDEN_DIM)
            assert cb.is_pinned()
        assert buf._buf.device.type == "cuda"

    def test_init_dtype(self, device):
        buf = self._make_buf(device, dtype=torch.float16)
        assert buf._buf.dtype == torch.float16
        for cb in buf._cpu_bufs:
            assert cb.dtype == torch.float16

    def test_offsets_correct(self, device):
        buf = self._make_buf(device)
        expected = torch.arange(0, NUM_LAYERS * MAX_TOKENS, MAX_TOKENS, dtype=torch.int64)
        assert torch.equal(buf._offsets.cpu(), expected)

    def test_get_write_args(self, device):
        buf = self._make_buf(device)
        n_tokens = 4
        for layer_idx in range(NUM_LAYERS):
            flat_buf, indices = buf.get_write_args(layer_idx, n_tokens)
            assert flat_buf is buf._buf
            expected_indices = torch.arange(n_tokens, dtype=torch.int64, device=device) + (
                layer_idx * MAX_TOKENS
            )
            assert torch.equal(indices, expected_indices)

    def test_reset(self, device):
        buf = self._make_buf(device)
        buf._buf.fill_(42.0)
        buf.reset()
        assert torch.all(buf._buf == 0)

    def test_copy_to_cpu(self, device):
        buf = self._make_buf(device)
        data = torch.randn_like(buf._buf)
        buf._buf.copy_(data)

        cpu_result = buf.copy_to_cpu()
        torch.cuda.synchronize()

        assert cpu_result.device.type == "cpu"
        assert torch.allclose(cpu_result, data.cpu(), atol=1e-6)

    def test_copy_to_cpu_ping_pong(self, device):
        """copy_to_cpu alternates between two ping-pong buffers."""
        buf = self._make_buf(device)
        ref1 = buf.copy_to_cpu()  # buf[0]
        ref2 = buf.copy_to_cpu()  # buf[1]
        ref3 = buf.copy_to_cpu()  # buf[0] again
        assert ref1.data_ptr() != ref2.data_ptr()
        assert ref1.data_ptr() == ref3.data_ptr()

    def test_copy_to_cpu_preserves_previous(self, device):
        """Data from call N is preserved when call N+1 writes to a different buffer."""
        buf = self._make_buf(device)
        # Write known data and copy
        data1 = torch.randn_like(buf._buf)
        buf._buf.copy_(data1)
        ref1 = buf.copy_to_cpu()
        torch.cuda.synchronize()
        expected1 = data1.cpu().clone()

        # Write different data and copy (goes to other buffer)
        data2 = torch.randn_like(buf._buf)
        buf._buf.copy_(data2)
        ref2 = buf.copy_to_cpu()
        torch.cuda.synchronize()

        # ref1 still holds data1 (not corrupted by second copy)
        assert torch.allclose(ref1, expected1, atol=1e-6)
        # ref2 holds data2
        assert torch.allclose(ref2, data2.cpu(), atol=1e-6)

    def test_write_and_read_back(self, device):
        """Write via index_copy_ (like observe op) and read back via copy_to_cpu."""
        buf = self._make_buf(device)
        n_tokens = 4
        x = torch.randn(n_tokens, HIDDEN_DIM, device=device)

        for layer_idx in range(NUM_LAYERS):
            flat_buf, indices = buf.get_write_args(layer_idx, n_tokens)
            flat_buf.index_copy_(0, indices, x)

        cpu_result = buf.copy_to_cpu()
        torch.cuda.synchronize()

        for layer_idx in range(NUM_LAYERS):
            start = layer_idx * MAX_TOKENS
            assert torch.allclose(cpu_result[start : start + n_tokens], x.cpu(), atol=1e-6)


# ─── MaskBuffer ──────────────────────────────────────────────────────────────


@requires_cuda
class TestMaskBuffer:
    def _make_buf(self, device, dtype=torch.float32):
        return MaskBuffer(
            num_layers=NUM_LAYERS,
            max_running_req=MAX_RUNNING_REQ,
            hidden_dim=HIDDEN_DIM,
            device=device,
            dtype=dtype,
        )

    def test_init_identity(self, device):
        buf = self._make_buf(device)
        assert torch.all(buf._scale == 1.0)
        assert torch.all(buf._add == 0.0)

    def test_init_shapes(self, device):
        buf = self._make_buf(device)
        num_slots = MAX_RUNNING_REQ + 1  # +1 sentinel
        assert buf._scale.shape == (NUM_LAYERS, num_slots, HIDDEN_DIM)
        assert buf._add.shape == (NUM_LAYERS, num_slots, HIDDEN_DIM)

    def test_init_dtype(self, device):
        buf = self._make_buf(device, dtype=torch.float16)
        assert buf._scale.dtype == torch.float16
        assert buf._add.dtype == torch.float16

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
