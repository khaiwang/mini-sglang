"""Tests for req_map plumbing and InterventionContext singleton."""

from __future__ import annotations

import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.engine.graph import GraphCaptureBuffer
from minisgl.intervention.buffers import MaskBuffer, ObservationBuffer
from minisgl.intervention.context import (
    InterventionContext,
    clear_intervention_ctx,
    get_intervention_ctx,
    set_intervention_ctx,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# Small test sizes
NUM_LAYERS = 4
HIDDEN_DIM = 16
MAX_RUNNING_REQ = 8
MAX_TOKENS = 8


# ─── Batch.req_map ───────────────────────────────────────────────────────────


class TestReqMapInBatch:
    def test_batch_has_req_map_field(self):
        """Batch dataclass should accept req_map as a set-after-init field."""
        dummy = SamplingParams()

        class FakeHandle:
            pass

        req = Req(
            input_ids=torch.tensor([1, 2, 3]),
            table_idx=0,
            cached_len=0,
            output_len=5,
            uid=0,
            sampling_params=dummy,
            cache_handle=FakeHandle(),
        )
        batch = Batch(reqs=[req], phase="prefill")
        # req_map is init=False, should be settable
        batch.req_map = torch.tensor([0, 0, 0], dtype=torch.int64)
        assert batch.req_map.shape == (3,)
        assert batch.req_map.dtype == torch.int64

    def test_batch_req_map_multi_request(self):
        """req_map correctly maps tokens to different table indices."""
        dummy = SamplingParams()

        class FakeHandle:
            pass

        reqs = [
            Req(torch.tensor([1, 2]), 0, 0, 5, 0, dummy, FakeHandle()),
            Req(torch.tensor([3, 4, 5]), 3, 0, 5, 1, dummy, FakeHandle()),
        ]
        batch = Batch(reqs=reqs, phase="prefill")
        # Simulate what _make_input_tuple would produce
        batch.req_map = torch.tensor([0, 0, 3, 3, 3], dtype=torch.int64)
        assert batch.req_map.shape == (5,)
        assert batch.req_map[0] == 0
        assert batch.req_map[2] == 3


# ─── GraphCaptureBuffer.req_map ──────────────────────────────────────────────


@requires_cuda
class TestGraphCaptureBuffer:
    def test_init_has_req_map(self):
        device = torch.device("cuda:0")
        buf = GraphCaptureBuffer.init(bs=8, vocab_size=32, device=device)
        assert hasattr(buf, "req_map")
        assert buf.req_map.shape == (8,)
        assert buf.req_map.dtype == torch.int64
        assert buf.req_map.device.type == "cuda"
        assert torch.all(buf.req_map == 0)

    def test_set_batch_copies_req_map(self):
        device = torch.device("cuda:0")
        buf = GraphCaptureBuffer.init(bs=8, vocab_size=32, device=device)

        dummy = SamplingParams()

        class FakeHandle:
            pass

        reqs = [Req(torch.tensor([i]), i, 0, 5, i, dummy, FakeHandle()) for i in range(4)]
        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = reqs + [reqs[0]] * 4  # pad to 8

        buf.set_batch(batch)

        # After set_batch, batch.req_map should be a slice of buf.req_map
        assert batch.req_map.shape == (8,)
        assert batch.req_map.data_ptr() == buf.req_map.data_ptr()

    def test_copy_from_transfers_req_map(self):
        device = torch.device("cuda:0")
        buf = GraphCaptureBuffer.init(bs=8, vocab_size=32, device=device)

        dummy = SamplingParams()

        class FakeHandle:
            pass

        reqs = [Req(torch.tensor([i]), i, 0, 5, i, dummy, FakeHandle()) for i in range(4)]
        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = reqs + [reqs[0]] * 4
        # Set all fields that copy_from reads
        batch.input_ids = torch.zeros(8, dtype=torch.int32, device=device)
        batch.out_loc = torch.zeros(8, dtype=torch.int32, device=device)
        batch.positions = torch.zeros(8, dtype=torch.int32, device=device)
        batch.req_map = torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], dtype=torch.int64, device=device)

        buf.copy_from(batch)

        expected = torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], dtype=torch.int64, device=device)
        assert torch.equal(buf.req_map, expected)


# ─── InterventionContext singleton ────────────────────────────────────────────


@requires_cuda
class TestInterventionContext:
    @pytest.fixture(autouse=True)
    def _clear_ctx(self):
        """Ensure clean state before and after each test."""
        clear_intervention_ctx()
        yield
        clear_intervention_ctx()

    def _make_ctx(self, device):
        obs = ObservationBuffer(NUM_LAYERS, MAX_TOKENS, HIDDEN_DIM, device)
        mask = MaskBuffer(NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM, device)
        obs_mask = torch.zeros(NUM_LAYERS, MAX_RUNNING_REQ + 1, device=device)
        return InterventionContext(
            obs_buffer=obs,
            mask_buffer=mask,
            obs_mask=obs_mask,
        )

    def test_default_is_none(self):
        assert get_intervention_ctx() is None

    def test_set_and_get(self):
        device = torch.device("cuda:0")
        ctx = self._make_ctx(device)
        set_intervention_ctx(ctx)
        assert get_intervention_ctx() is ctx

    def test_double_set_raises(self):
        device = torch.device("cuda:0")
        ctx = self._make_ctx(device)
        set_intervention_ctx(ctx)
        with pytest.raises(AssertionError, match="already set"):
            set_intervention_ctx(ctx)

    def test_clear(self):
        device = torch.device("cuda:0")
        ctx = self._make_ctx(device)
        set_intervention_ctx(ctx)
        clear_intervention_ctx()
        assert get_intervention_ctx() is None

    def test_clear_when_none_is_noop(self):
        """Clearing when nothing is set should not raise."""
        clear_intervention_ctx()
        assert get_intervention_ctx() is None

    def test_set_after_clear(self):
        device = torch.device("cuda:0")
        ctx1 = self._make_ctx(device)
        ctx2 = self._make_ctx(device)
        set_intervention_ctx(ctx1)
        clear_intervention_ctx()
        set_intervention_ctx(ctx2)
        assert get_intervention_ctx() is ctx2

    def test_ctx_fields(self):
        device = torch.device("cuda:0")
        ctx = self._make_ctx(device)
        set_intervention_ctx(ctx)
        retrieved = get_intervention_ctx()
        assert isinstance(retrieved.obs_buffer, ObservationBuffer)
        assert isinstance(retrieved.mask_buffer, MaskBuffer)
        assert retrieved.obs_mask.shape == (NUM_LAYERS, MAX_RUNNING_REQ + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
