"""Tests for InterventionManager. Requires CUDA."""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.intervention.buffers import MaskBuffer, ObservationBuffer
from minisgl.intervention.context import InterventionContext
from minisgl.intervention.manager import InterventionManager
from minisgl.intervention.request import InterventionRequest

NUM_LAYERS = 4
HIDDEN_DIM = 16
MAX_RUNNING_REQ = 8
MAX_TOKENS = 8

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ─── Helpers ─────────────────────────────────────────────────────────────────


class FakeHandle:
    """Minimal stand-in for BaseCacheHandle."""

    pass


def _make_req(uid: int, table_idx: int, input_len: int = 4, cached_len: int = 0) -> Req:
    return Req(
        input_ids=torch.zeros(input_len, dtype=torch.long),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=10,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=FakeHandle(),
    )


def _make_batch(
    reqs_info: List[Tuple[int, int, int, int]],
    phase: str,
    device: torch.device,
) -> Batch:
    """Build a Batch with req_map set.

    reqs_info: list of (uid, table_idx, input_len, cached_len).
    For decode: cached_len = input_len - 1 (one new token each).
    """
    reqs = []
    for uid, table_idx, input_len, cached_len in reqs_info:
        reqs.append(_make_req(uid, table_idx, input_len, cached_len))

    batch = Batch.__new__(Batch)
    batch.reqs = reqs
    batch.phase = phase
    batch.padded_reqs = reqs

    # Build req_map: [total_tokens] -> table_idx
    parts = []
    for req in reqs:
        parts.append(torch.full((req.extend_len,), req.table_idx, dtype=torch.long))
    if parts:
        batch.req_map = torch.cat(parts).to(device)
    else:
        batch.req_map = torch.empty(0, dtype=torch.long, device=device)

    return batch


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.fixture
def ictx(device):
    x_obs = ObservationBuffer(NUM_LAYERS, MAX_TOKENS, HIDDEN_DIM, device)
    res_obs = ObservationBuffer(NUM_LAYERS, MAX_TOKENS, HIDDEN_DIM, device)
    mb = MaskBuffer(NUM_LAYERS, MAX_RUNNING_REQ, HIDDEN_DIM, device)
    obs_mask = torch.zeros(NUM_LAYERS, MAX_RUNNING_REQ + 1, dtype=torch.float32, device=device)
    return InterventionContext(
        x_obs_buffer=x_obs, residual_obs_buffer=res_obs,
        mask_buffer=mb, obs_mask=obs_mask,
    )


@pytest.fixture
def manager(ictx):
    return InterventionManager(ictx)


# ─── TestSubmitRemove ────────────────────────────────────────────────────────


class TestSubmitRemove:
    def test_submit_registers(self, manager):
        req = InterventionRequest().observe(0)
        manager.submit(1, req)
        assert 1 in manager._active

    def test_duplicate_raises(self, manager):
        req = InterventionRequest()
        manager.submit(1, req)
        with pytest.raises(ValueError, match="Duplicate"):
            manager.submit(1, req)

    def test_remove_clears(self, manager):
        manager.submit(1, InterventionRequest())
        manager.remove(1)
        assert 1 not in manager._active

    def test_remove_noop(self, manager):
        manager.remove(999)  # should not raise


# ─── TestPrepareStep ─────────────────────────────────────────────────────────


@requires_cuda
class TestPrepareStep:
    def test_resets_obs_buffers(self, manager, ictx, device):
        ictx.x_obs_buffer._buf.fill_(42.0)
        ictx.residual_obs_buffer._buf.fill_(42.0)
        batch = _make_batch([], "decode", device)
        manager.prepare_step(batch)
        assert torch.all(ictx.x_obs_buffer._buf == 0)
        assert torch.all(ictx.residual_obs_buffer._buf == 0)

    def test_resets_obs_mask(self, manager, ictx, device):
        ictx.obs_mask.fill_(1.0)
        batch = _make_batch([], "decode", device)
        manager.prepare_step(batch)
        assert torch.all(ictx.obs_mask == 0)

    def test_resets_mask_buffer(self, manager, ictx, device):
        ictx.mask_buffer.set_ablate(0, 0)
        batch = _make_batch([], "decode", device)
        manager.prepare_step(batch)
        assert torch.allclose(ictx.mask_buffer._scale, torch.ones_like(ictx.mask_buffer._scale))
        assert torch.allclose(ictx.mask_buffer._add, torch.zeros_like(ictx.mask_buffer._add))

    def test_sets_obs_mask_for_observed_layers(self, manager, ictx, device):
        req = InterventionRequest().observe(1).observe(3)
        manager.submit(10, req)
        batch = _make_batch([(10, 2, 5, 4)], "decode", device)
        manager.prepare_step(batch)
        assert ictx.obs_mask[1, 2].item() == 1.0
        assert ictx.obs_mask[3, 2].item() == 1.0
        # Other layers should be 0
        assert ictx.obs_mask[0, 2].item() == 0.0
        assert ictx.obs_mask[2, 2].item() == 0.0

    def test_sets_obs_mask_for_conditional_write_read_layer(self, manager, ictx, device):
        fn = lambda x, r: x
        req = InterventionRequest().conditional_write(1, 3, fn)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)
        # read_layer=1 should be observed
        assert ictx.obs_mask[1, 0].item() == 1.0

    def test_sets_mask_for_ablation(self, manager, ictx, device):
        req = InterventionRequest().ablate(2)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)
        assert ictx.mask_buffer._scale[2, 0].sum().item() == 0.0
        assert ictx.mask_buffer._add[2, 0].sum().item() == 0.0

    def test_sets_mask_for_steering(self, manager, ictx, device):
        v = torch.randn(HIDDEN_DIM, device=device)
        req = InterventionRequest().steer(1, v, alpha=2.0)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)
        expected = 2.0 * v
        assert torch.allclose(ictx.mask_buffer._add[1, 0], expected)
        assert torch.allclose(
            ictx.mask_buffer._scale[1, 0], torch.ones(HIDDEN_DIM, device=device)
        )

    def test_sets_mask_for_patching(self, manager, ictx, device):
        v = torch.randn(HIDDEN_DIM, device=device)
        req = InterventionRequest().patch(1, v)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)
        assert ictx.mask_buffer._scale[1, 0].sum().item() == 0.0
        assert torch.allclose(ictx.mask_buffer._add[1, 0], v)

    def test_uid_not_in_batch_ignored(self, manager, ictx, device):
        req = InterventionRequest().ablate(0)
        manager.submit(99, req)
        batch = _make_batch([(1, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)
        # mask_buffer should be identity since uid=99 is not in batch
        assert torch.allclose(ictx.mask_buffer._scale, torch.ones_like(ictx.mask_buffer._scale))

    def test_multiple_requests(self, manager, ictx, device):
        req1 = InterventionRequest().ablate(0)
        req2 = InterventionRequest().observe(2)
        manager.submit(10, req1)
        manager.submit(20, req2)
        batch = _make_batch([(10, 0, 5, 4), (20, 1, 5, 4)], "decode", device)
        manager.prepare_step(batch)
        # req1: ablate layer 0, table_idx 0
        assert ictx.mask_buffer._scale[0, 0].sum().item() == 0.0
        # req2: observe layer 2, table_idx 1
        assert ictx.obs_mask[2, 1].item() == 1.0
        # Cross-check: req2 should not ablate
        assert torch.allclose(
            ictx.mask_buffer._scale[:, 1], torch.ones(NUM_LAYERS, HIDDEN_DIM, device=device)
        )

    def test_applies_pending_patches(self, manager, ictx, device):
        req = InterventionRequest().observe(0)
        manager.submit(10, req)
        # Manually queue a pending patch
        v = torch.randn(HIDDEN_DIM, device=device)
        manager._active[10].pending_patches.append((2, v))
        manager._needs_rerun.add(10)

        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)
        # Pending patch at layer 2 should be applied as set_patch
        assert ictx.mask_buffer._scale[2, 0].sum().item() == 0.0
        assert torch.allclose(ictx.mask_buffer._add[2, 0], v)
        # needs_rerun cleared
        assert not manager.needs_rerun(10)


# ─── TestProcessStep ─────────────────────────────────────────────────────────


@requires_cuda
class TestProcessStep:
    def test_extracts_decode_observations(self, manager, ictx, device):
        """Decode: 1 token per req, extract correct slice."""
        req = InterventionRequest().observe(1)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)

        # Write known data to GPU obs buffer at layer 1, token 0
        known = torch.randn(1, HIDDEN_DIM, device=device)
        start = 1 * MAX_TOKENS  # layer 1 offset
        ictx.x_obs_buffer._buf[start : start + 1] = known
        ictx.residual_obs_buffer._buf[start : start + 1] = known * 2

        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()

        manager.process_step(x_cpu, res_cpu, batch)
        obs = manager.get_observations(10)
        assert 1 in obs
        x_obs, res_obs = obs[1]
        assert x_obs.shape == (1, HIDDEN_DIM)
        assert torch.allclose(x_obs, known.cpu(), atol=1e-6)
        assert torch.allclose(res_obs, (known * 2).cpu(), atol=1e-6)

    def test_extracts_prefill_observations(self, manager, ictx, device):
        """Prefill: multi-token per req, correct offsets."""
        req = InterventionRequest().observe(0)
        manager.submit(10, req)
        # uid=10: 3 tokens (input_len=3, cached_len=0)
        # uid=20: 2 tokens (input_len=2, cached_len=0), no intervention
        batch = _make_batch([(10, 0, 3, 0), (20, 1, 2, 0)], "prefill", device)
        manager.prepare_step(batch)

        # Write known data for first 5 tokens at layer 0
        data = torch.randn(5, HIDDEN_DIM, device=device)
        ictx.x_obs_buffer._buf[:5] = data

        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()

        manager.process_step(x_cpu, res_cpu, batch)
        obs = manager.get_observations(10)
        assert 0 in obs
        x_obs, _ = obs[0]
        # uid=10 occupies tokens 0..2 (3 tokens)
        assert x_obs.shape == (3, HIDDEN_DIM)
        assert torch.allclose(x_obs, data[:3].cpu(), atol=1e-6)

    def test_noop_when_obs_none(self, manager, ictx, device):
        """process_step(None, None, batch) should not crash."""
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.process_step(None, None, batch)  # no crash

    def test_conditional_write_creates_pending_and_needs_rerun(self, manager, ictx, device):
        fn = lambda x, r: x + r  # simple fn
        req = InterventionRequest().conditional_write(0, 2, fn)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)

        # Write known data to GPU buffer at layer 0
        known_x = torch.randn(1, HIDDEN_DIM, device=device)
        known_r = torch.randn(1, HIDDEN_DIM, device=device)
        ictx.x_obs_buffer._buf[:1] = known_x
        ictx.residual_obs_buffer._buf[:1] = known_r

        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()

        manager.process_step(x_cpu, res_cpu, batch)
        assert manager.needs_rerun(10)
        # Pending patch should exist
        entry = manager._active[10]
        assert len(entry.pending_patches) == 1
        write_layer, activation = entry.pending_patches[0]
        assert write_layer == 2
        expected = known_x.cpu() + known_r.cpu()
        assert torch.allclose(activation, expected, atol=1e-6)

    def test_conditional_write_full_cycle(self, manager, ictx, device):
        """Full cycle: prepare1→process1(rerun)→prepare2(applies patch, no rerun)."""
        fn = lambda x, r: x * 2
        req = InterventionRequest().conditional_write(0, 2, fn)
        manager.submit(10, req)

        batch = _make_batch([(10, 0, 5, 4)], "decode", device)

        # Pass 1: observe
        manager.prepare_step(batch)
        known = torch.ones(1, HIDDEN_DIM, device=device) * 3.0
        ictx.x_obs_buffer._buf[:1] = known
        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()
        manager.process_step(x_cpu, res_cpu, batch)
        assert manager.needs_rerun(10)

        # Pass 2: apply pending patch
        manager.prepare_step(batch)
        assert not manager.needs_rerun(10)
        # Patch should be applied at layer 2
        assert ictx.mask_buffer._scale[2, 0].sum().item() == 0.0
        expected_add = (known * 2).squeeze(0)  # fn = x * 2, keep on same device
        assert torch.allclose(ictx.mask_buffer._add[2, 0], expected_add, atol=1e-6)


# ─── TestGetObservations ─────────────────────────────────────────────────────


@requires_cuda
class TestGetObservations:
    def test_returns_empty_for_unknown_uid(self, manager):
        assert manager.get_observations(999) == {}

    def test_returns_stored_after_process_step(self, manager, ictx, device):
        req = InterventionRequest().observe(0)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)

        known = torch.randn(1, HIDDEN_DIM, device=device)
        ictx.x_obs_buffer._buf[:1] = known
        ictx.residual_obs_buffer._buf[:1] = known
        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()

        manager.process_step(x_cpu, res_cpu, batch)
        obs = manager.get_observations(10)
        assert 0 in obs
        assert obs[0][0].shape == (1, HIDDEN_DIM)


# ─── TestNeedsRerun ──────────────────────────────────────────────────────────


@requires_cuda
class TestNeedsRerun:
    def test_false_by_default(self, manager):
        assert not manager.needs_rerun(10)

    def test_true_after_conditional_write_processed(self, manager, ictx, device):
        fn = lambda x, r: x
        req = InterventionRequest().conditional_write(0, 2, fn)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)

        known = torch.randn(1, HIDDEN_DIM, device=device)
        ictx.x_obs_buffer._buf[:1] = known
        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()

        manager.process_step(x_cpu, res_cpu, batch)
        assert manager.needs_rerun(10)

    def test_cleared_after_next_prepare_step(self, manager, ictx, device):
        fn = lambda x, r: x
        req = InterventionRequest().conditional_write(0, 2, fn)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)

        # Pass 1
        manager.prepare_step(batch)
        known = torch.randn(1, HIDDEN_DIM, device=device)
        ictx.x_obs_buffer._buf[:1] = known
        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()
        manager.process_step(x_cpu, res_cpu, batch)
        assert manager.needs_rerun(10)

        # Pass 2
        manager.prepare_step(batch)
        assert not manager.needs_rerun(10)


# ─── TestLifecycle ───────────────────────────────────────────────────────────


@requires_cuda
class TestLifecycle:
    def test_full_request_lifecycle(self, manager, ictx, device):
        """submit -> prepare -> simulate forward -> process -> get_obs -> remove."""
        v = torch.randn(HIDDEN_DIM, device=device)
        req = InterventionRequest().observe(1).steer(2, v)
        manager.submit(10, req)

        batch = _make_batch([(10, 0, 5, 4)], "decode", device)
        manager.prepare_step(batch)

        # Verify masks set
        assert ictx.obs_mask[1, 0].item() == 1.0
        assert torch.allclose(ictx.mask_buffer._add[2, 0], v)

        # Simulate forward: write data to buffer
        known = torch.randn(1, HIDDEN_DIM, device=device)
        start = 1 * MAX_TOKENS
        ictx.x_obs_buffer._buf[start : start + 1] = known
        ictx.residual_obs_buffer._buf[start : start + 1] = known

        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()

        manager.process_step(x_cpu, res_cpu, batch)
        obs = manager.get_observations(10)
        assert 1 in obs

        manager.remove(10)
        assert manager.get_observations(10) == {}

    def test_conditional_write_lifecycle(self, manager, ictx, device):
        """submit -> prepare(obs) -> process(rerun) -> prepare(patch) -> process(emit) -> remove."""
        fn = lambda x, r: x + r
        req = InterventionRequest().conditional_write(0, 2, fn)
        manager.submit(10, req)
        batch = _make_batch([(10, 0, 5, 4)], "decode", device)

        # Pass 1: observe
        manager.prepare_step(batch)
        assert ictx.obs_mask[0, 0].item() == 1.0
        known_x = torch.ones(1, HIDDEN_DIM, device=device)
        known_r = torch.ones(1, HIDDEN_DIM, device=device) * 2
        ictx.x_obs_buffer._buf[:1] = known_x
        ictx.residual_obs_buffer._buf[:1] = known_r
        x_cpu = ictx.x_obs_buffer.copy_to_cpu()
        res_cpu = ictx.residual_obs_buffer.copy_to_cpu()
        torch.cuda.synchronize()
        manager.process_step(x_cpu, res_cpu, batch)
        assert manager.needs_rerun(10)

        # Pass 2: apply patch
        manager.prepare_step(batch)
        assert not manager.needs_rerun(10)
        # Patch at layer 2: scale=0, add = known_x + known_r = 3.0
        assert ictx.mask_buffer._scale[2, 0].sum().item() == 0.0
        expected = torch.ones(HIDDEN_DIM, device=device) * 3.0
        assert torch.allclose(ictx.mask_buffer._add[2, 0], expected, atol=1e-6)

        # Pass 2 process: no pending -> no rerun
        manager.process_step(None, None, batch)
        assert not manager.needs_rerun(10)

        manager.remove(10)
        assert 10 not in manager._active


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
