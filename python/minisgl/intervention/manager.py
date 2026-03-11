"""CPU-side intervention orchestration.

Call sequence per step: prepare_step(batch) -> GPU forward -> process_step(x_obs_cpu, res_obs_cpu, batch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
from minisgl.intervention.context import InterventionContext
from minisgl.intervention.request import InterventionRequest, WriteOp


@dataclass
class _ActiveEntry:
    """Internal tracking for one active intervention."""

    request: InterventionRequest
    table_idx: Optional[int] = None
    observations: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    pending_patches: List[Tuple[int, torch.Tensor]] = field(default_factory=list)


class InterventionManager:
    """Manages intervention lifecycles for all active requests.

    Sits between the scheduler and engine: sets GPU masks before forward,
    extracts observations after forward, and handles conditional writes.
    """

    def __init__(self, ctx: InterventionContext) -> None:
        self._ctx = ctx
        self._active: Dict[int, _ActiveEntry] = {}
        self._needs_rerun: Set[int] = set()

    # --- Lifecycle ---

    def submit(self, uid: int, request: InterventionRequest) -> None:
        """Register an intervention request for the given uid."""
        if uid in self._active:
            raise ValueError(f"Duplicate intervention uid: {uid}")
        self._active[uid] = _ActiveEntry(request=request)

    def remove(self, uid: int) -> None:
        """Remove an intervention. No-op if uid not found."""
        self._active.pop(uid, None)
        self._needs_rerun.discard(uid)

    # --- Per-step hooks ---

    def prepare_step(self, batch) -> None:
        """Set GPU masks/buffers before forward pass.

        1. Reset obs buffers, obs_mask, mask_buffer to identity.
        2. For each req in batch with active intervention:
           - Set obs_mask for observed layers + conditional_write read_layers.
           - Apply write ops (ablate/steer/patch) to mask_buffer.
        3. Apply pending patches from previous conditional writes.
        4. Clear needs_rerun for uids whose patches are now applied.
        """
        from minisgl.scheduler.prefill import ChunkedReq

        ictx = self._ctx

        # 1. Reset everything to identity
        ictx.x_obs_buffer.reset()
        ictx.residual_obs_buffer.reset()
        ictx.obs_mask.zero_()
        ictx.mask_buffer.reset()

        # Build uid -> table_idx mapping from batch
        uid_to_table: Dict[int, int] = {}
        for req in batch.reqs:
            if isinstance(req, ChunkedReq):
                continue
            uid_to_table[req.uid] = req.table_idx

        # 2. Apply interventions for active entries present in batch
        for uid, entry in self._active.items():
            if uid not in uid_to_table:
                entry.table_idx = None
                continue
            table_idx = uid_to_table[uid]
            entry.table_idx = table_idx

            req = entry.request

            # Set obs_mask for explicit observations
            for obs_op in req.observations:
                ictx.obs_mask[obs_op.layer, table_idx] = 1.0

            # Set obs_mask for conditional write read_layers (auto-observe)
            for cw_op in req.conditional_writes:
                ictx.obs_mask[cw_op.read_layer, table_idx] = 1.0

            # Apply static write ops
            for write_op in req.writes:
                self._apply_write(write_op, table_idx)

            # 3. Apply pending patches (from previous conditional write pass)
            for write_layer, activation in entry.pending_patches:
                ictx.mask_buffer.set_patch(write_layer, table_idx, activation)
            if entry.pending_patches:
                entry.pending_patches.clear()
                # 4. Patch applied — this uid no longer needs rerun
                self._needs_rerun.discard(uid)

    def process_step(
        self,
        x_obs_cpu: Optional[torch.Tensor],
        res_obs_cpu: Optional[torch.Tensor],
        batch,
    ) -> None:
        """Extract observations and run conditional write callbacks.

        1. Compute per-req token offsets in flat buffer.
        2. Extract layer slices from CPU buffers for observed layers.
        3. Run conditional_write callbacks and queue resulting patches.
        """
        from minisgl.scheduler.prefill import ChunkedReq

        if x_obs_cpu is None or res_obs_cpu is None:
            return

        max_tokens = self._ctx.x_obs_buffer._max_tokens_per_slot

        # Build offset map: uid -> (start_offset, length) in flat buffer token dim
        uid_offsets: Dict[int, Tuple[int, int]] = {}
        offset = 0
        for req in batch.reqs:
            if isinstance(req, ChunkedReq):
                continue
            if batch.is_decode:
                length = 1
            else:
                length = req.extend_len
            uid_offsets[req.uid] = (offset, length)
            offset += length

        # Process each active entry that has a table_idx in this batch
        for uid, entry in self._active.items():
            if entry.table_idx is None:
                continue
            if uid not in uid_offsets:
                continue

            tok_start, tok_len = uid_offsets[uid]
            req = entry.request

            # Collect observed layers (explicit + conditional read_layers)
            observed_layers: set = {op.layer for op in req.observations}
            for cw_op in req.conditional_writes:
                observed_layers.add(cw_op.read_layer)

            # Extract observations per layer
            for layer in observed_layers:
                buf_start = layer * max_tokens + tok_start
                buf_end = buf_start + tok_len
                x_slice = x_obs_cpu[buf_start:buf_end].clone()
                res_slice = res_obs_cpu[buf_start:buf_end].clone()
                entry.observations[layer] = (x_slice, res_slice)

            # Run conditional write callbacks
            for cw_op in req.conditional_writes:
                obs = entry.observations.get(cw_op.read_layer)
                if obs is None:
                    continue
                x_obs, res_obs = obs
                activation = cw_op.fn(x_obs, res_obs)
                entry.pending_patches.append((cw_op.write_layer, activation))
                self._needs_rerun.add(uid)

    # --- Query ---

    def needs_rerun(self, uid: int) -> bool:
        """True if this uid's token should NOT be emitted (stall for conditional write)."""
        return uid in self._needs_rerun

    def get_observations(self, uid: int) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns {layer: (x_obs_cpu, res_obs_cpu)}. Empty dict if not found."""
        entry = self._active.get(uid)
        if entry is None:
            return {}
        return dict(entry.observations)

    # --- Internal ---

    def _apply_write(self, write_op: WriteOp, table_idx: int) -> None:
        mb = self._ctx.mask_buffer
        if write_op.kind == "ablate":
            mb.set_ablate(write_op.layer, table_idx)
        elif write_op.kind == "steer":
            assert write_op.vector is not None
            mb.set_steer(write_op.layer, table_idx, write_op.vector, write_op.alpha)
        elif write_op.kind == "patch":
            assert write_op.vector is not None
            mb.set_patch(write_op.layer, table_idx, write_op.vector)
        else:
            raise ValueError(f"Unknown write kind: {write_op.kind}")
