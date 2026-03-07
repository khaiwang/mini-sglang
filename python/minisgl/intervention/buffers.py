from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class ObsEntry:
    """Tracks an in-flight async D2H copy."""

    layer_idx: int
    offset: int
    length: int
    event: torch.cuda.Event


class ObservationRingBuffer:
    """Streaming ring buffer for prefill observations (eager execution only).

    GPU writes one layer's observation at a time. A dedicated copy stream
    asynchronously copies completed observations to CPU pinned memory.
    Ring space is reused once the copy completes.
    """

    def __init__(
        self,
        ring_size: int,
        hidden_dim: int,
        num_layers: int,
        max_running_req: int,
        device: torch.device,
    ) -> None:
        self._ring_size = ring_size
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._device = device

        # GPU ring buffer
        self._buf = torch.zeros(ring_size, hidden_dim, dtype=torch.float32, device=device)
        # Per-request per-layer observation enable mask
        self._obs_mask = torch.zeros(
            num_layers, max_running_req, dtype=torch.float32, device=device
        )
        # Current write position in ring (CPU-tracked)
        self._write_ptr: int = 0
        # Dedicated stream for async D2H copies
        self._copy_stream = torch.cuda.Stream(device=device)
        # Pinned CPU memory for staging
        self._cpu_staging = torch.zeros(ring_size, hidden_dim, dtype=torch.float32, pin_memory=True)
        # In-flight copies
        self._pending: List[ObsEntry] = []

    def write(
        self,
        x: torch.Tensor,
        layer_idx: int,
        obs_mask: torch.Tensor,
        req_map: torch.Tensor,
    ) -> None:
        """Write one layer's observation into the ring buffer.

        1. Compute masked activations
        2. Copy into ring at _write_ptr
        3. Launch async D2H copy on _copy_stream
        4. Advance _write_ptr
        """
        n = x.shape[0]
        offset = self._write_ptr

        # Write masked activations into ring (handle wrap-around)
        per_token_mask = obs_mask[layer_idx, req_map]  # [total_tokens]
        masked = x * per_token_mask.unsqueeze(-1)  # [total_tokens, hidden_dim]
        end = offset + n
        if end <= self._ring_size:
            self._buf[offset:end] = masked
        else:
            first = self._ring_size - offset
            self._buf[offset : self._ring_size] = masked[:first]
            self._buf[: end - self._ring_size] = masked[first:]

        # Record event on compute stream so copy stream can wait
        compute_event = torch.cuda.Event()
        compute_event.record()

        # Async copy on dedicated stream (handle wrap-around)
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(compute_event)
            if end <= self._ring_size:
                self._cpu_staging[offset:end].copy_(self._buf[offset:end], non_blocking=True)
            else:
                first = self._ring_size - offset
                self._cpu_staging[offset : self._ring_size].copy_(
                    self._buf[offset : self._ring_size], non_blocking=True
                )
                self._cpu_staging[: end - self._ring_size].copy_(
                    self._buf[: end - self._ring_size], non_blocking=True
                )
            copy_done = torch.cuda.Event()
            copy_done.record(self._copy_stream)

        self._pending.append(
            ObsEntry(layer_idx=layer_idx, offset=offset, length=n, event=copy_done)
        )

        # Advance write pointer (wrap around)
        self._write_ptr = (offset + n) % self._ring_size

    def _read_staging(self, offset: int, length: int) -> torch.Tensor:
        """Read from CPU staging, handling wrap-around."""
        end = offset + length
        if end <= self._ring_size:
            return self._cpu_staging[offset:end].clone()
        return torch.cat(
            [
                self._cpu_staging[offset : self._ring_size].clone(),
                self._cpu_staging[: end - self._ring_size].clone(),
            ]
        )

    def poll(self) -> List[Tuple[int, torch.Tensor]]:
        """Check for completed copies. Returns (layer_idx, cpu_tensor) pairs.

        Non-blocking: only returns entries whose copy events have completed.
        """
        completed = []
        still_pending = []
        for entry in self._pending:
            if entry.event.query():
                cpu_data = self._read_staging(entry.offset, entry.length)
                completed.append((entry.layer_idx, cpu_data))
            else:
                still_pending.append(entry)
        self._pending = still_pending
        return completed

    def flush(self) -> List[Tuple[int, torch.Tensor]]:
        """Synchronize all pending copies and return all results."""
        self._copy_stream.synchronize()
        results = []
        for entry in self._pending:
            cpu_data = self._read_staging(entry.offset, entry.length)
            results.append((entry.layer_idx, cpu_data))
        self._pending.clear()
        return results

    def reset(self) -> None:
        """Zero buffer, reset write pointer, clear pending list."""
        self._buf.zero_()
        self._write_ptr = 0
        self._pending.clear()

    def available_space(self) -> int:
        """Ring size minus in-flight entries' total length."""
        in_flight = sum(e.length for e in self._pending)
        return self._ring_size - in_flight


class DecodeObservationBuffer:
    """Simple flat buffer for decode observations (CUDA graph compatible).

    Pre-computed per-layer offsets allow index_copy_ writes during graph replay.
    Bulk-copied to CPU after replay.
    """

    def __init__(
        self,
        num_layers: int,
        max_decode_bs: int,
        hidden_dim: int,
        max_running_req: int,
        device: torch.device,
    ) -> None:
        self._num_layers = num_layers
        self._max_decode_bs = max_decode_bs
        self._hidden_dim = hidden_dim
        self._device = device

        total_rows = num_layers * max_decode_bs
        # GPU flat buffer
        self._buf = torch.zeros(total_rows, hidden_dim, dtype=torch.float32, device=device)
        # Per-request per-layer observation enable mask
        self._obs_mask = torch.zeros(
            num_layers, max_running_req, dtype=torch.float32, device=device
        )
        # Pre-computed layer offsets: [0, max_decode_bs, 2*max_decode_bs, ...]
        self._offsets = torch.arange(0, total_rows, max_decode_bs, dtype=torch.int64, device=device)
        # Base indices: [0, 1, 2, ..., max_decode_bs-1]
        self._base_indices = torch.arange(max_decode_bs, dtype=torch.int64, device=device)
        # Pinned CPU buffer for async copy
        self._cpu_buf = torch.zeros(total_rows, hidden_dim, dtype=torch.float32, pin_memory=True)

    def get_write_args(self, layer_idx: int, bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (buf, indices) for observe_decode.

        indices = base_indices[:bs] + offsets[layer_idx]
        """
        indices = self._base_indices[:bs] + self._offsets[layer_idx]
        return self._buf, indices

    def reset(self) -> None:
        """Zero the buffer."""
        self._buf.zero_()

    def copy_to_cpu(self) -> torch.Tensor:
        """Async copy entire buffer to CPU (non_blocking). Returns CPU tensor."""
        self._cpu_buf.copy_(self._buf, non_blocking=True)
        return self._cpu_buf


class MaskBuffer:
    """Per-request, per-layer intervention masks for blend operations.

    Default state is identity: scale=1.0, add=0.0 (no-op blend).
    """

    def __init__(
        self,
        num_layers: int,
        max_running_req: int,
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        self._num_layers = num_layers
        self._max_running_req = max_running_req
        self._hidden_dim = hidden_dim
        self._device = device

        # Multiplicative mask — default 1.0 (identity)
        self._scale = torch.ones(
            num_layers, max_running_req, hidden_dim, dtype=torch.float32, device=device
        )
        # Additive mask — default 0.0 (identity)
        self._add = torch.zeros(
            num_layers, max_running_req, hidden_dim, dtype=torch.float32, device=device
        )

    def reset(self) -> None:
        """Reset to identity: scale=1, add=0."""
        self._scale.fill_(1.0)
        self._add.zero_()

    def set_ablate(self, layer: int, table_idx: int) -> None:
        """Ablation: zero out activations at (layer, table_idx)."""
        self._scale[layer, table_idx] = 0.0
        self._add[layer, table_idx] = 0.0

    def set_steer(
        self,
        layer: int,
        table_idx: int,
        vector: torch.Tensor,
        alpha: float = 1.0,
    ) -> None:
        """Steering: add alpha * vector to activations at (layer, table_idx).

        scale=1 (pass through original), add=alpha*vector.
        """
        self._scale[layer, table_idx] = 1.0
        self._add[layer, table_idx] = alpha * vector.to(self._device)

    def set_patch(self, layer: int, table_idx: int, activation: torch.Tensor) -> None:
        """Patching: replace activations at (layer, table_idx) with given activation.

        scale=0 (discard original), add=activation.
        """
        self._scale[layer, table_idx] = 0.0
        self._add[layer, table_idx] = activation.to(self._device)
