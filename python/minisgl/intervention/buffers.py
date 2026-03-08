from __future__ import annotations

from typing import Tuple

import torch


class ObservationBuffer:
    """Flat pre-allocated buffer for layer observations.

    Single instance at runtime, sized for the larger of prefill/decode token counts.
    Uses index_copy_ for CUDA-graph-safe writes. Bulk-copied to CPU after use.
    """

    def __init__(
        self,
        num_layers: int,
        max_tokens_per_slot: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._num_layers = num_layers
        self._max_tokens_per_slot = max_tokens_per_slot
        self._hidden_dim = hidden_dim
        self._device = device

        total_rows = num_layers * max_tokens_per_slot
        # GPU flat buffer
        self._buf = torch.zeros(total_rows, hidden_dim, dtype=dtype, device=device)
        # Pre-computed layer offsets: [0, max_tokens_per_slot, 2*max_tokens_per_slot, ...]
        self._offsets = torch.arange(
            0, total_rows, max_tokens_per_slot, dtype=torch.int64, device=device
        )
        # Base indices: [0, 1, 2, ..., max_tokens_per_slot-1]
        self._base_indices = torch.arange(
            max_tokens_per_slot, dtype=torch.int64, device=device
        )
        # Pinned CPU buffer for async copy
        self._cpu_buf = torch.zeros(total_rows, hidden_dim, dtype=dtype, pin_memory=True)

    def get_write_args(self, layer_idx: int, n_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (buf, indices) for the observe op.

        indices = base_indices[:n_tokens] + offsets[layer_idx]
        """
        indices = self._base_indices[:n_tokens] + self._offsets[layer_idx]
        return self._buf, indices

    def reset(self) -> None:
        """Zero the buffer."""
        self._buf.zero_()

    def copy_to_cpu(self) -> torch.Tensor:
        """Async copy entire buffer to CPU (non_blocking).

        Returns a reference to the internal pinned staging buffer. Caller must
        process the returned tensor before the next copy_to_cpu() call, which
        overwrites it.
        """
        self._cpu_buf.copy_(self._buf, non_blocking=True)
        return self._cpu_buf


class MaskBuffer:
    """Per-request, per-layer intervention masks for blend operations.

    Default state is identity: scale=1.0, add=0.0 (no-op blend).

    Internally allocates ``max_running_req + 1`` slots so that the dummy
    request (``table_idx = max_running_req``) used for CUDA-graph padding
    indexes into a valid sentinel slot that stays at identity.
    """

    def __init__(
        self,
        num_layers: int,
        max_running_req: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._num_layers = num_layers
        self._max_running_req = max_running_req
        self._hidden_dim = hidden_dim
        self._device = device

        num_slots = max_running_req + 1  # +1 sentinel for dummy/padding requests
        # Multiplicative mask — default 1.0 (identity)
        self._scale = torch.ones(
            num_layers, num_slots, hidden_dim, dtype=dtype, device=device
        )
        # Additive mask — default 0.0 (identity)
        self._add = torch.zeros(
            num_layers, num_slots, hidden_dim, dtype=dtype, device=device
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
