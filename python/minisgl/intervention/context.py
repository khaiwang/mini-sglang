from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.intervention.buffers import MaskBuffer, ObservationBuffer


@dataclass
class InterventionContext:
    """Holds all intervention state: observation buffers, mask buffer, and obs_mask.

    Two observation buffers capture both sides of the fused-norm split:
    ``x_obs_buffer`` for MLP output, ``residual_obs_buffer`` for residual stream.
    ``obs_mask`` is standalone — not inside buffer classes.
    """

    x_obs_buffer: ObservationBuffer
    residual_obs_buffer: ObservationBuffer
    mask_buffer: MaskBuffer
    obs_mask: torch.Tensor  # [num_layers, max_running_req + 1] (+1 sentinel for padding)


_INTERVENTION_CTX: InterventionContext | None = None


def get_intervention_ctx() -> InterventionContext | None:
    """Returns None when intervention is disabled."""
    return _INTERVENTION_CTX


def set_intervention_ctx(ctx: InterventionContext) -> None:
    global _INTERVENTION_CTX
    assert _INTERVENTION_CTX is None, "Intervention context is already set"
    _INTERVENTION_CTX = ctx


def clear_intervention_ctx() -> None:
    global _INTERVENTION_CTX
    _INTERVENTION_CTX = None
