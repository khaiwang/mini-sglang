from __future__ import annotations

from minisgl.core import get_global_ctx
from minisgl.intervention.context import InterventionContext
from minisgl.intervention.ops import blend, observe


def wrap_layers(layers: list, ictx: InterventionContext) -> None:
    """Wrap each layer's forward() to run observe + blend after execution.

    Call before CUDA graph capture so graphs include intervention ops.
    Identity masks (scale=1, add=0, obs_mask=0) make ops numerical no-ops.
    """
    for layer_idx, layer in enumerate(layers):
        _wrap_one(layer, layer_idx, ictx)


def unwrap_layers(layers: list) -> None:
    """Restore original forward() on each layer."""
    for layer in layers:
        if hasattr(layer, "_original_forward"):
            del layer.forward
            del layer._original_forward


def _wrap_one(layer, layer_idx: int, ictx: InterventionContext) -> None:
    original_forward = layer.forward
    layer._original_forward = original_forward  # _ hides from BaseOP.state_dict()

    x_buf = ictx.x_obs_buffer
    res_buf = ictx.residual_obs_buffer
    mb = ictx.mask_buffer
    obs_mask = ictx.obs_mask

    def hooked_forward(x, residual=None):
        x, residual = original_forward(x, residual)
        req_map = get_global_ctx().batch.req_map
        observe(
            x, layer_idx, x_buf._buf, obs_mask,
            req_map, x_buf._base_indices, x_buf._offsets,
        )
        observe(
            residual, layer_idx, res_buf._buf, obs_mask,
            req_map, res_buf._base_indices, res_buf._offsets,
        )
        x, residual = blend(x, residual, layer_idx, mb._scale, mb._add, req_map)
        return x, residual

    layer.forward = hooked_forward
