from __future__ import annotations

import torch


def observe(
    x: torch.Tensor,
    layer_idx: int,
    flat_buf: torch.Tensor,
    obs_mask: torch.Tensor,
    req_map: torch.Tensor,
    base_indices: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    """Write observation into flat buffer using index_copy_. CUDA-graph safe.

    Works for both prefill and decode — the only difference is buffer sizing.

    Args:
        x: [n_tokens, hidden_dim] activations
        layer_idx: which transformer layer
        flat_buf: [num_layers * max_tokens_per_slot, hidden_dim] GPU flat buffer
        obs_mask: [num_layers, max_running_req] per-request per-layer enable
        req_map: [n_tokens] maps each token to its request's table_idx
        base_indices: [max_tokens_per_slot] pre-allocated [0, 1, 2, ...]
        offsets: [num_layers] pre-computed layer offsets into flat_buf
    """
    per_token_mask = obs_mask[layer_idx, req_map]  # [n_tokens]
    masked = x * per_token_mask.unsqueeze(-1)  # [n_tokens, hidden_dim]
    indices = base_indices[: x.shape[0]] + offsets[layer_idx]
    flat_buf.index_copy_(0, indices, masked)


def blend(
    x: torch.Tensor,
    residual: torch.Tensor,
    layer_idx: int,
    scale: torch.Tensor,
    add: torch.Tensor,
    req_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-request intervention to both x and residual.

    Split blend: scale applies to both x and residual, add applies to x only.
    Algebraically equivalent to h' = (x + residual) * scale + add on the full
    hidden state, without breaking the fused norm optimization.

    Args:
        x: [total_tokens, hidden_dim] MLP output
        residual: [total_tokens, hidden_dim] residual stream
        layer_idx: which transformer layer
        scale: [num_layers, max_running_req, hidden_dim] multiplicative mask (default 1.0)
        add: [num_layers, max_running_req, hidden_dim] additive mask (default 0.0)
        req_map: [total_tokens] maps each token to its request's table_idx

    Returns:
        (x_blended, residual_blended) — both [total_tokens, hidden_dim]
    """
    s = scale[layer_idx, req_map]
    return x * s + add[layer_idx, req_map], residual * s
