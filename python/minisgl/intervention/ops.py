from __future__ import annotations

import torch


def observe_prefill(
    x: torch.Tensor,
    layer_idx: int,
    ring_buf: torch.Tensor,
    obs_mask: torch.Tensor,
    req_map: torch.Tensor,
    write_offset: int,
) -> None:
    """Write observation into ring buffer at write_offset. Eager mode only.

    Args:
        x: [total_tokens, hidden_dim] activations
        layer_idx: which transformer layer
        ring_buf: [ring_size, hidden_dim] GPU ring buffer
        obs_mask: [num_layers, max_running_req] per-request per-layer enable
        req_map: [total_tokens] maps each token to its request's table_idx
        write_offset: position in ring to start writing
    """
    per_token_mask = obs_mask[layer_idx, req_map]  # [total_tokens]
    masked = x * per_token_mask.unsqueeze(-1)  # [total_tokens, hidden_dim]
    n = x.shape[0]
    ring_buf[write_offset : write_offset + n] = masked


def observe_decode(
    x: torch.Tensor,
    layer_idx: int,
    flat_buf: torch.Tensor,
    obs_mask: torch.Tensor,
    req_map: torch.Tensor,
    base_indices: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    """Write observation into flat buffer using index_copy_. CUDA-graph safe.

    Args:
        x: [batch_size, hidden_dim] activations (one token per request)
        layer_idx: which transformer layer
        flat_buf: [num_layers * max_decode_bs, hidden_dim] GPU flat buffer
        obs_mask: [num_layers, max_running_req] per-request per-layer enable
        req_map: [batch_size] maps each token to its request's table_idx
        base_indices: [max_decode_bs] pre-allocated [0, 1, 2, ...]
        offsets: [num_layers] pre-computed layer offsets into flat_buf
    """
    per_token_mask = obs_mask[layer_idx, req_map]  # [batch_size]
    masked = x * per_token_mask.unsqueeze(-1)  # [batch_size, hidden_dim]
    indices = base_indices[: x.shape[0]] + offsets[layer_idx]
    flat_buf.index_copy_(0, indices, masked)


def mask_blend(
    x: torch.Tensor,
    layer_idx: int,
    scale: torch.Tensor,
    add: torch.Tensor,
    req_map: torch.Tensor,
) -> torch.Tensor:
    """Apply per-request intervention. Works in both eager and graph mode.

    Args:
        x: [total_tokens, hidden_dim] activations
        layer_idx: which transformer layer
        scale: [num_layers, max_running_req, hidden_dim] multiplicative mask (default 1.0)
        add: [num_layers, max_running_req, hidden_dim] additive mask (default 0.0)
        req_map: [total_tokens] maps each token to its request's table_idx

    Returns:
        [total_tokens, hidden_dim] blended activations
    """
    return x * scale[layer_idx, req_map] + add[layer_idx, req_map]
