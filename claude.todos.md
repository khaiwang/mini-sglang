# Intervention TODOs

## Ring buffer for prefill observations

Replace flat `ObservationBuffer` (prefill instance) with streaming ring buffer to reduce GPU memory from `num_layers * max_tokens * hidden_dim` to `ring_size * hidden_dim`.

**Design:**
- Ring streams layer-by-layer with async D2H copies via dedicated copy stream
- Ring sized at 2-3x `max_extend_tokens` so GPU never stalls waiting for copy drain
- Per-write tag (single int32 at write offset, copied with data) for corruption detection — CPU checks tag matches expected `layer_idx` after D2H
- Pre-check: `available_space()` assertion
- Post-check: tag verification

**Why deferred:** The flat buffer works correctly and is simpler to reason about. Ring buffer is a memory optimization for large models (8B+ params where `num_layers * max_tokens * hidden_dim` exceeds several GB).
