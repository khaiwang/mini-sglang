# Buffer+Mask Intervention System — Roadmap

## Goal

Build an activation intervention system for mini-sglang that supports observe (read), steer/ablate (write), and conditional (read-then-write) interventions — unified across prefill and decode, compatible with CUDA graphs, with zero control flow in the hot path.

## Core Design

### Two Fixed Ops Per Layer

After each transformer layer's `forward()`, three tensor ops execute unconditionally:

```python
# 1. Observe x (MLP output) into flat observation buffer
observe(x, layer_idx, x_flat_buf, obs_mask, req_map, base_indices, offsets)

# 2. Observe residual into separate flat observation buffer
observe(residual, layer_idx, res_flat_buf, obs_mask, req_map, base_indices, offsets)

# 3. Blend: apply per-request intervention masks (split blend)
#    scale applies to both x and residual, add applies to x only
x, residual = blend(x, residual, layer_idx, scale, add, req_map)
# Equivalent to: x = x * s + add[...], residual = residual * s
```

**Why dual observe + split blend?** Mini-sglang's fused norm splits the hidden state into `(x, residual)` — an internal optimization. In HuggingFace models (which nnsight targets), layer `.output` is the full hidden state `h = x + residual`. By observing both tensors separately, users can reconstruct the full hidden state on CPU. The split blend (`scale` on both, `add` on `x` only) is algebraically equivalent to `h' = (x + residual) * scale + add` — matching nnsight's hidden-state-level intervention — without breaking the fused norm optimization.

All ops:
- **Always execute** — no branches, no conditionals, no hooks
- **All are CUDA-graph safe** — `observe` uses `index_copy_` with pre-computed offsets, `blend` is pure elementwise
- **CUDA graphs are only used for decode** (`graph.py:149` checks `batch.is_decode`). Prefill always runs eagerly, even with chunked prefill.
- **Reduce to no-ops via data**: `obs_mask=0` skips observation, `scale=1/add=0` is identity blend
- **Support per-request, per-layer granularity** via `req_map` gather
- **Two `ObservationBuffer` instances per tensor** at runtime: one sized for prefill (`max_extend_tokens`), one for decode (`max_decode_bs`). Same class, different sizes.

### Unified Prefill + Decode via `req_map`

`req_map: [total_tokens] → table_idx` maps each token in the flattened batch to its owning request's stable slot.

| Phase | `x` shape | `req_map` | Notes |
|-------|-----------|-----------|-------|
| Prefill | `[total_tokens, hidden]` | `[tok0→R0, tok1→R0, ..., tokN→R1, ...]` | Multiple tokens per request |
| Decode | `[batch_size, hidden]` | `[table_idx_0, table_idx_1, ...]` | One token per request |
| Chunked prefill | Same as prefill | Same | `total_tokens ≤ max_extend_tokens` guaranteed |

All mask/observation tensors are indexed by `(layer, table_idx)`. The gather `mask[layer, req_map]` fans out per-request values to per-token automatically. Requests in the same batch with different intervention targets (e.g., request A observes layer 5, request B observes layer 12) coexist without branching — the per-request masks encode the difference.

### `req_map` Plumbing

`req_map` is already computed in `_make_input_tuple()` (`scheduler.py:254-261`) as the `table_idx` mapping. To make it available inside `model.forward()`:

1. Add `req_map: torch.Tensor` field to `Batch` (`core.py`)
2. Set it in `Scheduler._prepare_batch()` from `_make_input_tuple()`
3. Access via `get_global_ctx().batch.req_map` in intervention ops
4. Add to `GraphCaptureBuffer` (`graph.py`), copied in `copy_from()` like `input_ids`

### Observation Granularity

Any observation pattern is expressible via the `obs_mask[layer, table_idx]` tensor:

- **All tokens for request R at layer L**: `obs_mask[L, R] = 1.0`, all other entries 0
- **Last token only**: set `obs_mask` such that only the last token position accumulates (via a separate position-aware index, or post-process from full observation)
- **Specific position P**: use a position-filtered index tensor
- **Bulk harvest** (à la Goodfire): `obs_mask[:, :] = 1.0` for all layers, all requests

### Performance Characteristics

**No existing fusions are broken.** Each transformer layer is already a sequence of discrete kernel calls (fused_norm → attention → fused_norm → MLP). The intervention ops are inserted between two layers that already communicate through global memory. No cross-layer fusion exists in mini-sglang today.

**Cost model for the two inserted ops per layer:**
- Kernel launch overhead: ~5-10µs × 2 ops × `num_layers` = 320-640µs for 32 layers
- Memory bandwidth: negligible in decode (bs × hidden_dim ≈ 8KB per op), hidden behind matmuls in prefill
- Compute: elementwise multiply+add, negligible vs attention/MLP matmuls

**CUDA graph implications:**
- Ops at every layer (fixed topology), identity masks make them numerical no-ops
- No graph topology change when interventions are enabled/disabled — purely data-driven
- Extra kernels increase graph size slightly but graph capture is one-time cost

**Possible optimization (future):** fuse observe+blend into a single custom kernel per layer, halving kernel launch count.

## Async Pipeline

Both prefill and decode use the same flat `ObservationBuffer` pattern (different instances with different sizing). The pipeline is:

```
Stage 1 (GPU, per layer)          Stage 2 (after forward)          Stage 3 (CPU, overlapped)
─────────────────────────────     ─────────────────────────        ──────────────────────────
observe() index_copy_ into        copy_to_cpu() bulk copies        InterventionManager.step()
flat buffer at each layer.        entire flat buffer to CPU         reads observations, runs
Pre-computed offsets, graph-safe. pinned memory. Non-blocking.      handlers, updates masks.
```

**Prefill buffer**: `ObservationBuffer(num_layers, max_extend_tokens, hidden_dim, device, dtype)` — may be large for big models, but simple and correct.

**Decode buffer**: `ObservationBuffer(num_layers, max_decode_bs, hidden_dim, device, dtype)` — small (~144MB for 8B models).

**Future optimization (see `claude.todos.md`):** Replace the prefill instance with a streaming ring buffer to reduce GPU memory from `num_layers * max_tokens * hidden_dim` to `ring_size * hidden_dim`, with per-layer async D2H copies.

## Components

### `intervention/buffers.py` — Pre-allocated GPU Tensors

```python
class ObservationBuffer:
    # Flat pre-allocated buffer for layer observations (prefill or decode)
    # Two instances at runtime with different max_tokens_per_slot sizing
    _buf: Tensor              # [num_layers * max_tokens_per_slot, hidden_dim]
    _offsets: Tensor          # [num_layers] — pre-computed layer offsets
    _base_indices: Tensor     # [max_tokens_per_slot] — [0, 1, 2, ...]
    _cpu_buf: Tensor          # pinned CPU staging

    def get_write_args(layer_idx, n_tokens) -> Tuple[Tensor, Tensor]: ...
    def reset() -> None: ...
    def copy_to_cpu() -> Tensor: ...  # returns ref to internal pinned buf (aliased!)

class MaskBuffer:
    # Per-request, per-layer intervention masks
    _scale: Tensor            # [num_layers, max_running_req, hidden_dim] — default 1.0
    _add: Tensor              # [num_layers, max_running_req, hidden_dim] — default 0.0

    def reset(self) -> None: ...            # scale=1, add=0 (identity)
    def set_ablate(self, layer, table_idx) -> None: ...
    def set_steer(self, layer, table_idx, vector, alpha) -> None: ...
    def set_patch(self, layer, table_idx, activation) -> None: ...
```

All buffer classes accept a `dtype` parameter (default `torch.float32`) to match model precision and avoid type promotion overhead.

All attributes `_`-prefixed to hide from `BaseOP.state_dict()`.

`obs_mask` is a standalone tensor `[num_layers, max_running_req]` — not stored inside buffer classes. Passed as an argument to the `observe` op.

Buffer sizes: `max_obs_tokens = max_extend_tokens`, `max_running_req` from `EngineConfig`. Pre-allocated once at engine init, reused across all forward passes.

### `intervention/ops.py` — Pure Tensor Functions

```python
def observe(x, layer_idx, flat_buf, obs_mask, req_map, base_indices, offsets):
    """Write observation into flat buffer using index_copy_. CUDA-graph safe.
    Works for both prefill and decode — only buffer sizing differs.
    Called twice per layer: once for x, once for residual."""
    per_token_mask = obs_mask[layer_idx, req_map]
    masked = x * per_token_mask.unsqueeze(-1)
    indices = base_indices[:x.shape[0]] + offsets[layer_idx]
    flat_buf.index_copy_(0, indices, masked)

def blend(x, residual, layer_idx, scale, add, req_map):
    """Split blend: scale on both, add on x only. Matches nnsight semantics."""
    s = scale[layer_idx, req_map]
    return x * s + add[layer_idx, req_map], residual * s
```

### `intervention/context.py` — Global Singleton

```python
@dataclass
class InterventionContext:
    x_obs_buffer: ObservationBuffer        # MLP output observations
    residual_obs_buffer: ObservationBuffer  # residual stream observations
    mask_buffer: MaskBuffer
    obs_mask: Tensor  # [num_layers, max_running_req + 1]

_INTERVENTION_CTX: InterventionContext | None = None

def get_intervention_ctx() -> InterventionContext | None:
    """Returns None when intervention is disabled (vanilla mode)."""
    return _INTERVENTION_CTX

def set_intervention_ctx(ctx: InterventionContext) -> None: ...
```

### `intervention/manager.py` — CPU-side Async Logic

```python
class InterventionManager:
    def __init__(self, ctx: InterventionContext): ...

    def submit(self, uid: int, request: InterventionRequest) -> None:
        """Register intervention for a request."""

    def remove(self, uid: int) -> None:
        """Cleanup when request finishes."""

    def prepare_step(self) -> None:
        """Called before forward: reset obs_buf, update routes/masks from pending requests."""

    def process_step(self, obs_cpu: Tensor | None) -> None:
        """Called after forward (in _process_last_data):
        1. Dispatch observations to per-request handlers
        2. Run conditional_write callbacks
        3. Queue mask updates for next step
        """
```

### `intervention/request.py` — User-facing API

```python
@dataclass
class InterventionRequest:
    uid: int
    observations: List[ObserveOp]
    writes: List[WriteOp]
    conditional_writes: List[ConditionalWriteOp]

    def observe(self, layer: int, positions: slice | list[int] | None = None) -> Self: ...
    def ablate(self, layer: int) -> Self: ...                           # scale=0, add=0
    def steer(self, layer: int, vector: Tensor, alpha: float = 1.0) -> Self: ...  # scale=1, add=alpha*v
    def patch(self, layer: int, activation: Tensor) -> Self: ...        # scale=0, add=activation
    def conditional_write(self, read_layer: int, write_layer: int,
                         fn: Callable[[Tensor], Tuple[Tensor, Tensor]]) -> Self: ...
```

## Integration Points

### Modified existing files

| File | Change |
|------|--------|
| `core.py` | Add `req_map: torch.Tensor` field to `Batch` |
| `models/llama.py` | Add intervention ops to `LlamaModel.forward()` layer loop |
| `models/qwen2.py` | Same for `Qwen2Model.forward()` |
| `models/qwen3.py` | Same for `Qwen3Model.forward()` |
| `models/qwen3_moe.py` | Same for `Qwen3Model.forward()` |
| `engine/config.py` | Add `enable_intervention: bool = False` to `EngineConfig` |
| `engine/engine.py` | Allocate buffers in `__init__`, add obs to `ForwardOutput` |
| `engine/graph.py` | Add `req_map` to `GraphCaptureBuffer` |
| `scheduler/scheduler.py` | Set `batch.req_map` in `_prepare_batch()`, call manager in `_process_last_data()` |
| `server/args.py` | Add `--enable-intervention` CLI flag |

### Model forward modification (identical pattern for all models)

```python
# Before (e.g., llama.py LlamaModel.forward):
def forward(self, input_ids):
    x = self.embed_tokens.forward(input_ids)
    residual = None
    for layer in self.layers.op_list:
        x, residual = layer.forward(x, residual)
    return self.norm.forward(x, residual)[0]

# After (using hook-based wrapping — no model code changes needed):
# In engine init, before CUDA graph capture:
wrap_layers(model.layers.op_list, ictx)
# Each layer's forward() is replaced with a closure that calls:
#   x, residual = original_forward(x, residual)
#   observe(x, ...)           # observe MLP output
#   observe(residual, ...)    # observe residual stream
#   x, residual = blend(x, residual, ...)  # split blend
#   return x, residual
```

The wrapping happens before CUDA graph capture, so graphs permanently include intervention ops. Identity masks (`scale=1, add=0, obs_mask=0`) make them numerical no-ops. The `wrap_layers()` approach avoids modifying model files directly.

### New file structure

```
python/minisgl/intervention/
├── __init__.py          # Public exports: get/set_intervention_ctx, observe, blend
├── buffers.py           # ObservationBuffer, MaskBuffer
├── ops.py               # observe(), blend() — pure tensor functions
├── context.py           # InterventionContext singleton
├── manager.py           # InterventionManager (CPU-side async)
└── request.py           # InterventionRequest (user-facing API)

tests/intervention/
├── test_buffers.py      # ObservationBuffer (shapes, offsets, write/read, copy_to_cpu), MaskBuffer
├── test_ops.py          # observe (unified), blend (split), end-to-end observe+blend
├── test_req_map.py      # req_map plumbing: prefill packing, decode identity, graph buffer
└── test_e2e.py          # End-to-end with real model (spawn scheduler subprocess)

benchmark/intervention/
└── bench_overhead.py    # Vanilla vs identity-mask vs active intervention throughput
```

## Implementation Steps

### Step 1: Buffers + Ops (no integration yet) — DONE

**Files:** `intervention/buffers.py`, `intervention/ops.py`, `tests/intervention/test_buffers.py`, `tests/intervention/test_ops.py`

Built and tested in isolation with mock tensors (no model, no engine):
- `ObservationBuffer`: unified flat buffer for both prefill and decode, with `get_write_args()`, `copy_to_cpu()` (aliased return documented), `reset()`, `dtype` parameterization
- `MaskBuffer`: allocation, `reset()`, `set_ablate/steer/patch`, `dtype` parameterization
- `observe()`: single unified function using `index_copy_` — tested with various `obs_mask` patterns (single layer, multi-layer, multi-request, prefill-style, decode-style)
- `mask_blend()`: verified identity, ablation, steering, patching, multi-request isolation via `req_map`
- `obs_mask` is a standalone tensor, not stored inside buffer classes
- 37 tests, 100% coverage on intervention module

### Step 2: Context + `req_map` Plumbing

**Files:** `intervention/context.py`, `core.py`, `engine/graph.py`, `scheduler/scheduler.py`

- Add `InterventionContext` singleton with get/set
- Add `req_map` field to `Batch` dataclass
- Add `req_map` to `GraphCaptureBuffer` (init, `set_batch`, `copy_from`)
- Set `batch.req_map` in `Scheduler._prepare_batch()` using table_idx from `_make_input_tuple()`
- Test: verify `req_map` values are correct for prefill (packed multi-request) and decode (one per request)

### Step 3: Model Integration

**Files:** `models/llama.py`, `models/qwen2.py`, `models/qwen3.py`, `models/qwen3_moe.py`

- Modify `*Model.forward()` layer loops to call `observe()` + `mask_blend()` when intervention context is set
- Extract shared helper if possible to avoid duplicating the pattern
- Test: with `enable_intervention=False`, verify no behavior change (intervention context not set → vanilla path)

### Step 4: Engine Integration

**Files:** `engine/config.py`, `engine/engine.py`, `engine/engine.py` (ForwardOutput)

- Add `enable_intervention: bool = False` to `EngineConfig`
- In `Engine.__init__`, after model load and before graph capture:
  - Allocate `ObservationBuffer` and `MaskBuffer`
  - Create and set `InterventionContext`
- Add `obs_buf_cpu` to `ForwardOutput` (async-copied obs buffer from inactive ping-pong slot)
- Add `--enable-intervention` to `server/args.py`
- Test: engine init with intervention enabled, verify buffers allocated, graphs captured with intervention ops

### Step 5: Manager + Request API

**Files:** `intervention/manager.py`, `intervention/request.py`

- `InterventionRequest`: builder API for `observe()`, `ablate()`, `steer()`, `patch()`, `conditional_write()`
- `InterventionManager`:
  - `submit(uid, request)`: register, translate request into buffer updates
  - `remove(uid)`: cleanup masks/routes for finished request
  - `prepare_step()`: reset obs_buf, apply pending mask updates
  - `process_step(obs_cpu)`: dispatch observations to handlers, run conditional callbacks
- Test with mock forward passes (no real model)

### Step 6: Scheduler Integration

**Files:** `scheduler/scheduler.py`

- Create `InterventionManager` in `Scheduler.__init__` when intervention is enabled
- In `_prepare_batch()`: call `manager.prepare_step()` (reset obs, update routes)
- In `_process_last_data()`: after `copy_done.synchronize()`, call `manager.process_step(obs_cpu)`
- Wire intervention requests from incoming `UserMsg` (or separate intervention message type)
- Test: full scheduler loop with intervention

### Step 7: End-to-End Tests

**File:** `tests/intervention/test_e2e.py`

Following `tests/core/test_scheduler.py` pattern (spawn subprocess, send messages, check results):

1. **Identity test**: intervention enabled, default masks → output matches vanilla
2. **Ablation test**: scale=0, add=0 at layer N → output changes, verify deterministically
3. **Steering test**: scale=1, add=vector at layer N → output shifts
4. **Observation test**: observe layer N → obs_buf contains correct hidden states
5. **Multi-request isolation**: two requests, different interventions → each gets correct treatment
6. **Prefill observation**: observe during prefill → all tokens captured correctly
7. **Chunked prefill**: long prompt chunked → intervention applies correctly across chunks

### Step 8: Benchmark

**File:** `benchmark/intervention/bench_overhead.py`

Three configurations, same model (Qwen3-0.6B), same workload:

1. **Vanilla**: `--enable-intervention` off
2. **Identity overhead**: intervention on, all masks identity — measures pure op overhead
3. **Active intervention**: intervention on, observation at 4 layers + steering at 2 layers

Metrics: tokens/sec (prefill + decode), per-step latency, GPU utilization.

## Design Decisions Log

| Decision | Rationale |
|----------|-----------|
| `req_map` gather instead of per-token masks | Memory-efficient (`[layers, max_req, hidden]` vs `[layers, max_tokens, hidden]`), semantically cleaner (interventions are per-request) |
| Masks indexed by `table_idx` not batch position | Stable per-request slot, masks persist as batch composition changes |
| `_`-prefixed buffer attributes | `BaseOP.state_dict()` skips `_`-prefixed names — buffers hidden from model checkpoints |
| `if ictx is not None` branch at capture time | Graph topology fixed at capture. Vanilla mode (no intervention) avoids all overhead |
| Dual observe (x + residual) | Captures both sides of fused norm split; users reconstruct full hidden state on CPU |
| Split blend (scale both, add x only) | Algebraically equivalent to nnsight's `h' = h * scale + add` without breaking fused norm |
| Intervene after full layer (post-MLP, post-allreduce) | Clean hidden state, TP-merged activations, no fusion broken |
| Ops at every layer, data-driven enable | CUDA graphs require fixed topology. Per-request per-layer `obs_mask` controls which layers are active for which requests |
| Unified `ObservationBuffer` for prefill and decode | Same class, two instances with different sizing. Simpler than separate ring + flat buffer classes. Ring buffer optimization deferred (see `claude.todos.md`) |
| `obs_mask` as standalone tensor, not in buffer | Avoids dead state — buffer classes are pure storage, masking logic lives in `observe()` op |
| `dtype` parameter on all buffers | Match model precision to avoid type promotion in `mask_blend` and 2x memory waste |
| `copy_to_cpu()` returns aliased internal buffer | Intentional for async pipeline — avoids allocation per call. Documented that caller must process before next call |

## What This Proves

- Buffer+mask ops have measurable but bounded overhead when inserted into the forward pass
- CUDA graph capture works with intervention ops included (no hooks, no dynamic control flow)
- Multiple concurrent requests with different intervention targets run in the same batch without interference
- `req_map` gather unifies prefill and decode under one code path
- Async observation pipeline overlaps with GPU compute via overlap scheduling

## Future Work (Not in This Demo)

- Ring buffer optimization for prefill observations (see `claude.todos.md`) — reduce GPU memory from `num_layers * max_tokens * hidden_dim` to `ring_size * hidden_dim`
- KV cache dirty page tracking / prefix cache pollution from interventions
- `torch.compile` fusion of intervention ops with adjacent kernels
- Real nnsight API translation layer
- Multi-GPU tensor parallel buffer distribution (currently: intervene post-allreduce only)
- Custom fused observe+blend kernel to halve kernel launch overhead
- Position-specific observation (per-token mask within a request, not just per-request)
- Streaming observation API for real-time activation visualization
