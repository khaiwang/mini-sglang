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

### `intervention/request.py` — User-facing API

```python
@dataclass
class ObserveOp:
    layer: int

@dataclass
class WriteOp:
    layer: int
    kind: Literal["ablate", "steer", "patch"]  # validated in __post_init__
    vector: Tensor | None = None
    alpha: float = 1.0

@dataclass
class ConditionalWriteOp:
    read_layer: int
    write_layer: int
    fn: Callable[[Tensor, Tensor], Tensor]  # fn(x_obs, res_obs) -> activation
    once: bool = True  # True: fire once then auto-remove; False: per-token (SAE steering)

@dataclass
class InterventionRequest:
    observations: List[ObserveOp]
    writes: List[WriteOp]
    conditional_writes: List[ConditionalWriteOp]

    # Builder API (fluent chaining, returns self):
    def observe(self, layer: int) -> Self: ...
    def ablate(self, layer: int) -> Self: ...                           # scale=0, add=0
    def steer(self, layer: int, vector: Tensor, alpha: float = 1.0) -> Self: ...  # scale=1, add=alpha*v
    def patch(self, layer: int, activation: Tensor) -> Self: ...        # scale=0, add=activation
    def conditional_write(self, read_layer: int, write_layer: int,
                         fn: Callable[[Tensor, Tensor], Tensor],
                         once: bool = True) -> Self: ...
    # conditional_write auto-adds ObserveOp for read_layer if not present
    # once=True: fires once then auto-removes (activation patching, diagnostics)
    # once=False: fires every decode step (SAE steering, adaptive transforms)
    # Conditional writes are decode-only — skipped during prefill
```

No `uid` in `InterventionRequest` — the spec is independent; `manager.submit(uid, request)` binds them.

### `intervention/manager.py` — CPU-side Orchestration

Call sequence per step: `prepare_step(batch)` → GPU forward → `process_step(x_obs_cpu, res_obs_cpu, batch)`.

```python
class InterventionManager:
    def __init__(self, ctx: InterventionContext): ...

    # --- Lifecycle ---
    def submit(self, uid: int, request: InterventionRequest) -> None:
        """Register. Raises ValueError on duplicate uid."""

    def remove(self, uid: int) -> None:
        """Cleanup. No-op if uid not found."""

    # --- Per-step hooks ---
    def prepare_step(self, batch: Batch) -> None:
        """1. Reset obs buffers, obs_mask, mask_buffer to identity.
        2. Pre-compute token offsets (immune to complete_one mutations).
        3. For each req in batch with active intervention:
           - Set obs_mask for observed layers + conditional_write read_layers.
           - Apply write ops (ablate/steer/patch) to mask_buffer.
        4. Apply pending patches from previous conditional writes.
        5. Clear needs_rerun for uids whose patches are now applied."""

    def process_step(self, x_obs_cpu, res_obs_cpu, batch) -> None:
        """1. Use pre-computed token offsets from prepare_step.
        2. Extract layer slices from CPU buffers per observed layer.
        3. Run conditional_write callbacks (decode only): fn(x_obs, res_obs) -> activation.
        4. Auto-remove one-shot (once=True) ops after firing.
        5. Queue resulting patches as pending, mark uid in _needs_rerun."""

    # --- Query ---
    def needs_rerun(self, uid: int) -> bool:
        """True if this uid's token should NOT be emitted (stall for conditional write)."""

    def get_observations(self, uid: int) -> Dict[int, Tuple[Tensor, Tensor]]:
        """Returns {layer: (x_obs_cpu, res_obs_cpu)}. Empty dict if not found."""
```

**Internal state**: `_active: Dict[int, _ActiveEntry]` where `_ActiveEntry` holds:
- `request: InterventionRequest`
- `table_idx: int | None` (set each step from batch)
- `observations: Dict[int, Tuple[Tensor, Tensor]]` (layer → (x_obs, res_obs))
- `pending_patches: List[Tuple[int, Tensor]]` (write_layer, activation)

Plus: `_needs_rerun: Set[int]` tracking uids whose tokens should not be emitted, and `_token_offsets: Dict[int, Tuple[int, int]]` (uid → (start, length)) pre-computed in `prepare_step`.

**ChunkedReq handling**: Import `ChunkedReq` from `minisgl.scheduler.prefill`. Skip in `prepare_step` (offset computation and intervention application).

**Token offset pre-computation**: Offsets are computed in `prepare_step` (before `forward_batch` / `complete_one` can mutate `cached_len`/`device_len`), then reused in `process_step`. This prevents the bug where `complete_one()` makes `extend_len` stale for prefill reqs. Flat buffer layout is `[L * max_tokens_per_slot + token_offset, hidden_dim]`. Real reqs come first in `padded_reqs` (padding appended by `pad_batch`), so iterating `batch.reqs` with cumulative offsets is correct.

**Conditional writes are decode-only**: Conditional write callbacks only fire during decode. Prefill observations are still extracted (useful for `get_observations`), but the rerun mechanism is decode-only since prefill processes all tokens at once and can't stall individual positions. One-shot ops (`once=True`) are auto-removed after firing.

**Pending patch ordering**: In `prepare_step`, regular writes are applied first, then pending patches. Conditional patches override static writes at the same `(layer, table_idx)` if there's a conflict.

#### Conditional Write Rerun Mechanism

Each request with conditional writes follows a two-pass state machine:

```
Pass 1 (OBSERVING):
  prepare_step: set obs_mask for read_layer. mask_buffer at write_layer = identity.
  Forward: observe captures read_layer activation. write_layer blend = identity (no-op).
  process_step: fn(x_obs, res_obs) -> activation. Store as pending patch.
                Mark uid in _needs_rerun.
  Scheduler (Step 6): sees needs_rerun(uid) = True
    → reverts complete_one (device_len -= 1, cached_len -= 1)
    → skips append_host and DetokenizeMsg
    → request stays in decode batch

Pass 2 (PATCHED):
  prepare_step: apply pending patch at write_layer via set_patch().
                Clear uid from _needs_rerun.
  Forward: re-runs same position T. write_layer blend applies patch.
           KV cache at position T is re-computed (correct: layers before write_layer
           produce identical KV, layers at/after produce updated KV).
  process_step: no pending conditional writes → normal.
  Scheduler: needs_rerun(uid) = False → process normally (emit token).
```

**Why re-run is correct**: Since `read_layer < write_layer`, layers 0..write_layer-1 produce identical output in both passes (same input token, same blend=identity). The observation at `read_layer` is the same in both passes. The patch at `write_layer` only affects layers write_layer..N-1 in pass 2, giving the correct result as if the patch were applied mid-forward.

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
├── test_hooks.py        # wrap_layers/unwrap_layers, identity pass-through, observe+blend in wrapped forward
├── test_req_map.py      # req_map plumbing: prefill packing, decode identity, graph buffer
├── test_engine_integration.py  # EngineConfig field, ForwardOutput fields, CLI flag
├── test_request.py      # InterventionRequest builder API, dataclass fields (pure Python, no CUDA)
├── test_manager.py      # InterventionManager lifecycle, prepare/process, conditional write rerun
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

### Step 3: Hook Wrapping for Model Integration — DONE

**Files:** `intervention/hooks.py`, `tests/intervention/test_hooks.py`

- `wrap_layers()` replaces each layer's `forward()` with a closure that calls observe (x + residual) + blend
- `unwrap_layers()` restores original forward methods
- No model file modifications needed — wrapping happens at engine init before CUDA graph capture
- Tests verify wrapping/unwrapping, identity pass-through, and observation correctness

### Step 4: Engine Integration — DONE

**Files:** `engine/config.py`, `engine/engine.py`, `server/args.py`

- Added `enable_intervention: bool = False` to `EngineConfig`
- In `Engine.__init__`, after model load and before graph capture:
  - Allocate `ObservationBuffer` (x + residual) and `MaskBuffer`
  - Create `obs_mask` tensor, `InterventionContext` singleton
  - `wrap_layers()` on `model.model.layers.op_list`
- `ForwardOutput` expanded with `x_obs_cpu` and `res_obs_cpu` (async-copied CPU tensors, default `None`)
- `forward_batch()` calls `copy_to_cpu()` on both observation buffers when intervention is active
- `shutdown()` calls `unwrap_layers()` and `clear_intervention_ctx()`
- Added `--enable-intervention` CLI flag to `server/args.py`
- Tests: config field, ForwardOutput fields, backward compat, CLI flag parsing

### Step 5: Manager + Request API — DONE

**Files:** `intervention/request.py`, `intervention/manager.py`, `intervention/__init__.py`, `tests/intervention/test_request.py`, `tests/intervention/test_manager.py`

- `InterventionRequest`: builder API with fluent chaining for `observe()`, `ablate()`, `steer()`, `patch()`, `conditional_write()`. No `uid` — spec is independent, `manager.submit(uid, req)` binds them. `conditional_write` auto-adds `ObserveOp` for `read_layer`.
- `InterventionManager`:
  - `submit(uid, request)` / `remove(uid)`: lifecycle management
  - `prepare_step(batch)`: reset all buffers/masks to identity, set `obs_mask` and `mask_buffer` from active entries in batch, apply pending conditional write patches, clear `_needs_rerun`
  - `process_step(x_obs_cpu, res_obs_cpu, batch)`: extract per-req observations from flat CPU buffers using token offsets, run conditional write callbacks, queue resulting patches
  - `needs_rerun(uid)` / `get_observations(uid)`: query methods for scheduler integration
- ChunkedReq filtering in both prepare and process paths
- Two-pass conditional write rerun mechanism: observe pass → compute patch → apply pass
- 14 pure-Python tests (request API), 27 CUDA tests (manager lifecycle, resets, mask setting, observation extraction, conditional write full cycle)

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
