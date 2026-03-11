# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini-SGLang is a lightweight (~5,000 lines of Python) LLM inference framework, a compact implementation of [SGLang](https://github.com/sgl-project/sglang). It supports Llama-3, Qwen-2.5, Qwen-3 (including MoE) model architectures. Linux-only (requires CUDA).

## Common Commands

### Installation
```bash
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Running the Server
```bash
python -m minisgl --model "Qwen/Qwen3-0.6B"                    # single GPU
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --port 30000 # multi-GPU
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell            # interactive shell
```

### Testing
```bash
pytest                              # run all tests (requires GPU)
pytest tests/kernel/test_tensor.py  # run a single test file
```

Tests are in `tests/` organized by subsystem: `core/`, `kernel/`, `misc/`, `intervention/`. Some tests (e.g., `test_scheduler.py`) spawn subprocesses and require a GPU with a real model. Tests use `call_if_main` decorator pattern — they can also be run directly as scripts.

### Environment Setup

Use conda (no `uv` on this machine):
```bash
conda create -n minisgl python=3.12 -y
conda run -n minisgl pip install -e ".[dev]"
conda run -n minisgl pytest tests/              # run tests
conda run -n minisgl ruff check .               # lint
```

Git remote uses SSH (`git@github.com:khaiwang/mini-sglang.git`).

### Linting / Formatting
```bash
ruff check .        # lint (line-length=100, py310)
ruff format .       # format
black .             # alternative formatter (line-length=100)
mypy python/        # type checking (strict mode)
```

## Architecture

### Multi-Process Design

The system runs as multiple processes communicating via **ZeroMQ** (control messages) and **NCCL** (GPU tensor data):

```
User → API Server (FastAPI/uvicorn) → Tokenizer → Scheduler(s) → Detokenizer → API Server → User
```

- **API Server** (`server/api_server.py`): FastAPI frontend with `/v1/chat/completions` endpoint
- **Tokenizer/Detokenizer** (`tokenizer/`): Separate worker processes for text↔token conversion
- **Scheduler** (`scheduler/scheduler.py`): One per GPU (TP rank). Rank 0 coordinates with others. Manages request lifecycle, batching, and cache allocation
- **Engine** (`engine/engine.py`): Per-GPU worker owned by Scheduler. Manages model, KV cache, CUDA graphs, and runs forward passes

Launch flow: `__main__.py` → `server/launch.py:launch_server()` spawns Scheduler processes (one per TP rank) + Tokenizer/Detokenizer processes, then starts the API server.

### Source Code Layout (`python/minisgl/`)

All source is under `python/minisgl/`. Key module relationships:

- **`core.py`**: Central dataclasses — `Req` (request state), `Batch` (forward batch), `Context` (global inference context with page table, KV cache, attention backend), `SamplingParams`. A global `Context` singleton is set via `set_global_ctx()`.
- **`layers/`**: TP-aware building blocks (`Linear`, `Embedding`, `RMSNorm`, `RotaryEmbedding`, `Attention`, `MoE`). All inherit from `BaseOP` (in `layers/base.py`), which provides custom `state_dict()`/`load_state_dict()` — **not** `nn.Module`.
- **`models/`**: Model implementations (Llama, Qwen2, Qwen3, Qwen3MoE). Registered in `register.py` via `_MODEL_REGISTRY` dict mapping HF architecture strings to classes. `ModelConfig` is parsed from HuggingFace `PretrainedConfig`.
- **`attention/`**: Backend interface (`BaseAttnBackend`) with implementations: FlashAttention (`fa`), FlashInfer (`fi`), TensorRT-LLM (`trtllm`). Supports hybrid prefill/decode backends (e.g., `fa,fi`). Uses `Registry` pattern.
- **`kvcache/`**: KV cache pool (`MHAKVCache`) and prefix cache managers: `NaivePrefixCache` and `RadixPrefixCache`. Uses `Registry` pattern.
- **`scheduler/`**: Scheduling logic split into `PrefillManager` (chunked prefill), `DecodeManager`, `CacheManager` (page allocation/eviction), `TableManager` (page table slots). Supports overlap scheduling (CPU scheduling overlapped with GPU compute).
- **`engine/`**: `Engine` initializes model, KV cache, attention backend, CUDA graphs. `GraphRunner` handles CUDA graph capture/replay. `Sampler` handles top-k/top-p/temperature sampling.
- **`message/`**: Typed messages for ZMQ IPC (`UserMsg`, `DetokenizeMsg`, `BatchBackendMsg`, etc.) with msgpack serialization.
- **`kernel/`**: Custom CUDA kernels via `tvm-ffi` for JIT compilation and Python binding.
- **`distributed/`**: Tensor parallelism primitives (all-reduce, all-gather) with optional PyNCCL backend.
- **`env.py`**: Environment variable configuration via `EnvClassSingleton`. All env vars are prefixed with `MINISGL_` (e.g., `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`).

### Config Hierarchy

`EngineConfig` (frozen dataclass) → `SchedulerConfig` (adds scheduling params, ZMQ addresses) → `ServerArgs` (adds host/port). All use frozen dataclasses with `@cached_property` for lazy-loaded derived values like `model_config`.

### Key Design Patterns

- **`BaseOP` instead of `nn.Module`**: All layers/models use a custom `BaseOP` base class with its own `state_dict()`/`load_state_dict()`. `OPList` replaces `nn.ModuleList`. `StateLessOP` for parameter-free ops.
- **`Registry` pattern**: Used for attention backends, cache managers, MoE backends — allows string-based selection with lazy imports.
- **Global context**: `core.py` maintains a global `Context` singleton accessed via `get_global_ctx()`. Layers read batch/KV cache state from this context rather than receiving it as arguments.
- **Overlap scheduling**: `Scheduler.overlap_loop()` overlaps GPU execution of the current batch with CPU processing of the previous batch's results, using separate CUDA streams.

## Workflow Rules

- **Plan before coding**: When starting work on an implementation step (e.g., "step 3"), always enter plan mode first. Propose a concrete plan with exact file changes, code snippets, and test strategy before writing any code.

## Intervention System (Buffer+Mask)

See **[`docs/intervention_roadmap.md`](docs/intervention_roadmap.md)** for the full design and implementation plan.

### nnsight Reference (`~/nnsight/`)

The nnsight library (`~/nnsight/`) is the reference for intervention API design. Key docs:
- `~/nnsight/CLAUDE.md` — comprehensive agent guide (common patterns, vLLM integration, gotchas)
- `~/nnsight/NNsight.md` — deep design doc (tracing, interleaving, Envoy, vLLM architecture)
- `~/nnsight/NNsight_Walkthrough.ipynb` — walkthrough notebook with examples

**nnsight intervention scenarios (constrained scope for mini-sglang):**

| nnsight Pattern | mini-sglang Equivalent | In Scope? |
|----------------|----------------------|-----------|
| `model.layers[L].output.save()` — observe activations | `observe_prefill` / `observe_decode` with `obs_mask[L]=1` | Yes |
| `model.layers[L].output[:] = 0` — ablation | `MaskBuffer.set_ablate(L, table_idx)` (scale=0, add=0) | Yes |
| `model.layers[L].output += vector` — steering | `MaskBuffer.set_steer(L, table_idx, vector, alpha)` | Yes |
| `model.layers[L].output = activation` — patching | `MaskBuffer.set_patch(L, table_idx, activation)` (scale=0, add=act) | Yes |
| Activation patching across prompts (clean→corrupt) | `conditional_write`: observe layer L from req A, patch into req B | Yes |
| Multi-request batching with different interventions | `req_map` + per-`(layer, table_idx)` masks — different requests target different layers | Yes |
| `.input` access (module inputs) | Out of scope — only post-layer residual stream | No |
| `.grad` / backward pass gradients | Out of scope — inference only | No |
| `model.edit()` — persistent model edits | Out of scope — per-request masks reset each step | No |
| Source tracing (sub-module internals) | Out of scope — only full-layer granularity | No |
| `tracer.stop()` — early stopping | Out of scope — all layers always execute | No |
| Module skipping (`layer.skip()`) | Out of scope — blend can zero output but layer still runs | No |
| Multi-token generation `.iter` steps | Implicit — decode loop runs per-token, masks persist across steps | N/A |

### Summary

Three fixed tensor ops inserted after each transformer layer via hook wrapping: **dual observe** (read both `x` and `residual`) and **split blend** (modify activations). This captures both sides of the fused-norm split to match nnsight's hidden-state semantics:

```python
# After each layer's forward() (injected by wrap_layers):
# 1. Observe x (MLP output) into flat buffer
observe(x, layer_idx, x_flat_buf, obs_mask, req_map, base_indices, offsets)
# 2. Observe residual into separate flat buffer
observe(residual, layer_idx, res_flat_buf, obs_mask, req_map, base_indices, offsets)
# 3. Split blend: scale on both, add on x only
x, residual = blend(x, residual, layer_idx, scale, add, req_map)
```

Key design decisions:
- **`req_map: [total_tokens] → table_idx`** unifies prefill (packed multi-request) and decode (one token per request). Already computed in `_make_input_tuple()`. Added to `Batch` and `GraphCaptureBuffer`.
- **Masks indexed by `(layer, table_idx)`** — per-request, per-layer granularity. `req_map` gather broadcasts to per-token. Different requests in the same batch can target different layers without branching.
- **CUDA graphs only apply to decode** (`graph.py:149` checks `batch.is_decode`). Prefill always runs eagerly. This allows different observation strategies: ring buffer (prefill) vs flat buffer (decode).
- **Dual observe (x + residual)**: captures both sides of the fused-norm split. Users reconstruct full hidden state `h = x + residual` on CPU if needed.
- **Split blend**: `scale` applies to both `x` and `residual`, `add` applies to `x` only. Algebraically equivalent to `h' = (x + residual) * scale + add` — matching nnsight's hidden-state-level intervention without breaking fused norm.
- **Flat buffer for observations**: pre-allocated buffer (`num_layers × max_tokens_per_slot × hidden_dim`). Uses `index_copy_` with pre-computed offsets for CUDA graph compatibility. `obs_mask` is always float32; `observe()` casts `per_token_mask` to `x.dtype` to keep computation in model dtype and avoid `index_copy_` dtype mismatches.
- **Data-driven, no control flow**: ops execute at every layer; identity masks (`scale=1, add=0, obs_mask=0`) make them no-ops. CUDA graph topology is fixed.
- **`_`-prefixed buffer attributes** hide from `BaseOP.state_dict()`.
- **Three-stage async pipeline**: GPU observe → async GPU→CPU copy (bulk) → CPU processing in `_process_last_data()`. `ObservationBuffer.copy_to_cpu()` uses **ping-pong CPU buffers** (two pinned tensors, alternating via XOR flip) so overlap scheduling never corrupts the previous step's observation data.

### Manager + Request API (CPU-side)

**`InterventionRequest`** (`intervention/request.py`): user-facing spec with fluent builder:
```python
req = InterventionRequest().observe(0).steer(1, vec).ablate(2)
req = InterventionRequest().conditional_write(read_layer=0, write_layer=2, fn=my_fn)
# conditional_write auto-adds ObserveOp for read_layer if not present
```
No `uid` in the request — `manager.submit(uid, request)` binds spec to request identity.

**`InterventionManager`** (`intervention/manager.py`): CPU-side orchestration per step:
- `prepare_step(batch)`: reset buffers/masks → set `obs_mask` + `mask_buffer` for active entries → apply pending patches from conditional writes → clear `_needs_rerun`
- `process_step(x_obs_cpu, res_obs_cpu, batch)`: extract per-req observations → run conditional write callbacks → queue pending patches
- `needs_rerun(uid)` / `get_observations(uid)`: query methods for scheduler

**Conditional write rerun mechanism** — two-pass state machine for `conditional_write(read_layer, write_layer, fn)`:
1. **Pass 1 (OBSERVING)**: `obs_mask` set for `read_layer`, `write_layer` blend = identity. After forward, `fn(x_obs, res_obs) -> activation`. Activation stored as pending patch. `needs_rerun(uid) = True` → scheduler stalls token emission (reverts `complete_one`, skips `append_host`/`DetokenizeMsg`, request stays in decode batch).
2. **Pass 2 (PATCHED)**: pending patch applied at `write_layer` via `set_patch()`. `needs_rerun(uid) = False`. Forward re-runs same position with patch active. Scheduler processes normally (emits token).

Why re-run is correct: since `read_layer < write_layer`, layers 0..write_layer-1 produce identical output in both passes. The patch at `write_layer` only affects layers write_layer..N-1 in pass 2.
