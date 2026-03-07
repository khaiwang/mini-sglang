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

Tests are in `tests/` organized by subsystem: `core/`, `kernel/`, `misc/`. Some tests (e.g., `test_scheduler.py`) spawn subprocesses and require a GPU with a real model. Tests use `call_if_main` decorator pattern — they can also be run directly as scripts.

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

## Intervention System (Buffer+Mask)

See **[`docs/intervention_roadmap.md`](docs/intervention_roadmap.md)** for the full design and implementation plan.

### Summary

Two fixed tensor ops inserted after each transformer layer: **observe** (read activations) and **blend** (modify activations). Separate observation strategies for prefill vs decode:

```python
# After each layer's forward():
# Prefill (eager): ring buffer with async D2H drain
observe_prefill(x, layer_idx, ring_buf, obs_mask, req_map, write_offset)
# Decode (CUDA graph): flat buffer with index_copy_
observe_decode(x, layer_idx, flat_buf, obs_mask, req_map, base_indices, offsets)
# Blend (same for both): per-request scale + add
x = x * scale[layer_idx, req_map] + add[layer_idx, req_map]
```

Key design decisions:
- **`req_map: [total_tokens] → table_idx`** unifies prefill (packed multi-request) and decode (one token per request). Already computed in `_make_input_tuple()`. Added to `Batch` and `GraphCaptureBuffer`.
- **Masks indexed by `(layer, table_idx)`** — per-request, per-layer granularity. `req_map` gather broadcasts to per-token. Different requests in the same batch can target different layers without branching.
- **CUDA graphs only apply to decode** (`graph.py:149` checks `batch.is_decode`). Prefill always runs eagerly. This allows different observation strategies: ring buffer (prefill) vs flat buffer (decode).
- **Ring buffer for prefill observations**: fixed-size GPU ring buffer with a dedicated CUDA copy stream for async D2H. Avoids pre-allocating `[num_layers × max_tokens × hidden_dim]` (9+ GB for 8B models). Ring space is reused as async copies complete.
- **Flat buffer for decode observations**: small pre-allocated buffer (`num_layers × max_decode_bs × hidden_dim`, ~144MB for 8B models). Uses `index_copy_` with pre-computed offsets for CUDA graph compatibility.
- **Data-driven, no control flow**: ops execute at every layer; identity masks (`scale=1, add=0, obs_mask=0`) make them no-ops. CUDA graph topology is fixed.
- **`_`-prefixed buffer attributes** hide from `BaseOP.state_dict()`.
- **Three-stage async pipeline**: GPU observe → async GPU→CPU copy (per-layer for prefill via ring, bulk for decode) → CPU processing in `_process_last_data()`.
