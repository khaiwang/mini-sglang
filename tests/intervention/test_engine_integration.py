"""Tests for Step 4: Engine integration of intervention system.

Tests config plumbing, ForwardOutput expansion, and CLI flag parsing.
Full Engine.__init__ tests require GPU + model weights — deferred to Step 7 E2E tests.
"""

from __future__ import annotations

import torch
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import ForwardOutput


def test_engine_config_has_intervention_field():
    """EngineConfig accepts enable_intervention and defaults to False."""
    from minisgl.distributed import DistributedInfo

    config = EngineConfig(
        model_path="dummy",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float16,
    )
    assert config.enable_intervention is False

    config2 = EngineConfig(
        model_path="dummy",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float16,
        enable_intervention=True,
    )
    assert config2.enable_intervention is True


def test_forward_output_has_obs_fields():
    """ForwardOutput has x_obs_cpu and res_obs_cpu fields, defaulting to None."""
    gpu = torch.tensor([1, 2, 3])
    cpu = torch.tensor([1, 2, 3])
    event = torch.cuda.Event()

    out = ForwardOutput(gpu, cpu, event)
    assert out.x_obs_cpu is None
    assert out.res_obs_cpu is None

    x_obs = torch.randn(4, 8)
    res_obs = torch.randn(4, 8)
    out2 = ForwardOutput(gpu, cpu, event, x_obs, res_obs)
    assert out2.x_obs_cpu is x_obs
    assert out2.res_obs_cpu is res_obs


def test_forward_output_backward_compat():
    """3-arg construction still works (old callers that don't pass obs fields)."""
    gpu = torch.tensor([1])
    cpu = torch.tensor([1])
    event = torch.cuda.Event()

    out = ForwardOutput(gpu, cpu, event)
    assert len(out) == 5
    assert out.next_tokens_gpu is gpu
    assert out.next_tokens_cpu is cpu
    assert out.copy_done_event is event
    assert out.x_obs_cpu is None
    assert out.res_obs_cpu is None


def test_forward_output_unpack_3():
    """Unpacking first 3 elements works for backward compat."""
    gpu = torch.tensor([1])
    cpu = torch.tensor([1])
    event = torch.cuda.Event()
    out = ForwardOutput(gpu, cpu, event)
    tokens_gpu, tokens_cpu, evt, *rest = out
    assert tokens_gpu is gpu
    assert tokens_cpu is cpu
    assert evt is event
    assert rest == [None, None]


def test_parse_args_enable_intervention():
    """CLI --enable-intervention flag sets the field correctly."""
    from minisgl.server.args import parse_args

    args, _ = parse_args(["--model", "Qwen/Qwen3-0.6B", "--dummy-weight"])
    assert args.enable_intervention is False

    args2, _ = parse_args(
        ["--model", "Qwen/Qwen3-0.6B", "--dummy-weight", "--enable-intervention"]
    )
    assert args2.enable_intervention is True
