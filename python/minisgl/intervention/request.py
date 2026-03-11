"""Intervention request dataclasses and builder API.

No CUDA dependency — pure Python + torch types for serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


@dataclass
class ObserveOp:
    """Observe activations at a specific layer."""

    layer: int


@dataclass
class WriteOp:
    """Static write intervention at a specific layer.

    kind: "ablate" | "steer" | "patch"
    vector: required for "steer" and "patch", None for "ablate"
    alpha: scaling factor for "steer" (default 1.0)
    """

    layer: int
    kind: str  # "ablate" | "steer" | "patch"
    vector: Optional[torch.Tensor] = None
    alpha: float = 1.0


@dataclass
class ConditionalWriteOp:
    """Two-pass conditional write: observe at read_layer, then patch at write_layer.

    fn(x_obs, res_obs) -> activation tensor to patch at write_layer.
    """

    read_layer: int
    write_layer: int
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class InterventionRequest:
    """User-facing intervention specification.

    Builder methods return self for fluent chaining:
        req = InterventionRequest().observe(0).steer(1, vec).ablate(2)
    """

    observations: List[ObserveOp] = field(default_factory=list)
    writes: List[WriteOp] = field(default_factory=list)
    conditional_writes: List[ConditionalWriteOp] = field(default_factory=list)

    def observe(self, layer: int) -> InterventionRequest:
        self.observations.append(ObserveOp(layer=layer))
        return self

    def ablate(self, layer: int) -> InterventionRequest:
        self.writes.append(WriteOp(layer=layer, kind="ablate"))
        return self

    def steer(
        self, layer: int, vector: torch.Tensor, alpha: float = 1.0
    ) -> InterventionRequest:
        self.writes.append(WriteOp(layer=layer, kind="steer", vector=vector, alpha=alpha))
        return self

    def patch(self, layer: int, activation: torch.Tensor) -> InterventionRequest:
        self.writes.append(WriteOp(layer=layer, kind="patch", vector=activation))
        return self

    def conditional_write(
        self,
        read_layer: int,
        write_layer: int,
        fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> InterventionRequest:
        # Auto-add ObserveOp for read_layer if not already present
        observed_layers = {op.layer for op in self.observations}
        if read_layer not in observed_layers:
            self.observations.append(ObserveOp(layer=read_layer))
        self.conditional_writes.append(
            ConditionalWriteOp(read_layer=read_layer, write_layer=write_layer, fn=fn)
        )
        return self
