"""A ``torch.compile`` backend that runs the traced graph unchanged and records every graph it was asked to compile."""

from __future__ import annotations

import torch

from collections.abc import Callable


CompileBackend = Callable[[torch.fx.GraphModule, list[torch.Tensor]], Callable[..., object]]


def counting_compile_backend() -> tuple[CompileBackend, list[torch.fx.GraphModule]]:
    """Return a compile backend and the list that receives one graph module per compilation."""

    compiled_graphs: list[torch.fx.GraphModule] = []

    def counting_backend(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> Callable[..., object]:
        # example_inputs: (...) the traced graph's input tensors, deleted unused
        del example_inputs
        compiled_graphs.append(graph_module)
        return graph_module.forward

    return counting_backend, compiled_graphs
