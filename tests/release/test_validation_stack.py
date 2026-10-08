"""Exact, low-cost validation-stack checks for CUDA release hardware."""

from __future__ import annotations

import sys
import pytest
import torch

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from packaging.requirements import Requirement
from tests.conftest import validation_pins
from tests.structure.support.hardware import device_fingerprint


ROOT = Path(__file__).resolve().parents[2]


def test_candidate_prefers_the_pytorch_cudnn_runtime() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    wheel_library = "/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
    environment = next(
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip().startswith("LD_LIBRARY_PATH=")
    )
    assert environment.split("=", maxsplit=1)[1].split(":", maxsplit=1)[0] == (wheel_library)


@pytest.mark.gpu
def test_fp8_validation_stack_uses_the_cuda13_transformer_engine_core() -> None:
    expected = "2.12.0"
    assert version("transformer-engine") == expected
    assert version("transformer-engine-cu13") == expected
    assert version("transformer-engine-torch") == expected

    with pytest.raises(PackageNotFoundError):
        version("transformer-engine-cu12")


def _declared(relative_path: str, distribution: str) -> Requirement:
    """The one declaration of a distribution in a file under requirements/."""
    path = ROOT / "requirements" / relative_path
    declarations = [
        Requirement(line.partition("#")[0].strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.partition("#")[0].strip()
    ]
    (requirement,) = [item for item in declarations if item.name == distribution]
    return requirement


@pytest.mark.gpu
def test_gpu_validation_stack_is_exactly_pinned() -> None:
    import transformers

    pins = validation_pins()
    assert sys.version_info[:2] == (3, 12)
    assert version("torch").split("+", maxsplit=1)[0] == pins["torch"]
    assert torch.__version__.split("+", maxsplit=1)[0] == pins["torch"]
    assert version("transformers") == pins["transformers"]
    assert transformers.__version__ == pins["transformers"]
    # validation.txt pins Torch and Transformers alone. The image resolves the Hub client and
    # the kernel loader within their declared ranges.
    assert _declared("core.in", "huggingface-hub").specifier.contains(version("huggingface-hub"))
    assert _declared("features/flash.in", "kernels").specifier.contains(version("kernels"))
    assert torch.version.cuda is not None
    assert torch.version.cuda.startswith("13.0")


@pytest.mark.gpu
def test_release_cuda_gpu_is_available_without_running_a_model() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
    properties = torch.cuda.get_device_properties(0)
    device_fingerprint(
        {
            "cuda_device": properties.name,
            "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_total_memory": int(properties.total_memory),
        }
    )
    # probe: (1,)
    probe = torch.ones(1, device="cuda")
    assert probe.item() == 1.0
