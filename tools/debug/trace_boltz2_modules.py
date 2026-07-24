"""Export bounded Boltz2 activation traces for official/local diagnostics."""

from __future__ import annotations

import argparse
import gc
import json
import torch
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from safetensors.torch import load_file, save_file

from tests.structure.support import boltz2_bundle


_MSA_LAYER_PATHS = tuple(
    f"msa_module.layers.{layer_index}.{module_name}"
    for layer_index in range(8)
    for module_name in (
        "pair_weighted_averaging",
        "msa_transition",
        "outer_product_mean",
        "outer_product_mean.norm",
        "outer_product_mean.proj_a",
        "outer_product_mean.proj_b",
        "outer_product_mean.proj_o",
        "pairformer_layer",
    )
)

_MODULE_PATHS = (
    "input_embedder",
    "s_init",
    "z_init_1",
    "z_init_2",
    "rel_pos",
    "token_bonds",
    "contact_conditioning",
    "s_norm",
    "z_norm",
    "s_recycle",
    "z_recycle",
    "msa_module.msa_proj",
    "msa_module.s_proj",
    "msa_module",
    "pairformer_module",
    "distogram_module",
    "diffusion_conditioning",
    "structure_module.score_model",
    "confidence_module.s_inputs_norm",
    "confidence_module.s_norm",
    "confidence_module.z_norm",
    "confidence_module.rel_pos",
    "confidence_module.token_bonds",
    "confidence_module.contact_conditioning",
    "confidence_module.s_to_z",
    "confidence_module.s_to_z_transpose",
    "confidence_module.dist_bin_pairwise_embed",
    "confidence_module.pairformer_stack",
    "confidence_module.confidence_heads",
    *_MSA_LAYER_PATHS,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", choices=("reference", "candidate"), required=True)
    parser.add_argument("--exchange-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        help="Override prepared inputs with feature tensors from this bundle.",
    )
    return parser


def _tensor_leaves(value: Any, path: str = "value") -> dict[str, torch.Tensor]:
    if torch.is_tensor(value):
        return {path: value.detach().cpu().contiguous().clone()}  # value: (...)
    if isinstance(value, Mapping):
        leaves: dict[str, torch.Tensor] = {}
        for key in sorted(value, key=str):
            leaves.update(_tensor_leaves(value[key], f"{path}.{key}"))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_tensor_leaves(item, f"{path}.{index}"))
        return leaves
    return {}


def _register_trace_hooks(
    core: torch.nn.Module,
    traces: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    call_counts: defaultdict[str, int] = defaultdict(int)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for module_path in _MODULE_PATHS:
        try:
            module = core.get_submodule(module_path)
        except AttributeError:
            continue

        for parameter_name, parameter in module.named_parameters(recurse=False):
            # parameter: (...)
            key = f"{module_path}__parameter__{parameter_name}".replace(".", "__")
            if key not in traces:
                traces[key] = parameter.detach().cpu().contiguous().clone()  # (...)
        for buffer_name, buffer in module.named_buffers(recurse=False):
            # buffer: (...)
            key = f"{module_path}__buffer__{buffer_name}".replace(".", "__")
            if key not in traces:
                traces[key] = buffer.detach().cpu().contiguous().clone()  # (...)

        def hook(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
            *,
            path: str = module_path,
        ) -> None:
            call_index = call_counts[path]
            call_counts[path] += 1
            prefix = f"{path}__call_{call_index:03d}"
            values = {
                **_tensor_leaves(args, "args"),
                **_tensor_leaves(kwargs, "kwargs"),
                **_tensor_leaves(output, "output"),
            }  # values[path]: (...)
            for value_path, X in values.items():
                # X: (...)
                key = f"{prefix}__{value_path}".replace(".", "__")
                if key in traces:
                    raise RuntimeError(f"Duplicate trace tensor key: {key}")
                traces[key] = X

        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    return handles


def _load(
    producer: Literal["reference", "candidate"],
    request: Mapping[str, Any],
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    if producer == "reference":
        archive = boltz2_bundle._download_official_file(
            request,
            boltz2_bundle._molecule_archive,
        )
        checkpoint = boltz2_bundle._download_official_file(request, "boltz2_conf.ckpt")
        molecule_dir = boltz2_bundle._extract_molecules(archive, str(request["sequence"]))
        features = boltz2_bundle._prepare_reference_features(  # values: (...)
            request, molecule_dir
        )
        model = boltz2_bundle._load_reference_model(request, checkpoint)
    else:
        features = boltz2_bundle._prepare_candidate_features(request)  # values: (...)
        model = boltz2_bundle._load_candidate_model(request)
    return model, features


def main(argv: Sequence[str] | None = None) -> int:
    """Run one deterministic forward and export the selected module trace."""

    arguments = _parser().parse_args(argv)
    producer: Literal["reference", "candidate"] = arguments.producer
    request_path = (
        arguments.exchange_root
        / "structure"
        / "requests"
        / boltz2_bundle.reference_container
        / f"{boltz2_bundle.model_id}.json"
    )
    request = boltz2_bundle.load_request(request_path)
    model, features = _load(producer, request)  # features values: (...)
    if arguments.feature_bundle is not None:
        stored = load_file(arguments.feature_bundle, device="cpu")  # values: (...)
        features = {
            name.removeprefix("feature__"): X  # (...)
            for name, X in stored.items()
            if name.startswith("feature__")
        }  # values: (...)
        if set(features) != set(boltz2_bundle._feature_names):
            raise RuntimeError("Feature override does not match the Boltz2 contract.")
    core = model.core if hasattr(model, "core") else model
    traces: dict[str, torch.Tensor] = {}
    handles = _register_trace_hooks(core, traces)
    try:
        bundle = boltz2_bundle._run_model(model, features, request)  # values: (...)
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for name, X in bundle.items():
        # X: (...)
        if name.startswith(("noise__", "output__")):
            traces[f"bundle__{name}"] = X.detach().cpu().contiguous().clone()  # (...)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(sorted(traces.items())), arguments.output)
    metadata = {
        "producer": producer,
        "request_sha256": request["request_sha256"],
        "environment": boltz2_bundle._environment_metadata(),
        "tensor_count": len(traces),
        "tensor_sha256": boltz2_bundle.tensor_set_sha256(traces),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
