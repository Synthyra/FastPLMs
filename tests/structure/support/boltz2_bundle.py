"""Produce isolated official and candidate Boltz2 structure bundles.

The reference producer uses the pinned Boltz public parser, tokenizer,
featurizer, checkpoint constructor, and forward method. The candidate producer
uses the installed FastPLMs package. They communicate only through a
manifest-derived JSON request and hash-verified safetensors bundles.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import shutil
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from tests.structure.support.state_contract import (
    exact_state_contract,
    semantic_config_contract,
    validate_exact_state_contract,
    validate_semantic_config_contract,
)

schema_version = 2
model_id = "boltz2"
reference_container = "reference-boltz2"
fold_sequence = "ACDEFGHIK"
feature_seed = 42
fold_seed = 17
fold_recycling_steps = 1
fold_sampling_steps = 10
fold_diffusion_samples = 1
fold_parameter_dtype = "float32"
fold_compute_dtype = "bfloat16"
fold_execution = "fp32_parameters_cuda_bf16_autocast"
fold_dtype = fold_compute_dtype
diffusion_noise_generator = "numpy-pcg64-standard-normal-float32"
conformer_policy = "first-pinned-official-conformer"
_steering_policy = {
    "fk_steering": False,
    "physical_guidance_update": False,
    "contact_guidance_update": False,
    "num_gd_steps": 16,
}

_molecule_archive = "mols.tar"
_feature_names = (
    "affinity_token_mask",
    "asym_id",
    "atom_backbone_feat",
    "atom_pad_mask",
    "atom_resolved_mask",
    "atom_to_token",
    "bfactor",
    "contact_conditioning",
    "contact_threshold",
    "coords",
    "cyclic_period",
    "deletion_mean",
    "deletion_value",
    "disto_center",
    "disto_coords_ensemble",
    "disto_target",
    "entity_id",
    "frame_resolved_mask",
    "frames_idx",
    "has_deletion",
    "method_feature",
    "modified",
    "mol_type",
    "msa",
    "msa_mask",
    "msa_paired",
    "plddt",
    "profile",
    "query_to_template",
    "r_set_to_rep_atom",
    "ref_atom_name_chars",
    "ref_charge",
    "ref_chirality",
    "ref_element",
    "ref_pos",
    "ref_space_uid",
    "res_type",
    "residue_index",
    "sym_id",
    "template_ca",
    "template_cb",
    "template_frame_rot",
    "template_frame_t",
    "template_mask",
    "template_mask_cb",
    "template_mask_frame",
    "template_restype",
    "token_bonds",
    "token_disto_mask",
    "token_index",
    "token_pad_mask",
    "token_resolved_mask",
    "token_to_center_atom",
    "token_to_rep_atom",
    "type_bonds",
    "visibility_ids",
)
_required_outputs = (
    "complex_plddt",
    "iptm",
    "pae",
    "pae_logits",
    "pde",
    "pde_logits",
    "pdistogram",
    "plddt",
    "plddt_logits",
    "ptm",
    "sample_atom_coords",
)
_exact_features = (
    "affinity_token_mask",
    "asym_id",
    "atom_backbone_feat",
    "atom_pad_mask",
    "atom_resolved_mask",
    "atom_to_token",
    "contact_conditioning",
    "contact_threshold",
    "cyclic_period",
    "deletion_mean",
    "deletion_value",
    "entity_id",
    "frame_resolved_mask",
    "frames_idx",
    "has_deletion",
    "method_feature",
    "modified",
    "mol_type",
    "msa",
    "msa_mask",
    "msa_paired",
    "profile",
    "query_to_template",
    "r_set_to_rep_atom",
    "ref_atom_name_chars",
    "ref_charge",
    "ref_chirality",
    "ref_element",
    "ref_space_uid",
    "res_type",
    "residue_index",
    "sym_id",
    "template_mask",
    "template_mask_cb",
    "template_mask_frame",
    "template_restype",
    "token_bonds",
    "token_disto_mask",
    "token_index",
    "token_pad_mask",
    "token_resolved_mask",
    "token_to_center_atom",
    "token_to_rep_atom",
    "type_bonds",
    "visibility_ids",
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _request_fingerprint(request: Mapping[str, Any]) -> str:
    payload = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return the exact byte digest of one tensor."""

    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def tensor_set_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and values in stable order."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    return {
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "files": [
            {"path": item.path, "algorithm": item.algorithm, "digest": item.digest}
            for item in checkpoint.files
        ],
    }


def _upstream_metadata(upstream: Any) -> dict[str, Any]:
    return {
        "id": upstream.id,
        "path": upstream.path,
        "url": upstream.url,
        "revision": upstream.revision,
        "license_expression": upstream.license_expression,
    }


def prepare_request(exchange_root: Path) -> Path:
    """Write one manifest-derived request for the isolated Boltz service."""

    from fastplms.registry import get_model_registry

    registry = get_model_registry()
    spec = registry[model_id]
    if spec.family.reference_container != reference_container:
        raise RuntimeError("Boltz2 reference container disagrees with models.toml.")
    if spec.family.upstreams != ("boltz",):
        raise RuntimeError("Boltz2 must declare exactly the pinned Boltz upstream.")
    if spec.family.bf16_execution != "fp32_parameters_autocast":
        raise RuntimeError("Boltz2 must retain FP32 parameters under CUDA BF16 autocast.")
    if _molecule_archive not in spec.official.file_map:
        raise RuntimeError("Boltz2 official provenance omits its molecule archive.")
    request = {
        "schema_version": schema_version,
        "model_id": model_id,
        "architecture": spec.family.architecture,
        "reference_container": spec.family.reference_container,
        "official": _checkpoint_metadata(spec.official),
        "candidate": _checkpoint_metadata(spec.fast),
        "candidate_auto_model": spec.auto_map["AutoModel"],
        "upstream": _upstream_metadata(registry.upstreams["boltz"]),
        "state_transform": spec.family.state_transform,
        "sequence": fold_sequence,
        "feature_seed": feature_seed,
        "seed": fold_seed,
        "recycling_steps": fold_recycling_steps,
        "sampling_steps": fold_sampling_steps,
        "diffusion_samples": fold_diffusion_samples,
        "diffusion_noise_generator": diffusion_noise_generator,
        "conformer_policy": conformer_policy,
        "steering": dict(_steering_policy),
        "dtype": fold_dtype,
        "parameter_dtype": fold_parameter_dtype,
        "compute_dtype": fold_compute_dtype,
        "execution": fold_execution,
        "feature_names": list(_feature_names),
        "output_names": list(_required_outputs),
    }
    request["request_sha256"] = _request_fingerprint(request)
    path = exchange_root / "structure" / "requests" / reference_container / f"{model_id}.json"
    _atomic_write_text(path, _canonical_json(request))
    return path


def _validate_checkpoint(source: object, *, label: str) -> None:
    if not isinstance(source, Mapping):
        raise ValueError(f"Boltz2 request omits {label} checkpoint metadata.")
    revision = source.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError(f"Boltz2 {label} revision is not immutable.")
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Boltz2 {label} checkpoint has no pinned files.")
    for item in files:
        if not isinstance(item, Mapping) or item.get("algorithm") not in {
            "git-sha1",
            "sha256",
        }:
            raise ValueError(f"Boltz2 {label} contains an invalid file identity.")


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema_version") != schema_version:
        raise ValueError("Unsupported Boltz2 structure-bundle schema.")
    if request.get("model_id") != model_id:
        raise ValueError(f"Unsupported Boltz2 model ID: {request.get('model_id')!r}")
    if request.get("reference_container") != reference_container:
        raise ValueError("Boltz2 request names the wrong reference container.")
    if request.get("dtype") != fold_dtype:
        raise ValueError("Boltz2 native structure parity requires BF16 mixed precision.")
    if request.get("parameter_dtype") != fold_parameter_dtype:
        raise ValueError("Boltz2 native structure parity requires FP32 parameters.")
    if request.get("compute_dtype") != fold_compute_dtype:
        raise ValueError("Boltz2 native structure parity requires BF16 compute.")
    if request.get("execution") != fold_execution:
        raise ValueError("Boltz2 native structure parity requires CUDA BF16 autocast.")
    if request.get("steering") != _steering_policy:
        raise ValueError("Boltz2 structure parity requires guidance-disabled sampling.")
    if request.get("diffusion_noise_generator") != diffusion_noise_generator:
        raise ValueError("Boltz2 structure parity requires portable diffusion noise.")
    if request.get("conformer_policy") != conformer_policy:
        raise ValueError("Boltz2 structure parity requires its pinned conformer policy.")
    if tuple(request.get("feature_names", ())) != _feature_names:
        raise ValueError("Boltz2 request feature schema mismatch.")
    if tuple(request.get("output_names", ())) != _required_outputs:
        raise ValueError("Boltz2 request output schema mismatch.")
    expected = dict(request)
    observed_fingerprint = expected.pop("request_sha256", None)
    if observed_fingerprint != _request_fingerprint(expected):
        raise ValueError("Boltz2 request fingerprint mismatch.")
    _validate_checkpoint(request.get("official"), label="official")
    _validate_checkpoint(request.get("candidate"), label="candidate")
    upstream = request.get("upstream")
    if not isinstance(upstream, Mapping) or upstream.get("id") != "boltz":
        raise ValueError("Boltz2 request omits its pinned upstream.")
    if len(str(upstream.get("revision", ""))) != 40:
        raise ValueError("Boltz2 upstream revision is not immutable.")
    official_files = {
        item.get("path"): item for item in request["official"]["files"] if isinstance(item, Mapping)
    }
    archive = official_files.get(_molecule_archive)
    if not isinstance(archive, Mapping) or archive.get("algorithm") != "sha256":
        raise ValueError("Boltz2 molecule archive is not SHA-256 pinned.")


def load_request(path: Path) -> dict[str, Any]:
    """Load and validate one manifest-derived Boltz2 request."""

    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError(f"Boltz2 request must be a JSON object: {path}")
    _validate_request(request)
    return request


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_official_file(request: Mapping[str, Any], filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    source = request["official"]
    identities = {item["path"]: item for item in source["files"] if isinstance(item, Mapping)}
    if filename not in identities:
        raise ValueError(f"Boltz2 request does not pin {filename!r}.")
    path = Path(
        hf_hub_download(
            repo_id=source["repo_id"],
            filename=filename,
            revision=source["revision"],
        )
    )
    identity = identities[filename]
    if identity["algorithm"] == "sha256" and _file_sha256(path) != identity["digest"]:
        raise RuntimeError(f"Boltz2 official asset hash mismatch: {filename}")
    return path


def _required_residue_names(sequence: str) -> tuple[str, ...]:
    from boltz.data import const

    names = set()
    for residue in sequence:
        token = const.prot_letter_to_token[residue]
        names.add(token if isinstance(token, str) else const.tokens[token])
    return tuple(sorted(names))


def _extract_molecules(archive_path: Path, sequence: str) -> Path:
    wanted_names = _required_residue_names(sequence)
    archive_hash = _file_sha256(archive_path)
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    output = cache_root / "fastplms-boltz2" / archive_hash / "mols"
    if all((output / f"{name}.pkl").is_file() for name in wanted_names):
        return output

    output.mkdir(parents=True, exist_ok=True)
    wanted = {f"mols/{name}.pkl": name for name in wanted_names}
    found: set[str] = set()
    with tarfile.open(archive_path, mode="r") as archive:
        for member in archive:
            name = wanted.get(member.name)
            if name is None:
                continue
            if not member.isfile() or member.size <= 0:
                raise RuntimeError(f"Invalid Boltz2 molecule member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read Boltz2 molecule member: {member.name}")
            target = output / f"{name}.pkl"
            handle, temporary_name = tempfile.mkstemp(
                dir=output,
                prefix=f".{name}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    shutil.copyfileobj(source, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, target)
            except BaseException:
                Path(temporary_name).unlink(missing_ok=True)
                raise
            found.add(name)
            if len(found) == len(wanted_names):
                break
    missing = sorted(set(wanted_names).difference(found))
    if missing:
        raise RuntimeError(f"Boltz2 molecule archive omits residues: {missing}")
    return output


def _normalize_features(features: Mapping[str, object]) -> dict[str, torch.Tensor]:
    missing = sorted(set(_feature_names).difference(features))
    if missing:
        raise RuntimeError(f"Boltz2 featurizer omitted required inputs: {missing}")
    tensors: dict[str, torch.Tensor] = {}
    for name in _feature_names:
        value = features[name]
        if not torch.is_tensor(value):
            raise TypeError(f"Boltz2 feature {name!r} is not a tensor.")
        tensors[f"feature__{name}"] = value.detach().cpu().contiguous().clone()
    return tensors


def _prepare_reference_features(
    request: Mapping[str, Any],
    molecule_dir: Path,
) -> dict[str, torch.Tensor]:
    import numpy as np
    from boltz.data import const
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.module.inferencev2 import collate
    from boltz.data.mol import load_molecules
    from boltz.data.parse.fasta import parse_fasta
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Input

    residue_names = _required_residue_names(str(request["sequence"]))
    molecules = load_molecules(molecule_dir, list(residue_names))
    for molecule in molecules.values():
        conformer_ids = sorted(conformer.GetId() for conformer in molecule.GetConformers())
        if not conformer_ids:
            raise RuntimeError("Boltz2 molecule has no pinned conformer.")
        for conformer_id in conformer_ids[1:]:
            molecule.RemoveConformer(conformer_id)
    with tempfile.TemporaryDirectory(prefix="boltz2-reference-input-") as directory:
        fasta = Path(directory) / "boltz2.fasta"
        fasta.write_text(
            f">A|protein|empty\n{request['sequence']}\n",
            encoding="utf-8",
            newline="\n",
        )
        target = parse_fasta(fasta, molecules, molecule_dir, boltz2=True)
    input_data = Input(
        structure=target.structure,
        msa={},
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )
    tokenized = Boltz2Tokenizer().tokenize(input_data)
    all_molecules = dict(molecules)
    all_molecules.update(target.extra_mols or {})
    torch.manual_seed(int(request["feature_seed"]))
    features = Boltz2Featurizer().process(
        tokenized,
        molecules=all_molecules,
        random=np.random.default_rng(int(request["feature_seed"])),
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=const.max_msa_seqs,
        pad_to_max_seqs=False,
        single_sequence_prop=0.0,
        compute_frames=True,
        inference_pocket_constraints=None,
        inference_contact_constraints=None,
        compute_constraint_features=True,
        override_method=None,
        compute_affinity=False,
    )
    features = collate([features])
    normalized = _normalize_features(features)
    return {name.removeprefix("feature__"): tensor for name, tensor in normalized.items()}


def _prepare_candidate_features(request: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    from fastplms.models.boltz.minimal_featurizer import build_boltz2_features

    torch.manual_seed(int(request["feature_seed"]))
    features, template = build_boltz2_features(str(request["sequence"]))
    if template.sequence != request["sequence"]:
        raise RuntimeError("Boltz2 candidate normalized the sequence unexpectedly.")
    normalized = _normalize_features(features)
    return {name.removeprefix("feature__"): tensor for name, tensor in normalized.items()}


def _require_floating_parameter_dtype(
    model: torch.nn.Module,
    expected: torch.dtype,
) -> None:
    floating_parameters = {
        name: parameter.dtype
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point()
    }
    if not floating_parameters:
        raise RuntimeError("Boltz2 model has no floating parameters to validate.")
    mismatches = {name: dtype for name, dtype in floating_parameters.items() if dtype != expected}
    if mismatches:
        sample = ", ".join(
            f"{name}={dtype}" for name, dtype in list(sorted(mismatches.items()))[:8]
        )
        raise RuntimeError(
            f"Boltz2 requires {expected} parameter storage before BF16 autocast; found {sample}."
        )


def _load_reference_model(
    request: Mapping[str, Any],
    checkpoint_path: Path,
) -> torch.nn.Module:
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2

    predict_args = {
        "recycling_steps": int(request["recycling_steps"]),
        "sampling_steps": int(request["sampling_steps"]),
        "diffusion_samples": int(request["diffusion_samples"]),
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": True,
    }
    msa_args = MSAModuleArgs(
        subsample_msa=True,
        num_subsampled_msa=1024,
        use_paired_feature=True,
    )
    steering_args = asdict(BoltzSteeringParams())
    steering_args.update(request["steering"])
    model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(msa_args),
        steering_args=steering_args,
    )
    model = model.eval().to(device="cuda", dtype=torch.float32)
    _require_floating_parameter_dtype(model, torch.float32)
    return model


def _load_candidate_model(request: Mapping[str, Any]) -> torch.nn.Module:
    from fastplms.registry import get_model_registry

    spec = get_model_registry()[model_id]
    source = request["candidate"]
    if source["repo_id"] != spec.fast.repo_id or source["revision"] != spec.fast.revision:
        raise RuntimeError("Boltz2 candidate request disagrees with models.toml.")
    auto_model = spec.auto_map["AutoModel"]
    if request["candidate_auto_model"] != auto_model:
        raise RuntimeError("Boltz2 candidate AutoModel request disagrees with models.toml.")
    module_name, class_name = auto_model.rsplit(".", maxsplit=1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class.from_pretrained(
        source["repo_id"],
        revision=source["revision"],
        dtype=torch.float32,
    )
    if getattr(model.core, "use_kernels", False):
        raise RuntimeError("Boltz2 compliance must use the declared eager implementation.")
    for name, value in request["steering"].items():
        if model.core.steering_args.get(name) != value:
            raise RuntimeError(f"Boltz2 candidate steering policy mismatch: {name}")
    model = model.eval().to(device="cuda", dtype=torch.float32)
    _require_floating_parameter_dtype(model, torch.float32)
    return model


@contextmanager
def _stable_cuda_numerics():
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_deterministic


@contextmanager
def _portable_random_draws(seed: int):
    """Replace every normal draw with a portable, recorded tensor stream.

    NumPy's PCG64 generator produces float32 values on the CPU. Each N tensor
    is then copied to the device and dtype requested by the official or local
    Boltz2 sampler. Neither wrapper calls PyTorch's random-number generator, so
    the injected values are independent of PyTorch, CUDA, and global RNG state.
    """
    import numpy as np

    captured: list[torch.Tensor] = []
    original_randn = torch.randn
    original_randn_like = torch.randn_like
    portable_rng = np.random.default_rng(seed)

    def portable_draw(template: torch.Tensor) -> torch.Tensor:
        # N is the portable normal tensor with shape matching the sampler draw.
        N = portable_rng.standard_normal(tuple(template.shape), dtype=np.float32)
        result = torch.from_numpy(N).to(device=template.device, dtype=template.dtype)
        result.requires_grad_(template.requires_grad)
        captured.append(result.detach().cpu().contiguous().clone())
        return result

    def recording_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = kwargs.pop("out", None)
        kwargs.pop("generator", None)
        template = torch.empty(*args, **kwargs)
        result = portable_draw(template)
        if out is not None:
            out.copy_(result)
            captured[-1] = out.detach().cpu().contiguous().clone()
            return out
        return result

    def recording_randn_like(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("generator", None)
        template = torch.empty_like(*args, **kwargs)
        return portable_draw(template)

    torch.randn = recording_randn
    torch.randn_like = recording_randn_like
    try:
        yield captured
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like


def _run_model(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
    request: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    device_features = {
        name: tensor.to(device="cuda", non_blocking=False) for name, tensor in features.items()
    }

    torch.manual_seed(int(request["seed"]))
    torch.cuda.manual_seed_all(int(request["seed"]))
    with (
        _portable_random_draws(int(request["seed"])) as captured_noise,
        torch.inference_mode(),
        _stable_cuda_numerics(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        output = model(
            feats=device_features,
            recycling_steps=int(request["recycling_steps"]),
            num_sampling_steps=int(request["sampling_steps"]),
            diffusion_samples=int(request["diffusion_samples"]),
            max_parallel_samples=None,
            run_confidence_sequentially=True,
        )
    if not captured_noise:
        raise RuntimeError("Boltz2 sampling did not request coordinate noise.")
    if not isinstance(output, Mapping):
        raise TypeError("Boltz2 forward did not return a tensor mapping.")
    missing = sorted(set(_required_outputs).difference(output))
    if missing:
        raise RuntimeError(f"Boltz2 forward omitted required outputs: {missing}")
    tensors = {f"feature__{name}": tensor for name, tensor in features.items()}
    tensors["noise__initial_standard_normal"] = captured_noise[0]
    for index, tensor in enumerate(captured_noise):
        tensors[f"noise__draw_{index:03d}"] = tensor
    for name in _required_outputs:
        value = output[name]
        if not torch.is_tensor(value):
            raise TypeError(f"Boltz2 output {name!r} is not a tensor.")
        tensors[f"output__{name}"] = value.detach().cpu().contiguous().clone()
    return tensors


def _environment_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("boltz", "transformers", "pytorch_lightning", "rdkit"):
        try:
            module = importlib.import_module(package)
        except ImportError:
            versions[package] = None
        else:
            versions[package] = str(getattr(module, "__version__", "unknown"))
    cuda_properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": cuda_properties.name if cuda_properties is not None else None,
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_properties is not None else None
        ),
        "cuda_total_memory": (
            int(cuda_properties.total_memory) if cuda_properties is not None else None
        ),
        "packages": versions,
    }


def _metadata(
    request: Mapping[str, Any],
    *,
    producer: Literal["reference", "candidate"],
    model: torch.nn.Module,
) -> dict[str, Any]:
    if producer == "reference":
        raw_config = dict(model.hparams)
        state_schema = "official_raw"
    else:
        raw_config = model.config
        state_schema = "canonical"
    return {
        "schema_version": schema_version,
        "producer": producer,
        "model_id": model_id,
        "request_sha256": request["request_sha256"],
        "official": request["official"],
        "candidate": request["candidate"],
        "upstream": request["upstream"],
        "sequence": request["sequence"],
        "feature_seed": request["feature_seed"],
        "seed": request["seed"],
        "recycling_steps": request["recycling_steps"],
        "sampling_steps": request["sampling_steps"],
        "diffusion_samples": request["diffusion_samples"],
        "diffusion_noise_generator": request["diffusion_noise_generator"],
        "conformer_policy": request["conformer_policy"],
        "steering": request["steering"],
        "dtype": request["dtype"],
        "parameter_dtype": request["parameter_dtype"],
        "compute_dtype": request["compute_dtype"],
        "execution": request["execution"],
        "attention_backend": "eager",
        "state_transform": request["state_transform"],
        "state_schema": state_schema,
        "semantic_config": semantic_config_contract(raw_config),
        "state": exact_state_contract(model),
        "environment": _environment_metadata(),
    }


def canonicalize_reference_state_contract(
    contract: Mapping[str, Any],
    *,
    expected_keys: set[str],
) -> dict[str, Any]:
    """Apply the declared Boltz2 inference-core transform to compact metadata."""

    validate_exact_state_contract(contract)

    def map_name(source_name: str) -> str | None:
        if source_name.startswith("ema."):
            return None
        name = source_name.removeprefix("model.").removeprefix("module.")
        canonical = name if name.startswith("core.") else f"core.{name}"
        if canonical in expected_keys:
            return canonical
        bare = canonical.removeprefix("core.")
        if bare.startswith(("template_module.", "bfactor_module.")):
            return None
        raise ValueError(f"Undeclared Boltz2 checkpoint state key: {source_name!r}.")

    tensors: dict[str, Any] = {}
    for source_name, record in contract["tensors"].items():
        target = map_name(str(source_name))
        if target is None:
            continue
        if target in tensors:
            raise ValueError(f"Boltz2 state-contract key collision for {target!r}.")
        tensors[target] = record
    missing = sorted(expected_keys.difference(tensors))
    if missing:
        raise ValueError(f"Boltz2 reference state omits canonical keys: {missing[:20]}.")

    aliases: list[list[str]] = []
    for group in contract["aliases"]:
        mapped = sorted({target for name in group if (target := map_name(str(name))) is not None})
        if len(mapped) > 1:
            aliases.append(mapped)
    return {"aliases": sorted(aliases), "tensors": dict(sorted(tensors.items()))}


def normalize_inference_config_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Remove Boltz training and backend controls that cannot affect eager evaluation."""

    validate_semantic_config_contract(contract)
    fields = json.loads(json.dumps(contract["fields"]))
    core_kwargs = fields.get("core_kwargs")
    if not isinstance(core_kwargs, dict):
        raise ValueError("Boltz2 semantic configuration omits core_kwargs.")

    pairformer_args = core_kwargs.get("pairformer_args")
    if not isinstance(pairformer_args, dict):
        raise ValueError("Boltz2 semantic configuration omits pairformer_args.")
    for name in (
        "activation_checkpointing",
        "dropout",
        "offload_to_cpu",
        "use_trifast",
    ):
        pairformer_args.pop(name, None)
    pairformer_args.setdefault("post_layer_norm", False)

    msa_args = core_kwargs.get("msa_args")
    if not isinstance(msa_args, dict):
        raise ValueError("Boltz2 semantic configuration omits msa_args.")
    for name in (
        "activation_checkpointing",
        "msa_dropout",
        "num_subsampled_msa",
        "offload_to_cpu",
        "subsample_msa",
        "use_trifast",
        "z_dropout",
    ):
        msa_args.pop(name, None)
    msa_args.setdefault("miniformer_blocks", False)

    diffusion_args = core_kwargs.get("diffusion_process_args")
    if not isinstance(diffusion_args, dict):
        raise ValueError("Boltz2 semantic configuration omits diffusion_process_args.")
    # Neither key affects evaluation: the first is not consumed by the
    # inference module, and the second is sampled only while training.
    diffusion_args.pop("mse_rotational_alignment", None)
    diffusion_args.pop("step_scale_random", None)
    return semantic_config_contract(fields)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_bundle(
    output_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    """Atomically publish one normalized Boltz2 structure bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = {
        name: tensor.detach().cpu().contiguous().clone() for name, tensor in sorted(tensors.items())
    }
    features = {
        name.removeprefix("feature__"): tensor
        for name, tensor in normalized.items()
        if name.startswith("feature__")
    }
    noise_draws = {
        name: tensor for name, tensor in normalized.items() if name.startswith("noise__draw_")
    }
    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "feature_sha256": tensor_set_sha256(features),
            "diffusion_noise_sha256": tensor_set_sha256(noise_draws),
            "diffusion_noise_draw_count": len(noise_draws),
            "tensor_hashes": {name: tensor_sha256(tensor) for name, tensor in normalized.items()},
            "tensor_keys": sorted(normalized),
        }
    )
    handle, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=".bundle.",
        suffix=".safetensors.tmp",
    )
    os.close(handle)
    try:
        save_file(normalized, temporary_name)
        os.replace(temporary_name, output_dir / "bundle.safetensors")
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    _atomic_write_text(output_dir / "metadata.json", _canonical_json(complete_metadata))


def load_bundle(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load one Boltz2 bundle and verify every declared tensor hash."""

    tensor_path = path / "bundle.safetensors"
    metadata_path = path / "metadata.json"
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing Boltz2 structure bundle under {path}. Run native producers first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != schema_version:
        raise ValueError(f"Unsupported Boltz2 bundle schema under {path}.")
    tensors = load_file(tensor_path, device="cpu")
    if sorted(tensors) != metadata.get("tensor_keys"):
        raise ValueError(f"Tensor-key mismatch in Boltz2 bundle {path}.")
    observed_hashes = {name: tensor_sha256(tensor) for name, tensor in tensors.items()}
    if observed_hashes != metadata.get("tensor_hashes"):
        raise ValueError(f"Tensor hash mismatch in Boltz2 bundle {path}.")
    features = {
        name.removeprefix("feature__"): tensor
        for name, tensor in tensors.items()
        if name.startswith("feature__")
    }
    if tensor_set_sha256(features) != metadata.get("feature_sha256"):
        raise ValueError(f"Feature hash mismatch in Boltz2 bundle {path}.")
    noise_draws = {
        name: tensor for name, tensor in tensors.items() if name.startswith("noise__draw_")
    }
    if tensor_set_sha256(noise_draws) != metadata.get("diffusion_noise_sha256"):
        raise ValueError(f"Diffusion-noise hash mismatch in Boltz2 bundle {path}.")
    validate_exact_state_contract(metadata.get("state"))
    validate_semantic_config_contract(metadata.get("semantic_config"))
    return tensors, metadata


def produce_reference(request_path: Path, output_dir: Path) -> None:
    """Run the pinned upstream Boltz2 public feature and model APIs."""

    request = load_request(request_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Official Boltz2 structure bundles require CUDA.")
    archive = _download_official_file(request, _molecule_archive)
    checkpoint = _download_official_file(request, "boltz2_conf.ckpt")
    molecule_dir = _extract_molecules(archive, str(request["sequence"]))
    features = _prepare_reference_features(request, molecule_dir)
    model = _load_reference_model(request, checkpoint)
    try:
        metadata = _metadata(request, producer="reference", model=model)
        tensors = _run_model(model, features, request)
        write_bundle(
            output_dir,
            tensors,
            metadata,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def produce_candidate(request_path: Path, output_dir: Path) -> None:
    """Run the installed FastPLMs Boltz2 feature and model APIs."""

    request = load_request(request_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Candidate Boltz2 structure bundles require CUDA.")
    features = _prepare_candidate_features(request)
    model = _load_candidate_model(request)
    try:
        metadata = _metadata(request, producer="candidate", model=model)
        tensors = _run_model(model, features, request)
        write_bundle(
            output_dir,
            tensors,
            metadata,
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _default_request(exchange_root: Path) -> Path:
    return exchange_root / "structure" / "requests" / reference_container / f"{model_id}.json"


def _default_output(
    exchange_root: Path,
    producer: Literal["reference", "candidate"],
) -> Path:
    return exchange_root / "structure" / "results" / producer / model_id / fold_dtype


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--exchange-root", type=Path, required=True)
    for name in ("produce-reference", "produce-candidate"):
        producer = subparsers.add_parser(name)
        producer.add_argument("--exchange-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare a request or produce one isolated bundle."""

    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        output = prepare_request(arguments.exchange_root)
    else:
        request_path = _default_request(arguments.exchange_root)
        producer = "reference" if arguments.command == "produce-reference" else "candidate"
        output = _default_output(arguments.exchange_root, producer)
        if producer == "reference":
            produce_reference(request_path, output)
        else:
            produce_candidate(request_path, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_exact_features",
    "_feature_names",
    "_required_outputs",
    "conformer_policy",
    "diffusion_noise_generator",
    "feature_seed",
    "fold_compute_dtype",
    "fold_diffusion_samples",
    "fold_dtype",
    "fold_execution",
    "fold_parameter_dtype",
    "fold_recycling_steps",
    "fold_sampling_steps",
    "fold_seed",
    "fold_sequence",
    "load_bundle",
    "load_request",
    "main",
    "model_id",
    "normalize_inference_config_contract",
    "prepare_request",
    "produce_candidate",
    "produce_reference",
    "reference_container",
    "schema_version",
    "tensor_set_sha256",
    "tensor_sha256",
    "write_bundle",
]
