import base64
import hashlib
import io
import json
import os
import runpy
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
import torch
from transformers import AutoModel

import fastplms.models.esm3.modeling_esm3 as esm3_module
from fastplms.models.esm3.modeling_esm3 import (
    _MAX_SAVED_RUNTIME_FILE_BYTES,
    _SAVED_RUNTIME_FILES,
    SEQUENCE_BOS_TOKEN,
    SEQUENCE_EOS_TOKEN,
    SEQUENCE_MASK_TOKEN,
    FastESM3Config,
    FastESM3GenerationConfig,
    FastESM3Model,
    _build_saved_runtime_archive,
    _render_saved_runtime_bundle,
    _saved_runtime_tree_hash,
    _validate_saved_runtime_relative_path,
)
from tests.conftest import strict_fp32_matmul


def _small_config() -> FastESM3Config:
    return FastESM3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_vector_heads=8,
        num_hidden_layers=2,
    )


def _small_model() -> FastESM3Model:
    return FastESM3Model(_small_config()).eval()


def _write_synthetic_runtime(package_root: Path) -> None:
    for relative in _SAVED_RUNTIME_FILES:
        path = package_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"validated runtime source: {relative}\n".encode())


def _saved_bundle_namespace(model_path: Path) -> dict[str, object]:
    return runpy.run_path(str(model_path / "fastplms_bundle.py"))


def _rewrite_saved_runtime_archive(
    model_path: Path,
    *,
    first_name: str | None = None,
    first_mode: int | None = None,
    corrupt_first_hash: bool = False,
) -> None:
    namespace = _saved_bundle_namespace(model_path)
    old_archive_hash = namespace["RUNTIME_HASH"]
    old_tree_hash = namespace["RUNTIME_TREE_HASH"]
    manifest = json.loads(json.dumps(namespace["RUNTIME_MANIFEST"]))
    archive = base64.b85decode("".join(namespace["RUNTIME_DATA"]))

    if corrupt_first_hash:
        first_relative = _SAVED_RUNTIME_FILES[0]
        manifest["files"][first_relative]["sha256"] = "0" * 64
    buffer = io.BytesIO()
    with (
        ZipFile(io.BytesIO(archive)) as source,
        ZipFile(
            buffer,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=9,
        ) as destination,
    ):
        for index, member in enumerate(source.infolist()):
            name = first_name if index == 0 and first_name is not None else member.filename
            info = ZipInfo(name, date_time=member.date_time)
            info.create_system = member.create_system
            info.compress_type = member.compress_type
            info.external_attr = (
                first_mode << 16 if index == 0 and first_mode is not None else member.external_attr
            )
            destination.writestr(
                info,
                source.read(member),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )
    poisoned_archive = buffer.getvalue()
    new_tree_hash = _saved_runtime_tree_hash(manifest)
    new_archive_hash, bundle = _render_saved_runtime_bundle(
        poisoned_archive,
        manifest,
        new_tree_hash,
    )
    (model_path / "fastplms_bundle.py").write_bytes(bundle)
    bridge_path = model_path / "modeling_fastplms.py"
    bridge = bridge_path.read_text(encoding="utf-8")
    assert isinstance(old_archive_hash, str)
    assert isinstance(old_tree_hash, str)
    bridge = bridge.replace(old_archive_hash, new_archive_hash).replace(
        old_tree_hash,
        new_tree_hash,
    )
    bridge_path.write_text(bridge, encoding="utf-8", newline="\n")


def _run_isolated_bridge_probe(
    model_path: Path,
    tmp_path: Path,
    body: str = "",
) -> subprocess.CompletedProcess[str]:
    probe = tmp_path / "esm3_runtime_bridge_probe.py"
    source = textwrap.dedent(
        """\
            import importlib.abc
            import importlib.util
            import sys
            import types
            from pathlib import Path


            class BlockInstalledFastPLMs(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "fastplms":
                        raise ModuleNotFoundError("installed FastPLMs is blocked")
                    return None


            artifact = Path(sys.argv[1])
            sys.modules.pop("fastplms", None)
            for name in tuple(sys.modules):
                if name.startswith("fastplms."):
                    sys.modules.pop(name)
            sys.meta_path.insert(0, BlockInstalledFastPLMs())

            def load_bridge(package_name):
                package = types.ModuleType(package_name)
                package.__package__ = package_name
                package.__path__ = [str(artifact)]
                sys.modules[package_name] = package
                module_name = f"{package_name}.modeling_fastplms"
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    artifact / "modeling_fastplms.py",
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError("Unable to load generated ESM3 bridge")
                bridge = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = bridge
                spec.loader.exec_module(bridge)
                return bridge
            """
    )
    if body.strip():
        source += "\n" + textwrap.dedent(body).strip() + "\n"
    probe.write_text(
        source,
        encoding="utf-8",
        newline="\n",
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return subprocess.run(
        [sys.executable, "-I", str(probe), str(model_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_esm3_sequence_only_forward() -> None:
    model = _small_model()
    batch = model.tokenize_sequences(["MKTAYIAKQ", "GGGG"], device=model.device)

    with torch.inference_mode():
        output = model(**batch)

    assert output.logits is not None
    assert output.function_logits is not None
    assert output.residue_logits is not None
    assert output.logits.shape[:2] == batch["input_ids"].shape
    assert output.logits.shape[-1] == 64
    assert output.structure_logits.shape[-1] == 4096
    assert output.function_logits.shape[-2:] == (8, 260)
    assert output.residue_logits.shape[-1] == 1478
    assert not torch.isnan(output.logits).any()


def test_esm3_uses_hugging_face_initialization_and_only_retains_requested_states() -> None:
    model = _small_model()
    model.attn_backend = "eager"
    batch = model.tokenize_sequences(["MKTAYIAKQ"], device=model.device)

    embedding_std = model.esm3.encoder.sequence_embed.weight.detach().std().item()
    assert embedding_std == pytest.approx(model.config.initializer_range, rel=0.2)

    with torch.inference_mode():
        default_output = model(**batch)
        full_output = model(
            **batch,
            output_hidden_states=True,
            output_attentions=True,
        )
        tuple_output = model(
            **batch,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=False,
        )

    assert default_output.hidden_states is None
    assert full_output.hidden_states is not None
    assert full_output.attentions is not None
    assert len(full_output.hidden_states) == model.config.num_hidden_layers
    assert len(full_output.attentions) == model.config.num_hidden_layers
    assert tuple(full_output.keys()) == (
        "last_hidden_state",
        "hidden_states",
        "attentions",
        "logits",
        "sequence_logits",
        "structure_logits",
        "secondary_structure_logits",
        "sasa_logits",
        "function_logits",
        "residue_logits",
        "embeddings",
    )
    assert isinstance(tuple_output, tuple)
    assert torch.equal(tuple_output[0], full_output.last_hidden_state)
    assert isinstance(tuple_output[1], tuple)
    assert isinstance(tuple_output[2], tuple)
    for actual, expected in zip(tuple_output, full_output.to_tuple(), strict=True):
        if isinstance(expected, tuple):
            assert isinstance(actual, tuple)
            for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
                torch.testing.assert_close(actual_tensor, expected_tensor)
            continue
        torch.testing.assert_close(actual, expected)

    labels = batch["input_ids"].clone()
    with torch.inference_mode():
        labeled_output = model(
            **batch,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        labeled_tuple = model(
            **batch,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=False,
        )
    assert tuple(labeled_output.keys())[:3] == (
        "loss",
        "last_hidden_state",
        "hidden_states",
    )
    assert labeled_output.loss is not None
    assert torch.isfinite(labeled_output.loss)
    torch.testing.assert_close(labeled_tuple[0], labeled_output.loss)
    torch.testing.assert_close(labeled_tuple[1], labeled_output.last_hidden_state)
    assert len(labeled_tuple) == len(labeled_output.to_tuple())
    for actual, expected in zip(
        labeled_tuple,
        labeled_output.to_tuple(),
        strict=True,
    ):
        if isinstance(expected, tuple):
            assert isinstance(actual, tuple)
            for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
                torch.testing.assert_close(actual_tensor, expected_tensor)
            continue
        torch.testing.assert_close(actual, expected)


def test_esm3_config_drives_hidden_state_output() -> None:
    config = _small_config()
    config.output_hidden_states = True
    model = FastESM3Model(config).eval()
    batch = model.tokenize_sequences(["MKTAYIAKQ"], device=model.device)

    with torch.inference_mode():
        output = model(**batch)

    assert output.hidden_states is not None
    assert len(output.hidden_states) == config.num_hidden_layers


def test_esm3_resize_updates_sequence_input_and_output_embeddings() -> None:
    model = _small_model()
    original_vocab_size = model.config.vocab_size
    resized_vocab_size = original_vocab_size + 7

    model.resize_token_embeddings(resized_vocab_size)

    assert model.get_input_embeddings().num_embeddings == resized_vocab_size
    assert model.get_output_embeddings().out_features == resized_vocab_size
    assert model.config.vocab_size == resized_vocab_size
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[SEQUENCE_BOS_TOKEN, original_vocab_size, SEQUENCE_EOS_TOKEN]])
        )
    assert output.sequence_logits.shape[-1] == resized_vocab_size


def test_esm3_accepts_function_tokens_argument() -> None:
    model = _small_model()
    batch = model.tokenize_sequences(["MKTAYIAKQ"], device=model.device)
    function_tokens = batch["input_ids"].new_zeros((*batch["input_ids"].shape, 8))

    with torch.inference_mode():
        output = model(**batch, function_tokens=function_tokens)

    assert output.logits is not None
    assert output.logits.shape[:2] == batch["input_ids"].shape


def test_esm3_rejects_attention_mask_row_without_a_valid_key() -> None:
    model = _small_model()
    input_ids = torch.tensor(
        [
            [SEQUENCE_BOS_TOKEN, 4, SEQUENCE_EOS_TOKEN],
            [SEQUENCE_BOS_TOKEN, 5, SEQUENCE_EOS_TOKEN],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 1], [0, 0, 0]])

    with pytest.raises(
        ValueError,
        match="attention_mask must keep at least one valid key per batch row",
    ):
        model(input_ids=input_ids, attention_mask=attention_mask)


def test_esm3_saved_runtime_archive_is_fixed_bounded_and_deterministic(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "fastplms"
    _write_synthetic_runtime(package_root)
    (package_root / ".secrets.env").write_text("TOKEN=excluded\n", encoding="utf-8")
    injected = package_root / "models" / "esm3" / "untracked_injected.py"
    injected.write_text("raise RuntimeError('must not be bundled')\n", encoding="utf-8")
    external = tmp_path / "external.py"
    external.write_text("raise RuntimeError('must not be followed')\n", encoding="utf-8")
    (package_root / "unknown_symlink.py").symlink_to(external)

    first_archive, first_manifest, first_tree_hash = _build_saved_runtime_archive(package_root)
    second_archive, second_manifest, second_tree_hash = _build_saved_runtime_archive(package_root)

    assert first_archive == second_archive
    assert first_manifest == second_manifest
    assert first_tree_hash == second_tree_hash
    assert first_manifest["schema_version"] == 1
    assert set(first_manifest["files"]) == set(_SAVED_RUNTIME_FILES)
    assert first_manifest["total_size"] == sum(
        record["size"] for record in first_manifest["files"].values()
    )
    assert ".secrets.env" not in first_manifest["files"]
    assert "models/esm3/untracked_injected.py" not in first_manifest["files"]
    assert "unknown_symlink.py" not in first_manifest["files"]
    with ZipFile(io.BytesIO(first_archive)) as archive:
        assert set(archive.namelist()) == {
            f"fastplms/{relative}" for relative in _SAVED_RUNTIME_FILES
        }
        for relative, record in first_manifest["files"].items():
            payload = archive.read(f"fastplms/{relative}")
            assert len(payload) == record["size"]
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]


@pytest.mark.parametrize(
    "value",
    (
        "../escape.py",
        "/absolute.py",
        "C:/escape.py",
        "models\\escape.py",
        "models//escape.py",
    ),
)
def test_esm3_saved_runtime_rejects_noncanonical_allowlist_paths(value: str) -> None:
    with pytest.raises(RuntimeError, match="runtime path is unsafe"):
        _validate_saved_runtime_relative_path(value)


def test_esm3_saved_runtime_rejects_missing_allowlisted_file(tmp_path: Path) -> None:
    package_root = tmp_path / "fastplms"
    _write_synthetic_runtime(package_root)
    package_root.joinpath(*_SAVED_RUNTIME_FILES[-1].split("/")).unlink()

    with pytest.raises(RuntimeError, match="runtime file is missing"):
        _build_saved_runtime_archive(package_root)


def test_esm3_saved_runtime_rejects_allowlisted_symlink(tmp_path: Path) -> None:
    package_root = tmp_path / "fastplms"
    _write_synthetic_runtime(package_root)
    target = tmp_path / "outside.py"
    target.write_text("outside = True\n", encoding="utf-8")
    allowlisted = package_root / "__init__.py"
    allowlisted.unlink()
    allowlisted.symlink_to(target)

    with pytest.raises(RuntimeError, match="must not contain a symlink"):
        _build_saved_runtime_archive(package_root)


def test_esm3_saved_runtime_rejects_oversize_allowlisted_file(tmp_path: Path) -> None:
    package_root = tmp_path / "fastplms"
    _write_synthetic_runtime(package_root)
    (package_root / "__init__.py").write_bytes(b"x" * (_MAX_SAVED_RUNTIME_FILE_BYTES + 1))

    with pytest.raises(RuntimeError, match="exceeds its size limit"):
        _build_saved_runtime_archive(package_root)


def test_esm3_saved_runtime_rejects_oversize_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "fastplms"
    _write_synthetic_runtime(package_root)
    monkeypatch.setattr(esm3_module, "_MAX_SAVED_RUNTIME_TOTAL_BYTES", 1)

    with pytest.raises(RuntimeError, match="total expanded size limit"):
        _build_saved_runtime_archive(package_root)


def test_esm3_loads_with_automodel(tmp_path: Path) -> None:
    model = _small_model()
    model.save_pretrained(tmp_path)
    assert (tmp_path / "modeling_esm3.py").is_file()
    assert (tmp_path / "modeling_fastplms.py").is_file()
    assert (tmp_path / "fastplms_bundle.py").is_file()
    assert not (tmp_path / "fastplms").exists()
    assert not list(tmp_path.glob("_fastplms_runtime_*"))
    bundle = _saved_bundle_namespace(tmp_path)
    assert set(bundle["RUNTIME_MANIFEST"]["files"]) == set(_SAVED_RUNTIME_FILES)
    bridge = (tmp_path / "modeling_fastplms.py").read_text(encoding="utf-8")
    assert "extractall" not in bridge
    assert "rglob" not in bridge
    assert "TemporaryDirectory" in bridge
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["auto_map"] == {
        "AutoConfig": "modeling_fastplms.FastESM3Config",
        "AutoModel": "modeling_fastplms.FastESM3Model",
    }

    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    batch = loaded.tokenize_sequences(["MKTAYIAKQ"], device=loaded.device)

    with torch.inference_mode():
        output = loaded(**batch)

    assert output.logits is not None
    assert output.logits.shape[:2] == batch["input_ids"].shape


def test_esm3_repeated_save_removes_stale_runtime_outputs(tmp_path: Path) -> None:
    model = _small_model()
    model.save_pretrained(tmp_path)
    expected_bundle = (tmp_path / "fastplms_bundle.py").read_bytes()
    expected_bridge = (tmp_path / "modeling_fastplms.py").read_bytes()

    stale_tree = tmp_path / "fastplms" / "models" / "esm3"
    stale_tree.mkdir(parents=True)
    (stale_tree / "poison.py").write_text("raise RuntimeError('stale')\n", encoding="utf-8")
    stale_cache = tmp_path / "_fastplms_runtime_deadbeef" / "fastplms"
    stale_cache.mkdir(parents=True)
    (stale_cache / "__init__.py").write_text(
        "raise RuntimeError('stale cache')\n",
        encoding="utf-8",
    )
    (tmp_path / "fastplms_bundle.py").write_text("stale bundle\n", encoding="utf-8")
    (tmp_path / "modeling_fastplms.py").write_text("stale bridge\n", encoding="utf-8")

    model.save_pretrained(tmp_path)

    assert not (tmp_path / "fastplms").exists()
    assert not (tmp_path / "_fastplms_runtime_deadbeef").exists()
    assert (tmp_path / "fastplms_bundle.py").read_bytes() == expected_bundle
    assert (tmp_path / "modeling_fastplms.py").read_bytes() == expected_bridge


def test_esm3_saved_bridge_reuses_same_runtime_in_process(tmp_path: Path) -> None:
    model_path = tmp_path / "saved"
    _small_model().save_pretrained(model_path)
    result = _run_isolated_bridge_probe(
        model_path,
        tmp_path,
        """
        first = load_bridge("artifact_first")
        runtime = sys.modules["fastplms"]
        runtime_root = Path(runtime.__file__).absolute().parent.parent
        second = load_bridge("artifact_second")
        assert sys.modules["fastplms"] is runtime
        assert len(first._RUNTIME_TEMPORARIES) == 1
        assert second._RUNTIME_TEMPORARIES == []
        assert runtime.__fastplms_saved_runtime_tree_hash__ == first.RUNTIME_TREE_HASH
        first._cleanup_runtime_temporaries()
        assert not runtime_root.exists()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_esm3_saved_bridge_rejects_preimported_runtime_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "saved"
    _small_model().save_pretrained(model_path)
    result = _run_isolated_bridge_probe(
        model_path,
        tmp_path,
        """
        fake_root = artifact / "mismatched_fastplms"
        fake_root.mkdir()
        fake_init = fake_root / "__init__.py"
        fake_init.write_text("__version__ = 'different'\\n", encoding="utf-8")
        fake = types.ModuleType("fastplms")
        fake.__file__ = str(fake_init)
        fake.__path__ = [str(fake_root)]
        sys.modules["fastplms"] = fake
        load_bridge("artifact_mismatch")
        """,
    )

    assert result.returncode != 0
    assert "Loaded FastPLMs version/runtime mismatch" in result.stderr


@pytest.mark.parametrize(
    ("poison", "message"),
    (
        ({"first_name": "fastplms/../escape.py"}, "unsafe path"),
        ({"first_name": "fastplms/unknown.py"}, "inventory is unexpected"),
        ({"first_mode": stat.S_IFLNK | 0o777}, "member is not canonical"),
        ({"corrupt_first_hash": True}, "member hash mismatch"),
    ),
)
def test_esm3_saved_bridge_rejects_poisoned_archive(
    tmp_path: Path,
    poison: dict[str, object],
    message: str,
) -> None:
    model_path = tmp_path / "saved"
    _small_model().save_pretrained(model_path)
    _rewrite_saved_runtime_archive(model_path, **poison)

    result = _run_isolated_bridge_probe(
        model_path,
        tmp_path,
        'load_bridge("artifact_poisoned")',
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_esm3_seeded_generation_is_repeatable_and_preserves_context() -> None:
    model = _small_model()
    config = FastESM3GenerationConfig(num_steps=2, temperature=1.0, seed=73)

    first = model.generate("MK__A", config)
    second = model.generate("MK__A", config)

    assert isinstance(first, str)
    assert first == second
    assert len(first) == 5
    assert first[:2] == "MK"
    assert first[-1] == "A"
    assert "_" not in first


@pytest.mark.parametrize("num_steps", (0, -1))
def test_esm3_generation_rejects_nonpositive_num_steps(num_steps: int) -> None:
    model = _small_model()

    with pytest.raises(ValueError, match="num_steps must be positive"):
        model.generate("M_K", FastESM3GenerationConfig(num_steps=num_steps))


@pytest.mark.parametrize("num_steps", (True, False, 1.5, "2"))
def test_esm3_generation_rejects_noninteger_num_steps(num_steps: object) -> None:
    model = _small_model()

    with pytest.raises(TypeError, match="num_steps must be an integer or None"):
        model.generate("M_K", FastESM3GenerationConfig(num_steps=num_steps))


def test_esm3_generation_none_num_steps_uses_mask_count() -> None:
    model = _small_model()
    observed_steps = 0
    original_forward = model.forward

    def count_forward_calls(*args, **kwargs):
        nonlocal observed_steps
        observed_steps += 1
        return original_forward(*args, **kwargs)

    model.forward = count_forward_calls
    generated = model.generate("M__K", FastESM3GenerationConfig(num_steps=None, seed=19))

    assert isinstance(generated, str)
    assert "_" not in generated
    assert observed_steps == 2


def test_esm3_generation_preserves_every_supported_conditioning_track() -> None:
    model = _small_model()
    input_ids = torch.tensor([[SEQUENCE_BOS_TOKEN, SEQUENCE_MASK_TOKEN, SEQUENCE_EOS_TOKEN]])
    shape = input_ids.shape
    conditioning = {
        "attention_mask": torch.ones(shape, dtype=torch.long),
        "structure_tokens": torch.zeros(shape, dtype=torch.long),
        "ss8_tokens": torch.zeros(shape, dtype=torch.long),
        "sasa_tokens": torch.zeros(shape, dtype=torch.long),
        "function_tokens": torch.zeros((*shape, 8), dtype=torch.long),
        "residue_annotation_tokens": torch.zeros((*shape, 16), dtype=torch.long),
        "average_plddt": torch.ones(shape),
        "per_res_plddt": torch.zeros(shape),
        "structure_coords": torch.full((*shape, 3, 3), float("nan")),
        "chain_id": torch.zeros(shape, dtype=torch.long),
        "sequence_id": torch.ones(shape, dtype=torch.bool),
    }
    observed: list[dict[str, torch.Tensor]] = []
    original_forward = model.forward

    def capture_forward(*args, **kwargs):
        observed.append({name: value for name, value in kwargs.items() if torch.is_tensor(value)})
        return original_forward(*args, **kwargs)

    model.forward = capture_forward
    generated = model.generate(
        {"sequence_tokens": input_ids, **conditioning},
        FastESM3GenerationConfig(num_steps=1, seed=17),
    )

    assert torch.is_tensor(generated)
    assert not generated.eq(SEQUENCE_MASK_TOKEN).any()
    assert len(observed) == 1
    assert set(conditioning).issubset(observed[0])
    for name, expected in conditioning.items():
        torch.testing.assert_close(observed[0][name], expected, equal_nan=True)


def test_esm3_generation_rejects_unknown_or_ambiguous_inputs() -> None:
    model = _small_model()
    input_ids = torch.tensor([[SEQUENCE_BOS_TOKEN, SEQUENCE_MASK_TOKEN, SEQUENCE_EOS_TOKEN]])

    with pytest.raises(TypeError, match="Unsupported ESM3 generation inputs: labels"):
        model.generate({"input_ids": input_ids, "labels": input_ids})
    with pytest.raises(ValueError, match="only one of input_ids or sequence_tokens"):
        model.generate({"input_ids": input_ids, "sequence_tokens": input_ids})


def test_esm3_saved_model_loads_without_installed_fastplms(tmp_path: Path) -> None:
    model_path = tmp_path / "saved"
    _small_model().save_pretrained(model_path)
    runtime_hash = _saved_bundle_namespace(model_path)["RUNTIME_HASH"]
    assert isinstance(runtime_hash, str)
    poisoned_cache = model_path / f"_fastplms_runtime_{runtime_hash[:16]}" / "fastplms"
    poisoned_cache.mkdir(parents=True)
    (poisoned_cache / "__init__.py").write_text(
        "raise RuntimeError('persistent model-local cache was trusted')\n",
        encoding="utf-8",
    )
    runtime_record = tmp_path / "runtime-path.txt"
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from pathlib import Path

        class BlockInstalledFastPLMs(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "fastplms":
                    raise ModuleNotFoundError("installed FastPLMs is blocked")
                return None

        sys.modules.pop("fastplms", None)
        for name in tuple(sys.modules):
            if name.startswith("fastplms."):
                sys.modules.pop(name)
        sys.meta_path.insert(0, BlockInstalledFastPLMs())

        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            sys.argv[1],
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        assert type(model).__module__ == "fastplms.models.esm3.modeling_esm3"
        package_file = Path(sys.modules["fastplms"].__file__).resolve()
        package_root = package_file.parent
        runtime_root = package_root.parent
        model_root = Path(sys.argv[1]).resolve()
        assert runtime_root.name.startswith("fastplms-esm3-runtime-")
        assert runtime_root != model_root
        assert model_root not in runtime_root.parents
        assert not any(path.name == "__pycache__" for path in package_root.rglob("*"))
        Path(sys.argv[2]).write_text(str(runtime_root), encoding="utf-8")
        batch = model.tokenize_sequences(["MKTAYIAKQ"], device=model.device)
        with torch.inference_mode():
            output = model(**batch)
        assert output.logits is not None
        assert output.logits.shape[:2] == batch["input_ids"].shape
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(model_path), str(runtime_record)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    extracted_runtime = Path(runtime_record.read_text(encoding="utf-8"))
    assert not extracted_runtime.exists()
    assert (
        (poisoned_cache / "__init__.py")
        .read_text(encoding="utf-8")
        .startswith("raise RuntimeError")
    )


def test_esm3_embed_dataset(tmp_path: Path) -> None:
    model = _small_model()
    save_path = tmp_path / "embeddings"

    result = model.embed_dataset(
        inputs=["MKTAYIAKQ", "GGGG"],
        batch_size=2,
        max_length=16,
        pooling=("mean", "cls"),
        output=save_path,
    )

    embeddings = result.as_dict(key="sequence")
    assert set(embeddings) == {"MKTAYIAKQ", "GGGG"}
    assert embeddings["MKTAYIAKQ"].shape == (128,)
    assert (save_path / "index.json").is_file()


@pytest.mark.gpu
def test_esm3_flex_matches_sdpa() -> None:
    model = _small_model().to(torch.device("cuda"))
    batch = model.tokenize_sequences(["MKTAYIAKQ", "GGGG"], device=model.device)

    with torch.inference_mode(), strict_fp32_matmul():
        model.set_attn_implementation("sdpa")
        sdpa_output = model(**batch).last_hidden_state
        model.set_attn_implementation("flex_attention")
        flex_output = model(**batch).last_hidden_state

    max_abs = (sdpa_output - flex_output).float().abs().max().item()
    mse = ((sdpa_output - flex_output).float() ** 2).mean().item()
    assert max_abs < 1e-4
    assert mse < 1e-8
