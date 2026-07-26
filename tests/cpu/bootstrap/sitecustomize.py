"""Python-startup policy for the hermetic CPU contract lane.

This module is intentionally standard-library-only.  The CPU workflow adds its
directory to ``PYTHONPATH``, so Python imports it before pytest can import the
repository-level ``tests/conftest.py`` (and therefore before Torch or the model
registry).  Child Python processes inherit the same startup policy.
"""

from __future__ import annotations

import builtins
import io
import os
import shlex
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_CACHE_ENVIRONMENT = {
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
    "TRANSFORMERS_CACHE": "huggingface/transformers",
    "HF_DATASETS_CACHE": "huggingface/datasets",
    "TORCH_HOME": "torch",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "TORCHINDUCTOR_CACHE_DIR": "torch-inductor",
    "TRITON_CACHE_DIR": "triton",
    "XDG_CACHE_HOME": "xdg",
}
_FIXED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "DO_NOT_TRACK": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTEST_XDIST_AUTO_NUM_WORKERS": "4",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_WORKSPACE = Path(__file__).resolve().parents[3]
_FORBIDDEN_READ_ROOTS = tuple(
    path.resolve()
    for path in (
        _WORKSPACE / "vendor" / "upstream",
        _WORKSPACE / ".git" / "modules",
        _WORKSPACE / "official",
    )
)
_CHECKPOINT_SUFFIXES = frozenset({".bin", ".ckpt", ".pt", ".pth", ".safetensors"})
_FORBIDDEN_CONTAINER_EXECUTABLES = frozenset(
    {"buildx", "docker", "docker-compose", "podman"}
)
_SHELL_EXECUTABLES = frozenset({"bash", "cmd", "dash", "powershell", "pwsh", "sh", "zsh"})
_COMMAND_WRAPPERS = frozenset({"command", "env", "nohup", "sudo"})


def _network_blocked(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Network access is forbidden in tests/cpu")


def _dir_fd_path(file: object, dir_fd: int | None) -> object:
    if dir_fd is None or not isinstance(file, (str, bytes, os.PathLike)):
        return file
    decoded = os.fsdecode(file)
    if os.path.isabs(decoded):
        return decoded
    for descriptor_root in ("/proc/self/fd", "/dev/fd"):
        try:
            directory = Path(os.readlink(f"{descriptor_root}/{dir_fd}"))
        except OSError:
            continue
        return directory / decoded
    raise RuntimeError(f"CPU contracts could not resolve directory descriptor {dir_fd}")


def _assert_portable_path(file: object, *, dir_fd: int | None = None) -> None:
    if not isinstance(file, (str, bytes, os.PathLike)):
        return
    try:
        resolved = Path(_dir_fd_path(file, dir_fd)).resolve()
    except (OSError, TypeError, ValueError):
        return
    if any(resolved == root or root in resolved.parents for root in _FORBIDDEN_READ_ROOTS):
        raise RuntimeError(f"CPU contracts may not access submodule/reference path: {resolved}")
    if (
        (resolved == _WORKSPACE or _WORKSPACE in resolved.parents)
        and (
            resolved.suffix.lower() in _CHECKPOINT_SUFFIXES
            or resolved.name.endswith(".safetensors.index.json")
        )
    ):
        raise RuntimeError(f"CPU contracts may not access checkpoint path: {resolved}")


def _executable_basename(value: object) -> str:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return ""
    name = Path(os.fsdecode(value)).name.lower()
    return name.removesuffix(".exe")


def _command_tokens(command: object) -> list[str] | None:
    if isinstance(command, (str, bytes, os.PathLike)):
        try:
            return shlex.split(os.fsdecode(command), posix=os.name != "nt")
        except ValueError:
            return None
    if isinstance(command, (list, tuple)):
        try:
            return [os.fsdecode(value) for value in command]
        except TypeError:
            return None
    return None


def _command_starts(tokens: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in {"&", "&&", ";", "|", "||"}:
            if current:
                commands.append(current)
                current = []
        else:
            current.append(token)
    if current:
        commands.append(current)
    return commands


def _unwrap_command(tokens: list[str]) -> list[str]:
    remaining = list(tokens)
    while remaining and _executable_basename(remaining[0]) in _COMMAND_WRAPPERS:
        wrapper = _executable_basename(remaining.pop(0))
        while remaining and (
            remaining[0].startswith("-")
            or (wrapper == "env" and "=" in remaining[0])
        ):
            remaining.pop(0)
    return remaining


def _forbidden_container_command(command: object, *, executable: object = None) -> str | None:
    explicit = _executable_basename(executable)
    if explicit in _FORBIDDEN_CONTAINER_EXECUTABLES:
        return explicit
    tokens = _command_tokens(command)
    if tokens is None:
        return None
    for candidate in _command_starts(tokens):
        unwrapped = _unwrap_command(candidate)
        if not unwrapped:
            continue
        executable_name = _executable_basename(unwrapped[0])
        if executable_name in _FORBIDDEN_CONTAINER_EXECUTABLES:
            return executable_name
        if executable_name in _SHELL_EXECUTABLES:
            lowered = [token.lower() for token in unwrapped]
            for flag in ("-c", "-ec", "-lc", "/c", "-command"):
                try:
                    index = lowered.index(flag)
                except ValueError:
                    continue
                if index + 1 < len(unwrapped):
                    nested = _forbidden_container_command(unwrapped[index + 1])
                    if nested is not None:
                        return nested
    return None


def _assert_portable_spawn(command: object, *, executable: object = None) -> None:
    forbidden = _forbidden_container_command(command, executable=executable)
    if forbidden is not None:
        raise RuntimeError(
            f"Container execution is forbidden in tests/cpu: {forbidden}"
        )


def _install_spawn_guards() -> None:
    if getattr(builtins, "_fastplms_cpu_spawn_guard", False):
        return
    original_popen = subprocess.Popen
    original_system = os.system

    class GuardedPopen(original_popen):  # type: ignore[misc, valid-type]
        def __init__(self, args: Any, *popen_args: Any, **popen_kwargs: Any) -> None:
            _assert_portable_spawn(args, executable=popen_kwargs.get("executable"))
            super().__init__(args, *popen_args, **popen_kwargs)

    def guarded_system(command: object) -> int:
        _assert_portable_spawn(command)
        return original_system(command)

    subprocess.Popen = GuardedPopen  # type: ignore[assignment]
    os.system = guarded_system  # type: ignore[assignment]

    for name in (
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
    ):
        original = getattr(os, name, None)
        if original is None:
            continue

        def guarded_spawn(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            command = args[1] if len(args) > 1 else kwargs.get("file")
            _assert_portable_spawn(command, executable=command)
            return _original(*args, **kwargs)

        setattr(os, name, guarded_spawn)
    builtins.__dict__["_fastplms_cpu_spawn_guard"] = True


def _install_open_guards() -> None:
    if getattr(builtins, "_fastplms_cpu_open_guard", False):
        return
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open

    def guarded_builtin_open(file: object, *args: Any, **kwargs: Any) -> Any:
        _assert_portable_path(file)
        return original_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: object, *args: Any, **kwargs: Any) -> Any:
        _assert_portable_path(file)
        return original_io_open(file, *args, **kwargs)

    def guarded_os_open(file: object, *args: Any, **kwargs: Any) -> int:
        _assert_portable_path(file, dir_fd=kwargs.get("dir_fd"))
        return original_os_open(file, *args, **kwargs)

    builtins.open = guarded_builtin_open
    io.open = guarded_io_open
    os.open = guarded_os_open  # type: ignore[assignment]
    builtins.__dict__["_fastplms_cpu_open_guard"] = True


def _install_hub_guards() -> None:
    # Hub is present in the required environment.  Keeping this import optional
    # lets the startup bootstrap remain harmless while an environment is built.
    try:
        import huggingface_hub
        import huggingface_hub._snapshot_download
        import huggingface_hub.file_download
    except ImportError:
        return
    huggingface_hub.hf_hub_download = _network_blocked  # type: ignore[assignment]
    huggingface_hub.snapshot_download = _network_blocked  # type: ignore[assignment]
    huggingface_hub.file_download.hf_hub_download = (  # type: ignore[assignment]
        _network_blocked
    )
    huggingface_hub._snapshot_download.snapshot_download = (  # type: ignore[assignment]
        _network_blocked
    )
    huggingface_hub.file_download.http_get = _network_blocked  # type: ignore[assignment]


def _install() -> None:
    bootstrap_root = str(Path(__file__).resolve().parent)
    python_path = os.environ.get("PYTHONPATH", "")
    python_path_entries = [entry for entry in python_path.split(os.pathsep) if entry]
    if bootstrap_root not in python_path_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join((bootstrap_root, *python_path_entries))
    cache_root_value = os.environ.get("FASTPLMS_CPU_CACHE_ROOT")
    if cache_root_value is None:
        cache_root = Path(tempfile.mkdtemp(prefix="fastplms-cpu-contract-cache-")).resolve()
        os.environ["FASTPLMS_CPU_CACHE_ROOT"] = str(cache_root)
        os.environ["FASTPLMS_CPU_CACHE_STARTED_EMPTY"] = "1"
    else:
        cache_root = Path(cache_root_value).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
    for name, relative in _CACHE_ENVIRONMENT.items():
        path = cache_root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(_FIXED_ENVIRONMENT)
    os.environ["FASTPLMS_CPU_BOOTSTRAPPED"] = "1"

    socket.create_connection = _network_blocked  # type: ignore[assignment]
    socket.getaddrinfo = _network_blocked  # type: ignore[assignment]
    socket.socket.connect = _network_blocked  # type: ignore[assignment]
    socket.socket.connect_ex = _network_blocked  # type: ignore[assignment]
    socket.socket.sendto = _network_blocked  # type: ignore[assignment]
    if hasattr(socket.socket, "sendmsg"):
        socket.socket.sendmsg = _network_blocked  # type: ignore[assignment]
    _install_open_guards()
    _install_spawn_guards()
    _install_hub_guards()
    builtins.__dict__["_fastplms_cpu_assert_portable_path"] = _assert_portable_path
    builtins.__dict__["_fastplms_cpu_process_bootstrapped"] = True


_install()
