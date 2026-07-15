"""Filesystem and subprocess types used by optional structure utilities."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any, TypeAlias

PathLike: TypeAlias = str | Path
PathOrBuffer: TypeAlias = PathLike | io.StringIO


def _stdout_destination(*, capture_output: bool, quiet: bool) -> int | None:
    if capture_output:
        return subprocess.PIPE
    if quiet:
        return subprocess.DEVNULL
    return None


def _stderr_text(error: subprocess.CalledProcessError) -> str:
    stderr = error.stderr
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode(errors="replace")
    return str(stderr)


def run_subprocess_with_errorcheck(
    *popenargs: Any,
    capture_output: bool = False,
    quiet: bool = False,
    env: dict[str, str] | None = None,
    shell: bool = False,
    executable: str | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run a command and include captured standard error in failures."""

    stdout = _stdout_destination(capture_output=capture_output, quiet=quiet)
    try:
        return subprocess.run(
            *popenargs,
            check=True,
            env=env,
            executable=executable,
            shell=shell,
            stderr=subprocess.PIPE,
            stdout=stdout,
            **kwargs,
        )
    except subprocess.CalledProcessError as error:
        message = f"Command failed with errorcode {error.returncode}.\n\n{_stderr_text(error)}"
        raise RuntimeError(message) from error
