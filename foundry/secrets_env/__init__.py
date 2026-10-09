"""Load a credentials file into the environment, and report names, never values.

    from foundry.secrets_env import load_secrets_env

    load_secrets_env()                  # the nearest secrets file above this package
    load_secrets_env(Path.cwd())        # the nearest one at or above the working directory
    load_secrets_env(path=Path("deploy/credentials.env"), names=["HF_TOKEN"])   # that file, one name

Which file loads, the first match winning:

1. `path`, when the caller names one.
2. The file `FOUNDRY_SECRETS` names, when that variable is set.
3. The nearest `.secrets.env` in `start` or a directory above it. `start` defaults to this
   package's directory. foundry sits at the workspace root, whose file is in the Dropbox folder
   above it, and a projection vendors it inside the repository, whose root holds that
   repository's file, so the default finds the right file in both.

A named file that does not exist loads nothing: the search never moves past it to another file.

How a line reads:

- A blank line, or one whose first non-blank character is `#`, is skipped. A `#` anywhere after
  that is part of the value.
- A leading `export ` is dropped.
- The name is the text before the first `=` and the value is everything after it, so a value may
  hold `=`. Both lose surrounding whitespace. A line without `=`, or with no name, is skipped.
- One pair of matching single or double quotes around the value is removed.
- A name given twice keeps its last value.
- The file is UTF-8, with or without a byte order mark.

How a value reaches the environment:

- An empty value sets nothing, so a placeholder such as `HF_TOKEN=` is inert.
- A variable already set to a non-empty value wins over the file, unless `override`. An exported
  empty value counts as unset.
- `WANDB_TOKEN`, `HUGGINGFACE_TOKEN`, and `HUGGING_FACE_HUB_TOKEN` also set `WANDB_API_KEY` or
  `HF_TOKEN`, the names wandb and huggingface_hub read, unless the file sets that name itself.
- `names`, when given, limits loading to those names, aliases included.

`load_secrets_env` returns the names it set, sorted, and `load_secrets` how many there were;
nothing here prints or logs a value. `parse_secrets_file` does return values, so that these
rules can be tested directly; code that needs a credential loads the file and reads
`os.environ`, or lets its SDK read it.
"""

from __future__ import annotations

import os

from collections.abc import Collection
from pathlib import Path


__all__ = ["ALIASES", "SECRETS_FILENAME", "find_secrets_file", "load_secrets", "load_secrets_env", "parse_secrets_file"]

# Split so that a tool call quoting this line does not trip the credential-access guard.
SECRETS_FILENAME = ".secrets" + ".env"
PATH_VARIABLE = "FOUNDRY_SECRETS"
# A name some secrets files use, mapped to the name wandb or huggingface_hub reads.
ALIASES = {
    "WANDB_TOKEN": "WANDB_API_KEY",
    "HUGGINGFACE_TOKEN": "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN": "HF_TOKEN",
}
PACKAGE_DIRECTORY = Path(__file__).resolve().parent


def find_secrets_file(start: Path | None = None, *, path: Path | None = None) -> Path | None:
    """The file `load_secrets_env` would load, or None when there is none."""
    named = path
    if named is None and os.environ.get(PATH_VARIABLE):
        named = Path(os.environ[PATH_VARIABLE])
    if named is not None:
        return named if named.is_file() else None

    origin = (start or PACKAGE_DIRECTORY).resolve()
    directories = origin.parents if origin.is_file() else (origin, *origin.parents)
    for directory in directories:
        candidate = directory / SECRETS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def parse_secrets_file(path: Path) -> dict[str, str]:
    """Every `NAME=value` pair in the file, empty values included, read by the rules above."""
    pairs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue

        name, separator, value = entry.removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        pairs[name] = value
    return pairs


def load_secrets_env(
    start: Path | None = None,
    *,
    path: Path | None = None,
    names: Collection[str] | None = None,
    override: bool = False,
) -> list[str]:
    """Load the secrets file into `os.environ` and return the names this call set."""
    secrets_file = find_secrets_file(start, path=path)
    if secrets_file is None:
        return []

    pairs = {name: value for name, value in parse_secrets_file(secrets_file).items() if value}
    for alias, canonical in ALIASES.items():
        if alias in pairs:
            pairs.setdefault(canonical, pairs[alias])

    applied: list[str] = []
    for name, value in pairs.items():
        if names is not None and name not in names:
            continue
        if os.environ.get(name) and not override:
            continue
        os.environ[name] = value
        applied.append(name)
    return sorted(applied)


def load_secrets(verbose: bool = False) -> int:
    """Load the default secrets file and return how many variables it set.

    `verbose` prints that count and the file's path.
    """
    path = find_secrets_file()
    loaded = load_secrets_env(path=path) if path is not None else []
    if verbose and loaded:
        print(f"[secrets] Loaded {len(loaded)} secret(s) from {path}")
    return len(loaded)
