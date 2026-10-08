"""Credential-shaped names, configs with them removed, and files that hold credentials.

616 runs in this W&B account carry credential-shaped config keys, some nested. A config sent
to W&B is copied into Dropbox by `ws capture` and from there into a GitHub projection, so a
credential in it travels far. `redact` drops such keys by name at any depth. It errs toward
dropping: a missing setting costs a line of a summary, a missed credential costs a rotation.
Values are not inspected; a secret stored under an ordinary name is out of reach here.

`is_credential_path` recognizes a credential file by its path, so code that walks a tree,
ships it to a machine, or mounts it into a container can leave the file out unopened.
"""

from __future__ import annotations

import re

from collections.abc import Mapping
from pathlib import PurePosixPath


# Segments that name a credential wherever they appear, so `client_secret`, `db_password`,
# and `runtime.auth` all match.
CREDENTIAL_SEGMENTS = frozenset({"secret", "password", "passwd", "credential", "auth", "authorization", "bearer"})
# Segments that name a credential only when they end the name: `hf_token`, `wandb_api_key`,
# and `private_key` hold one, while `token_budget`, `token_dropout`, and `key_dim` are settings.
CREDENTIAL_ENDINGS = frozenset({"token", "key"})
SEPARATORS = re.compile(r"[._\-/]")
# Files that hold credentials, by name, as the credential guard in ~/.claude/hooks lists them,
# plus `gha-creds-*.json`, the service-account key google-github-actions/auth writes into a
# CI checkout.
CREDENTIAL_FILE = re.compile(
    r"^[\w.-]*\.env(\.|$)|^\.envrc$|^\.secrets(\.|$)|^id_(rsa|ed25519|ecdsa|dsa)"
    r"|\.(pem|key|p12|pfx|keystore|jks)$|^\.(netrc|npmrc|pypirc|git-credentials)$"
    r"|credentials\.json$|service[-_]?account.*\.json$|firebase[-_]adminsdk|^gha-creds-.*\.json$",
    re.IGNORECASE,
)
CREDENTIAL_DIRECTORIES = frozenset({".ssh", ".aws", ".gnupg", "gcloud"})
# A placeholder such as a `.template` copy holds no values and is ordinary content.
PLACEHOLDER = re.compile(r"\.(example|sample|template|dist)$", re.IGNORECASE)


def is_credential(name: str) -> bool:
    """Whether a config key names a credential, by the segment rules above."""
    lowered = name.lower()
    if "api_key" in lowered or "apikey" in lowered:
        return True
    segments = SEPARATORS.split(lowered)
    return segments[-1] in CREDENTIAL_ENDINGS or any(segment in CREDENTIAL_SEGMENTS for segment in segments)


def is_credential_path(path: str) -> bool:
    """Whether a relative posix path names a credential file, which is moved but never opened."""
    *directories, name = PurePosixPath(path).parts
    if PLACEHOLDER.search(name):
        return False
    return bool(CREDENTIAL_FILE.search(name)) or any(part.lower() in CREDENTIAL_DIRECTORIES for part in directories)


def redact(values: Mapping[str, object]) -> dict[str, object]:
    """Drop every credential-shaped key, at any depth, including inside lists."""
    return {name: _redacted(value) for name, value in values.items() if not is_credential(str(name))}


def _redacted(value: object) -> object:
    if isinstance(value, Mapping):
        return redact(value)
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redacted(item) for item in value)
    return value
