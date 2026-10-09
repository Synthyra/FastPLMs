---
id: foundry.modal_image
kind: library_module
status: active
aliases: [foundry.modal_image]
---
# foundry.modal_image

One way to build a Modal image: from a workspace environment's lock, with the code the project
imports mounted by the rules `ws run` uses. It replaces the image definitions that projects wrote
out one by one, which pinned torch 2.11 in five places and let numpy, pandas, and scikit-learn
float.

```python
import modal

from foundry.modal_image import workspace_image

image = workspace_image("fastplms_runtime", packages=["biotite"], apt=["git"], env={"HF_HOME": "/cache/hf"})
app = modal.App("fold", image=image)
```

Launch through `ws run`, which puts the project's import roots on PYTHONPATH:

```
ws run folding_service -- python -m modal run src/fold/app.py
```

## MMseqs2

`with_mmseqs(image)` adds the MMseqs2 release the workspace's clustering is pinned to (`MMSEQS_RELEASE`,
checked against the SHA-256 GitHub records) at `MMSEQS_BINARY`; pass `MMSEQS_COMMIT` as the expected version
to `foundry.datasets`. The two dataset builders that run MMseqs on Modal share it.

## What the image holds

1. `modal_base` from the environment's manifest, with its `python`.
2. `apt` packages.
3. The torch family pinned in the lock, from `torch_index`.
4. The rest of the lock and the app's `packages`, resolved together, so a version the lock pins
   is the one installed.
5. `commands`, for a build step that needs the packages.
6. `env`, and PYTHONPATH naming the mounted roots.
7. Each workspace import root at `/ws/tree/<workspace path>` and foundry at
   `/ws/import-roots/foundry`, added at container start unless `copy=True`. Caches, version
   control, and credential files are left out by name.

Inside a container Modal imports the app module again. There `workspace_image` returns a bare
image without reading anything, because the container already runs the image built where the
app was launched.

## Any other directory

No image may carry a credential file, and a clone of a projection keeps `.secrets.env` at its
root. Every other directory an image adds takes `ignore_with_credentials`, which leaves out what
its patterns match and, at any depth, every cache and version control directory and every file
`foundry.logging.credentials.is_credential_path` names:

```python
from foundry.modal_image import ignore_with_credentials

image = image.add_local_dir("src", "/root/src", ignore=ignore_with_credentials(["results", "**/*.csv"]))
```

Patterns are dockerignore globs relative to the directory added. A bare name such as `results`
matches only at its top level and `**/results` at every depth; a leading `/` matches nothing on
Windows. The rule is a `modal.FilePatternMatcher`, so Modal still skips a directory the patterns
match without walking it. `tests/tier0_smoke/test_modal_mounts.py` fails on a directory added to an
image without a credential rule.

## Foundry in an app's own image

An app that builds its own image and imports foundry, at module level or in the code it runs,
needs foundry in the container. A clone of a projection vendors it into the source directory the
image adds. The workspace keeps one copy at its root, which a launch from the workspace has to
add itself:

```python
from foundry.modal_image import ignore_with_credentials, workspace_foundry

workspace_copy = workspace_foundry("src")
if workspace_copy is not None:
    image = image.add_local_dir(workspace_copy, "/root/src/foundry", ignore=ignore_with_credentials())
```

`workspace_foundry` is None in a clone, whose source directory holds foundry, and inside the
container.

A model family the app imports, such as FastPLMs, follows the same rule through
`workspace_model_family(source, "fastplms")`, which names `models/fastplms`:

```python
fastplms = workspace_model_family("src", "fastplms")
if fastplms is not None:
    image = image.add_local_dir(fastplms, "/root/src/fastplms", ignore=ignore_with_credentials(["tests", "vendor"]))
```

## Limits

- An environment needs `modal_base` and `python` in its manifest. `fastplms_runtime` has them.
- A lock recorded from a constraints file pins only what that file pins, so the other packages
  resolve to their newest compatible versions at build time.
