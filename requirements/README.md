# Dependency profiles

FastPLMs is loaded from Hugging Face model repositories with
`trust_remote_code=True`. This repository is a source, test, and dependency
workspace, not an installable Python distribution.

`core.in` and `features/*.in` are the direct dependency declarations.
`profiles/*.in` compose those declarations for the environments exercised by
the repository. Validation commands constrain Torch and Transformers with
`constraints/validation.txt`.

Create a local validation environment with:

```bash
uv venv
uv pip install \
  -r requirements/profiles/cpu-validation.in \
  -c requirements/constraints/validation.txt \
  --torch-backend cpu
```

CUDA container profiles use the same command without `--torch-backend cpu`.
The FP8 profiles additionally pass
`--overrides requirements/overrides/cuda.txt`.

When a fully transitive lock is needed for a maintained environment, compile
the corresponding profile rather than locking every mutually incompatible
feature together:

```bash
uv pip compile \
  requirements/profiles/candidate.in \
  -c requirements/constraints/validation.txt \
  --universal \
  --generate-hashes \
  -o requirements/locks/candidate.txt
```

Generated locks are environment-specific test inputs. Public Hugging Face
artifacts should expose only their direct runtime dependencies.
