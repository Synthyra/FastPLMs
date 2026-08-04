# Dependency profiles

FastPLMs loads from Hugging Face model repositories with
`trust_remote_code=True`. This repository is a source, test, and dependency
workspace. It is not an installable Python distribution.

`core.in` and `features/*.in` declare direct dependencies. `profiles/*.in`
combine those declarations for repository environments. Validation commands
constrain Torch and Transformers with
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
FP8 profiles also pass
`--overrides requirements/overrides/cuda.txt`.

When you need a full transitive lock for a maintained environment, compile the
required profile. Do not lock all incompatible features together:

```bash
uv pip compile \
  requirements/profiles/candidate.in \
  -c requirements/constraints/validation.txt \
  --universal \
  --generate-hashes \
  -o requirements/locks/candidate.txt
```

Generated locks are environment-specific test inputs. Public Hugging Face
artifacts must expose only direct runtime dependencies.
