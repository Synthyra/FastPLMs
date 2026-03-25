"""
Data-driven HuggingFace upload script for all FastPLMs models.

Builds composite single-file modeling scripts by concatenating shared modules
(attention.py, embedding_mixin.py, entrypoint_setup.py) with model-specific code,
then uploads to each HF repo.

Usage:
    py -m update_HF
    py -m update_HF --hf_token YOUR_TOKEN
    py -m update_HF --families esm2 dplm
    py -m update_HF --skip-weights
    py -m update_HF --files-only
    py -m update_HF --config-only
"""

import argparse
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, login

_REPO_ROOT = Path(__file__).resolve().parent

# Regex to strip the try/except import guards that reference fastplms.*
# These are only needed for local development; composites have the code inline.
_IMPORT_GUARD_PATTERN = re.compile(
    r"try:\s*\n"
    r"(?:\s+from fastplms\.\w+ import[^\n]*\n|\s+from fastplms\.\w+ import \(\n(?:\s+[^\n]*\n)*?\s+\)\n)+"
    r"\s*except ImportError:\s*\n"
    r"\s*pass[^\n]*\n",
    re.MULTILINE,
)

COMPOSITE_SHARED_MODULES = [
    "entrypoint_setup.py",
    "fastplms/embedding_mixin.py",
    "fastplms/attention.py",
]


_FUTURE_IMPORT_RE = re.compile(r"^from __future__ import annotations\s*\n?", re.MULTILINE)


def build_composite(modeling_path: str, include_embedding_mixin: bool = True) -> str:
    """Build a single self-contained modeling file for HF Hub upload.

    Concatenates shared modules + model code, stripping the try/except
    import guards from the model code since shared definitions are inlined above.
    Hoists `from __future__ import annotations` to the top (must be first statement).
    """
    parts = []
    for shared_path in COMPOSITE_SHARED_MODULES:
        if not include_embedding_mixin and "embedding_mixin" in shared_path:
            continue
        content = (_REPO_ROOT / shared_path).read_text(encoding="utf-8")
        content = _FUTURE_IMPORT_RE.sub("", content)
        parts.append(content)

    model_code = (_REPO_ROOT / modeling_path).read_text(encoding="utf-8")
    model_code = _IMPORT_GUARD_PATTERN.sub("", model_code)
    model_code = _FUTURE_IMPORT_RE.sub("", model_code)
    parts.append(model_code)

    return "from __future__ import annotations\n\n" + "\n".join(parts)


MODEL_REGISTRY = [
    {
        "family": "e1",
        "repo_ids": [
            "Synthyra/Profluent-E1-150M",
            "Synthyra/Profluent-E1-300M",
            "Synthyra/Profluent-E1-600M",
        ],
        "modeling_src": "fastplms/e1/modeling_e1.py",
        "modeling_dest": "modeling_e1.py",
        "composite": True,
        "include_embedding_mixin": True,
        "extra_files": {
            "fastplms/e1/tokenizer.json": "tokenizer.json",
        },
        "readme_map": {
            "Synthyra/Profluent-E1-150M": "fastplms/e1/README.md",
            "Synthyra/Profluent-E1-300M": "fastplms/e1/README.md",
            "Synthyra/Profluent-E1-600M": "fastplms/e1/README.md",
        },
        "license_map": {
            "Synthyra/Profluent-E1-150M": "fastplms/e1/LICENSE",
            "Synthyra/Profluent-E1-300M": "fastplms/e1/LICENSE",
            "Synthyra/Profluent-E1-600M": "fastplms/e1/LICENSE",
        },
        "weight_module": "fastplms.e1.get_weights",
    },
    {
        "family": "esmplusplus",
        "repo_ids": [
            "Synthyra/ESMplusplus_small",
            "Synthyra/ESMplusplus_large",
        ],
        "modeling_src": "fastplms/esm_plusplus/modeling_esm_plusplus.py",
        "modeling_dest": "modeling_esm_plusplus.py",
        "composite": True,
        "include_embedding_mixin": True,
        "extra_files": {},
        "readme_map": {
            "Synthyra/ESMplusplus_small": "fastplms/esm_plusplus/README_small.md",
            "Synthyra/ESMplusplus_large": "fastplms/esm_plusplus/README_large.md",
        },
        "license_map": {
            "Synthyra/ESMplusplus_small": "fastplms/esm_plusplus/LICENSE_small",
            "Synthyra/ESMplusplus_large": "fastplms/esm_plusplus/LICENSE_large",
        },
        "weight_module": "fastplms.esm_plusplus.get_weights",
    },
    {
        "family": "esm2",
        "repo_ids": [
            "Synthyra/ESM2-8M",
            "Synthyra/ESM2-35M",
            "Synthyra/ESM2-150M",
            "Synthyra/ESM2-650M",
            "Synthyra/ESM2-3B",
            "Synthyra/FastESM2_650",
        ],
        "modeling_src": "fastplms/esm2/modeling_fastesm.py",
        "modeling_dest": "modeling_fastesm.py",
        "composite": True,
        "include_embedding_mixin": True,
        "extra_files": {},
        "readme_map": {
            "Synthyra/ESM2-8M": "fastplms/esm2/README.md",
            "Synthyra/ESM2-35M": "fastplms/esm2/README.md",
            "Synthyra/ESM2-150M": "fastplms/esm2/README.md",
            "Synthyra/ESM2-650M": "fastplms/esm2/README.md",
            "Synthyra/ESM2-3B": "fastplms/esm2/README.md",
            "Synthyra/FastESM2_650": "fastplms/esm2/README_650.md",
        },
        "license_map": {
            "Synthyra/ESM2-8M": "fastplms/esm2/LICENSE",
            "Synthyra/ESM2-35M": "fastplms/esm2/LICENSE",
            "Synthyra/ESM2-150M": "fastplms/esm2/LICENSE",
            "Synthyra/ESM2-650M": "fastplms/esm2/LICENSE",
            "Synthyra/ESM2-3B": "fastplms/esm2/LICENSE",
            "Synthyra/FastESM2_650": "fastplms/esm2/LICENSE",
        },
        "weight_module": "fastplms.esm2.get_weights",
    },
    {
        "family": "dplm",
        "repo_ids": [
            "Synthyra/DPLM-150M",
            "Synthyra/DPLM-650M",
            "Synthyra/DPLM-3B",
        ],
        "modeling_src": "fastplms/dplm/modeling_dplm.py",
        "modeling_dest": "modeling_dplm.py",
        "composite": True,
        "include_embedding_mixin": True,
        "extra_files": {},
        "readme_map": {
            "Synthyra/DPLM-150M": "fastplms/dplm/README.md",
            "Synthyra/DPLM-650M": "fastplms/dplm/README.md",
            "Synthyra/DPLM-3B": "fastplms/dplm/README.md",
        },
        "license_map": {},
        "weight_module": "fastplms.dplm.get_weights",
    },
    {
        "family": "dplm2",
        "repo_ids": [
            "Synthyra/DPLM2-150M",
            "Synthyra/DPLM2-650M",
            "Synthyra/DPLM2-3B",
        ],
        "modeling_src": "fastplms/dplm2/modeling_dplm2.py",
        "modeling_dest": "modeling_dplm2.py",
        "composite": True,
        "include_embedding_mixin": True,
        "extra_files": {},
        "readme_map": {
            "Synthyra/DPLM2-150M": "fastplms/dplm2/README.md",
            "Synthyra/DPLM2-650M": "fastplms/dplm2/README.md",
            "Synthyra/DPLM2-3B": "fastplms/dplm2/README.md",
        },
        "license_map": {},
        "weight_module": "fastplms.dplm2.get_weights",
    },
    {
        "family": "ankh",
        "repo_ids": [
            "Synthyra/ANKH_base",
            "Synthyra/ANKH_large",
            "Synthyra/ANKH2_large",
            "Synthyra/ANKH3_large",
            "Synthyra/ANKH3_xl",
        ],
        "modeling_src": "fastplms/ankh/modeling_ankh.py",
        "modeling_dest": "modeling_ankh.py",
        "composite": True,
        "include_embedding_mixin": True,
        "extra_files": {},
        "readme_map": {
            "Synthyra/ANKH_base": "fastplms/ankh/README.md",
            "Synthyra/ANKH_large": "fastplms/ankh/README.md",
            "Synthyra/ANKH2_large": "fastplms/ankh/README.md",
            "Synthyra/ANKH3_large": "fastplms/ankh/README.md",
            "Synthyra/ANKH3_xl": "fastplms/ankh/README.md",
        },
        "license_map": {
            "Synthyra/ANKH_base": "fastplms/ankh/ankh_license.txt",
            "Synthyra/ANKH_large": "fastplms/ankh/ankh_license.txt",
            "Synthyra/ANKH2_large": "fastplms/ankh/ankh_license.txt",
            "Synthyra/ANKH3_large": "fastplms/ankh/ankh_license.txt",
            "Synthyra/ANKH3_xl": "fastplms/ankh/ankh_license.txt",
        },
        "weight_module": "fastplms.ankh.get_weights",
    },
    {
        "family": "esmfold",
        "repo_ids": [
            "Synthyra/FastESMFold",
        ],
        "modeling_src": "fastplms/esmfold/modeling_fast_esmfold.py",
        "modeling_dest": "modeling_fast_esmfold.py",
        "composite": True,
        "include_embedding_mixin": False,
        "extra_files": {},
        "readme_map": {
            "Synthyra/FastESMFold": "fastplms/esmfold/README.md",
        },
        "license_map": {},
        "weight_module": "fastplms.esmfold.get_weights",
    },
    {
        "family": "boltz",
        "repo_ids": [
            "Synthyra/Boltz2",
        ],
        "modeling_src": None,
        "modeling_dest": None,
        "composite": False,
        "include_embedding_mixin": False,
        "extra_files": {
            "fastplms/boltz/modeling_boltz2.py": "modeling_boltz2.py",
            "fastplms/boltz/__init__.py": "__init__.py",
            "fastplms/boltz/minimal_featurizer.py": "minimal_featurizer.py",
            "fastplms/boltz/minimal_structures.py": "minimal_structures.py",
            "fastplms/boltz/cif_writer.py": "cif_writer.py",
            "fastplms/boltz/vb_const.py": "vb_const.py",
            "fastplms/boltz/vb_layers_attention.py": "vb_layers_attention.py",
            "fastplms/boltz/vb_layers_attentionv2.py": "vb_layers_attentionv2.py",
            "fastplms/boltz/vb_layers_confidence_utils.py": "vb_layers_confidence_utils.py",
            "fastplms/boltz/vb_layers_dropout.py": "vb_layers_dropout.py",
            "fastplms/boltz/vb_layers_initialize.py": "vb_layers_initialize.py",
            "fastplms/boltz/vb_layers_outer_product_mean.py": "vb_layers_outer_product_mean.py",
            "fastplms/boltz/vb_layers_pair_averaging.py": "vb_layers_pair_averaging.py",
            "fastplms/boltz/vb_layers_pairformer.py": "vb_layers_pairformer.py",
            "fastplms/boltz/vb_layers_transition.py": "vb_layers_transition.py",
            "fastplms/boltz/vb_layers_triangular_mult.py": "vb_layers_triangular_mult.py",
            "fastplms/boltz/vb_loss_diffusionv2.py": "vb_loss_diffusionv2.py",
            "fastplms/boltz/vb_modules_confidencev2.py": "vb_modules_confidencev2.py",
            "fastplms/boltz/vb_modules_diffusion_conditioning.py": "vb_modules_diffusion_conditioning.py",
            "fastplms/boltz/vb_modules_diffusionv2.py": "vb_modules_diffusionv2.py",
            "fastplms/boltz/vb_modules_encodersv2.py": "vb_modules_encodersv2.py",
            "fastplms/boltz/vb_modules_transformersv2.py": "vb_modules_transformersv2.py",
            "fastplms/boltz/vb_modules_trunkv2.py": "vb_modules_trunkv2.py",
            "fastplms/boltz/vb_modules_utils.py": "vb_modules_utils.py",
            "fastplms/boltz/vb_potentials_potentials.py": "vb_potentials_potentials.py",
            "fastplms/boltz/vb_potentials_schedules.py": "vb_potentials_schedules.py",
            "fastplms/boltz/vb_tri_attn_attention.py": "vb_tri_attn_attention.py",
            "fastplms/boltz/vb_tri_attn_primitives.py": "vb_tri_attn_primitives.py",
            "fastplms/boltz/vb_tri_attn_utils.py": "vb_tri_attn_utils.py",
        },
        "readme_map": {
            "Synthyra/Boltz2": "fastplms/boltz/README.md",
        },
        "license_map": {
            "Synthyra/Boltz2": "fastplms/boltz/LICENSE",
        },
        "weight_module": "fastplms.boltz.get_weights",
    },
]


def _run_weight_scripts(
    families: Optional[list], hf_token: Optional[str], skip_weights: bool
) -> None:
    python_cmd = "python" if platform.system().lower() == "linux" else "py"
    for entry in MODEL_REGISTRY:
        if families is not None and entry["family"] not in families:
            continue
        module = entry["weight_module"]
        if module is None:
            continue
        command = [python_cmd, "-m", module]
        if hf_token is not None:
            command.extend(["--hf_token", hf_token])
        if skip_weights:
            command.append("--skip-weights")
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)


def _upload_files(api: HfApi, families: Optional[list]) -> None:
    for entry in MODEL_REGISTRY:
        if families is not None and entry["family"] not in families:
            continue

        # Build composite file if needed
        composite_path = None
        if entry["composite"] and entry["modeling_src"] is not None:
            composite_code = build_composite(
                entry["modeling_src"],
                include_embedding_mixin=entry["include_embedding_mixin"],
            )
            # Verify composite compiles
            compile(composite_code, entry["modeling_dest"], "exec")
            composite_path = os.path.join(tempfile.gettempdir(), entry["modeling_dest"])
            with open(composite_path, "w", encoding="utf-8") as f:
                f.write(composite_code)
            print(f"Built composite: {entry['modeling_dest']} ({len(composite_code)} chars)")

        for repo_id in entry["repo_ids"]:
            print(f"\nUploading to {repo_id}")

            # Upload composite modeling file
            if composite_path is not None:
                api.upload_file(
                    path_or_fileobj=composite_path,
                    path_in_repo=entry["modeling_dest"],
                    repo_id=repo_id,
                    repo_type="model",
                )

            # Upload extra files (Boltz vb_* modules, E1 tokenizer, etc.)
            for local_path, repo_path in entry["extra_files"].items():
                abs_local = str(_REPO_ROOT / local_path)
                api.upload_file(
                    path_or_fileobj=abs_local,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model",
                )

            # Upload license
            license_path = entry["license_map"].get(repo_id)
            if license_path:
                abs_license = str(_REPO_ROOT / license_path)
                if os.path.exists(abs_license):
                    api.upload_file(
                        path_or_fileobj=abs_license,
                        path_in_repo="LICENSE",
                        repo_id=repo_id,
                        repo_type="model",
                    )

            # Upload readme
            readme_path = entry["readme_map"].get(repo_id)
            if readme_path:
                abs_readme = str(_REPO_ROOT / readme_path)
                if os.path.exists(abs_readme):
                    api.upload_file(
                        path_or_fileobj=abs_readme,
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        repo_type="model",
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload FastPLMs models to HuggingFace")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Run weight scripts without downloading/pushing model weights",
    )
    parser.add_argument("--files-only", action="store_true", help="Only upload files, skip weight conversion")
    parser.add_argument("--config-only", action="store_true", help="Only upload config+tokenizer via --skip-weights, skip file uploads")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    if args.config_only:
        _run_weight_scripts(args.families, args.hf_token, skip_weights=True)
    elif not args.files_only:
        _run_weight_scripts(args.families, args.hf_token, args.skip_weights)

    if not args.config_only:
        api = HfApi()
        _upload_files(api, args.families)

    print("\nDone.")
