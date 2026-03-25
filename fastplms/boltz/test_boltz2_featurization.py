"""Featurization parity test: minimal_featurizer vs official Boltz2 pipeline.

Compares feature tensors produced by both pipelines for the same input sequence.
Requires the `boltz` package to be installed for the official pipeline.

Usage:
    python -m fastplms.boltz.test_boltz2_featurization
"""

import sys
from pathlib import Path
from typing import Dict, List

import torch

from .minimal_featurizer import build_boltz2_features
from . import vb_const as const


def _load_official_features(sequence: str) -> Dict[str, torch.Tensor]:
    """Generate features using the official Boltz2 data pipeline."""
    try:
        from boltz.main import check_inputs, process_inputs, BoltzProcessedInput, Manifest
        from boltz.data.module.inferencev2 import BoltzInferenceDataModule
    except ImportError:
        print("ERROR: boltz package not installed. Install with: pip install boltz")
        sys.exit(1)

    import tempfile
    import yaml

    cache = Path("~/.boltz/").expanduser()
    input_yaml = yaml.dump({
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}}
        ],
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        input_path = out_dir / "input.yaml"
        input_path.write_text(input_yaml)

        data = check_inputs(input_path)
        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=cache / "ccd.pkl",
            mol_dir=cache / "mols",
            use_msa_server=False,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
        )

        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=Manifest.load(processed_dir / "manifest.json"),
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
        )

        data_module = BoltzInferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            num_workers=0,
        )

        features_dict = list(data_module.predict_dataloader())[0]

    features = {}
    for k, v in features_dict.items():
        if k == "record":
            continue
        if torch.is_tensor(v):
            features[k] = v.float() if v.is_floating_point() else v
        else:
            features[k] = torch.tensor(v)
    return features


def _compare_features(
    fast_feats: Dict[str, torch.Tensor],
    official_feats: Dict[str, torch.Tensor],
) -> List[Dict]:
    """Compare two feature dicts key-by-key. Returns list of mismatch reports."""
    all_keys = sorted(set(fast_feats.keys()) | set(official_feats.keys()))
    mismatches = []

    for key in all_keys:
        if key not in fast_feats:
            mismatches.append({"key": key, "issue": "MISSING in FastPLMs"})
            continue
        if key not in official_feats:
            mismatches.append({"key": key, "issue": "EXTRA in FastPLMs (not in official)"})
            continue

        f = fast_feats[key]
        o = official_feats[key]

        if f.shape != o.shape:
            mismatches.append({
                "key": key,
                "issue": f"SHAPE MISMATCH: FastPLMs {tuple(f.shape)} vs official {tuple(o.shape)}",
            })
            continue

        if f.dtype != o.dtype:
            mismatches.append({
                "key": key,
                "issue": f"DTYPE MISMATCH: FastPLMs {f.dtype} vs official {o.dtype}",
            })

        if f.is_floating_point():
            max_err = (f.float() - o.float()).abs().max().item()
            mean_err = (f.float() - o.float()).abs().mean().item()
            if max_err > 1e-4:
                mismatches.append({
                    "key": key,
                    "issue": f"VALUE MISMATCH: max_err={max_err:.6f}, mean_err={mean_err:.6f}",
                })
        else:
            n_diff = (f != o).sum().item()
            if n_diff > 0:
                mismatches.append({
                    "key": key,
                    "issue": f"VALUE MISMATCH: {n_diff} differing elements out of {f.numel()}",
                })

    return mismatches


# Features where random augmentation makes exact comparison impossible;
# compare only shape and dtype.
_RANDOM_AUGMENTATION_KEYS = {"ref_pos", "coords", "disto_coords_ensemble"}


def main():
    test_sequence = "AAAAAAAAAAAAAAAAAAAAGGGGGGGGGGLLLLLLLLLLL"
    print(f"Test sequence ({len(test_sequence)} residues): {test_sequence}")
    print()

    print("Building FastPLMs features...")
    fast_feats, _ = build_boltz2_features(test_sequence)
    print(f"  Keys: {len(fast_feats)}")

    print("Building official Boltz2 features...")
    official_feats = _load_official_features(test_sequence)
    print(f"  Keys: {len(official_feats)}")
    print()

    # Compare
    mismatches = _compare_features(fast_feats, official_feats)

    # Filter out expected random augmentation differences
    structural_mismatches = []
    augmentation_mismatches = []
    for m in mismatches:
        if m["key"] in _RANDOM_AUGMENTATION_KEYS and "VALUE MISMATCH" in m["issue"]:
            augmentation_mismatches.append(m)
        else:
            structural_mismatches.append(m)

    if augmentation_mismatches:
        print(f"Expected differences (random augmentation, {len(augmentation_mismatches)} keys):")
        for m in augmentation_mismatches:
            print(f"  {m['key']}: {m['issue']}")
        print()

    if structural_mismatches:
        print(f"STRUCTURAL MISMATCHES ({len(structural_mismatches)}):")
        for m in structural_mismatches:
            print(f"  {m['key']}: {m['issue']}")
        print()
        print("FAIL: Feature parity not achieved.")
        return 1
    else:
        print("PASS: All features match (excluding expected random augmentation).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
