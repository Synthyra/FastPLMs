import argparse
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, login

from boltz_fastplms.modeling_boltz2 import Boltz2Model


BOLTZ2_CKPT_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"


def _download_checkpoint_if_needed(checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        urllib.request.urlretrieve(BOLTZ2_CKPT_URL, str(checkpoint_path))  # noqa: S310
    return checkpoint_path


def _copy_runtime_package(output_dir: Path) -> None:
    source_pkg = Path(__file__).resolve().parent
    runtime_files = [
        "__init__.py",
        "modeling_boltz2.py",
        "minimal_featurizer.py",
        "minimal_structures.py",
        "cif_writer.py",
    ]
    for filename in runtime_files:
        shutil.copyfile(source_pkg / filename, output_dir / filename)
    for flat_module in source_pkg.glob("vb_*.py"):
        shutil.copyfile(flat_module, output_dir / flat_module.name)


if __name__ == "__main__":
    # py -m boltz_fastplms.get_boltz2_weights
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="boltz_fastplms/weights/boltz2_conf.ckpt")
    parser.add_argument("--output_dir", type=str, default="boltz2_automodel_export")
    parser.add_argument("--repo_id", type=str, default="Synthyra/Boltz2")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--use_kernels", action="store_true")
    args = parser.parse_args()

    checkpoint_path = _download_checkpoint_if_needed(Path(args.checkpoint_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Boltz2Model.from_boltz_checkpoint(
        checkpoint_path=str(checkpoint_path),
        use_kernels=args.use_kernels,
    )
    model.config.auto_map = {
        "AutoConfig": "modeling_boltz2.Boltz2Config",
        "AutoModel": "modeling_boltz2.Boltz2Model",
    }
    model.save_pretrained(str(output_dir))
    _copy_runtime_package(output_dir=output_dir)

    if args.repo_id is not None:
        if args.token is not None:
            login(token=args.token)
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.repo_id,
            repo_type="model",
        )
