import tempfile
import yaml
import torch
import numpy as np
import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List, Union

from transformers import PreTrainedModel
from basic_boltzgen import Boltz
from boltzgen_config import BoltzGenConfig
from boltzgen_flat.task_predict_data_from_yaml import DataConfig, FromYamlDataModule
from boltzgen_flat.data_tokenize_tokenizer import Tokenizer
from boltzgen_flat.data_feature_featurizer import Featurizer
from boltzgen_flat.data_write_mmcif import to_mmcif
from boltzgen_flat.data_data import Structure
from boltzgen_flat import data_const as const
from boltzgen_flat import cli_boltzgen
import huggingface_hub


class BoltzGen(PreTrainedModel):
    config_class = BoltzGenConfig
    
    def __init__(self, config: BoltzGenConfig):
        super().__init__(config)
        self.config = config
        self.boltz = Boltz(**config.__dict__).eval()
        
        # Initialize tokenizer and featurizer for data processing
        self.tokenizer = Tokenizer()
        self.featurizer = Featurizer()

    def _resolve_artifact_path(self, artifact: str, repo_type: str = "dataset") -> str:
        """
        Resolve artifact path, downloading from HuggingFace if necessary.
        """
        if artifact.startswith("huggingface:"):
            try:
                _, repo_id, filename = artifact.split(":")
            except ValueError:
                raise ValueError(
                    f"Invalid artifact: {artifact}. Expected format: huggingface:<repo_id>:<filename>"
                )
            
            # Use default cache dir and token from env
            result = huggingface_hub.hf_hub_download(
                repo_id,
                filename,
                repo_type=repo_type,
                library_name="boltzgen",
            )
            return str(Path(result))
        else:
            path = Path(artifact)
            if not path.exists():
                raise FileNotFoundError(f"Artifact not found: {path}")
            return str(path)

    @torch.inference_mode()
    def fold_proteins(self, sequences: Dict[str, str], output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Fold proteins from sequences.

        Args:
            sequences: Dictionary mapping chain IDs to sequences.
            output_path: Optional path to save the resulting CIF file.

        Returns:
            Dictionary containing raw model output.
        """
        # Create temporary YAML content
        entities = []
        for chain_id, seq in sequences.items():
            entities.append({
                "protein": {
                    "id": chain_id,
                    "sequence": seq
                }
            })
        
        yaml_content = {"entities": entities}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_yaml:
            yaml.dump(yaml_content, tmp_yaml)
            tmp_yaml_path = tmp_yaml.name

        try:
            # Setup data config
            moldir = getattr(self.config, 'moldir', None)
            if moldir is None:
                moldir = cli_boltzgen.ARTIFACTS["moldir"][0]
            
            moldir = self._resolve_artifact_path(moldir, repo_type="dataset")

            data_config = DataConfig(
                moldir=moldir,
                multiplicity=1,
                yaml_path=tmp_yaml_path,
                tokenizer=self.tokenizer,
                featurizer=self.featurizer,
                design=False, # We are folding, not designing
                compute_affinity=False,
            )

            # Create data module
            dm = FromYamlDataModule(
                cfg=data_config,
                batch_size=1,
                num_workers=0,
                pin_memory=False
            )
            dm.setup(stage="predict")
            dataloader = dm.predict_dataloader()

            # Run inference
            batch = next(iter(dataloader))
            
            # Move batch to device
            device = next(self.parameters()).device
            batch = dm.transfer_batch_to_device(batch, device)
            
            # Run prediction
            if not hasattr(self.boltz, 'predict_args') or self.boltz.predict_args is None:
                self.boltz.predict_args = {
                    "recycling_steps": 3,
                    "sampling_steps": 200,
                    "diffusion_samples": 1
                }
            
            out = self.boltz.predict_step(batch)
            
            if out.get("exception", False):
                raise RuntimeError("BoltzGen prediction failed.")

            if output_path:
                structures = self.output_to_structure(out)
                if structures:
                    self.save_to_cif(structures[0], output_path)

            return out

        finally:
            # Cleanup
            if Path(tmp_yaml_path).exists():
                Path(tmp_yaml_path).unlink()

    @torch.inference_mode()
    def design_proteins(
        self,
        entities: List[Dict[str, Any]],
        constraints: Optional[List[Dict[str, Any]]] = None,
        num_designs: int = 1,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Design proteins based on entities and constraints.

        Args:
            entities: List of entity dictionaries.
            constraints: List of constraint dictionaries.
            num_designs: Number of designs to generate.
            output_dir: Optional directory to save outputs. If None, uses a temporary directory.
            **kwargs: Additional arguments for boltzgen run.

        Returns:
            Dictionary containing raw model output.
        """
        # Construct YAML content
        yaml_content = {"entities": entities}
        if constraints:
            yaml_content["constraints"] = constraints

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_yaml:
            yaml.dump(yaml_content, tmp_yaml)
            tmp_yaml_path = tmp_yaml.name

        try:
            # Setup data config
            moldir = getattr(self.config, 'moldir', None)
            if moldir is None:
                moldir = cli_boltzgen.ARTIFACTS["moldir"][0]
            
            moldir = self._resolve_artifact_path(moldir, repo_type="dataset")

            data_config = DataConfig(
                moldir=moldir,
                multiplicity=1,
                yaml_path=tmp_yaml_path,
                tokenizer=self.tokenizer,
                featurizer=self.featurizer,
                design=True, # Enable design
                compute_affinity=False,
            )

            # Create data module
            dm = FromYamlDataModule(
                cfg=data_config,
                batch_size=1,
                num_workers=0,
                pin_memory=False
            )
            dm.setup(stage="predict")
            dataloader = dm.predict_dataloader()

            # Run inference
            batch = next(iter(dataloader))
            
            # Move batch to device
            device = next(self.parameters()).device
            batch = dm.transfer_batch_to_device(batch, device)
            
            # Update predict_args with num_designs and kwargs
            predict_args = {
                "recycling_steps": 3,
                "sampling_steps": 200,
                "diffusion_samples": num_designs
            }
            predict_args.update(kwargs)
            self.boltz.predict_args = predict_args
            
            out = self.boltz.predict_step(batch)
            
            if out.get("exception", False):
                raise RuntimeError("BoltzGen prediction failed.")
            
            if output_dir:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                structures = self.output_to_structure(out)
                for i, structure in enumerate(structures):
                    self.save_to_cif(structure, str(out_path / f"design_{i}.cif"))

            return out

        finally:
            if Path(tmp_yaml_path).exists():
                Path(tmp_yaml_path).unlink()

    def output_to_structure(self, output: Dict[str, Any]) -> List[Structure]:
        """
        Convert raw model output to a list of Structure objects.

        Args:
            output: Dictionary containing model output (from predict_step).

        Returns:
            List of Structure objects.
        """
        structures = []
        
        # Determine number of samples
        if "sample_atom_coords" in output:
            num_samples = output["sample_atom_coords"].shape[0]
        elif "coords" in output:
             # If coords has shape (samples, L, 3)
             if output["coords"].ndim == 3:
                 num_samples = output["coords"].shape[0]
             else:
                 num_samples = 1
        else:
            num_samples = 1

        for i in range(num_samples):
            struct_feat = {}
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    # Remove batch dim if present (usually dim 0 is batch=1)
                    # But for coords/sample_atom_coords, dim 0 is samples
                    if k in ["sample_atom_coords", "coords"]:
                        # These are handled separately or we need to pick the sample
                        pass
                    elif v.shape[0] == 1: 
                        struct_feat[k] = v[0].cpu()
                    else:
                        struct_feat[k] = v.cpu()
                    
                    # Ensure scalar tensors are converted to python scalars if needed
                    # Structure.from_feat might expect numpy arrays or scalars
                    if isinstance(struct_feat.get(k), torch.Tensor):
                         if struct_feat[k].numel() == 1:
                             # Keep as tensor but squeeze if needed, or let it be
                             pass
                else:
                    struct_feat[k] = v
            
            # Special handling for 'id' which might be a 0-d array/tensor
            if 'id' in struct_feat:
                val = struct_feat['id']
                if isinstance(val, (np.ndarray, np.generic)):
                    if val.size == 1:
                        struct_feat['id'] = str(val.item())
                elif isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        struct_feat['id'] = str(val.item())

            # Handle coords for this sample
            if "sample_atom_coords" in output:
                struct_feat["coords"] = output["sample_atom_coords"][i].cpu()
            elif "coords" in output:
                if output["coords"].ndim == 3:
                    struct_feat["coords"] = output["coords"][i].cpu()
                else:
                    struct_feat["coords"] = output["coords"].cpu()
            
            # Create Structure
            try:
                structure, _, _ = Structure.from_feat(struct_feat)
                
                # Add bfactor if available (plddt)
                if "plddt" in output:
                    plddt = output["plddt"][i].cpu() # (tokens,)
                    # Map to atoms
                    atom_to_token = struct_feat["atom_to_token"].float()
                    plddt_atom = atom_to_token @ plddt.float()
                    structure.atoms["bfactor"] = plddt_atom[struct_feat["atom_pad_mask"].bool()].numpy()
                
                structures.append(structure)
            except Exception as e:
                print(f"Error creating structure for sample {i}: {e}")
        
        return structures

    def save_to_cif(self, structure: Structure, path: str):
        """
        Save a Structure object to a CIF file.

        Args:
            structure: Structure object.
            path: Output file path.
        """
        cif_text = to_mmcif(structure)
        with open(path, "w") as f:
            f.write(cif_text)
