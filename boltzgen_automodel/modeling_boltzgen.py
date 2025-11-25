import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
import tempfile
import yaml
import torch
import numpy as np
import huggingface_hub
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


# Default design specification - a simple protein design (80-140 residues)
DEFAULT_DESIGN_ENTITIES = [
    {
        "protein": {
            "id": "A",
            "sequence": "80..140"  # Design between 80-140 residues
        }
    }
]


# MSA special values
MSA_AUTO = 0      # Auto-generate MSA (not actually supported in simplified implementation)
MSA_NONE = -1     # No MSA / single-sequence mode
MSA_EMPTY = "empty"  # Alias for no MSA


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

    def _get_moldir(self) -> str:
        """Get the moldir path, downloading if necessary."""
        moldir = getattr(self.config, 'moldir', None)
        if moldir is None:
            moldir = cli_boltzgen.ARTIFACTS["moldir"][0]
        return self._resolve_artifact_path(moldir, repo_type="dataset")

    def _run_prediction(
        self,
        yaml_content: Dict[str, Any],
        design: bool = True,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        num_designs: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Internal method to run prediction from YAML content.
        
        Args:
            yaml_content: Dictionary with 'entities' and optional 'constraints'.
            design: Whether this is a design task.
            recycling_steps: Number of recycling steps.
            sampling_steps: Number of sampling steps.
            diffusion_samples: Number of diffusion samples to generate.
            num_designs: Number of designs to generate.
            **kwargs: Additional predict args.
            
        Returns:
            Raw model output dictionary.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_yaml:
            yaml.dump(yaml_content, tmp_yaml)
            tmp_yaml_path = tmp_yaml.name

        try:
            moldir = self._get_moldir()
            
            data_config = DataConfig(
                moldir=moldir,
                multiplicity=1,
                yaml_path=tmp_yaml_path,
                tokenizer=self.tokenizer,
                featurizer=self.featurizer,
                design=design,
                compute_affinity=False,
            )

            dm = FromYamlDataModule(
                cfg=data_config,
                batch_size=1,
                num_workers=0,
                pin_memory=False
            )
            dm.setup(stage="predict")
            dataloader = dm.predict_dataloader()

            batch = next(iter(dataloader))
            
            device = next(self.parameters()).device
            batch = dm.transfer_batch_to_device(batch, device)
            
            # Store batch reference for structure conversion
            self._last_batch = batch
            
            predict_args = {
                "recycling_steps": recycling_steps,
                "sampling_steps": sampling_steps,
                "diffusion_samples": diffusion_samples
            }
            predict_args.update(kwargs)
            self.boltz.predict_args = predict_args
            
            out = self.boltz.predict_step(batch)
            
            if out.get("exception", False):
                raise RuntimeError("BoltzGen prediction failed.")

            return out

        finally:
            if Path(tmp_yaml_path).exists():
                Path(tmp_yaml_path).unlink()

    @torch.inference_mode()
    def fold_proteins(
        self, 
        sequences: Optional[Dict[str, str]] = None,
        msa_paths: Optional[Dict[str, Union[str, int]]] = None,
        output_path: Optional[str] = None,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fold proteins from sequences.

        Args:
            sequences: Dictionary mapping chain IDs to sequences.
                       If None, uses a default short sequence for testing.
            msa_paths: Optional dictionary mapping chain IDs to MSA file paths or special values:
                       - A file path (str): Path to an MSA file (a3m format)
                       - 0 or MSA_AUTO: Signal for auto-generation (not implemented in simplified version)
                       - -1 or MSA_NONE: No MSA / single-sequence mode (default)
                       - "empty": Alias for no MSA
                       If None, all chains use single-sequence mode.
            output_path: Optional path to save the resulting CIF file.
            recycling_steps: Number of recycling steps.
            sampling_steps: Number of sampling steps.
            diffusion_samples: Number of diffusion samples to generate.
            **kwargs: Additional arguments for boltzgen prediction.
            
        Returns:
            Dictionary containing raw model output.
            
        Example usage:
            # Simple folding without MSA
            output = model.fold_proteins(sequences={"A": "MKTAYIAKQRQISFVK"})
            
            # Folding with MSA file
            output = model.fold_proteins(
                sequences={"A": "MKTAYIAKQRQISFVK"},
                msa_paths={"A": "/path/to/chain_a.a3m"}
            )
        """
        # Default sequence for testing
        if sequences is None:
            sequences = {"A": "MKTAYIAKQRQISFVK"}
        
        # Create YAML content
        entities = []
        for chain_id, seq in sequences.items():
            entity = {
                "protein": {
                    "id": chain_id,
                    "sequence": seq
                }
            }
            
            # Add MSA specification if provided
            if msa_paths is not None and chain_id in msa_paths:
                msa_value = msa_paths[chain_id]
                # Normalize MSA value
                if msa_value == "empty":
                    msa_value = -1
                entity["protein"]["msa"] = msa_value
            
            entities.append(entity)
        
        yaml_content = {"entities": entities}
        
        out = self._run_prediction(
            yaml_content=yaml_content,
            design=False,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            **kwargs
        )

        if output_path:
            structures = self.output_to_structure(out)
            if structures:
                self.save_to_cif(structures[0], output_path)

        return out

    @torch.inference_mode()
    def design_proteins(
        self,
        entities: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        msa_paths: Optional[Dict[str, Union[str, int]]] = None,
        num_designs: int = 1,
        diffusion_batch_size: Optional[int] = None,
        output_dir: Optional[str] = None,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Design proteins based on entities and constraints.

        Args:
            entities: List of entity dictionaries. If None, uses a default design
                     specification (protein with 80-140 designed residues).
            constraints: Optional list of constraint dictionaries (bonds, total_len, etc).
            msa_paths: Optional dictionary mapping chain IDs to MSA file paths or special values:
                       - A file path (str): Path to an MSA file (a3m format)
                       - 0 or MSA_AUTO: Signal for auto-generation (not implemented)
                       - -1 or MSA_NONE: No MSA / single-sequence mode (default for design)
                       - "empty": Alias for no MSA
                       Note: Designed chains automatically use single-sequence mode.
            num_designs: Total number of designs to generate. The implementation batches
                        these efficiently using diffusion_batch_size.
            diffusion_batch_size: Number of diffusion samples to generate per model run.
                                 If None, defaults to 1 if num_designs < 100, else 10.
                                 Note: All designs in the same batch share random parameters
                                 (e.g., sampled length for variable-length designs).
            output_dir: Optional directory to save outputs.
            recycling_steps: Number of recycling steps.
            sampling_steps: Number of sampling steps.
            **kwargs: Additional arguments for boltzgen prediction.

        Returns:
            Dictionary containing raw model output with all designs.
            The 'coords' key will have shape (num_designs, num_atoms, 3).
            
        Example usage:
            # Design with defaults (80-140 residue protein)
            output = model.design_proteins()
            
            # Design a specific protein with 10 designs
            output = model.design_proteins(
                entities=[{"protein": {"id": "A", "sequence": "50..100"}}],
                num_designs=10
            )
            
            # Design with target structure and MSA for target
            output = model.design_proteins(
                entities=[
                    {"protein": {"id": "B", "sequence": "80..140"}},
                    {"file": {"path": "target.cif", "include": [{"chain": {"id": "A"}}]}}
                ],
                msa_paths={"A": "/path/to/target_msa.a3m"},
                num_designs=100
            )
        """
        # Use default design entities if none provided
        if entities is None:
            entities = DEFAULT_DESIGN_ENTITIES
        
        # Calculate batching parameters (following boltzgen CLI logic)
        if diffusion_batch_size is None:
            diffusion_batch_size = 1 if num_designs < 100 else 10
        
        num_batches = math.ceil(num_designs / diffusion_batch_size)
        
        # Apply MSA specifications to entities
        if msa_paths is not None:
            entities = self._apply_msa_to_entities(entities, msa_paths)
        
        # Construct YAML content
        yaml_content = {"entities": entities}
        if constraints:
            yaml_content["constraints"] = constraints

        # Collect all outputs across batches
        all_outputs = []
        
        for batch_idx in range(num_batches):
            # Calculate how many samples for this batch
            remaining = num_designs - batch_idx * diffusion_batch_size
            batch_samples = min(diffusion_batch_size, remaining)
            
            out = self._run_prediction(
                yaml_content=yaml_content,
                design=True,
                diffusion_samples=batch_samples,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps,
                **kwargs
            )
            all_outputs.append(out)
        
        # Merge outputs from all batches
        merged_output = self._merge_batch_outputs(all_outputs)
        
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            structures = self.output_to_structure(merged_output)
            for i, structure in enumerate(structures):
                self.save_to_cif(structure, str(out_path / f"design_{i}.cif"))

        return merged_output
    
    def _apply_msa_to_entities(
        self, 
        entities: List[Dict[str, Any]], 
        msa_paths: Dict[str, Union[str, int]]
    ) -> List[Dict[str, Any]]:
        """
        Apply MSA specifications to entities.
        
        Args:
            entities: List of entity dictionaries.
            msa_paths: Dictionary mapping chain IDs to MSA paths or special values.
            
        Returns:
            Modified entities list with MSA specifications added.
        """
        import copy
        modified_entities = copy.deepcopy(entities)
        
        for entity in modified_entities:
            entity_type = next(iter(entity.keys())).lower()
            
            if entity_type == "protein":
                chain_id = entity["protein"].get("id")
                if chain_id and chain_id in msa_paths:
                    msa_value = msa_paths[chain_id]
                    # Normalize MSA value
                    if msa_value == "empty":
                        msa_value = -1
                    entity["protein"]["msa"] = msa_value
            elif entity_type == "file":
                # Handle file entities - MSAs can be specified per chain in include
                include = entity["file"].get("include", [])
                if isinstance(include, list):
                    for item in include:
                        if "chain" in item:
                            chain_id = item["chain"].get("id")
                            if chain_id and chain_id in msa_paths:
                                msa_value = msa_paths[chain_id]
                                if msa_value == "empty":
                                    msa_value = -1
                                item["chain"]["msa"] = msa_value
        
        return modified_entities
    
    def _merge_batch_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge outputs from multiple batches into a single output dictionary.
        
        Args:
            outputs: List of output dictionaries from separate batches.
            
        Returns:
            Merged output dictionary.
        """
        if len(outputs) == 1:
            return outputs[0]
        
        merged = {}
        
        # Keys that should be concatenated along the sample dimension
        concat_keys = {"coords", "plddt", "pae", "ptm", "iptm", "confidence_score",
                      "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
                      "ligand_iptm", "protein_iptm"}
        
        for key in outputs[0].keys():
            values = [out[key] for out in outputs if key in out]
            
            if not values:
                continue
            
            first_val = values[0]
            
            if key in concat_keys and isinstance(first_val, torch.Tensor):
                # Concatenate along sample dimension (dim 0 for most metrics)
                if first_val.ndim >= 1:
                    merged[key] = torch.cat(values, dim=0)
                else:
                    merged[key] = torch.stack(values)
            elif isinstance(first_val, torch.Tensor):
                # For non-concatenated tensors, just use first batch's value
                # (these are typically shared features like atom masks)
                merged[key] = first_val
            elif isinstance(first_val, list):
                # Extend lists
                merged[key] = []
                for v in values:
                    if isinstance(v, list):
                        merged[key].extend(v)
                    else:
                        merged[key].append(v)
            else:
                # For other types, use first value
                merged[key] = first_val
        
        return merged

    def output_to_structure(
        self, 
        output: Dict[str, Any],
        sample_indices: Optional[List[int]] = None
    ) -> List[Structure]:
        """
        Convert raw model output to a list of Structure objects.

        Args:
            output: Dictionary containing model output (from predict_step).
            sample_indices: Optional list of sample indices to convert. 
                           If None, converts all samples.

        Returns:
            List of Structure objects.
        """
        structures = []
        
        # Determine number of samples from coords shape
        coords = output.get("coords")
        if coords is None:
            raise ValueError("Output must contain 'coords' key")
        
        if coords.ndim == 3:
            num_samples = coords.shape[0]
        else:
            num_samples = 1
            
        if sample_indices is None:
            sample_indices = list(range(num_samples))

        # Keys that are NOT stacked during collation (remain as lists)
        # These need special handling - take [0] to get the actual value
        list_keys = {
            "all_coords", "all_resolved_mask", "crop_to_all_atom_map",
            "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries",
            "activity_name", "activity_qualifier", "sid", "cid", "aid",
            "normalized_protein_accession", "pair_id", "record", "id",
            "structure", "tokenized", "structure_bonds", "extra_mols", "data_sample_idx"
        }

        for i in sample_indices:
            try:
                # Prepare feature dict for Structure.from_feat
                # Following the pattern from writer.py - take [0] for batch dim
                struct_feat = {}
                
                for k, v in output.items():
                    if k == "coords":
                        # For coords, select the sample
                        if coords.ndim == 3:
                            struct_feat[k] = coords[i].cpu()
                        else:
                            struct_feat[k] = coords.cpu()
                    elif k == "structure_bonds":
                        # structure_bonds is a list [array] from collation
                        # Need to extract the numpy structured array
                        if isinstance(v, list):
                            struct_feat[k] = v[0] if v else np.array([], dtype=const.bond_dtype if hasattr(const, 'bond_dtype') else object)
                        elif isinstance(v, torch.Tensor):
                            struct_feat[k] = v.cpu().numpy()
                        else:
                            struct_feat[k] = v
                    elif k == "id":
                        # id is typically a list from collation
                        if isinstance(v, list):
                            struct_feat[k] = str(v[0]) if v else "structure"
                        else:
                            struct_feat[k] = str(v)
                    elif k == "extra_mols":
                        # extra_mols is a list from collation
                        if isinstance(v, list):
                            struct_feat[k] = v[0] if v else None
                        else:
                            struct_feat[k] = v
                    elif k in list_keys:
                        # Other list keys from collation - take first element
                        if isinstance(v, list) and len(v) > 0:
                            struct_feat[k] = v[0]
                        else:
                            struct_feat[k] = v
                    elif isinstance(v, torch.Tensor):
                        # For tensors, remove batch dimension if present
                        if v.ndim > 0 and v.shape[0] == 1:
                            struct_feat[k] = v[0].cpu()
                        else:
                            struct_feat[k] = v.cpu()
                    elif isinstance(v, np.ndarray):
                        struct_feat[k] = v
                    elif isinstance(v, list):
                        # Unknown lists - take first element if batch size 1
                        if len(v) == 1:
                            struct_feat[k] = v[0]
                        else:
                            struct_feat[k] = v
                    else:
                        struct_feat[k] = v
                
                # Ensure structure_bonds is proper numpy structured array
                if 'structure_bonds' in struct_feat:
                    bonds = struct_feat['structure_bonds']
                    if isinstance(bonds, torch.Tensor):
                        struct_feat['structure_bonds'] = bonds.cpu().numpy()
                    elif bonds is None:
                        # Create empty bonds array with proper dtype
                        from boltzgen_flat.data_data import Bond
                        struct_feat['structure_bonds'] = np.array([], dtype=Bond)
                
                # Create Structure using from_feat
                structure, designed_atoms, designed_residues = Structure.from_feat(struct_feat)
                
                # Add bfactor from plddt if available
                if "plddt" in output:
                    plddt = output["plddt"]
                    if plddt.ndim > 1:
                        plddt_sample = plddt[i].cpu() if i < plddt.shape[0] else plddt[0].cpu()
                    else:
                        plddt_sample = plddt.cpu()
                    
                    # Map token-level plddt to atoms
                    atom_to_token = struct_feat["atom_to_token"]
                    if isinstance(atom_to_token, torch.Tensor):
                        atom_to_token = atom_to_token.float()
                    else:
                        atom_to_token = torch.from_numpy(atom_to_token).float()
                    
                    plddt_atom = atom_to_token @ plddt_sample.float()
                    atom_pad_mask = struct_feat["atom_pad_mask"]
                    if isinstance(atom_pad_mask, torch.Tensor):
                        atom_pad_mask = atom_pad_mask.bool()
                    else:
                        atom_pad_mask = torch.from_numpy(atom_pad_mask).bool()
                    
                    structure.atoms["bfactor"] = plddt_atom[atom_pad_mask].numpy()
                
                structures.append(structure)
                
            except Exception as e:
                import traceback
                print(f"Error creating structure for sample {i}: {e}")
                traceback.print_exc()
        
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

    def get_confidence_scores(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract confidence scores from model output.
        
        Args:
            output: Dictionary containing model output.
            
        Returns:
            Dictionary with confidence metrics.
        """
        confidence = {}
        
        confidence_keys = [
            "plddt", "ptm", "iptm", "pae", 
            "confidence_score", "complex_plddt", "complex_iplddt",
            "complex_pde", "complex_ipde", "ligand_iptm", "protein_iptm"
        ]
        
        for key in confidence_keys:
            if key in output:
                val = output[key]
                if isinstance(val, torch.Tensor):
                    confidence[key] = val.cpu().numpy()
                else:
                    confidence[key] = val
        
        return confidence

    def select_best_design(
        self, 
        output: Dict[str, Any],
        metric: str = "confidence_score"
    ) -> int:
        """
        Select the best design based on a confidence metric.
        
        Args:
            output: Model output dictionary.
            metric: Metric to use for selection. Options: 'confidence_score', 
                   'iptm', 'ptm', 'complex_plddt'.
                   
        Returns:
            Index of the best sample.
        """
        if metric not in output:
            # Fall back to weighted combination of iptm and ptm
            if "iptm" in output and "ptm" in output:
                scores = 0.8 * output["iptm"].cpu().numpy() + 0.2 * output["ptm"].cpu().numpy()
            elif "ptm" in output:
                scores = output["ptm"].cpu().numpy()
            else:
                return 0  # No confidence metrics available
        else:
            scores = output[metric]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
        
        return int(np.argmax(scores))


if __name__ == "__main__":
    # py -m modeling_boltzgen
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--design", action="store_true")
    args = parser.parse_args()

    recycling_steps = 3
    sampling_steps = 200
    diffusion_samples = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BoltzGen.from_pretrained('Synthyra/boltzgen').eval().to(device)


    sequence = "MTLRCLEPSGNGGEGTRSQWGTAGSAEEPSPQAARLAKALRELGQTGWYWGSMTVNEAKEKLKEAPEGTFLIRDSSHSDYLLTISVKTSAGPTNLRIEYQDGKFRLDSIICVKSKLKQFDSVVHLIDYYVQMCKDKRTGPEAPRNGTVHLYLTKPLYTSAPSLQHLCRLTINKCTGAIWGLPLPTRLKDYLEEYKFQV"


    if args.design:
        num_designs = 10
        diffusion_batch_size = 1
        output_dir = "design_proteins"
        entities = [{"protein": {"id": "A", "sequence": sequence}}]

        output = model.design_proteins(
            entities=entities,
            num_designs=num_designs,
            diffusion_batch_size=diffusion_batch_size,
            output_dir=output_dir,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
        )

    else:
        output = model.fold_proteins(
            sequences={"A": sequence},
            recycling_steps=recycling_steps, 
            sampling_steps=sampling_steps,

        )

        for k, v in output.items():
            try:
                print(f"{k}: {v.shape}")
            except:
                print(f"{k}: {type(v)}")

        print(output['plddt'])
        pae = output['pae']
        plt.imshow(pae.cpu().numpy().squeeze())
        plt.colorbar()
        plt.show()
