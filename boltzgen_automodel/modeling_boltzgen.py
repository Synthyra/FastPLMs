import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Dict, Optional, List, Any, Union
import numpy as np
from pathlib import Path
import huggingface_hub

from .basic_boltzgen import Boltz
from .boltzgen_config import BoltzGenConfig
from .boltzgen_flat.data_tokenize_tokenizer import Tokenizer
from .boltzgen_flat.data_feature_featurizer import Featurizer
from .boltzgen_flat.data_data import Input, Structure
from .boltzgen_flat.data_mol import load_canonicals
from .boltzgen_flat.data_template_features import load_dummy_templates
from .boltzgen_flat import data_const as const
from .boltzgen_flat.data_parse_mmcif import parse_mmcif


"""
from boltzgen_automodel.modeling_boltzgen import BoltzGen
from boltzgen_automodel.boltzgen_config import BoltzGenConfig
# Initialize
config = BoltzGenConfig()
model = BoltzGen(config)
# Fold
sequences = {"MKTAYIAKQRQISFVK": 1}
output = model.fold_proteins(sequences)
model.save_to_cif(output, "prediction.cif", sequence="MKTAYIAKQRQISFVK")
# Design
design_output = model.design_proteins(design_length=20)
"""



class BoltzGen(PreTrainedModel):
    config_class = BoltzGenConfig
    
    def __init__(self, config: BoltzGenConfig):
        super().__init__(config)
        self.config = config
        self.boltz = Boltz(**config.__dict__)
        
        # Initialize tokenizer and featurizer lazily or here
        # We'll initialize them in the methods to avoid pickling issues if any
        
    def fold_proteins(
        self,
        sequences: Dict[str, int],
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        diffusion_samples: int = 1,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fold multiple proteins with specified copy counts.
        
        Args:
            sequences: Dictionary mapping sequences to copy counts
                      e.g., {"MKTAYIAKQRQISFVK": 2, "SHFSRQLE": 1}
            recycling_steps: Number of recycling iterations (default: 3)
            sampling_steps: Number of diffusion sampling steps (default: 50)
            diffusion_samples: Number of samples per protein (default: 1)
            device: Device to run on (default: auto-detect CUDA)
            
        Returns:
            Dictionary containing the model output.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        self.eval()

        # Expand sequences based on counts
        expanded_sequences = []
        for seq, count in sequences.items():
            for _ in range(count):
                expanded_sequences.append(seq)
        
        print(f"Folding complex with {len(expanded_sequences)} chains...")
        
        # Create features
        tokenizer = Tokenizer()
        featurizer = Featurizer()
        features = self._create_input_features(expanded_sequences, tokenizer, featurizer)
        
        # Move to device and add batch dimension
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in features.items()}
        
        # Run inference
        with torch.no_grad():
            # Apply masker
            feat_masked = self.boltz.masker(batch)
            
            # Forward pass
            output = self.boltz(
                feat_masked,
                recycling_steps=recycling_steps,
                num_sampling_steps=sampling_steps,
                diffusion_samples=diffusion_samples,
                run_confidence_sequentially=True,
            )
            
        return output

    def design_proteins(
        self,
        design_length: int,
        target_cif_path: Optional[str] = None,
        secondary_structure: Optional[str] = None,
        recycling_steps: int = 3,
        sampling_steps: int = 100,
        step_scale: float = 1.8,
        noise_scale: float = 0.95,
        diffusion_samples: int = 1,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Design a protein de novo or against a target structure.
        
        Args:
            design_length: Number of residues to design
            target_cif_path: Optional path to target structure CIF file
            secondary_structure: Optional secondary structure string (H/E/L)
            recycling_steps: Number of recycling iterations
            sampling_steps: Number of diffusion steps
            step_scale: Diffusion step scale
            noise_scale: Diffusion noise scale
            diffusion_samples: Number of samples
            device: Device to run on
            
        Returns:
            Dictionary with designed coordinates and metrics
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.to(device)
        self.eval()
        
        print(f"Designing protein: {design_length} residues...")
        
        tokenizer = Tokenizer()
        featurizer = Featurizer()
        
        # Create design features
        features = self._create_design_input_features(
            design_length=design_length,
            target_cif_path=target_cif_path,
            secondary_structure=secondary_structure,
            tokenizer=tokenizer,
            featurizer=featurizer,
        )
        
        # Move to device and add batch dimension
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in features.items()}
        
        # Run inference
        with torch.no_grad():
            # Apply masker
            feat_masked = self.boltz.masker(batch)
            
            output = self.boltz(
                feat_masked,
                recycling_steps=recycling_steps,
                num_sampling_steps=sampling_steps,
                diffusion_samples=diffusion_samples,
                run_confidence_sequentially=True,
                step_scale=step_scale,
                noise_scale=noise_scale,
            )
            
        return output

    def save_to_cif(
        self,
        output: Dict,
        filepath: str,
        sequence: Optional[str] = None,
        chain_id: str = "A",
        include_metrics: bool = True,
    ) -> None:
        """
        Save prediction to CIF file with metrics.
        """
        # Extract coordinates
        coords = output['sample_atom_coords'][0].cpu().numpy()  # [num_atoms, 3]
        
        # Extract or decode sequence
        if sequence is None:
            # Try to extract from res_type_logits if available
            if 'res_type_logits' in output:
                res_types = output['res_type_logits'].argmax(dim=-1)[0].cpu().numpy()
                # Convert to sequence using token mapping
                canonical_tokens = const.canonical_tokens
                sequence = ''.join([const.prot_token_to_letter.get(canonical_tokens[rt], 'X') 
                                   for rt in res_types if rt < len(canonical_tokens)])
            else:
                # If we don't have logits (folding mode), we might not have the sequence easily accessible 
                # unless passed in. For now, we'll use 'X' or try to infer.
                # In folding mode, the sequence is known by the user.
                sequence = 'X' * (len(coords) // 4)
        
        num_residues = len(sequence)
        
        with open(filepath, 'w') as f:
            f.write("data_BOLTZGEN_PREDICTION\n")
            f.write("#\n")
            f.write("_entry.id BOLTZGEN_PREDICTION\n")
            f.write("#\n")
            
            f.write("loop_\n")
            f.write("_atom_site.group_PDB\n")
            f.write("_atom_site.id\n")
            f.write("_atom_site.type_symbol\n")
            f.write("_atom_site.label_atom_id\n")
            f.write("_atom_site.label_comp_id\n")
            f.write("_atom_site.label_asym_id\n")
            f.write("_atom_site.label_seq_id\n")
            f.write("_atom_site.Cartn_x\n")
            f.write("_atom_site.Cartn_y\n")
            f.write("_atom_site.Cartn_z\n")
            f.write("_atom_site.B_iso_or_equiv\n")
            
            atom_names = ['N', 'CA', 'C', 'O']
            elements = ['N', 'C', 'C', 'O']
            atom_id = 1
            
            plddt_values = None
            if include_metrics and 'plddt' in output:
                plddt_values = output['plddt'][0].cpu().numpy()
            
            for res_idx, aa in enumerate(sequence):
                res_name = const.prot_letter_to_token.get(aa, 'UNK')
                
                for atom_idx, (atom_name, element) in enumerate(zip(atom_names, elements)):
                    coord_idx = res_idx * 4 + atom_idx
                    if coord_idx >= len(coords):
                        break
                        
                    x, y, z = coords[coord_idx]
                    
                    if plddt_values is not None and res_idx < len(plddt_values):
                        b_factor = plddt_values[res_idx]
                    else:
                        b_factor = 100.0
                    
                    f.write(f"ATOM {atom_id} {element} {atom_name} {res_name} {chain_id} "
                           f"{res_idx + 1} {x:.3f} {y:.3f} {z:.3f} {b_factor:.2f}\n")
                    atom_id += 1
            
            if include_metrics:
                f.write("#\n")
                if 'plddt' in output:
                    f.write("loop_\n")
                    f.write("_pdbx_model_confidence.residue_id\n")
                    f.write("_pdbx_model_confidence.plddt\n")
                    plddt = output['plddt'][0].cpu().numpy()
                    for i, score in enumerate(plddt[:num_residues]):
                        f.write(f"{i + 1} {score:.2f}\n")
                    f.write("#\n")
                
                if 'ptm' in output:
                    ptm = output['ptm'].item()
                    f.write(f"_pdbx_model_confidence.ptm {ptm:.3f}\n")
                
                if 'iptm' in output:
                    iptm = output['iptm'].item()
                    f.write(f"_pdbx_model_confidence.iptm {iptm:.3f}\n")

    # Helper methods
    
    def _create_protein_tokens(self, sequence: str, chain_id: int = 0, entity_id: int = 0) -> np.ndarray:
        seq_len = len(sequence)
        dtype = [
            ('res_idx', 'i4'), ('asym_id', 'i4'), ('entity_id', 'i4'), ('sym_id', 'i4'),
            ('mol_type', 'i4'), ('res_type', 'i4'), ('modified', 'i4'), ('ccd', 'i4'),
            ('binding_type', 'i4'), ('structure_group', 'i4'), ('feature_res_idx', 'i4'),
            ('feature_asym_id', 'i4'), ('center_coords', 'f4', (3,)), ('is_standard', 'bool'),
            ('design_mask', 'bool'), ('target_msa_mask', 'bool'), ('design_ss_mask', 'bool'),
            ('resolved_mask', 'bool'), ('disto_mask', 'bool'), ('token_idx', 'i4'),
            ('atom_idx', 'i4'), ('atom_num', 'i4'), ('center_idx', 'i4'), ('disto_idx', 'i4'),
            ('res_name', 'U3'), ('cyclic_period', 'i4'),
        ]
        dtype = np.dtype(dtype, align=True)
        tokens = np.zeros(seq_len, dtype=dtype)
        tokens['token_idx'] = np.arange(seq_len)
        
        for i, aa in enumerate(sequence):
            tokens[i]['res_type'] = const.token_ids.get(const.prot_letter_to_token.get(aa, 'UNK'), const.token_ids['UNK'])
            tokens[i]['res_name'] = const.prot_letter_to_token.get(aa, 'UNK')
            tokens[i]['feature_asym_id'] = chain_id
            tokens[i]['atom_idx'] = i * 4
            tokens[i]['atom_num'] = 4
            tokens[i]['center_idx'] = i * 4 + 1
            tokens[i]['disto_idx'] = i * 4 + 1
            tokens[i]['cyclic_period'] = 0
        return tokens

    def _create_dummy_structure(self, num_tokens: int) -> Structure:
        num_atoms = num_tokens * 4
        ensemble = np.zeros(1, dtype=[('atom_coord_idx', 'i4', (2,))])
        ensemble[0]['atom_coord_idx'] = [0, num_atoms]
        coords = np.zeros(num_atoms, dtype=[('coords', 'f4', (3,))])
        
        atoms = np.zeros(num_atoms, dtype=[
            ('res_idx', 'i4'), ('name', 'U4'), ('element', 'U2'), ('charge', 'i4'),
            ('conformer', 'i4'), ('chirality', 'i4'), ('ref_space_uid', 'i4'),
            ('bfactor', 'f4'), ('is_present', 'bool'), ('plddt', 'f4'), ('coords', 'f4', (3,)),
        ])
        
        for i in range(num_atoms):
            token_idx = i // 4
            atom_idx = i % 4
            atom_names = ['N', 'CA', 'C', 'O']
            elements = ['N', 'C', 'C', 'O']
            atoms[i]['res_idx'] = token_idx
            atoms[i]['name'] = atom_names[atom_idx]
            atoms[i]['element'] = elements[atom_idx]
            atoms[i]['chirality'] = const.chirality_type_ids['CHI_UNSPECIFIED']
            atoms[i]['ref_space_uid'] = token_idx
            atoms[i]['bfactor'] = 100.0
            atoms[i]['is_present'] = True
            atoms[i]['plddt'] = 100.0
            
        bonds = np.zeros(0, dtype=[('atom1', 'i4'), ('atom2', 'i4'), ('bond_type', 'i4')])
        
        residues = np.zeros(num_tokens, dtype=[
            ('name', 'U8'), ('label_asym_id', 'U8'), ('label_seq_id', 'i4'),
            ('auth_asym_id', 'U8'), ('auth_seq_id', 'i4'), ('pdbx_PDB_ins_code', 'U8'),
            ('atom_idx', 'i4'), ('atom_num', 'i4'), ('res_type', 'i4'),
        ])
        
        for i in range(num_tokens):
            residues[i]['name'] = 'GLY'
            residues[i]['label_asym_id'] = 'A'
            residues[i]['label_seq_id'] = i + 1
            residues[i]['auth_asym_id'] = 'A'
            residues[i]['auth_seq_id'] = i + 1
            residues[i]['atom_idx'] = i * 4
            residues[i]['atom_num'] = 4
            residues[i]['res_type'] = const.token_ids['GLY']
            
        chains = np.zeros(1, dtype=[
            ('mol_type', 'i4'), ('label_asym_id', 'U8'), ('auth_asym_id', 'U8'),
            ('entity_id', 'i4'), ('asym_id', 'i4'), ('name', 'U4'),
            ('res_idx', 'i4'), ('res_num', 'i4'), ('cyclic_period', 'i4'),
        ])
        chains[0]['mol_type'] = const.chain_type_ids['PROTEIN']
        chains[0]['label_asym_id'] = 'A'
        chains[0]['auth_asym_id'] = 'A'
        chains[0]['name'] = 'A'
        chains[0]['res_num'] = num_tokens
        
        interfaces = np.zeros(0, dtype=[('chain1', 'i4'), ('chain2', 'i4')])
        mask = np.ones(1, dtype=bool)
        
        return Structure(atoms=atoms, bonds=bonds, residues=residues, chains=chains,
                         interfaces=interfaces, mask=mask, coords=coords, ensemble=ensemble)

    def _create_input_features(self, sequences: List[str], tokenizer: Tokenizer, featurizer: Featurizer) -> Dict[str, torch.Tensor]:
        all_tokens = []
        for chain_idx, seq in enumerate(sequences):
            tokens = self._create_protein_tokens(seq, chain_id=chain_idx, entity_id=chain_idx)
            all_tokens.append(tokens)
        
        tokens = np.concatenate(all_tokens)
        num_tokens = len(tokens)
        bonds = np.zeros(0, dtype=[('token_1', 'i4'), ('token_2', 'i4'), ('type', 'i4')])
        token_to_res = np.arange(num_tokens, dtype=np.int64)
        structure = self._create_dummy_structure(num_tokens)
        
        input_data = Input(
            tokens=tokens, bonds=bonds, token_to_res=token_to_res,
            structure=structure, msa={}, templates=None,
        )
        
        # Download moldir if needed
        try:
            moldir_path = huggingface_hub.hf_hub_download(
                "boltzgen/inference-data", "mols.zip", repo_type="dataset", library_name="boltzgen"
            )
            molecules = load_canonicals(moldir=Path(moldir_path))
        except Exception as e:
            print(f"Warning: Could not load molecules: {e}")
            molecules = {}

        features = featurizer.process(
            input_data, molecules=molecules, random=np.random.default_rng(42),
            training=False, max_seqs=1, backbone_only=False, atom14=False, atom37=False,
            design=False, override_method="X-RAY DIFFRACTION", disulfide_prob=1.0, disulfide_on=True,
        )
        
        templates_features = load_dummy_templates(tdim=1, num_tokens=num_tokens)
        features.update(templates_features)
        features["idx_dataset"] = torch.tensor(1)
        features["chain_design_mask"] = torch.zeros(num_tokens, dtype=torch.bool)
        
        return features

    def _create_design_tokens(self, design_length: int, chain_id: int = 0, entity_id: int = 0, secondary_structure: Optional[str] = None) -> np.ndarray:
        dtype = [
            ('res_idx', 'i4'), ('asym_id', 'i4'), ('entity_id', 'i4'), ('sym_id', 'i4'),
            ('mol_type', 'i4'), ('res_type', 'i4'), ('modified', 'i4'), ('ccd', 'i4'),
            ('binding_type', 'i4'), ('structure_group', 'i4'), ('feature_res_idx', 'i4'),
            ('feature_asym_id', 'i4'), ('center_coords', 'f4', (3,)), ('is_standard', 'bool'),
            ('design_mask', 'bool'), ('target_msa_mask', 'bool'), ('design_ss_mask', 'bool'),
            ('resolved_mask', 'bool'), ('disto_mask', 'bool'), ('token_idx', 'i4'),
            ('atom_idx', 'i4'), ('atom_num', 'i4'), ('center_idx', 'i4'), ('disto_idx', 'i4'),
            ('res_name', 'U3'), ('cyclic_period', 'i4'),
        ]
        dtype = np.dtype(dtype, align=True)
        tokens = np.zeros(design_length, dtype=dtype)
        tokens['token_idx'] = np.arange(design_length)
        
        for i in range(design_length):
            token_name = const.prot_letter_to_token['G']
            res_type_id = const.token_ids[token_name]
            
            tokens[i]['res_idx'] = i + 1
            tokens[i]['asym_id'] = chain_id
            tokens[i]['entity_id'] = entity_id
            tokens[i]['mol_type'] = const.chain_type_ids['PROTEIN']
            tokens[i]['res_type'] = res_type_id
            tokens[i]['is_standard'] = True
            tokens[i]['design_mask'] = True
            tokens[i]['ccd'] = res_type_id
            tokens[i]['binding_type'] = const.binding_type_ids['BINDING']
            tokens[i]['target_msa_mask'] = False
            
            if secondary_structure and i < len(secondary_structure):
                ss = secondary_structure[i]
                tokens[i]['design_ss_mask'] = ss in ['H', 'E', 'L']
            else:
                tokens[i]['design_ss_mask'] = False
            
            tokens[i]['resolved_mask'] = True
            tokens[i]['disto_mask'] = True
            tokens[i]['feature_res_idx'] = i + 1
            tokens[i]['feature_asym_id'] = chain_id
            tokens[i]['atom_idx'] = i * 4
            tokens[i]['atom_num'] = 4
            tokens[i]['center_idx'] = i * 4 + 1
            tokens[i]['disto_idx'] = i * 4 + 1
            tokens[i]['res_name'] = token_name
            tokens[i]['cyclic_period'] = 0
            
        return tokens

    def _load_target_from_cif(self, cif_path: str, chain_id: Optional[str] = None) -> Any:
        with open(cif_path, 'r') as f:
            cif_content = f.read()
        
        tokenizer = Tokenizer()
        record = parse_mmcif(cif_content)
        tokenized = tokenizer.tokenize(record)
        
        if chain_id is not None:
            chain_mask = tokenized.tokens['asym_id'] == chain_id
            tokens = tokenized.tokens[chain_mask]
            structure = tokenized.structure
        else:
            tokens = tokenized.tokens
            structure = tokenized.structure
            
        tokens['design_mask'][:] = False
        tokens['target_msa_mask'][:] = True
        tokens['binding_type'][:] = const.binding_type_ids['UNSPECIFIED']
        
        return tokens, structure

    def _create_design_input_features(
        self,
        design_length: int,
        target_cif_path: Optional[str] = None,
        secondary_structure: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None,
        featurizer: Optional[Featurizer] = None,
    ) -> Dict[str, torch.Tensor]:
        if tokenizer is None: tokenizer = Tokenizer()
        if featurizer is None: featurizer = Featurizer()
        
        if target_cif_path is not None:
            target_tokens, target_structure = self._load_target_from_cif(target_cif_path)
        else:
            target_tokens = np.array([], dtype=self._create_design_tokens(1).dtype)
            target_structure = None
            
        design_tokens = self._create_design_tokens(
            design_length, chain_id=0, entity_id=1, secondary_structure=secondary_structure
        )
        
        if len(target_tokens) > 0:
            all_tokens = np.concatenate([target_tokens, design_tokens])
        else:
            all_tokens = design_tokens
            
        num_tokens = len(all_tokens)
        bonds = np.zeros(0, dtype=[('token_1', 'i4'), ('token_2', 'i4'), ('type', 'i4')])
        token_to_res = np.arange(num_tokens, dtype=np.int64)
        
        if target_structure is not None:
            structure = target_structure
        else:
            structure = self._create_dummy_structure(num_tokens)
            
        input_data = Input(
            tokens=all_tokens, bonds=bonds, token_to_res=token_to_res,
            structure=structure, msa={}, templates=None,
        )
        
        try:
            moldir_path = huggingface_hub.hf_hub_download(
                "boltzgen/inference-data", "mols.zip", repo_type="dataset", library_name="boltzgen"
            )
            molecules = load_canonicals(moldir=Path(moldir_path))
        except Exception:
            molecules = {}
            
        features = featurizer.process(
            input_data, molecules=molecules, random=np.random.default_rng(42),
            training=False, max_seqs=1, backbone_only=True, atom14=False, atom37=False,
            design=True, override_method="X-RAY DIFFRACTION", disulfide_prob=1.0, disulfide_on=True,
        )
        
        templates_features = load_dummy_templates(tdim=1, num_tokens=num_tokens)
        features.update(templates_features)
        
        chain_design_mask = torch.zeros(num_tokens, dtype=torch.bool)
        if len(target_tokens) > 0:
            chain_design_mask[len(target_tokens):] = True
        else:
            chain_design_mask[:] = True
        features["chain_design_mask"] = chain_design_mask
        features["idx_dataset"] = torch.tensor(1)
        features["id"] = "design_example"
        
        return features