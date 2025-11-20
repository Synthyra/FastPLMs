# ============================================================================
# BoltzGen HuggingFace Model Class
# ============================================================================
#
# This file contains the BoltzGen class that should be added to modeling_boltzgen.py
# It provides a HuggingFace-compatible interface with inference methods.
#

import sys
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np


class BoltzGenConfig(PretrainedConfig):
    """Configuration class for BoltzGen model."""
    model_type = "boltzgen"
    
    def __init__(
        self,
        token_s=384,
        token_z=128,
        atom_s=128,
        atom_z=16,
        num_pairformer_blocks=64,
        num_pairformer_heads=16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_s = token_s
        self.token_z = token_z
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.num_pairformer_blocks = num_pairformer_blocks
        self.num_pairformer_heads = num_pairformer_heads


class BoltzGen(PreTrainedModel):
    """
    BoltzGen model for protein folding and design.
    
    This class provides a HuggingFace-compatible interface to the BoltzGen model,
    with convenient methods for protein folding, design, and structure export.
    
    Example usage:
        >>> model = BoltzGen.from_pretrained("boltzgen/boltzgen-1")
        >>> 
        >>> # Fold proteins
        >>> results = model.fold_proteins({"MKTAYIAKQRQISFVK": 2, "SHFSRQLE": 1})
        >>> 
        >>> # Design a protein
        >>> output = model.design_protein(50, secondary_structure="HHHHHEEELLL")
        >>> 
        >>> # Save to CIF
        >>> model.save_to_cif(output, "designed_protein.cif")
    """
    
    config_class = BoltzGenConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # The actual model will be loaded from the existing Boltz implementation
        # This is a wrapper class that adds inference methods
        
    def fold_proteins(
        self,
        sequences: Dict[str, int],
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        diffusion_samples: int = 1,
        device: Optional[str] = None,
    ) -> Dict[str, Dict]:
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
            Dictionary mapping sequences to their prediction outputs
            
        Example:
            >>> results = model.fold_proteins({
            ...     "MKTAYIAKQRQISFVK": 2,  # Fold 2 copies
            ...     "SHFSRQLEERLGLIEV": 1,  # Fold 1 copy
            ... })
            >>> for seq, output in results.items():
            ...     print(f"Sequence: {seq}")
            ...     print(f"PTM: {output['ptm'].item():.3f}")
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        results = {}
        
        for sequence, count in sequences.items():
            print(f"Folding {sequence} ({count} copies)...")
            
            # Import minimal fold functions
            from minimal_fold_inference import create_tokens_from_sequence, create_input_features, create_dummy_structure
            from boltzgen_flat.data_tokenize_tokenizer import Tokenizer
            from boltzgen_flat.data_feature_featurizer import Featurizer
            
            tokenizer = Tokenizer()
            featurizer = Featurizer()
            
            features = create_input_features([sequence], tokenizer, featurizer)
            
            # Move to device and add batch dimension
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in features.items()}
            
            # Run inference (using the existing Boltz forward method)
            with torch.no_grad():
                output = self.forward(batch)
            
            # Store results for each copy
            for i in range(count):
                copy_key = f"{sequence}_copy{i+1}" if count > 1 else sequence
                results[copy_key] = output
                
        return results
    
    def design_protein(
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
    ) -> Dict:
        """
        Design a protein de novo or against a target structure.
        
        Args:
            design_length: Number of residues to design
            target_cif_path: Optional path to target structure CIF file
            secondary_structure: Optional secondary structure string (H/E/L)
                               e.g., "HHHHHEEELLL" for helix-sheet-loop
            recycling_steps: Number of recycling iterations (default: 3)
            sampling_steps: Number of diffusion steps (default: 100)
            step_scale: Diffusion step scale (default: 1.8)
            noise_scale: Diffusion noise scale (default: 0.95)
            diffusion_samples: Number of samples (default: 1)
            device: Device to run on (default: auto-detect CUDA)
            
        Returns:
            Dictionary with designed coordinates and metrics
            
        Example:
            >>> output = model.design_protein(
            ...     design_length=50,
            ...     secondary_structure="HHHHHHHHHEEEEEEELLLLLHHHHHHHH",
            ... )
            >>> print(f"PTM: {output['ptm'].item():.3f}")
            >>> model.save_to_cif(output, "designed_protein.cif")
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Designing protein: {design_length} residues...")
        if secondary_structure:
            print(f"Secondary structure: {secondary_structure}")
        if target_cif_path:
            print(f"Target: {target_cif_path}")
            
        from minimal_design_inference import create_design_input_features
        from boltzgen_flat.data_tokenize_tokenizer import Tokenizer
        from boltzgen_flat.data_feature_featurizer import Featurizer
        
        tokenizer = Tokenizer()
        featurizer = Featurizer()
        
        # Create design features
        features = create_design_input_features(
            design_length=design_length,
            target_cif_path=target_cif_path,
            secondary_structure=secondary_structure,
            tokenizer=tokenizer,
            featurizer=featurizer,
        )
        
        # Move to device and add batch dimension
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in features.items()}
        
        # Run inference (using the existing Boltz forward method)
        with torch.no_grad():
            output = self.forward(batch)
            
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
        
        Args:
            output: Model output dictionary from fold_proteins() or design_protein()
            filepath: Path to save CIF file
            sequence: Optional sequence (for folding) or None (will extract from output)
            chain_id: Chain identifier (default: "A")
            include_metrics: Whether to include confidence metrics (default: True)
            
        Example:
            >>> output = model.fold_proteins({"MKTAYIAKQRQISFVK": 1})
            >>> model.save_to_cif(
            ...     output["MKTAYIAKQRQISFVK"],
            ...     "folded_protein.cif",
            ...     sequence="MKTAYIAKQRQISFVK"
            ... )
        """
        # Extract coordinates
        coords = output['sample_atom_coords'][0].cpu().numpy()  # [num_atoms, 3]
        
        # Extract or decode sequence
        if sequence is None:
            # Try to extract from res_type_logits if available
            if 'res_type_logits' in output:
                res_types = output['res_type_logits'].argmax(dim=-1)[0].cpu().numpy()
                # Convert to sequence using token mapping
                sequence = ''.join([prot_token_to_letter.get(canonical_tokens[rt], 'X') 
                                   for rt in res_types if rt < len(canonical_tokens)])
            else:
                sequence = 'X' * (len(coords) // 4)  # Approximate from coords
        
        # Calculate number of residues
        num_residues = len(sequence)
        
        # Open file for writing
        with open(filepath, 'w') as f:
            # Write header
            f.write("data_BOLTZGEN_PREDICTION\n")
            f.write("#\n")
            f.write("_entry.id BOLTZGEN_PREDICTION\n")
            f.write("#\n")
            
            # Write atom coordinates
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
            
            # Write atoms (assuming 4 atoms per residue: N, CA, C, O)
            atom_names = ['N', 'CA', 'C', 'O']
            elements = ['N', 'C', 'C', 'O']
            atom_id = 1
            
            # Get pLDDT if available
            plddt_values = None
            if include_metrics and 'plddt' in output:
                plddt_values = output['plddt'][0].cpu().numpy()
            
            for res_idx, aa in enumerate(sequence):
                res_name = prot_letter_to_token.get(aa, 'UNK')
                
                for atom_idx, (atom_name, element) in enumerate(zip(atom_names, elements)):
                    coord_idx = res_idx * 4 + atom_idx
                    if coord_idx >= len(coords):
                        break
                        
                    x, y, z = coords[coord_idx]
                    
                    # Get B-factor (use pLDDT if available)
                    if plddt_values is not None and res_idx < len(plddt_values):
                        b_factor = plddt_values[res_idx]
                    else:
                        b_factor = 100.0
                    
                    f.write(f"ATOM {atom_id} {element} {atom_name} {res_name} {chain_id} "
                           f"{res_idx + 1} {x:.3f} {y:.3f} {z:.3f} {b_factor:.2f}\n")
                    atom_id += 1
            
            # Write confidence metrics if requested
            if include_metrics:
                f.write("#\n")
                
                # Write per-residue pLDDT
                if 'plddt' in output:
                    f.write("loop_\n")
                    f.write("_pdbx_model_confidence.residue_id\n")
                    f.write("_pdbx_model_confidence.plddt\n")
                    plddt = output['plddt'][0].cpu().numpy()
                    for i, score in enumerate(plddt[:num_residues]):
                        f.write(f"{i + 1} {score:.2f}\n")
                    f.write("#\n")
                
                # Write global metrics
                if 'ptm' in output:
                    ptm = output['ptm'].item()
                    f.write(f"_pdbx_model_confidence.ptm {ptm:.3f}\n")
                
                if 'iptm' in output:
                    iptm = output['iptm'].item()
                    f.write(f"_pdbx_model_confidence.iptm {iptm:.3f}\n")
                
                if 'complex_plddt' in output:
                    complex_plddt = output['complex_plddt'].item()
                    f.write(f"_pdbx_model_confidence.complex_plddt {complex_plddt:.3f}\n")
        
        print(f"Saved prediction to {filepath}")
        if include_metrics and 'ptm' in output:
            print(f"  PTM: {output['ptm'].item():.3f}")
        if include_metrics and 'plddt' in output:
            avg_plddt = output['plddt'][0].mean().item()
            print(f"  Average pLDDT: {avg_plddt:.2f}")
