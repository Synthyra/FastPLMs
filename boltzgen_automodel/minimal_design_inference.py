"""
MINIMAL DESIGN INFERENCE: Design proteins against targets using BoltzGen

This script provides minimal functions to design proteins against target structures.

Usage:
    py minimal_design_inference.py
"""

import torch
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple


from boltzgen_flat.data_tokenize_tokenizer import Tokenizer
from boltzgen_flat.data_feature_featurizer import Featurizer
from boltzgen_flat.data_data import Input, Structure
from boltzgen_flat.data_parse_mmcif import parse_mmcif
from boltzgen_flat.data_template_features import load_dummy_templates
from boltzgen_flat.data_mol import load_canonicals
from boltzgen_flat import data_const as const

# Import our custom Boltz model and setup from minimal_working_example
from minimal_working_example import (
    create_dummy_module, Boltz, DummyEMA, DummyValidator
)
import types
import huggingface_hub


# ============================================================================
# Setup module redirects (same as minimal_working_example.py)
# ============================================================================
def setup_pickle_modules():
    """Create module structure for pickle to find our Boltz class"""
    boltzgen = create_dummy_module('boltzgen')
    boltzgen_data = create_dummy_module('boltzgen.data', boltzgen)
    boltzgen_data_const = create_dummy_module('boltzgen.data.const', boltzgen_data)
    boltzgen_model = create_dummy_module('boltzgen.model', boltzgen)
    boltzgen_model_models = create_dummy_module('boltzgen.model.models', boltzgen_model)
    boltzgen_model_models_boltz = create_dummy_module('boltzgen.model.models.boltz', boltzgen_model_models)
    boltzgen_model_models_boltz.Boltz = Boltz
    boltzgen_model_optim = create_dummy_module('boltzgen.model.optim', boltzgen_model)
    boltzgen_model_optim_ema = create_dummy_module('boltzgen.model.optim.ema', boltzgen_model_optim)
    boltzgen_model_optim_ema.EMA = DummyEMA
    boltzgen_model_validation = create_dummy_module('boltzgen.model.validation', boltzgen_model)
    boltzgen_model_validation_validator = create_dummy_module('boltzgen.model.validation.validator', boltzgen_model_validation)
    boltzgen_model_validation_validator.Validator = DummyValidator
    boltzgen_model_validation_design = create_dummy_module('boltzgen.model.validation.design', boltzgen_model_validation)
    boltzgen_model_validation_design.DesignValidator = DummyValidator
    boltzgen_model_validation_rcsb = create_dummy_module('boltzgen.model.validation.rcsb', boltzgen_model_validation)
    boltzgen_model_validation_rcsb.RCSBValidator = DummyValidator
    boltzgen_model_validation_refolding = create_dummy_module('boltzgen.model.validation.refolding', boltzgen_model_validation)
    boltzgen_model_validation_refolding.RefoldingValidator = DummyValidator


setup_pickle_modules()


# ============================================================================
# Helper Functions
# ============================================================================

def create_design_tokens(
    design_length: int,
    chain_id: int = 0,
    entity_id: int = 0,
    secondary_structure: Optional[str] = None,
) -> np.ndarray:
    """
    Create token array for designed residues.
    
    Args:
        design_length: Number of residues to design
        chain_id: Chain identifier
        entity_id: Entity identifier
        secondary_structure: Optional secondary structure specification
                            'H' = helix, 'E' = sheet, 'L' = loop, None = unspecified
        
    Returns:
        Structured numpy array with token information
    """
    dtype = [
        ('res_idx', 'i4'),
        ('asym_id', 'i4'),
        ('entity_id', 'i4'),
        ('sym_id', 'i4'),
        ('mol_type', 'i4'),
        ('res_type', 'i4'),
        ('modified', 'i4'),
        ('ccd', 'i4'),
        ('binding_type', 'i4'),
        ('structure_group', 'i4'),
        ('feature_res_idx', 'i4'),
        ('feature_asym_id', 'i4'),
        ('center_coords', 'f4', (3,)),
        ('is_standard', 'bool'),
        ('design_mask', 'bool'),
        ('target_msa_mask', 'bool'),
        ('design_ss_mask', 'bool'),
        ('resolved_mask', 'bool'),
        ('disto_mask', 'bool'),
        ('token_idx', 'i4'),
        ('atom_idx', 'i4'),
        ('atom_num', 'i4'),
        ('center_idx', 'i4'),
        ('disto_idx', 'i4'),
        ('res_name', 'U3'),
        ('cyclic_period', 'i4'),
    ]
    dtype = np.dtype(dtype, align=True)
    
    tokens = np.zeros(design_length, dtype=dtype)
    tokens['token_idx'] = np.arange(design_length)
    
    for i in range(design_length):
        # Use glycine as placeholder for design tokens
        token_name = const.prot_letter_to_token['G']  # Glycine
        res_type_id = const.token_ids[token_name]
        
        tokens[i]['res_idx'] = i + 1
        tokens[i]['asym_id'] = chain_id
        tokens[i]['entity_id'] = entity_id
        tokens[i]['sym_id'] = 0
        tokens[i]['mol_type'] = const.chain_type_ids['PROTEIN']
        tokens[i]['res_type'] = res_type_id
        tokens[i]['is_standard'] = True
        tokens[i]['design_mask'] = True  # Mark as design token
        tokens[i]['modified'] = 0
        tokens[i]['ccd'] = res_type_id
        tokens[i]['binding_type'] = const.binding_type_ids['BINDING']  # Design binder
        tokens[i]['structure_group'] = 0
        tokens[i]['center_coords'] = [0.0, 0.0, 0.0]
        tokens[i]['target_msa_mask'] = False  # No MSA for design tokens
        
        # Secondary structure conditioning
        if secondary_structure and i < len(secondary_structure):
            ss = secondary_structure[i]
            tokens[i]['design_ss_mask'] = ss in ['H', 'E', 'L']
        else:
            tokens[i]['design_ss_mask'] = False
        
        tokens[i]['resolved_mask'] = True
        tokens[i]['disto_mask'] = True
        
        tokens[i]['feature_res_idx'] = i + 1
        tokens[i]['feature_asym_id'] = chain_id
        
        # Initialize atom fields (assuming 4 atoms per residue: N, CA, C, O)
        tokens[i]['atom_idx'] = i * 4
        tokens[i]['atom_num'] = 4
        tokens[i]['center_idx'] = i * 4 + 1  # CA
        tokens[i]['disto_idx'] = i * 4 + 1   # CA (using CA as disto for minimal example)
        tokens[i]['res_name'] = token_name  # Set res_name to 'GLY'
        tokens[i]['cyclic_period'] = 0
    
    return tokens


def load_target_from_cif(cif_path: str, chain_id: Optional[str] = None) -> Tuple[np.ndarray, Structure]:
    """
    Load target structure from CIF file.
    
    Args:
        cif_path: Path to CIF file
        chain_id: Optional chain ID to extract (if None, uses all chains)
        
    Returns:
        Tuple of (tokens, structure)
    """
    # Parse CIF file
    with open(cif_path, 'r') as f:
        cif_content = f.read()
    
    tokenizer = Tokenizer()
    record = parse_mmcif(cif_content)
    
    # Tokenize
    tokenized = tokenizer.tokenize(record)
    
    # Filter to specific chain if requested
    if chain_id is not None:
        # Find tokens for this chain
        chain_mask = tokenized.tokens['asym_id'] == chain_id
        tokens = tokenized.tokens[chain_mask]
        structure = tokenized.structure  # Note: structure filtering is more complex
    else:
        tokens = tokenized.tokens
        structure = tokenized.structure
    
    # Mark all target tokens as NOT design tokens
    tokens['design_mask'][:] = False
    tokens['target_msa_mask'][:] = True
    tokens['binding_type'][:] = const.binding_type_ids['UNSPECIFIED']
    
    return tokens, structure


def create_design_input_features(
    design_length: int,
    target_cif_path: Optional[str] = None,
    target_tokens: Optional[np.ndarray] = None,
    target_structure: Optional[Structure] = None,
    secondary_structure: Optional[str] = None,
    tokenizer: Optional[Tokenizer] = None,
    featurizer: Optional[Featurizer] = None,
) -> Dict[str, torch.Tensor]:
    """
    Create input features for protein design.
    
    Args:
        design_length: Number of residues to design
        target_cif_path: Path to target structure CIF file (if provided)
        target_tokens: Pre-loaded target tokens (alternative to CIF path)
        target_structure: Pre-loaded target structure (alternative to CIF path)
        secondary_structure: Optional secondary structure for designed residues
        tokenizer: BoltzGen tokenizer (created if None)
        featurizer: BoltzGen featurizer (created if None)
        
    Returns:
        Dictionary of feature tensors ready for model input
    """
    if tokenizer is None:
        tokenizer = Tokenizer()
    if featurizer is None:
        featurizer = Featurizer()
    
    # Load or create target
    if target_cif_path is not None:
        print(f"Loading target from {target_cif_path}...")
        target_tokens, target_structure = load_target_from_cif(target_cif_path)
    elif target_tokens is not None and target_structure is not None:
        print("Using provided target...")
    else:
        print("No target provided - designing de novo protein...")
        target_tokens = np.array([], dtype=target_tokens.dtype if target_tokens is not None else None)
        target_structure = None
    
    # Create design tokens
    design_tokens = create_design_tokens(
        design_length,
        chain_id=0,  # Use chain 0 for simple design
        entity_id=1,
        secondary_structure=secondary_structure,
    )
    
    # Combine tokens
    if len(target_tokens) > 0:
        all_tokens = np.concatenate([target_tokens, design_tokens])
    else:
        all_tokens = design_tokens
    
    num_tokens = len(all_tokens)
    
    # Create bonds
    bonds_dtype = [
        ('token_1', 'i4'),
        ('token_2', 'i4'),
        ('type', 'i4'),
    ]
    bonds = np.zeros(0, dtype=bonds_dtype)
    
    # Create token_to_res mapping
    token_to_res = np.arange(num_tokens, dtype=np.int64)
    
    # Handle structure
    if target_structure is not None:
        # Extend structure with dummy atoms for design tokens
        structure = target_structure  # Simplified - real version would extend atoms
    else:
        # Create dummy structure
        structure = create_dummy_structure(num_tokens)
    
    # Create Input object
    input_data = Input(
        tokens=all_tokens,
        bonds=bonds,
        token_to_res=token_to_res,
        structure=structure,
        msa={},
        templates=None,
    )
    
    # Load canonical molecules
    import huggingface_hub
    
    # Download moldir
    print("Downloading molecule data...")
    moldir_path = huggingface_hub.hf_hub_download(
        "boltzgen/inference-data",
        "mols.zip",
        repo_type="dataset",
        library_name="boltzgen",
    )
    molecules = load_canonicals(moldir=Path(moldir_path))
    
    # Use featurizer to create features
    features = featurizer.process(
        input_data,
        molecules=molecules,
        random=np.random.default_rng(42),
        training=False,
        max_seqs=1,
        backbone_only=True,  # Use backbone_only instead of atom14 to avoid dtype issues
        atom14=False,
        atom37=False,
        design=True,  # Enable design mode
        override_method="X-RAY DIFFRACTION",
        disulfide_prob=1.0,
        disulfide_on=True,
    )
    
    # Add dummy template features
    templates_features = load_dummy_templates(tdim=1, num_tokens=num_tokens)
    features.update(templates_features)
    
    # Add chain design mask (which chains are being designed)
    chain_design_mask = torch.zeros(num_tokens, dtype=torch.bool)
    if len(target_tokens) > 0:
        chain_design_mask[len(target_tokens):] = True  # Only design chain is being designed
    else:
        chain_design_mask[:] = True  # All residues are being designed
    features["chain_design_mask"] = chain_design_mask
    
    # Add required features
    features["idx_dataset"] = torch.tensor(1)
    features["id"] = "design_example"  # Add ID for masker
    
    return features


def create_dummy_structure(num_tokens: int) -> Structure:
    """Create a minimal structure object with dummy coordinates"""
    num_atoms = num_tokens * 4  # Approximate: N, CA, C, O per residue
    
    # Create dummy ensemble data
    ensemble = np.zeros(1, dtype=[
        ('atom_coord_idx', 'i4', (2,)),
    ])
    ensemble[0]['atom_coord_idx'] = [0, num_atoms]
    
    # Create dummy coords
    coords = np.zeros(num_atoms, dtype=[('coords', 'f4', (3,))])
    
    # Create dummy atoms data
    atoms = np.zeros(num_atoms, dtype=[
        ('res_idx', 'i4'),
        ('name', 'U4'),
        ('element', 'U2'),
        ('charge', 'i4'),
        ('conformer', 'i4'),
        ('chirality', 'i4'),
        ('ref_space_uid', 'i4'),
        ('bfactor', 'f4'),
        ('is_present', 'bool'),
        ('plddt', 'f4'),
        ('coords', 'f4', (3,)),
    ])
    
    for i in range(num_atoms):
        token_idx = i // 4
        atom_idx = i % 4
        atom_names = ['N', 'CA', 'C', 'O']
        elements = ['N', 'C', 'C', 'O']
        
        atoms[i]['res_idx'] = token_idx
        atoms[i]['name'] = atom_names[atom_idx]
        atoms[i]['element'] = elements[atom_idx]
        atoms[i]['charge'] = 0
        atoms[i]['conformer'] = 0
        atoms[i]['chirality'] = const.chirality_type_ids['CHI_UNSPECIFIED']
        atoms[i]['ref_space_uid'] = token_idx
        atoms[i]['bfactor'] = 100.0
        atoms[i]['is_present'] = True
        atoms[i]['plddt'] = 100.0
        atoms[i]['coords'] = [0.0, 0.0, 0.0]
    
    # Create dummy bonds (no bonds for now)
    bonds = np.zeros(0, dtype=[
        ('atom1', 'i4'),
        ('atom2', 'i4'),
        ('bond_type', 'i4'),
    ])
    
    # Create dummy residues
    residues = np.zeros(num_tokens, dtype=[
        ('name', 'U8'),
        ('label_asym_id', 'U8'),
        ('label_seq_id', 'i4'),
        ('auth_asym_id', 'U8'),
        ('auth_seq_id', 'i4'),
        ('pdbx_PDB_ins_code', 'U8'),
        ('atom_idx', 'i4'),
        ('atom_num', 'i4'),
        ('res_type', 'i4'),
    ])
    
    for i in range(num_tokens):
        residues[i]['name'] = 'GLY'
        residues[i]['label_asym_id'] = 'A'
        residues[i]['label_seq_id'] = i + 1
        residues[i]['auth_asym_id'] = 'A'
        residues[i]['auth_seq_id'] = i + 1
        residues[i]['pdbx_PDB_ins_code'] = ''
        residues[i]['atom_idx'] = i * 4
        residues[i]['atom_num'] = 4
        residues[i]['res_type'] = const.token_ids['GLY']
    
    # Create dummy chains (single chain)
    chains = np.zeros(1, dtype=[
        ('mol_type', 'i4'),
        ('label_asym_id', 'U8'),
        ('auth_asym_id', 'U8'),
        ('entity_id', 'i4'),
        ('asym_id', 'i4'),
        ('name', 'U4'),
        ('res_idx', 'i4'),
        ('res_num', 'i4'),
        ('cyclic_period', 'i4'),
    ])
    
    chains[0]['mol_type'] = const.chain_type_ids['PROTEIN']
    chains[0]['label_asym_id'] = 'A'
    chains[0]['auth_asym_id'] = 'A'
    chains[0]['entity_id'] = 0
    chains[0]['asym_id'] = 0
    chains[0]['name'] = 'A'
    chains[0]['res_idx'] = 0
    chains[0]['res_num'] = num_tokens
    chains[0]['cyclic_period'] = 0
    
    # Create dummy interfaces (no interfaces)
    interfaces = np.zeros(0, dtype=[
        ('chain1', 'i4'),
        ('chain2', 'i4'),
    ])
    
    # Create mask (all valid)
    mask = np.ones(1, dtype=bool)
    
    structure = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        interfaces=interfaces,
        mask=mask,
        coords=coords,
        ensemble=ensemble,
    )
    
    return structure


def load_model(
    checkpoint_path: Optional[str] = None,
    checkpoint_name: str = "boltz2_conf_final.ckpt",
    device: str = "cuda"
) -> Boltz:
    """
    Load BoltzGen model.
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, downloads from HuggingFace.
        checkpoint_name: Name of checkpoint to download from HuggingFace.
                        Options:
                        - "boltz2_conf_final.ckpt" (default, full model with confidence)
                        - Custom checkpoint filename if you have your own
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Loaded Boltz model
        
    Available checkpoints on HuggingFace (boltzgen/boltzgen-1):
        - boltz2_conf_final.ckpt: Full BoltzGen model with confidence prediction
        - boltzgen1_structuretrained_small.ckpt: Small pretrained checkpoint (training only)
        
    Note: For inference, boltz2_conf_final.ckpt is the standard checkpoint.
    If you've trained your own model (small/large), pass the path to checkpoint_path.
    """
    if checkpoint_path is None:
        print(f"Downloading checkpoint '{checkpoint_name}' from HuggingFace...")
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id="boltzgen/boltzgen-1",
            filename=checkpoint_name,
            repo_type="model",
            library_name="boltzgen",
        )
        print(f"Downloaded to: {checkpoint_path}")
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract config
    config = checkpoint["hyper_parameters"]
    config["validators"] = None
    config["validate_structure"] = False
    config["structure_prediction_training"] = False
    config["inference_logging"] = False
    config["predict_args"] = {
        "recycling_steps": 3,
        "sampling_steps": 500,
        "diffusion_samples": 1,
    }
    
    # Override masker args for design
    config["masker_args"] = {
        "mask": True,
        "mask_backbone": False,
    }
    
    # Create model
    print("Creating model...")
    model = Boltz(**config)
    
    # Load weights (prefer EMA if available)
    state_dict = checkpoint["state_dict"]
    ema_keys = [k for k in state_dict.keys() if k.startswith("ema.")]
    
    if ema_keys:
        print(f"Using EMA weights ({len(ema_keys)} parameters)")
        state_dict = {
            k.replace("ema.", ""): v
            for k, v in state_dict.items()
            if k.startswith("ema.")
        }
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    # Display model architecture info
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(f"  Token embedding dim (token_s): {config.get('token_s', 'N/A')}")
    print(f"  Token pair dim (token_z): {config.get('token_z', 'N/A')}")
    print(f"  Atom embedding dim (atom_s): {config.get('atom_s', 'N/A')}")
    print(f"  Atom pair dim (atom_z): {config.get('atom_z', 'N/A')}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**3):.2f} GB (fp32)")
    
    # Show pairformer info
    if hasattr(model, 'pairformer_module'):
        pf_config = config.get('pairformer_args', {})
        print(f"  Pairformer blocks: {pf_config.get('num_blocks', 'N/A')}")
        print(f"  Pairformer heads: {pf_config.get('num_heads', 'N/A')}")
    
    print("="*60)
    print("\nModel loaded successfully!")
    return model


# ============================================================================
# Main Design Functions
# ============================================================================

def design_protein(
    design_length: int,
    target_cif_path: Optional[str] = None,
    secondary_structure: Optional[str] = None,
    model: Optional[Boltz] = None,
    recycling_steps: int = 3,
    sampling_steps: int = 500,
    diffusion_samples: int = 1,
    step_scale: float = 1.8,
    noise_scale: float = 0.95,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Design a protein against an optional target.
    
    Args:
        design_length: Number of residues to design
        target_cif_path: Optional path to target structure CIF file
        secondary_structure: Optional secondary structure specification (e.g., "HHHHEEELLL")
        model: Pre-loaded model (if None, will load from checkpoint)
        recycling_steps: Number of recycling iterations
        sampling_steps: Number of diffusion sampling steps
        diffusion_samples: Number of designs to generate
        step_scale: Diffusion step scale (higher = more diverse)
        noise_scale: Diffusion noise scale (lower = more diverse)
        device: Device to run on
        
    Returns:
        Dictionary containing:
            - sample_atom_coords: [num_samples, num_atoms, 3] predicted coordinates
            - res_type_logits: [num_samples, num_tokens, num_aa_types] sequence predictions
            - ptm: Predicted TM-score
            - Other confidence metrics
    """
    print("="*80)
    print(f"DESIGNING PROTEIN: {design_length} residues")
    if target_cif_path:
        print(f"  Target: {target_cif_path}")
    if secondary_structure:
        print(f"  Secondary structure: {secondary_structure}")
    print("="*80)
    
    # Load model if not provided
    if model is None:
        model = load_model(device=device)
    
    # Create features
    print("\nCreating input features...")
    tokenizer = Tokenizer()
    featurizer = Featurizer()
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
    
    # Run inference
    print(f"\nRunning inference...")
    print(f"  Recycling steps: {recycling_steps}")
    print(f"  Sampling steps: {sampling_steps}")
    print(f"  Diffusion samples: {diffusion_samples}")
    print(f"  Step scale: {step_scale}")
    print(f"  Noise scale: {noise_scale}")
    
    with torch.no_grad():
        # Apply masker (masks designed residue coordinates and sequences)
        feat_masked = model.masker(batch)
        
        # Forward pass
        output = model(
            feat_masked,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            run_confidence_sequentially=True,
            step_scale=step_scale,
            noise_scale=noise_scale,
        )
    
    print("\n[SUCCESS] Design complete!")
    print(f"\nOutput keys: {list(output.keys())}")
    print(f"Predicted coordinates shape: {output['sample_atom_coords'].shape}")
    if 'res_type_logits' in output:
        print(f"Sequence predictions shape: {output['res_type_logits'].shape}")
    
    return output


def extract_designed_sequence(output: Dict[str, torch.Tensor], design_mask: torch.Tensor) -> str:
    """
    Extract designed sequence from model output.
    
    Args:
        output: Model output dictionary
        design_mask: Boolean mask indicating designed positions
        
    Returns:
        Designed amino acid sequence
    """
    if 'res_type_logits' not in output:
        return "Sequence not available in output"
    
    # Get predicted residue types
    res_type_logits = output['res_type_logits'][0]  # Remove batch dim
    predicted_types = torch.argmax(res_type_logits, dim=-1)
    
    # Extract designed positions
    design_positions = design_mask.cpu().numpy()
    predicted_types = predicted_types.cpu().numpy()[design_positions]
    
    # Convert to sequence
    token_to_letter = {v: k for k, v in const.prot_letter_to_token.items()}
    sequence = ''.join([token_to_letter.get(t, 'X') for t in predicted_types])
    
    return sequence


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
MINIMAL DESIGN INFERENCE EXAMPLES
===================================

This script demonstrates minimal protein design with BoltzGen.

Running example predictions...
    """)
    
    # Example 1: Design a protein de novo (no target)
    print("\n" + "="*80)
    print("EXAMPLE 1: De novo protein design (no target)")
    print("="*80)
    
    try:
        output = design_protein(
            design_length=50,
            target_cif_path=None,
            secondary_structure="HHHHHHHHHEEEEEEELLLLLHHHHHHHHHEEEEEEELLLLL",
            recycling_steps=3,
            sampling_steps=100,  # Reduced for speed
            diffusion_samples=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        print(f"\nResults:")
        print(f"  Coordinates: {output['sample_atom_coords'].shape}")
        if 'ptm' in output:
            print(f"  PTM score: {output['ptm'].item():.3f}")
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Error: {e}")
    
    # Example 2: Design a protein against a target
    print("\n" + "="*80)
    print("EXAMPLE 2: Protein design against target")
    print("="*80)
    
    # Check if example target exists
    target_path = Path("boltzgen/example/7rpz.cif")
    
    if target_path.exists():
        try:
            output = design_protein(
                design_length=30,
                target_cif_path=str(target_path),
                recycling_steps=3,
                sampling_steps=100,
                diffusion_samples=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            
            print(f"\nResults:")
            print(f"  Coordinates: {output['sample_atom_coords'].shape}")
            if 'iptm' in output:
                print(f"  iPTM score: {output['iptm'].item():.3f}")
            if 'design_ptm' in output:
                print(f"  Design PTM: {output['design_ptm'].item():.3f}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Target file not found: {target_path}")
        print("Skipping this example.")
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)
    print("""
NEXT STEPS:
-----------
1. Extract coordinates: output['sample_atom_coords']
2. Extract sequences: output['res_type_logits'] 
3. Save to CIF files (implement separately)
4. Analyze confidence metrics (ptm, iptm, pae, etc.)
5. Run multiple samples and filter by metrics
    """)

