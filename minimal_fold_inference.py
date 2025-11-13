"""
MINIMAL FOLDING INFERENCE: Fold proteins from sequence using BoltzGen

This script provides minimal functions to:
1. Fold a single protein from sequence
2. Fold a protein complex (multiple chains)

Usage:
    py minimal_fold_inference.py
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Import BoltzGen components we need
sys.path.insert(0, str(Path(__file__).parent / "boltzgen" / "src"))
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.data import Input, Structure
from boltzgen.data.mol import load_canonicals
from boltzgen.data.template.features import load_dummy_templates
from boltzgen.data import const

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

def create_protein_tokens(sequence: str, chain_id: int = 0, entity_id: int = 0) -> np.ndarray:
    """
    Create token array for a protein sequence.
    
    Args:
        sequence: Amino acid sequence (e.g., "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL")
        chain_id: Chain identifier (asym_id)
        entity_id: Entity identifier
        
    Returns:
        Structured numpy array with token information
    """
    # Convert sequence to residue type indices
    seq_len = len(sequence)
    
    # Create structured array for tokens
    dtype = [
        ('res_idx', 'i4'),
        ('asym_id', 'i4'),
        ('entity_id', 'i4'),
        ('sym_id', 'i4'),
        ('mol_type', 'i4'),
        ('res_type', 'i4'),
        ('is_standard', 'bool'),
        ('design_mask', 'bool'),
        ('modified', 'i4'),
        ('ccd', 'i4'),
        ('binding_type', 'i4'),
        ('structure_group', 'i4'),
        ('center_coords', 'f4', (3,)),
        ('target_msa_mask', 'bool'),
        ('design_ss_mask', 'bool'),
        ('feature_res_idx', 'i4'),
        ('feature_asym_id', 'i4'),
    ]
    
    tokens = np.zeros(seq_len, dtype=dtype)
    
    for i, aa in enumerate(sequence):
        # Map amino acid to residue type
        if aa in const.prot_letter_to_token:
            token_name = const.prot_letter_to_token[aa]
        else:
            token_name = const.prot_letter_to_token['X']  # Unknown
        
        res_type_id = const.token_ids[token_name]
        
        tokens[i]['res_idx'] = i + 1
        tokens[i]['asym_id'] = chain_id
        tokens[i]['entity_id'] = entity_id
        tokens[i]['sym_id'] = 0
        tokens[i]['mol_type'] = const.chain_type_ids['PROTEIN']
        tokens[i]['res_type'] = res_type_id
        tokens[i]['is_standard'] = True
        tokens[i]['design_mask'] = False  # Not designing for folding
        tokens[i]['modified'] = 0
        tokens[i]['ccd'] = res_type_id
        tokens[i]['binding_type'] = const.binding_type_ids['UNSPECIFIED']
        tokens[i]['structure_group'] = 0
        tokens[i]['center_coords'] = [0.0, 0.0, 0.0]
        tokens[i]['target_msa_mask'] = True
        tokens[i]['design_ss_mask'] = False
        tokens[i]['feature_res_idx'] = i + 1
        tokens[i]['feature_asym_id'] = chain_id
    
    return tokens


def create_bonds(num_tokens: int) -> np.ndarray:
    """Create bond connectivity matrix (all zeros for now - will be populated by featurizer)"""
    return np.zeros((num_tokens, num_tokens), dtype=np.float32)


def create_dummy_structure(num_tokens: int) -> Structure:
    """Create a minimal structure object with dummy coordinates"""
    num_atoms = num_tokens * 4  # Approximate: N, CA, C, O per residue
    
    # Create dummy ensemble data
    ensemble = np.zeros(1, dtype=[
        ('atom_coord_idx', 'i4', (2,)),
    ])
    ensemble[0]['atom_coord_idx'] = [0, num_atoms]
    
    # Create dummy coords
    coords = np.zeros((1, num_atoms, 3), dtype=np.float32)
    
    # Create dummy atoms data
    atoms = np.zeros(num_atoms, dtype=[
        ('res_idx', 'i4'),
        ('atom_name', 'U4'),
        ('element', 'U2'),
        ('charge', 'i4'),
        ('conformer', 'i4'),
        ('chirality', 'i4'),
        ('ref_space_uid', 'i4'),
        ('bfactor', 'f4'),
    ])
    
    for i in range(num_atoms):
        token_idx = i // 4
        atom_idx = i % 4
        atom_names = ['N', 'CA', 'C', 'O']
        elements = ['N', 'C', 'C', 'O']
        
        atoms[i]['res_idx'] = token_idx
        atoms[i]['atom_name'] = atom_names[atom_idx]
        atoms[i]['element'] = elements[atom_idx]
        atoms[i]['charge'] = 0
        atoms[i]['conformer'] = 0
        atoms[i]['chirality'] = const.chirality_type_ids['CHI_UNSPECIFIED']
        atoms[i]['ref_space_uid'] = token_idx
        atoms[i]['bfactor'] = 100.0
    
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
    
    # Create dummy chains (single chain)
    chains = np.zeros(1, dtype=[
        ('mol_type', 'i4'),
        ('label_asym_id', 'U8'),
        ('auth_asym_id', 'U8'),
        ('entity_id', 'i4'),
        ('res_idx', 'i4'),
        ('res_num', 'i4'),
        ('cyclic_period', 'i4'),
    ])
    
    chains[0]['mol_type'] = const.chain_type_ids['PROTEIN']
    chains[0]['label_asym_id'] = 'A'
    chains[0]['auth_asym_id'] = 'A'
    chains[0]['entity_id'] = 0
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


def create_input_features(
    sequences: List[str],
    tokenizer: Tokenizer,
    featurizer: Featurizer,
) -> Dict[str, torch.Tensor]:
    """
    Create input features from sequences.
    
    Args:
        sequences: List of amino acid sequences (one per chain)
        tokenizer: BoltzGen tokenizer
        featurizer: BoltzGen featurizer
        
    Returns:
        Dictionary of feature tensors ready for model input
    """
    # Create tokens for all sequences
    all_tokens = []
    for chain_idx, seq in enumerate(sequences):
        tokens = create_protein_tokens(seq, chain_id=chain_idx, entity_id=chain_idx)
        all_tokens.append(tokens)
    
    # Concatenate all tokens
    tokens = np.concatenate(all_tokens)
    num_tokens = len(tokens)
    
    # Create bonds
    bonds = create_bonds(num_tokens)
    
    # Create token_to_res mapping
    token_to_res = np.arange(num_tokens, dtype=np.int64)
    
    # Create dummy structure
    structure = create_dummy_structure(num_tokens)
    
    # Create Input object
    input_data = Input(
        tokens=tokens,
        bonds=bonds,
        token_to_res=token_to_res,
        structure=structure,
        msa={},
        templates=None,
    )
    
    # Use featurizer to create features
    features = featurizer.process(
        input_data,
        molecules={},
        random=np.random.default_rng(42),
        training=False,
        max_seqs=1,
        backbone_only=False,
        atom14=False,
        atom37=False,
        design=False,
        override_method="X-RAY DIFFRACTION",
        disulfide_prob=1.0,
        disulfide_on=True,
    )
    
    # Add dummy template features
    templates_features = load_dummy_templates(tdim=1, num_tokens=num_tokens)
    features.update(templates_features)
    
    # Add required features
    features["idx_dataset"] = torch.tensor(1)
    features["chain_design_mask"] = torch.zeros(num_tokens, dtype=torch.bool)
    
    return features


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
        "sampling_steps": 200,
        "diffusion_samples": 5,
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
# Main Folding Functions
# ============================================================================

def fold_protein(
    sequence: str,
    model: Optional[Boltz] = None,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 5,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Fold a single protein from sequence.
    
    Args:
        sequence: Amino acid sequence
        model: Pre-loaded model (if None, will load from checkpoint)
        recycling_steps: Number of recycling iterations
        sampling_steps: Number of diffusion sampling steps
        diffusion_samples: Number of diffusion samples to generate
        device: Device to run on
        
    Returns:
        Dictionary containing:
            - sample_atom_coords: [num_samples, num_atoms, 3] predicted coordinates
            - ptm: Predicted TM-score
            - Other confidence metrics
    """
    print("="*80)
    print(f"FOLDING PROTEIN: {len(sequence)} residues")
    print("="*80)
    
    # Load model if not provided
    if model is None:
        model = load_model(device=device)
    
    # Create features
    print("\nCreating input features...")
    tokenizer = Tokenizer()
    featurizer = Featurizer()
    features = create_input_features([sequence], tokenizer, featurizer)
    
    # Move to device and add batch dimension
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in features.items()}
    
    # Run inference
    print(f"\nRunning inference...")
    print(f"  Recycling steps: {recycling_steps}")
    print(f"  Sampling steps: {sampling_steps}")
    print(f"  Diffusion samples: {diffusion_samples}")
    
    with torch.no_grad():
        # Apply masker
        feat_masked = model.masker(batch)
        
        # Forward pass
        output = model(
            feat_masked,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            run_confidence_sequentially=True,
        )
    
    print("\n[SUCCESS] Folding complete!")
    print(f"\nOutput keys: {list(output.keys())}")
    print(f"Predicted coordinates shape: {output['sample_atom_coords'].shape}")
    
    return output


def fold_complex(
    sequences: List[str],
    model: Optional[Boltz] = None,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 5,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Fold a protein complex from multiple sequences.
    
    Args:
        sequences: List of amino acid sequences (one per chain)
        model: Pre-loaded model (if None, will load from checkpoint)
        recycling_steps: Number of recycling iterations
        sampling_steps: Number of diffusion sampling steps
        diffusion_samples: Number of diffusion samples to generate
        device: Device to run on
        
    Returns:
        Dictionary containing predicted structure and confidence metrics
    """
    print("="*80)
    print(f"FOLDING COMPLEX: {len(sequences)} chains")
    for i, seq in enumerate(sequences):
        print(f"  Chain {i}: {len(seq)} residues")
    print("="*80)
    
    # Load model if not provided
    if model is None:
        model = load_model(device=device)
    
    # Create features
    print("\nCreating input features...")
    tokenizer = Tokenizer()
    featurizer = Featurizer()
    features = create_input_features(sequences, tokenizer, featurizer)
    
    # Move to device and add batch dimension
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in features.items()}
    
    # Run inference
    print(f"\nRunning inference...")
    print(f"  Recycling steps: {recycling_steps}")
    print(f"  Sampling steps: {sampling_steps}")
    print(f"  Diffusion samples: {diffusion_samples}")
    
    with torch.no_grad():
        # Apply masker
        feat_masked = model.masker(batch)
        
        # Forward pass
        output = model(
            feat_masked,
            recycling_steps=recycling_steps,
            num_sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            run_confidence_sequentially=True,
        )
    
    print("\n[SUCCESS] Complex folding complete!")
    print(f"\nOutput keys: {list(output.keys())}")
    print(f"Predicted coordinates shape: {output['sample_atom_coords'].shape}")
    
    return output


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
MINIMAL FOLDING INFERENCE EXAMPLES
====================================

This script demonstrates minimal folding inference with BoltzGen.

Running example predictions...
    """)
    
    # Example 1: Fold a single protein
    print("\n" + "="*80)
    print("EXAMPLE 1: Folding a single protein")
    print("="*80)
    
    sequence = "MKTAYIAKQRQISFVKSHFSRQLE"  # Short example
    
    try:
        output = fold_protein(
            sequence=sequence,
            recycling_steps=3,
            sampling_steps=50,  # Reduced for speed
            diffusion_samples=1,  # Single sample for speed
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        print(f"\nResults:")
        print(f"  Coordinates: {output['sample_atom_coords'].shape}")
        if 'ptm' in output:
            print(f"  PTM score: {output['ptm'].item():.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Fold a protein complex
    print("\n" + "="*80)
    print("EXAMPLE 2: Folding a protein complex (2 chains)")
    print("="*80)
    
    sequences = [
        "MKTAYIAKQRQISFVK",  # Chain A
        "SHFSRQLEERLGLIEV",  # Chain B
    ]
    
    try:
        output = fold_complex(
            sequences=sequences,
            recycling_steps=3,
            sampling_steps=50,
            diffusion_samples=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        print(f"\nResults:")
        print(f"  Coordinates: {output['sample_atom_coords'].shape}")
        if 'iptm' in output:
            print(f"  iPTM score: {output['iptm'].item():.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)

