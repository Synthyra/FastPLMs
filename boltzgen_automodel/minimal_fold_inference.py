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
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

from boltzgen_flat.data_tokenize_tokenizer import Tokenizer
from boltzgen_flat.data_feature_featurizer import Featurizer
from boltzgen_flat.data_data import Input, Structure
from boltzgen_flat.data_mol import load_canonicals
from boltzgen_flat.data_template_features import load_dummy_templates
from boltzgen_flat import data_const as const
from load_utils_native import setup_pickle_modules, load_model
from modeling_boltzgen import BoltzGen


setup_pickle_modules()


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
    
    tokens = np.zeros(seq_len, dtype=dtype)
    tokens['token_idx'] = np.arange(seq_len)
    
    for i, aa in enumerate(sequence):
        # Map amino acid to residue type
        tokens[i]['res_type'] = const.token_ids.get(const.prot_letter_to_token.get(aa, 'UNK'), const.token_ids['UNK'])
        tokens[i]['res_name'] = const.prot_letter_to_token.get(aa, 'UNK')
        tokens[i]['feature_asym_id'] = chain_id
        
        # Initialize atom fields (assuming 4 atoms per residue: N, CA, C, O)
        tokens[i]['atom_idx'] = i * 4
        tokens[i]['atom_num'] = 4
        tokens[i]['center_idx'] = i * 4 + 1  # CA
        tokens[i]['disto_idx'] = i * 4 + 1   # CA (using CA as disto for minimal example)
        tokens[i]['cyclic_period'] = 0
    
    return tokens


def create_bonds(num_tokens: int) -> np.ndarray:
    """Create bond connectivity list (empty for now)"""
    dtype = [
        ('token_1', 'i4'),
        ('token_2', 'i4'),
        ('type', 'i4'),
    ]
    return np.zeros(0, dtype=dtype)


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


def fold_protein(
    sequence: str,
    model: Optional[BoltzGen] = None,
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
    model: Optional[BoltzGen] = None,
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
    return output


if __name__ == "__main__":
    # py -m boltzgen_automodel.minimal_fold_inference
    # Example 1: Fold a single protein
    print("\n" + "="*80)
    print("EXAMPLE 1: Folding a single protein")
    print("="*80)
    
    recycling_steps = 3
    sampling_steps = 200
    diffusion_samples = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BoltzGen.from_pretrained('Synthyra/boltzgen').boltz.eval().to(device)


    sequence = "MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL"  # Short example
    output = fold_protein(
        sequence=sequence,
        model=model,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,  # Reduced for speed
        diffusion_samples=diffusion_samples,  # Single sample for speed
        device=device,
    )
    for k, v in output.items():
        try:
            print(f"{k}: {v.shape}")
        except:
            print(f"{k}: {type(v)}")

    plddt = output['plddt']
    print(plddt[0])

    pae = output['pae']
    plt.imshow(pae.cpu().numpy().squeeze())
    plt.colorbar()
    plt.show()

    
    # Example 2: Fold a protein complex
    print("\n" + "="*80)
    print("EXAMPLE 2: Folding a protein complex (2 chains)")
    print("="*80)
    
    sequences = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSYISSSSSYTNYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTASYYCARGLAGVWGIDVWGQGTLVTVSS",  # Chain A
        "QNYTRSTDNQAVIKDALQGIQQQIKGLADKIGTEIGPKVSLIDTSSTITIPANIGLLGSKISQSTASINENVNEKCKFTLPPLKIHECNISCPNPLPFREYRPQTEGVSNLVGLPNNICLQKTSNQILKPKLISYTLPVVGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCT",  # Chain B
    ]
    output = fold_complex(
        sequences=sequences,
        model=model,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        device=device,
    )

    plddt = output['plddt']
    print(plddt[0])

    pae = output['pae']
    plt.imshow(pae.cpu().numpy().squeeze())
    plt.colorbar()
    plt.show()