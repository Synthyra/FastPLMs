import torch
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())
# Add boltzgen_automodel to path so boltzgen_flat can be imported as top-level
sys.path.append(os.path.join(os.getcwd(), 'boltzgen_automodel'))

from boltzgen_automodel.modeling_boltzgen import BoltzGen
from boltzgen_automodel.boltzgen_config import BoltzGenConfig

def test_boltzgen_automodel():
    print("Testing BoltzGen AutoModel...")
    
    # 1. Initialize Config with smaller model for faster testing
    print("\n1. Initializing Config...")
    config = BoltzGenConfig(
        atom_s=64,
        atom_z=16,
        token_s=64,
        token_z=32,
        num_bins=64,
        pairformer_args={'num_blocks': 1, 'num_heads': 4},
        score_model_args={
            'atom_encoder_depth': 1, 'atom_encoder_heads': 4,
            'token_transformer_depth': 1, 'token_transformer_heads': 4,
            'atom_decoder_depth': 1, 'atom_decoder_heads': 4,
            'conditioning_transition_layers': 1,
        },
        diffusion_process_args={
            'sigma_min': 0.0004, 'sigma_max': 160.0, 'sigma_data': 16.0,
            'rho': 7, 'P_mean': -1.2, 'P_std': 1.5, 'gamma_0': 0.8, 'gamma_min': 1.0,
            'noise_scale': 1.0, 'step_scale': 1.0,
            'step_scale_random': [1.0],
            'mse_rotational_alignment': True, 'coordinate_augmentation': True,
            'alignment_reverse_diff': True, 'synchronize_sigmas': False,
        },
    )
    
    # 2. Initialize Model
    print("\n2. Initializing Model...")
    model = BoltzGen(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model initialized on {device}")
    
    base_dir = Path(".")

    # 3. Test fold_proteins with defaults
    print("\n3. Testing fold_proteins with defaults...")
    try:
        output = model.fold_proteins()  # Uses default sequence
        print("fold_proteins (default) successful!")
        print(f"  Output keys: {list(output.keys())[:10]}...")  # Show first 10 keys
        if 'coords' in output:
            print(f"  Coords shape: {output['coords'].shape}")
        
        # Test confidence extraction
        confidence = model.get_confidence_scores(output)
        print(f"  Confidence metrics: {list(confidence.keys())}")
        
    except Exception as e:
        print(f"fold_proteins (default) failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. Test fold_proteins with custom sequence
    print("\n4. Testing fold_proteins with custom sequence...")
    sequences = {"A": "MKTAYIAKQRQISFVKGDPRAEVPRA"}
    fold_output_path = base_dir / "test_fold.cif"
    try:
        output = model.fold_proteins(
            sequences=sequences,
            output_path=str(fold_output_path)
        )
        print("fold_proteins (custom) successful!")
        if fold_output_path.exists():
            print(f"  CIF file saved to {fold_output_path}")
            
    except Exception as e:
        print(f"fold_proteins (custom) failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Test output_to_structure
    print("\n5. Testing output_to_structure...")
    try:
        structures = model.output_to_structure(output)
        print(f"  Converted to {len(structures)} structures")
        if structures:
            struct = structures[0]
            print(f"  Structure has {len(struct.atoms)} atoms, {len(struct.residues)} residues, {len(struct.chains)} chains")
            
    except Exception as e:
        print(f"output_to_structure failed: {e}")
        import traceback
        traceback.print_exc()

    # 6. Test save_to_cif
    print("\n6. Testing save_to_cif...")
    if structures:
        try:
            save_cif_path = base_dir / "test_fold_saved.cif"
            model.save_to_cif(structures[0], str(save_cif_path))
            if save_cif_path.exists():
                print(f"  save_to_cif successful! Saved to {save_cif_path}")
        except Exception as e:
            print(f"save_to_cif failed: {e}")
            import traceback
            traceback.print_exc()

    # 7. Test design_proteins with defaults
    print("\n7. Testing design_proteins with defaults...")
    try:
        output = model.design_proteins()  # Uses default entities
        print("design_proteins (default) successful!")
        print(f"  Output keys: {list(output.keys())[:10]}...")
        if 'coords' in output:
            print(f"  Coords shape: {output['coords'].shape}")
            
    except Exception as e:
        print(f"design_proteins (default) failed: {e}")
        import traceback
        traceback.print_exc()

    # 8. Test design_proteins with custom entities
    print("\n8. Testing design_proteins with custom entities...")
    entities = [{"protein": {"id": "A", "sequence": "20..30"}}]  # Smaller for faster test
    design_output_dir = base_dir / "test_design_output"
    try:
        output = model.design_proteins(
            entities=entities,
            num_designs=1,
            output_dir=str(design_output_dir),
            sampling_steps=50  # Fewer steps for faster test
        )
        print("design_proteins (custom) successful!")
        if design_output_dir.exists():
            cif_files = list(design_output_dir.glob("*.cif"))
            print(f"  Generated {len(cif_files)} CIF files in {design_output_dir}")
            
    except Exception as e:
        print(f"design_proteins (custom) failed: {e}")
        import traceback
        traceback.print_exc()

    # 9. Test select_best_design
    print("\n9. Testing select_best_design...")
    try:
        best_idx = model.select_best_design(output)
        print(f"  Best design index: {best_idx}")
        
        # Get only the best structure
        best_structures = model.output_to_structure(output, sample_indices=[best_idx])
        print(f"  Best structure has {len(best_structures[0].atoms)} atoms")
        
    except Exception as e:
        print(f"select_best_design failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("Testing complete!")
    print("="*50)

if __name__ == "__main__":
    test_boltzgen_automodel()
