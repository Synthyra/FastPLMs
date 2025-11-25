import torch
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())
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
    print("Initializing Model...")
    model = BoltzGen(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model initialized on {device}")
    
    print("\nTesting fold_proteins...")
    sequences = {"A": "MKTAYIAKQRQISFVK"}
    fold_output_path = base_dir / "test_fold.cif"
    try:
        output = model.fold_proteins(
            sequences=sequences,
            output_path=str(fold_output_path)
        )
        print("fold_proteins successful!")
        print("Output type:", type(output))
        print("Output keys:", output.keys())
        if 'coords' in output:
            print("Coords shape:", output['coords'].shape)
            
        # Test output_to_structure
        print("Testing output_to_structure...")
        structures = model.output_to_structure(output)
        print(f"Converted to {len(structures)} structures")
        
        # Test save_to_cif
        print("Testing save_to_cif...")
        if structures:
            save_cif_path = base_dir / "test_fold_saved.cif"
            model.save_to_cif(structures[0], str(save_cif_path))
            print(f"save_to_cif successful! Saved to {save_cif_path}")
        
    except Exception as e:
        print(f"fold_proteins failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting design_proteins...")
    entities = [{"protein": {"id": "A", "sequence": "MKTAYIAKQRQISFVK"}}]
    design_output_dir = base_dir / "test_design_output"
    try:
        output = model.design_proteins(
            entities=entities,
            num_designs=1,
            output_dir=str(design_output_dir)
        )
        print("design_proteins successful!")
        print("Output type:", type(output))
        print("Output keys:", output.keys())
        if 'coords' in output:
            print("Coords shape:", output['coords'].shape)
            
    except Exception as e:
        print(f"design_proteins failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_boltzgen_automodel()
