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
    
    # 1. Initialize Config
    print("Initializing Config...")
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
    print("Initializing Model...")
    model = BoltzGen(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model initialized on {device}")
    
    # 3. Test fold_proteins
    print("\nTesting fold_proteins...")
    sequences = {"MKTAYIAKQRQISFVK": 1}
    try:
        output = model.fold_proteins(
            sequences=sequences,
            recycling_steps=1,
            sampling_steps=5,
            diffusion_samples=1,
            device=device
        )
        print("fold_proteins successful!")
        print("Output keys:", output.keys())
        if 'sample_atom_coords' in output:
            print("Coords shape:", output['sample_atom_coords'].shape)
            
        # Test save_to_cif
        print("Testing save_to_cif...")
        model.save_to_cif(output, "test_fold.cif", sequence="MKTAYIAKQRQISFVK")
        print("save_to_cif successful!")
        
    except Exception as e:
        print(f"fold_proteins failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. Test design_proteins
    print("\nTesting design_proteins...")
    try:
        output = model.design_proteins(
            design_length=20,
            recycling_steps=1,
            sampling_steps=5,
            diffusion_samples=1,
            device=device
        )
        print("design_proteins successful!")
        print("Output keys:", output.keys())
        if 'sample_atom_coords' in output:
            print("Coords shape:", output['sample_atom_coords'].shape)
            
    except Exception as e:
        print(f"design_proteins failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_boltzgen_automodel()
