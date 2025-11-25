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
    config = BoltzGenConfig()
    
    # 2. Initialize Model
    print("Initializing Model...")
    model = BoltzGen(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model initialized on {device}")
    
    # 3. Test fold_proteins
    print("\nTesting fold_proteins...")
    sequences = {"A": "MKTAYIAKQRQISFVK"}
    try:
        output = model.fold_proteins(
            sequences=sequences,
            output_path="test_fold.cif"
        )
        print("fold_proteins successful!")
        print("Output type:", type(output))
        print("Output keys:", output.keys())
        if 'sample_atom_coords' in output:
            print("Coords shape:", output['sample_atom_coords'].shape)
            
        # Test output_to_structure
        print("Testing output_to_structure...")
        structures = model.output_to_structure(output)
        print(f"Converted to {len(structures)} structures")
        
        # Test save_to_cif
        print("Testing save_to_cif...")
        if structures:
            model.save_to_cif(structures[0], "test_fold_saved.cif")
            print("save_to_cif successful!")
        
    except Exception as e:
        print(f"fold_proteins failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. Test design_proteins
    print("\nTesting design_proteins...")
    entities = [{"protein": {"id": "A", "sequence": "MKTAYIAKQRQISFVK"}}]
    try:
        output = model.design_proteins(
            entities=entities,
            num_designs=1,
            output_dir="test_design_output"
        )
        print("design_proteins successful!")
        print("Output type:", type(output))
        print("Output keys:", output.keys())
        if 'sample_atom_coords' in output:
            print("Coords shape:", output['sample_atom_coords'].shape)
            
    except Exception as e:
        print(f"design_proteins failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_boltzgen_automodel()
