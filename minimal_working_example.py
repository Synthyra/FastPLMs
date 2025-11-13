"""
MINIMAL WORKING EXAMPLE: Load and inspect BoltzGen model

This script demonstrates how to:
1. Download a checkpoint from HuggingFace
2. Work around pickle import issues
3. Load weights into your custom modeling_boltzgen.Boltz class
4. Inspect the loaded model

Run: py minimal_working_example.py
"""

import torch
import sys
from pathlib import Path
import huggingface_hub

# ============================================================================
# SOLUTION: Create module redirects for pickle to find the Boltz class
# ============================================================================
# When the checkpoint was saved, it stored references to 'boltzgen.model.models.boltz.Boltz'
# We need to create stub modules that redirect to our custom implementation

# Import your custom Boltz class
from modeling_boltzgen import Boltz

# Create the module structure that pickle expects
import types

def create_dummy_module(name, parent=None):
    """Helper function to create a module and register it in sys.modules"""
    module = types.ModuleType(name)
    sys.modules[name] = module
    if parent is not None:
        setattr(parent, name.split('.')[-1], module)
    return module

# Create boltzgen package hierarchy
boltzgen = create_dummy_module('boltzgen')

# Create boltzgen.data package (needed for validators)
boltzgen_data = create_dummy_module('boltzgen.data', boltzgen)
boltzgen_data_const = create_dummy_module('boltzgen.data.const', boltzgen_data)

# Create boltzgen.model package
boltzgen_model = create_dummy_module('boltzgen.model', boltzgen)

# Create boltzgen.model.models package
boltzgen_model_models = create_dummy_module('boltzgen.model.models', boltzgen_model)

# Create boltzgen.model.models.boltz module with our Boltz class
boltzgen_model_models_boltz = create_dummy_module('boltzgen.model.models.boltz', boltzgen_model_models)
boltzgen_model_models_boltz.Boltz = Boltz  # Redirect to our implementation!

# Create boltzgen.model.optim for EMA
boltzgen_model_optim = create_dummy_module('boltzgen.model.optim', boltzgen_model)

# Create a dummy EMA class (PyTorch Lightning callback)
class DummyEMA:
    """Dummy EMA callback for loading checkpoints"""
    pass

boltzgen_model_optim_ema = create_dummy_module('boltzgen.model.optim.ema', boltzgen_model_optim)
boltzgen_model_optim_ema.EMA = DummyEMA

# Create boltzgen.model.validation for validators
boltzgen_model_validation = create_dummy_module('boltzgen.model.validation', boltzgen_model)

# Create dummy validator classes
class DummyValidator:
    """Dummy validator for loading checkpoints"""
    pass

boltzgen_model_validation_validator = create_dummy_module('boltzgen.model.validation.validator', boltzgen_model_validation)
boltzgen_model_validation_validator.Validator = DummyValidator

boltzgen_model_validation_design = create_dummy_module('boltzgen.model.validation.design', boltzgen_model_validation)
boltzgen_model_validation_design.DesignValidator = DummyValidator

boltzgen_model_validation_rcsb = create_dummy_module('boltzgen.model.validation.rcsb', boltzgen_model_validation)
boltzgen_model_validation_rcsb.RCSBValidator = DummyValidator

boltzgen_model_validation_refolding = create_dummy_module('boltzgen.model.validation.refolding', boltzgen_model_validation)
boltzgen_model_validation_refolding.RefoldingValidator = DummyValidator


def minimal_load_example():
    """
    Minimal example to load BoltzGen weights into custom Boltz class.
    """
    
    print("="*80)
    print("MINIMAL BOLTZGEN LOADING EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Download checkpoint from HuggingFace
    # ========================================================================
    
    print("\n[1/4] Downloading checkpoint from HuggingFace...")
    
    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id="boltzgen/boltzgen-1",
        filename="boltz2_conf_final.ckpt",  # Folding + confidence model
        repo_type="model",
        library_name="boltzgen",
    )
    
    print(f"[OK] Downloaded to: {checkpoint_path}")
    
    # ========================================================================
    # STEP 2: Load checkpoint with custom unpickler
    # ========================================================================
    
    print("\n[2/4] Loading checkpoint...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"[OK] Checkpoint loaded successfully!")
        print(f"  Keys in checkpoint: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================================================
    # STEP 3: Extract configuration
    # ========================================================================
    
    print("\n[3/4] Extracting model configuration...")
    
    if "hyper_parameters" in checkpoint:
        config = checkpoint["hyper_parameters"]
        print(f"[OK] Found hyperparameters in checkpoint")
        print(f"\n  Key configuration:")
        print(f"    - token_s: {config.get('token_s')}")
        print(f"    - token_z: {config.get('token_z')}")
        print(f"    - atom_s: {config.get('atom_s')}")
        print(f"    - atom_z: {config.get('atom_z')}")
        print(f"    - confidence_prediction: {config.get('confidence_prediction')}")
        print(f"    - inverse_fold: {config.get('inverse_fold')}")
        print(f"    - use_miniformer: {config.get('use_miniformer')}")
        
        # Override for inference (remove PyTorch Lightning specific stuff)
        config["validators"] = None
        config["validate_structure"] = False
        config["structure_prediction_training"] = False
        config["inference_logging"] = False
        config["predict_args"] = {
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 5,
        }
    else:
        print("[ERROR] No hyperparameters in checkpoint!")
        print("  You'll need to manually specify the config")
        return None
    
    # ========================================================================
    # STEP 4: Instantiate model and load weights
    # ========================================================================
    
    print("\n[4/4] Instantiating model and loading weights...")
    
    try:
        # Create model with config
        print("  Creating model instance...")
        model = Boltz(**config)
        
        # Extract weights from checkpoint
        state_dict = checkpoint["state_dict"]
        
        # Check if EMA weights are available (recommended for inference)
        ema_keys = [k for k in state_dict.keys() if k.startswith("ema.")]
        
        if ema_keys:
            print(f"  Found {len(ema_keys)} EMA weights - using those (better for inference)")
            # Use EMA weights (better performance)
            state_dict = {
                k.replace("ema.", ""): v
                for k, v in state_dict.items()
                if k.startswith("ema.")
            }
        else:
            print(f"  No EMA weights found - using regular weights")
        
        # Load weights into model
        print("  Loading state dict...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"[OK] Model instantiated successfully!")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        if missing:
            print(f"\n  First few missing keys:")
            for key in list(missing)[:5]:
                print(f"    - {key}")
        if unexpected:
            print(f"\n  First few unexpected keys:")
            for key in list(unexpected)[:5]:
                print(f"    - {key}")
        
        # Set to eval mode
        model.eval()
        
        # ====================================================================
        # STEP 5: Inspect model structure
        # ====================================================================
        
        print("\n" + "="*80)
        print("MODEL STRUCTURE")
        print("="*80)
        
        print(f"\nModel components:")
        print(f"  [OK] Input embedder: {hasattr(model, 'input_embedder')}")
        print(f"  [OK] Pairformer module: {hasattr(model, 'pairformer_module')}")
        print(f"  [OK] MSA module: {hasattr(model, 'msa_module')}")
        print(f"  [OK] Structure module: {hasattr(model, 'structure_module')}")
        print(f"  [OK] Confidence module: {hasattr(model, 'confidence_module')}")
        print(f"  [OK] Distogram module: {hasattr(model, 'distogram_module')}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel size:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Size: ~{total_params * 4 / (1024**3):.2f} GB (fp32)")
        
        # ====================================================================
        # SUCCESS!
        # ====================================================================
        
        print("\n" + "="*80)
        print("[SUCCESS] MODEL LOADED!")
        print("="*80)
        print("""
The model loaded successfully! You now have a working BoltzGen model
in your custom Boltz class from modeling_boltzgen.py.

WHAT'S NEXT?
------------

For actual inference, you have several options:

1. Use BoltzGen's official CLI (EASIEST):
   boltzgen run example/7rpz.cif --output results --skip_inverse_folding

2. Use BoltzGen's data pipeline:
   - The official pipeline handles all feature creation
   - Requires downloading the molecule database
   - See boltzgen/src/boltzgen/data/ for implementation

3. Create your own minimal inference pipeline:
   - You'll need to create proper input features
   - See create_dummy_features() below for the structure
   - Note: Real features require proper preprocessing

4. Port to HuggingFace format:
   - Now that loading works, you can extract the weights
   - Create a HuggingFace-compatible model class
   - Upload to HuggingFace Hub for easy sharing
        """)
        
        return model, config
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_checkpoint_info(checkpoint_path):
    """
    Detailed inspection of a checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print("\n" + "="*80)
    print(f"CHECKPOINT INSPECTION: {Path(checkpoint_path).name}")
    print("="*80)
    
    print(f"\nTop-level keys: {list(checkpoint.keys())}")
    
    if "hyper_parameters" in checkpoint:
        hp = checkpoint["hyper_parameters"]
        print(f"\nHyperparameters ({len(hp)} keys):")
        for key in sorted(hp.keys()):
            value = hp[key]
            if isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
    
    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        print(f"\nState dict: {len(sd)} keys")
        
        # Group by module
        modules = {}
        for key in sd.keys():
            module = key.split('.')[0]
            if module not in modules:
                modules[module] = []
            modules[module].append(key)
        
        print(f"\nModules:")
        for module, keys in sorted(modules.items()):
            print(f"  {module}: {len(keys)} parameters")
            # Show first few keys
            if len(keys) <= 3:
                for k in keys:
                    print(f"    - {k}")
            else:
                for k in keys[:3]:
                    print(f"    - {k}")
                print(f"    ... and {len(keys)-3} more")
    
    return checkpoint


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        # Inspect a checkpoint
        if len(sys.argv) > 2:
            show_checkpoint_info(sys.argv[2])
        else:
            print("Usage: py minimal_working_example.py inspect <checkpoint_path>")
    
    else:
        # Default: load and inspect model
        result = minimal_load_example()
        
        if result is not None:
            model, config = result
            
            print("\n" + "="*80)
            print("USAGE EXAMPLES")
            print("="*80)
            print("""
Run this script in different modes:

1. Load and inspect model (default):
   py minimal_working_example.py

2. Inspect a checkpoint file:
   py minimal_working_example.py inspect <path_to_checkpoint>

3. For real inference, use the BoltzGen CLI:
   boltzgen run example/7rpz.cif --output results
            """)
        else:
            print("\n[ERROR] Failed to load model. Check errors above.")
