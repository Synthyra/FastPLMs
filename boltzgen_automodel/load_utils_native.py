import torch
import sys
import types
import huggingface_hub
from typing import Optional

from modeling_boltzgen import Boltz


def create_dummy_module(name, parent=None):
    """Helper function to create a module and register it in sys.modules"""
    module = types.ModuleType(name)
    sys.modules[name] = module
    if parent is not None:
        setattr(parent, name.split('.')[-1], module)
    return module


# Create dummy validator classes
class DummyValidator:
    """Dummy validator for loading checkpoints"""
    pass


# Create a dummy EMA class (PyTorch Lightning callback)
class DummyEMA:
    """Dummy EMA callback for loading checkpoints"""
    pass


def setup_pickle_modules():
    """
    Create module structure for pickle to find our Boltz class.
    
    This MUST be called before torch.load() to avoid ModuleNotFoundError.
    The modules are registered in sys.modules so pickle can import them.
    """
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
    boltzgen_model_optim_ema = create_dummy_module('boltzgen.model.optim.ema', boltzgen_model_optim)
    boltzgen_model_optim_ema.EMA = DummyEMA
    
    # Create boltzgen.model.validation for validators
    boltzgen_model_validation = create_dummy_module('boltzgen.model.validation', boltzgen_model)
    
    boltzgen_model_validation_validator = create_dummy_module('boltzgen.model.validation.validator', boltzgen_model_validation)
    boltzgen_model_validation_validator.Validator = DummyValidator
    
    boltzgen_model_validation_design = create_dummy_module('boltzgen.model.validation.design', boltzgen_model_validation)
    boltzgen_model_validation_design.DesignValidator = DummyValidator
    
    boltzgen_model_validation_rcsb = create_dummy_module('boltzgen.model.validation.rcsb', boltzgen_model_validation)
    boltzgen_model_validation_rcsb.RCSBValidator = DummyValidator
    
    boltzgen_model_validation_refolding = create_dummy_module('boltzgen.model.validation.refolding', boltzgen_model_validation)
    boltzgen_model_validation_refolding.RefoldingValidator = DummyValidator


# Set up dummy modules at import time (required before any torch.load() calls)
setup_pickle_modules()


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
    
    # Create model
    print("Creating model...")
    model = Boltz(**config)
    
    state_dict = checkpoint["state_dict"]
    
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

    print("="*60)
    print("\nModel loaded successfully!")
    return model


def minimal_load_example():
    print("="*80)
    print("MINIMAL BOLTZGEN LOADING EXAMPLE")
    print("="*80)
    
    print("\n[1/4] Downloading checkpoint from HuggingFace...")
    
    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id="boltzgen/boltzgen-1",
        filename="boltz2_conf_final.ckpt",  # Folding + confidence model
        repo_type="model",
        library_name="boltzgen",
    )
    
    print(f"[OK] Downloaded to: {checkpoint_path}")
    print("\n[2/4] Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"[OK] Checkpoint loaded successfully!")
    print(f"  Keys in checkpoint: {list(checkpoint.keys())}")


    print("\n[3/4] Extracting model configuration...")
    
    config = checkpoint["hyper_parameters"]

    print("\n[4/4] Instantiating model and loading weights...")
    
    # Create model with config
    print("  Creating model instance...")
    model = Boltz(**config)
    
    # Extract weights from checkpoint
    state_dict = checkpoint["state_dict"]
    
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

    print("\n" + "="*80)
    print("[SUCCESS] MODEL LOADED!")
    print("="*80)
    return model, config


if __name__ == "__main__":    
    result = minimal_load_example()
