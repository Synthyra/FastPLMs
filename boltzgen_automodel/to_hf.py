import torch
import sys
import types
import huggingface_hub
import json
import copy

from modeling_boltzgen import Boltz
from boltzgen_config import BoltzGenConfig
from modeling_boltzgen import BoltzGen


def convert_sets_to_lists(obj, visited=None):
    """Recursively convert sets to lists and DictConfig to dict (safe for model initialization)."""
    if visited is None:
        visited = set()
    
    # Handle basic types first (fast path)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # Prevent infinite recursion with circular references
    obj_id = id(obj)
    if obj_id in visited:
        return obj  # Return as-is for circular refs (they'll be handled during serialization)
    visited.add(obj_id)
    
    try:
        # Handle DictConfig (from Hydra/OmegaConf)
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'DictConfig':
            # Convert DictConfig to regular dict, then recurse
            try:
                # Try OmegaConf.to_container first (most reliable)
                from omegaconf import OmegaConf
                obj = OmegaConf.to_container(obj, resolve=True)
                # Now recurse on the converted dict
                return convert_sets_to_lists(obj, visited)
            except (ImportError, AttributeError):
                # Fallback: try dict() constructor
                try:
                    obj = dict(obj)
                    # Now recurse on the converted dict
                    return convert_sets_to_lists(obj, visited)
                except:
                    # Last resort: use __dict__ if available
                    if hasattr(obj, '__dict__'):
                        obj = obj.__dict__
                        return convert_sets_to_lists(obj, visited)
                    else:
                        return obj
        
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets_to_lists(v, visited) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            result = [convert_sets_to_lists(item, visited) for item in obj]
            return tuple(result) if isinstance(obj, tuple) else result
        else:
            # Leave everything else as-is (don't convert to string)
            return obj
    finally:
        visited.discard(obj_id)


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


def to_huggingface():
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

    # Convert sets to lists (safe for model initialization)
    # Other serialization issues will be handled by to_dict() during push_to_hub
    print("\n[5/5] Preparing config for HuggingFace upload...")
    # Make a deep copy to avoid modifying the original config
    try:
        config_copy = copy.deepcopy(config)
    except Exception as e:
        print(f"  Warning: deepcopy failed ({e}), using shallow copy")
        config_copy = copy.copy(config)
    config_clean = convert_sets_to_lists(config_copy)
    
    # Validate that dict-type arguments are actually dicts (convert DictConfig if needed)
    dict_args = ['embedder_args', 'msa_args', 'pairformer_args', 'score_model_args', 
                 'diffusion_process_args', 'diffusion_loss_args', 'affinity_model_args',
                 'affinity_model_args1', 'affinity_model_args2', 'confidence_model_args',
                 'training_args', 'validation_args', 'masker_args', 'template_args',
                 'token_distance_args', 'inverse_fold_args', 'predict_args', 'dynamic_args']
    
    for arg_name in dict_args:
        if arg_name in config_clean:
            value = config_clean[arg_name]
            if value is not None and not isinstance(value, dict):
                # Handle DictConfig (from Hydra/OmegaConf)
                if hasattr(value, '__class__') and value.__class__.__name__ == 'DictConfig':
                    try:
                        from omegaconf import OmegaConf
                        config_clean[arg_name] = OmegaConf.to_container(value, resolve=True)
                        print(f"  Converted {arg_name} from DictConfig to dict")
                    except (ImportError, AttributeError):
                        try:
                            config_clean[arg_name] = dict(value)
                            print(f"  Converted {arg_name} from DictConfig to dict (fallback)")
                        except:
                            print(f"  Warning: Could not convert {arg_name} from DictConfig, using empty dict")
                            config_clean[arg_name] = {}
                elif isinstance(value, str):
                    print(f"  Warning: {arg_name} is a string, attempting to parse as JSON...")
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            config_clean[arg_name] = parsed
                            print(f"    Successfully parsed as dict.")
                        else:
                            print(f"    Parsed but not a dict, using empty dict.")
                            config_clean[arg_name] = {}
                    except:
                        print(f"    Could not parse, using empty dict.")
                        config_clean[arg_name] = {}
                else:
                    print(f"  Warning: {arg_name} is {type(value).__name__}, expected dict. Using empty dict.")
                    config_clean[arg_name] = {}
    
    hf_config = BoltzGenConfig(**config_clean)

    hf_model = BoltzGen(hf_config)
    hf_model.boltz.load_state_dict(model.state_dict())
    hf_model.push_to_hub('Synthyra/boltzgen', safe_serialization=False)


if __name__ == "__main__":
    to_huggingface()
