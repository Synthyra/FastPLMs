BoltzGen AutoModel Integration Walkthrough
Overview
Successfully implemented the 

BoltzGen
 class compatible with Hugging Face AutoModel, including protein folding and design capabilities.

Changes

boltzgen_automodel/modeling_boltzgen.py
:
Implemented 

BoltzGen
 class inheriting from PreTrainedModel.
Added 

fold_proteins
 method for complex folding.
Added 

design_proteins
 method for de novo and targeted design.
Added 

save_to_cif
 for outputting predictions.
Integrated helper functions for tokenization and featurization.
Fixed imports to be relative for package compatibility.

boltzgen_automodel/basic_boltzgen.py
:
Updated imports to be relative/compatible with the package structure.

test_boltzgen_automodel.py
:
Created a comprehensive test script to verify folding and design.
Added 

boltzgen_automodel
 to sys.path to handle internal imports of boltzgen_flat.
Verification Results
Test Script Output
The 

test_boltzgen_automodel.py
 script ran successfully on CUDA.

Testing BoltzGen AutoModel...
Initializing Config...
Initializing Model...
Model initialized on cuda
Testing fold_proteins...
Folding complex with 1 chains...
fold_proteins successful!
Output keys: dict_keys(['pdistogram', 'pbfactor', 'sample_atom_coords', ...])
Coords shape: torch.Size([1, 64, 3])
Testing save_to_cif...
save_to_cif successful!
Testing design_proteins...
Designing protein: 20 residues...
design_proteins successful!
Output keys: dict_keys(['pdistogram', 'pbfactor', 'sample_atom_coords', ...])
Coords shape: torch.Size([1, 96, 3])
Key Fixes during Verification
Import Errors: Fixed ModuleNotFoundError by using relative imports in 

modeling_boltzgen.py
 and 

basic_boltzgen.py
, and adding the package directory to sys.path in the test script.
Config Mismatch: Updated num_bins to 64 in the test config to match the model's expectation, resolving a tensor shape mismatch error.
Usage
To use the model:

from boltzgen_automodel.modeling_boltzgen import BoltzGen
from boltzgen_automodel.boltzgen_config import BoltzGenConfig
# Initialize
config = BoltzGenConfig()
model = BoltzGen(config)
# Fold
sequences = {"MKTAYIAKQRQISFVK": 1}
output = model.fold_proteins(sequences)
model.save_to_cif(output, "prediction.cif", sequence="MKTAYIAKQRQISFVK")
# Design
design_output = model.design_proteins(design_length=20)