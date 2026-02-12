import torch
from transformers import AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    "Synthyra/Boltz2",
    trust_remote_code=True,
    dtype=torch.float32,
    device_map="cuda"
).eval()


out = model.predict_structure(
    amino_acid_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    recycling_steps=3,
    num_sampling_steps=200,
    diffusion_samples=1,
)

print(out.sample_atom_coords.shape)
print(None if out.plddt is None else out.plddt.shape)