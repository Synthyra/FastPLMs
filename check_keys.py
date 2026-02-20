import torch
from e1_fastplms.modeling_e1 import E1Config, E1ForMaskedLM

config = E1Config()
model = E1ForMaskedLM(config)
print("Keys in our E1ForMaskedLM state_dict:")
for name in list(model.state_dict().keys())[:10]:
    print(name)

from e1_fastplms.load_official import load_official_model
# Just load a small one to check keys
official_model, _ = load_official_model("Profluent-Bio/E1-150m", device=torch.device("cpu"))
print("\nKeys in official E1ForMaskedLM state_dict:")
for name in list(official_model.model.state_dict().keys())[:10]:
    print(name)
