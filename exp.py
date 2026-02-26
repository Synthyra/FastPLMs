import torch
from torch.nn.functional import mse_loss
from huggingface_hub import login

from e1_fastplms.modeling_e1 import E1ForMaskedLM, E1Config


MODEL_PATH = 'Profluent-Bio/E1-150m'


if __name__ == "__main__":
    # py -m exp
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_token', type=str, default=None)
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)
        
    config = E1Config.from_pretrained(MODEL_PATH)
    config.auto_map = {
        "AutoConfig": "modeling_e1.E1Config",
        "AutoModel": "modeling_e1.E1Model",
        "AutoModelForMaskedLM": "modeling_e1.E1ForMaskedLM",
        "AutoModelForSequenceClassification": "modeling_e1.E1ForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_e1.E1ForTokenClassification"
    }
    model1 = E1ForMaskedLM.from_pretrained(MODEL_PATH, config=config, dtype=torch.float32).eval()
    model2 = E1ForMaskedLM.from_pretrained(MODEL_PATH, config=config, dtype=torch.float32).eval()

    for name1, param1 in model1.state_dict().items():
        for name2, param2 in model2.state_dict().items():
            if name1 == name2:
                diff = mse_loss(param1, param2).item()
                if diff > 0.0:
                    print(f"{name1}: {diff}")
                    assert diff < 1e-3, f"Parameter {name1} has a large difference: {diff}"


    model1.push_to_hub('Synthyra/Profluent-E1-150M')
    model3 = E1ForMaskedLM.from_pretrained('Synthyra/Profluent-E1-150M', config=config, dtype=torch.float32).eval()
    for name1, param1 in model2.state_dict().items():
        for name3, param3 in model3.state_dict().items():
            if name1 == name3:
                diff = mse_loss(param1, param3).item()
                if diff > 0.0:
                    print(f"{name1}: {diff}")
                    assert diff < 1e-3, f"Parameter {name1} has a large difference: {diff}"