import torch    
from huggingface_hub import login

from e1.modeling_e1 import E1ForMaskedLM, E1Config


model_dict = {
    'Profluent-E1-150M': 'Profluent-Bio/E1-150m',
    'Profluent-E1-300M': 'Profluent-Bio/E1-300m',
    'Profluent-E1-600M': 'Profluent-Bio/E1-600m',
}


if __name__ == "__main__":
    # py -m e1.get_e1_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    args = parser.parse_args()

    if args.token:
        login(token=args.token)
        
    for model_name in model_dict:
        config = E1Config.from_pretrained(model_dict[model_name])
        config.auto_map = {
            "AutoConfig": "modeling_e1.E1Config",
            "AutoModel": "modeling_e1.E1Model",
            "AutoModelForMaskedLM": "modeling_e1.E1ForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_e1.E1ForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_e1.E1ForTokenClassification"
        }
        model = E1ForMaskedLM.from_pretrained(model_dict[model_name], config=config, dtype=torch.bfloat16)
        model.push_to_hub('Synthyra/' + model_name)
