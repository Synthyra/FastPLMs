import argparse
import torch    
from huggingface_hub import login

from fastplms.e1.modeling_e1 import E1ForMaskedLM, E1Config


model_dict = {
    'Profluent-E1-150M': 'Profluent-Bio/E1-150m',
    'Profluent-E1-300M': 'Profluent-Bio/E1-300m',
    'Profluent-E1-600M': 'Profluent-Bio/E1-600m',
}


if __name__ == "__main__":
    # py -m fastplms.e1.get_weights

    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    for model_name in model_dict:
        repo_id = "Synthyra/" + model_name
        config = E1Config.from_pretrained(model_dict[model_name])
        config.auto_map = {
            "AutoConfig": "modeling_e1.E1Config",
            "AutoModel": "modeling_e1.E1Model",
            "AutoModelForMaskedLM": "modeling_e1.E1ForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_e1.E1ForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_e1.E1ForTokenClassification"
        }
        if args.skip_weights:
            config.push_to_hub(repo_id)
            print(f"[skip-weights] uploaded config for {repo_id}")
            continue
        model = E1ForMaskedLM.from_pretrained(model_dict[model_name], config=config, dtype=torch.float32)
        model.push_to_hub(repo_id)