import copy
import torch    
from huggingface_hub import HfApi, login
from transformers import EsmForMaskedLM
from esm2.modeling_fastesm import FastEsmForMaskedLM, FastEsmConfig


model_dict = {
    # Synthyra/ESM2-8M
    'ESM2-8M': 'facebook/esm2_t6_8M_UR50D',
    # Synthyra/ESM2-35M
    'ESM2-35M': 'facebook/esm2_t12_35M_UR50D',
    # Synthyra/ESM2-150M
    'ESM2-150M': 'facebook/esm2_t30_150M_UR50D',
    # Synthyra/ESM2-650M
    'ESM2-650M': 'facebook/esm2_t33_650M_UR50D',
    # Synthyra/ESM2-3B
    'ESM2-3B': 'facebook/esm2_t36_3B_UR50D',
}


if __name__ == "__main__":
    # py -m esm2.get_esm2_weights
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default=None)
    args = parser.parse_args()
    api = HfApi()

    if args.token:
        login(token=args.token)
    
    for model_name in model_dict:
        config = FastEsmConfig.from_pretrained(model_dict[model_name])
        config.auto_map = {
            "AutoConfig": "modeling_fastesm.FastEsmConfig",
            "AutoModel": "modeling_fastesm.FastEsmModel",
            "AutoModelForMaskedLM": "modeling_fastesm.FastEsmForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_fastesm.FastEsmForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_fastesm.FastEsmForTokenClassification"
        }
        config.tie_word_embeddings = False
        original_model = EsmForMaskedLM.from_pretrained(model_dict[model_name])
        model = FastEsmForMaskedLM(config=config).from_pretrained(model_dict[model_name], config=config)
        model.lm_head.load_state_dict(original_model.lm_head.state_dict())
        model.lm_head = copy.deepcopy(model.lm_head)
        repo_id = 'Synthyra/' + model_name
        model.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj="esm2/modeling_fastesm.py",
            path_in_repo="modeling_fastesm.py",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="embedding_mixin.py",
            path_in_repo="embedding_mixin.py",
            repo_id=repo_id,
            repo_type="model",
        )

        # Compare only the lm_head parameters and print warnings for mismatches in other parameters
        mismatched_params = []
        base_lm_head_state = dict(original_model.lm_head.named_parameters())
        fast_lm_head_state = dict(model.lm_head.named_parameters())
        for name, param in fast_lm_head_state.items():
            if name in base_lm_head_state:
                base_param = base_lm_head_state[name]
                assert param.shape == base_param.shape, f'{name} {param.shape} != {name} {base_param.shape}'
                assert torch.equal(param.data, base_param.data), f"Parameter {name} weights differ after transfer!"
            else:
                print(f'Warning: {name} not found in original_model.lm_head!')

 