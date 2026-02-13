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
        # decoder is the same as word_embeddings, and not loaded correctly by default
        model.lm_head.decoder.load_state_dict(original_model.esm.embeddings.word_embeddings.state_dict())
        # deep copy so they are not tied 
        model.lm_head = copy.deepcopy(model.lm_head)
        repo_id = 'Synthyra/' + model_name
        tokenizer = model.tokenizer
        tokenizer.push_to_hub(repo_id)
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

 