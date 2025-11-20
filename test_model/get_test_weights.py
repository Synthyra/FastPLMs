import copy
import torch    
from huggingface_hub import login
from transformers import EsmForMaskedLM
from test_model import FastEsmForMaskedLM, FastEsmConfig


model_dict = {
    # lhallee/test_auto_model
    'test_auto_model': 'facebook/esm2_t6_8M_UR50D',
}


if __name__ == "__main__":
    #login()
    for model_name in model_dict:
        config = FastEsmConfig.from_pretrained(model_dict[model_name])
        config.auto_map = {
            "AutoConfig": "test_model.FastEsmConfig",
            "AutoModel": "test_model.FastEsmModel",
            "AutoModelForMaskedLM": "test_model.FastEsmForMaskedLM",
            "AutoModelForSequenceClassification": "test_model.FastEsmForSequenceClassification",
            "AutoModelForTokenClassification": "test_model.FastEsmForTokenClassification"
        }
        config.tie_word_embeddings = False
        original_model = EsmForMaskedLM.from_pretrained(model_dict[model_name])
        model = FastEsmForMaskedLM(config=config).from_pretrained(model_dict[model_name], config=config)
        model.lm_head.load_state_dict(original_model.lm_head.state_dict())
        model.lm_head = copy.deepcopy(model.lm_head)
        model.push_to_hub('lhallee/' + model_name)

        for name1, param1 in model.named_parameters():
            for name2, param2 in original_model.named_parameters():
                if name1 == name2:
                    assert param1.shape == param2.shape, f'{name1} {param1.shape} != {name2} {param2.shape}'
                    assert torch.equal(param1.data.clone(), param2.data.clone()), f'{name1} {name2}'