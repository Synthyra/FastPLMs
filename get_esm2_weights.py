from modeling_fastesm import FastEsmForMaskedLM, FastEsmConfig


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


for model_name in model_dict:
    config = FastEsmConfig.from_pretrained(model_dict[model_name])
    config.auto_map = {
        "AutoConfig": "modeling_fastesm.FastEsmConfig",
        "AutoModel": "modeling_fastesm.FastEsmModel",
        "AutoModelForMaskedLM": "modeling_fastesm.FastEsmForMaskedLM",
        "AutoModelForSequenceClassification": "modeling_fastesm.FastEsmForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_fastesm.FastEsmForTokenClassification"
    }
    model = FastEsmForMaskedLM(config=config).from_pretrained(model_dict[model_name], config=config)
    model.push_to_hub('Synthyra/' + model_name)
