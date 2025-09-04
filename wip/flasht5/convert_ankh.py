import re
import os
from safetensors import safe_open
from safetensors.torch import save_file


def convert_ankh(current_path, save_path):
    tensors = {}
    with safe_open(current_path, framework="pt") as f:
        for k in f.keys():
            new_k = re.sub(".layer.*.SelfAttention.q", ".self_attention_layer.self_attention.Wq", k)
            new_k = re.sub(".layer.*.SelfAttention.k", ".self_attention_layer.self_attention.Wk", new_k)
            new_k = re.sub(".layer.*.SelfAttention.v", ".self_attention_layer.self_attention.Wv", new_k)
            new_k = re.sub(".layer.*.SelfAttention.o", ".self_attention_layer.self_attention.o", new_k)
            new_k = re.sub(".layer.*.EncDecAttention.q", ".cross_attention_layer.cross_attention.Wq", new_k)
            new_k = re.sub(".layer.*.EncDecAttention.k", ".cross_attention_layer.cross_attention.Wk", new_k)
            new_k = re.sub(".layer.*.EncDecAttention.v", ".cross_attention_layer.cross_attention.Wv", new_k)
            new_k = re.sub(".layer.*.EncDecAttention.o", ".cross_attention_layer.cross_attention.o", new_k)
            new_k = re.sub(".layer.*.SelfAttention.relative_attention_bias.", ".self_attention_layer.self_attention.pe_encoding.relative_attention_bias.", new_k)
            new_k = new_k.replace(".layer.0.layer_norm.", ".self_attention_layer.layer_norm.")
            if "encoder" in new_k:
                new_k = new_k.replace(".layer.1.layer_norm.", ".ff_layer.layer_norm.")
            else:
                new_k = new_k.replace(".layer.1.layer_norm.", ".cross_attention_layer.layer_norm.")
            new_k = new_k.replace(".layer.2.layer_norm.", ".ff_layer.layer_norm.")
            new_k = re.sub(".layer.*.DenseReluDense.", ".ff_layer.", new_k)
            new_k = new_k.replace(".wi_", ".act.wi_")
            tensors[new_k] = f.get_tensor(k).clone()

    save_file(tensors, save_path)


if __name__ == "__main__":
    import shutil
    from transformers import T5EncoderModel
    
    model_path_base = 'Synthyra'
    model_path = os.path.join(model_path_base, 'ANKH_base')
    save_path_base = os.path.join(model_path_base, 'ANKH_base_flash')
    save_path = os.path.join(save_path_base, 'model.safetensors')

    if os.path.exists(model_path_base):
        shutil.rmtree(model_path_base)

    model = T5EncoderModel.from_pretrained('Synthyra/ANKH_base')
    model.save_pretrained(model_path, push_to_hub=False)

    current_safetensors = os.path.join(model_path, 'model.safetensors')
    os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
    convert_ankh(current_safetensors, save_path)