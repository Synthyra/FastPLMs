from transformers import PretrainedConfig


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 20,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        ffn_type: str = "swiglu",
        expansion_ratio: float = 3.0,
        dropout: float = 0.0,
        bias: bool = False,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attn_backend: str = "auto",
        clip_qkv: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.ffn_type = ffn_type
        self.expansion_ratio = expansion_ratio
        self.dropout = dropout
        self.bias = bias
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attn_backend = attn_backend
        assert clip_qkv is None or clip_qkv > 0
        self.clip_qkv = clip_qkv
