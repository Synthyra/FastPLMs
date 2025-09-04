"""
TODO
Future addition to current T5 PLM models to use much more efficient flex attention
Waiting for official support of learned relative position embeddings in the score_mode
https://github.com/pytorch-labs/attention-gym/issues/20
https://pytorch.org/blog/flexattention/
https://github.com/pytorch-labs/attention-gym/pull/84/commits/4d045e172474e2b964e3961d794177ceca1549f8
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from torch.nn.attention.flex_attention import flex_attention
from .t5_attention import T5AttentionTransformers


class T5FlexAttentionMixin:
    """
    A mixin class to enable flex attention in T5 models.
    
    This mixin replaces the standard attention mechanism in T5Attention with flex attention,
    preserving the position bias mechanism that T5 uses.
    """
    
    def __init__(self, compile_flex=None, softcap=None):
        """
        Initialize the T5FlexAttentionMixin.
        
        Args:
            compile_flex: Whether to compile the flex attention function.
            kernel_options: Optional kernel options for flex attention.
        """
        self.compile_flex = compile_flex and (torch.cuda.is_available() or torch.backends.mps.is_available())
        self.flex_attention_fn = torch.compile(flex_attention) if self.compile_flex else flex_attention
        self.softcap = softcap

    def _apply_flex_attention(self, query, key, value, position_bias=None, attention_mask=None, head_mask=None):
        # adapted from https://github.com/huggingface/transformers/blob/752ef3fd4e70869626ec70657a770a85c0ad9219/src/transformers/integrations/flex_attention.py
        def score_mod(score, b, h, q_idx, kv_idx):
            if self.softcap is not None:
                score = self.softcap * torch.tanh(score / self.softcap)
            if position_bias is not None:
                score = score + position_bias[0][h][q_idx][kv_idx]
            if attention_mask is not None:
                score = score + attention_mask[b][0][q_idx][kv_idx]
            if head_mask is not None:
                score = score + head_mask[b][h][q_idx][kv_idx]
            return score

        # Apply flex attention
        attn_output, attention_weights = self.flex_attention_fn(
            query=query,
            key=key,
            value=value,
            score_mod=score_mod,
            enable_gqa=True,
            scale=1.0,
            return_lse=True,
        )
        return attn_output, attention_weights


class T5FlexAttention(nn.Module, T5FlexAttentionMixin):
    """
    Drop-in replacement for T5Attention that uses flex attention.
    
    This class preserves the interface of T5Attention but uses flex attention
    for the core attention computation.
    """
    
    def __init__(self, config, has_relative_attention_bias=False, layer_idx=None, compile_flex=False):
        nn.Module.__init__(self)
        T5FlexAttentionMixin.__init__(self, compile_flex=compile_flex)
        
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.layer_idx = layer_idx
        
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        
        # Regular T5 projection layers
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        
        self.pruned_heads = set()
        
    def prune_heads(self, heads):
        # Implementation similar to T5Attention.prune_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from T5Attention._relative_position_bucket
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / torch.log(torch.tensor(max_distance / max_exact, device=relative_position.device))
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """
        Compute binned relative position bias, same as in T5Attention
        """
        if device is None:
            device = self.relative_attention_bias.weight.device
        
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
            
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values
    
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Forward pass, similar to T5Attention but using flex attention
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None
        
        # Project hidden states to query vectors
        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        # Handle past key values for caching
        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache
        
        # Get current states for key and value
        current_states = key_value_states if is_cross_attention else hidden_states
        
        # Reuse cached key/value states if available and updated
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            # Compute new key and value states
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            
            # Update cache if needed
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True
        
        # Compute position bias if not provided
        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else (cache_position[-1] + 1 if cache_position is not None else seq_length)
            
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=query_states.device, dtype=query_states.dtype
                )
                if getattr(self, 'gradient_checkpointing', False) and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=query_states.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]
            
            # Add mask to position bias if provided
            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask
        
        # Apply pruned heads if any
        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias
        
        # Apply flex attention
        attn_output, attention_weights = self._apply_flex_attention(
            query=query_states, 
            key=key_states, 
            value=value_states, 
            position_bias=position_bias_masked,
            attention_mask=mask,
            head_mask=layer_head_mask
        )
        
        # Reshape output and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)
        
        # Prepare outputs
        outputs = (attn_output, past_key_value, position_bias)
        if output_attentions:
            outputs = outputs + (attention_weights,)
        return outputs


def replace_t5_attention_with_flex(model, compile_flex=False):
    """
    Replace all T5Attention modules in a T5 model with T5FlexAttention.
    
    Args:
        model: A T5 model instance
        
    Returns:
        The modified model with flex attention
    """    
    # Recursively replace all T5Attention modules
    for name, module in model.named_children():
        if isinstance(module, T5AttentionTransformers):
            # Create a new T5FlexAttention with the same parameters
            flex_attn = T5FlexAttention(
                config=model.config,
                has_relative_attention_bias=module.has_relative_attention_bias,
                layer_idx=module.layer_idx,
                compile_flex=compile_flex
            )
            
            # Copy weights
            flex_attn.q.weight.data = module.q.weight.data.clone()
            flex_attn.k.weight.data = module.k.weight.data.clone()
            flex_attn.v.weight.data = module.v.weight.data.clone()
            flex_attn.o.weight.data = module.o.weight.data.clone()
            
            if module.has_relative_attention_bias:
                flex_attn.relative_attention_bias.weight.data = module.relative_attention_bias.weight.data.clone()
            
            # Replace the module
            setattr(model, name, flex_attn)
        else:
            # Recursively process child modules
            replace_t5_attention_with_flex(module)
    
    return model



if __name__ == "__main__":
    # py -m wip.t5.t5_flex_attention
    from transformers import T5Config

    config = T5Config()

    attention_layer = T5FlexAttention(config=config, has_relative_attention_bias=True, layer_idx=0, compile_flex=False)
    hidden_states = torch.randn(1, 10, config.d_model)

    output = attention_layer(hidden_states)

    print(output)
