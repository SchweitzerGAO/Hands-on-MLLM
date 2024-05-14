import torch 
import torch.nn as nn
from config import ViTConfig
from attention import ViTAttention
from mlp import ViTMLP
from typing import Optional

class ViTEncoderBlock(nn.Module):
    """
    Encoder block of ViT
    Derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L327
    """
    def __init__(self,
                 config: ViTConfig,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        # attention sub-module
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # pre-norm
        self.attn_layer = ViTAttention(config)
        self.attn_dropout = nn.Dropout(config.p_dropout)

        # FFN sub-module
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # pre-norm
        self.ffn = ViTMLP(config)
        self.ffn_dropout = nn.Dropout(config.p_dropout)
    def forward(self, 
                hidden_state: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None,  # whether to mask some inputs of a head, transformers exxclusive built-in method but not usually used
                output_attention: bool = False):
        """
        hidden_state.shape = [batch_size, num_patches + 1 = seq_len, hidden_size]
        head_mask.shape = [batch_size, n_heads, seq_len, seq_len]
        """
        attn_out = self.attn_norm(hidden_state)
        attn_out = self.attn_layer(attn_out, 
                                   head_mask=head_mask, 
                                   output_attention=output_attention)
        context_layer = attn_out[0] # context_layer.shape = [bs, seq_len, hidden_size]
        attn_score = attn_out[1:] # attn_score.shape = [bs, n_heads, seq_len, seq_len]
        hidden_state = hidden_state + self.attn_dropout(context_layer) # residual connection

        ffn_out = self.ffn_norm(hidden_state)
        ffn_out = self.ffn(ffn_out) # ffn_out.shape = [bs, seq_len, hidden_size]
        hidden_state = hidden_state + self.ffn_dropout(ffn_out) # residual connection
        outputs = (hidden_state,) + attn_score
        return outputs

class ViTEncoder(nn.Module):
    """
    The encoder part of ViT
    Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L369
    """
    def __init__(self, 
                 config: ViTConfig,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.layers = nn.ModuleList([ViTEncoderBlock(config) for _ in range(config.num_layers)])
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Input: 
        hidden_state.shape = [batch_size, num_patches + 1 = seq_len, hidden_size]
        head_mask.shape = [num_layers, batch_size, n_heads, seq_len, seq_len]
        Output:
        hidden_state.shape = [batch_size, seq_len, hidden_size]
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_state,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_state, layer_head_mask, output_attentions)

            hidden_state = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_self_attentions] if v is not None)
        return dict(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

