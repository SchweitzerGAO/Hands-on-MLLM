import torch 
import torch.nn as nn
import torch.nn.functional as F
from hparams import ViTConfig
from typing import Optional

class ViTAttention(nn.Module):
    """
    Attention for ViT
    Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L179
    """
    def __init__(self,
                 config: ViTConfig,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.qkv_bias = config.qkv_bias # whether to add bias to the linear projection layer of qkv

        self.Wq = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.dropout_attn = nn.Dropout(config.p_dropout)

        self.Wo = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout_output = nn.Dropout(config.p_dropout)

    def forward(self, 
                hidden_state: torch.Tensor, 
                head_mask: Optional[torch.Tensor] = None,  # whether to mask some inputs of a head, transformers exxclusive built-in method but not usually used
                output_attention: bool = False
                ):
        """
        hidden_state.shape = [batch_size, num_patches + 1 = seq_len, hidden_size]
        head_mask.shape = [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len = hidden_state.shape[:-1]
        # qkv projection
        q = self.Wq(hidden_state)
        k = self.Wk(hidden_state)
        v = self.Wv(hidden_state)

        # split the head, q.shape = k.shape = v.shape = [bs, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # do dot-product MHA, attn_score.shape = [bs, n_heads, seq_len, seq_len]
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        attn_score = F.softmax(attn_score, dim=-1) # this do not change the shape of attn_score
        attn_score = self.dropout_attn(attn_score)

        # mask some heads
        if head_mask is not None:
            attn_score = attn_score * head_mask
        
        # do the 'retrieval' operation
        context_layer = torch.matmul(attn_score, v) # context_layer.shape = [bs, n_heads, seq_len, head_dim]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_size)
        outputs = (context_layer, attn_score) if output_attention else (context_layer,)
        return outputs   