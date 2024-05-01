import torch 
import torch.nn as nn
import hparams

# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/MyTransformer.py

from attention import TransformerMultiHeadAttention
from mlp import TransformerMLP

class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
                 d_model,
                 n_heads,
                 d_ff, # dimension of FFN upsample, usually 4 * d_model
                 p_dropout=0.1,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        # Self-attention layer
        self.attn = TransformerMultiHeadAttention(d_model=self.hidden_size, n_heads=self.n_heads)
        self.dropout_1 = nn.Dropout(p_dropout)
        self.norm_1 = nn.LayerNorm(self.hidden_size)

        # FFN layer
        self.ffn = TransformerMLP(d_model=self.hidden_size, d_ff=self.d_ff)
        self.dropout_2 = nn.Dropout(p_dropout)
        self.norm_2 = nn.LayerNorm(self.hidden_size)
    
    def forward(x: torch.Tensor,src_key_padding_mask=None):
        pass

    if __name__ == '__main__':
        pass

        