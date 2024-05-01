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
                 p_dropout,
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
        self.ffn = TransformerMLP(d_model=self.hidden_size, d_ff=self.d_ff, p_dropout=p_dropout)
        self.dropout_2 = nn.Dropout(p_dropout)
        self.norm_2 = nn.LayerNorm(self.hidden_size)
    def forward(self, 
                src: torch.Tensor,
                src_key_padding_mask:torch.Tensor=None):
        # Self attention sub-layer
        attn_score = self.attn(src, src, src, padding_mask=src_key_padding_mask) # attn_score.shape = [batch_size, seq_len, hidden_size]
        add_norm_1 = src + self.dropout_1(attn_score)
        add_norm_1 = self.norm_1(add_norm_1) # add_norm_1.shape = attn_score.shape

        # FFN sub-layer
        ffn_out = self.ffn(add_norm_1)
        add_norm_2 = add_norm_1 + self.dropout_2(ffn_out)
        add_norm_2 = self.norm_2(add_norm_2) # add_norm_2.shape = add_norm_1.shape
        return add_norm_2

class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 num_layers,
                 p_dropout=0.1, 
                 *args, 
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        encoder_block = TransformerEncoderBlock(d_model=d_model,
                                                n_heads=n_heads,
                                                d_ff=4 * d_model,
                                                p_dropout=p_dropout)
        self.layers = nn.ModuleList([encoder_block for _ in range(num_layers)])
    def forward(self, 
                src:torch.Tensor,
                src_key_padding_mask:torch.Tensor=None):
        for block in self.layers:
            src = block(src, src_key_padding_mask)
        return src

        

if __name__ == '__main__':
    encoder = TransformerEncoder(d_model=hparams.hidden_size,n_heads=hparams.n_heads,num_layers=hparams.num_encoder_layers)
    batch_size = 2
    seq_len = 1024
    src = torch.randn(batch_size, seq_len, hparams.hidden_size)
    print(encoder(src).shape)