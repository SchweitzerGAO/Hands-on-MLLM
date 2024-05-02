import torch 
import torch.nn as nn
import hparams

# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/MyTransformer.py

from attention import TransformerMultiHeadAttention
from mlp import TransformerMLP

class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 d_ff,
                 p_dropout, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Self attention sub-layer
        self.self_attn = TransformerMultiHeadAttention(d_model=self.hidden_size, n_heads=self.n_heads)
        self.dropout_1 = nn.Dropout(p_dropout)
        self.norm_1 = nn.LayerNorm(self.hidden_size)

        # Cross attention sub-layer
        self.cross_attn = TransformerMultiHeadAttention(d_model=self.hidden_size, n_heads=self.n_heads)
        self.dropout_2 = nn.Dropout(p_dropout)
        self.norm_2 = nn.LayerNorm(self.hidden_size)

        # FFN Layer
        self.ffn = TransformerMLP(d_model=self.hidden_size, d_ff=self.d_ff,p_dropout=p_dropout)
        self.dropout_3 = nn.Dropout(p_dropout)
        self.norm_3 = nn.LayerNorm(self.hidden_size)
    
    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor, # encoder output as K and V
                tgt_causal_mask: torch.Tensor=None,
                tgt_key_padding_mask: torch.Tensor=None,
                memory_key_padding_mask: torch.Tensor=None):
        # Self attention
        self_attn_score = self.self_attn(tgt, tgt, tgt, padding_mask=tgt_key_padding_mask,causal_mask=tgt_causal_mask)
        self_attn_score = tgt + self.dropout_1(self_attn_score)
        self_attn_score = self.norm_1(self_attn_score) # self_attn_score.shape = [batch_size, seq_len, hidden_size]

        # Cross attention
        cross_attn_score = self.cross_attn(self_attn_score, memory, memory,padding_mask=memory_key_padding_mask,causal_mask=None)
        cross_attn_score = self_attn_score + self.dropout_2(cross_attn_score)
        cross_attn_score = self.norm_2(cross_attn_score)

        # FFN
        ffn_out = self.ffn(cross_attn_score)
        ffn_out = cross_attn_score + self.dropout_3(ffn_out)
        ffn_out = self.norm_3(ffn_out)

        return ffn_out

class TransformerDecoder(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 num_layers,
                 p_dropout=0.1, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        decoder_block = TransformerDecoderBlock(d_model=d_model, 
                                                n_heads=n_heads, 
                                                d_ff=4 * d_model,
                                                p_dropout=p_dropout)
        self.layers = nn.ModuleList([decoder_block for _ in range(num_layers)]) # nn.Sequential may also do

    def forward(self,
                tgt:torch.Tensor, 
                memory: torch.Tensor,
                tgt_causal_mask: torch.Tensor=None,
                tgt_key_padding_mask: torch.Tensor=None,
                memory_key_padding_mask: torch.Tensor=None):
        for layer in self.layers:
            tgt = layer(tgt,memory,tgt_causal_mask,tgt_key_padding_mask,memory_key_padding_mask)
        return tgt




