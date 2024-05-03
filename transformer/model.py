import torch 
import torch.nn as nn

# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/MyTransformer.py

from embedding import TransformerEmbedding
from encoder import TransformerEncoder
from decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 num_layers: int,
                 p_dropout: float=0.1,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = TransformerEmbedding(vocab_size, d_model, p_dropout=p_dropout)
        self.encoder = TransformerEncoder(d_model, n_heads, num_layers, p_dropout=p_dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, num_layers,p_dropout=p_dropout)

        # Linear LM head
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, 
                src_tokens: torch.LongTensor,
                tgt_tokens: torch.LongTensor,
                tgt_causal_mask: torch.Tensor=None,
                src_key_padding_mask: torch.Tensor=None, 
                tgt_key_padding_mask: torch.Tensor=None):
        src_embed = self.embedding(src_tokens)
        tgt_embed = self.embedding(tgt_tokens)

        memory = self.encoder(src_embed,src_key_padding_mask)
        out = self.decoder(tgt_embed, memory, tgt_causal_mask, tgt_key_padding_mask, src_key_padding_mask)
        # return out because the pipelines of training and inferencing are different. The lm_head layer will be used outside this function
        return out


