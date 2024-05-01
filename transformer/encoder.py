import torch 
import torch.nn as nn
import hparams

from attention import TransformerMultiHeadAttention

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

        self.attn = TransformerMultiHeadAttention(self.hidden_size, self.n_heads)
        
