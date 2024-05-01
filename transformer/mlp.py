import torch 
import torch.nn as nn
import hparams

# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/MyTransformer.py

class TransformerMLP(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff, # dimension of FFN upsample, usually 4 * d_model
                 p_dropout,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = d_model
        self.d_ff = d_ff

        # Layers
        self.linear1 = nn.Linear(self.hidden_size, self.d_ff) # after this shape = [batch_size, seq_len, d_ff]
        self.dropout = nn.Dropout(p_dropout) # Why ?
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.d_ff, self.hidden_size) # after this shape = [batch_size, seq_len, hidden_size]
    
    def forward(self, x: torch.Tensor):
        x = self.linear2(self.activation(self.dropout(self.linear1(x))))
        return x
    