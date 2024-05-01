import torch 
import torch.nn as nn
import hparams

# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/MyTransformer.py

class TransformerMLP(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff, # dimension of FFN upsample, usually 4 * d_model
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = d_model
        self.d_ff = d_ff

        # Layers
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor):
        x = self.linear2(self.linear1(x))
        return x
    