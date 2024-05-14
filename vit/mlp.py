import torch 
import torch.nn as nn
from config import ViTConfig

class ViTMLP(nn.Module):
    """
    Derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L296
    """
    def __init__(self,
                 config: ViTConfig,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        self.up_sample = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.p_dropout)
        self.down_sample = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.down_sample(self.activation(self.dropout(self.up_sample(hidden_state))))
        return hidden_state