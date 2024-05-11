import torch 
import torch.nn as nn
import torch.nn.functional as F
from hparams import ViTConfig

class ViTAttention(nn.Module):
    def __init__(self,
                  
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)