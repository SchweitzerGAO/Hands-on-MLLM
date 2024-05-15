import torch 
import torch.nn as nn
from config import ViTConfig

class ViTPooler(nn.Module):
    """
    A pooler of the last hidden state of [CLS], where does this idea come from?
    A reasonable explanation is: This pooler converts BERT-style embeddings to downstream task embedding, which may improve the performance
    Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L594
    """
    def __init__(self, 
                 config: ViTConfig,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        self.projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_state: torch.Tensor):
        """
        hidden_state.shape = [batch_size, num_patches + 1 = seq_len, hidden_size]
        """
        cls_hidden = hidden_state[:, 0] # cls_hidden.shape = [bs, hidden_size]
        return self.activation(self.projector(cls_hidden))

