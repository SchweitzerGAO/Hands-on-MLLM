# ViT model related
class ViTConfig:
    def __init__(self,
                 hidden_size: int = 768,
                 num_layers: int = 3,
                 n_heads: int = 8,
                 image_size: int = 224,
                 patch_size: int = 16,
                 layer_norm_eps: float = 1e-12,
                 num_channels: int = 3, # RGB channel by default
                 qkv_bias: bool = True) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.p_dropout = 0.1
        self.image_size = image_size #  size of the input image. actual image size = image_size ** 2
        self.patch_size = patch_size #  size of the patch size, actual patch size = patch_size ** 2
        self.layer_norm_eps = layer_norm_eps
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias # if the bias shall be set in a linear layer
