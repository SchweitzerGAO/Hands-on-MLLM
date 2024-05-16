# ViT model related
class ViTConfig:
    def __init__(self,
                 hidden_size: int = 768,
                 num_layers: int = 3,
                 n_heads: int = 8,
                 image_size: int = 224,
                 patch_size: int = 16,
                 interpolate: bool = False,
                 layer_norm_eps: float = 1e-12,
                 num_channels: int = 3, # RGB channel by default
                 qkv_bias: bool = True,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 return_dict: bool = True,
                 num_labels: int = 0) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = 4 * hidden_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.p_dropout = 0.0
        self.image_size = image_size  # size of the input image. actual image size = image_size ** 2
        self.patch_size = patch_size  # size of the patch size, actual patch size = patch_size ** 2
        self.interpolate = interpolate
        self.layer_norm_eps = layer_norm_eps
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias  # if the bias shall be set in a linear layer
        self.output_attentions = output_attentions  # whether to output the attention scores
        self.output_hidden_states = output_hidden_states  # whether to output all of the hidden states
        self.return_dict = return_dict # whether return the output in a form of dict, if set False, return in a form of tuple

        """
        Classification head config
        """
        assert num_labels > 0
        self.num_labels = num_labels
        
        """
        Masked image modeling head config
        """

        
