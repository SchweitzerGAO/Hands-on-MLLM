import torch 
import torch.nn as nn
import torch.nn.functional as F
from config import ViTConfig
from typing import Optional
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image

class ViTPatchEmbedding(nn.Module):
    """
    Embed each patch of an image
    Derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L140

    """
    def __init__(self,
                 config: ViTConfig,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.image_size = (config.image_size, config.image_size)
        self.patch_size = (config.patch_size, config.patch_size)
        self.hidden_size = config.hidden_size
        self.num_patches = (self.image_size[1] // self.patch_size[1]) * (self.image_size[0] // self.patch_size[0])
        self.num_channels = config.num_channels
        
        """
        The projection layer
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, dilation, padding) x = 1,2,3
        input to Conv2d shape = [batch_size, in_channel, in_height, in_width] 
        output from Conv2d.shape = [batch_size, out_channel, out_height, out_width]
        the calculation of out_height and out_width: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        """
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
    
    def forward(self,
                image_pixels: torch.Tensor,
                ):
        """
        image_pixel.shape = [batch_size, num_channels, height, width]
        """

        embeddings = self.projection(image_pixels) # embeddings.shape = [batch_size, hidden_size, height / patch_size, width / patch_size]
        embeddings = embeddings.flatten(2).transpose(1, 2) # embeddings.shape = [batch_size, height * width / patch_size ** 2 = num_patches, hidden_size] 
        return embeddings

class ViTEmbedding(nn.Module):
    """
    Embedding layer of ViT
    Derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L60
    """
    def __init__(self,
                 config: ViTConfig,
                 use_mask_token: bool = False, # Currently unused, whether to mask the image
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config

        self.patch_embedding = ViTPatchEmbedding(config)
        # The learnable mask token, currently unused
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        # The learnable [class] token for image classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Number of patches
        num_patches = self.patch_embedding.num_patches

        # The learnable positional embedding
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        
        # The dropout
        self.dropout = nn.Dropout(config.p_dropout)


    def _interpolate_pos_embedding(self, 
                                   embeddings: torch.Tensor, 
                                   height: int, 
                                   width: int):
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        embedding.shape = [batch_size, num_patches + 1, hidden_size]
        """
        num_patches = embeddings.shape[1] - 1 # number of patches of the input image, maybe larger than the trained image size
        num_positions = self.position_embedding.shape[1] - 1 # number of the patches of trained model
        
        if num_patches == num_positions and height == width: # the input is a square image and the number of the patches is equal to the trained one
            return self.position_embedding
        
        cls_pos_embed = self.position_embedding[:, 0, :] # the [CLS] embedding should not be interpolated shape = [1, hidden_size]
        patch_pos_embed = self.position_embedding[:, 1:, :] # shape = [1, num_patches, hidden_size]

        hidden_size = embeddings.shape[-1]

        # There are at most h0 * w0 patches in total for the input
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(num_positions ** 0.5), int(num_positions ** 0.5), hidden_size)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2) # shape = [1, hidden_size, sqrt(num_positions), sqrt(num_positions)]
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / (num_positions ** 0.5), w0 / (num_positions ** 0.5)),
            mode="bicubic",
            align_corners=False,
        )  # shape = [1, hidden_size, h0, w0]

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, hidden_size) # shape = [1, h0 * w0, hidden_size]
        return torch.cat((cls_pos_embed.unsqueeze(0), patch_pos_embed), dim=1) # shape = [1, h0 * w0 + 1, hidden_size]
        

    def forward(self,
                image_pixels: torch.Tensor,
                image_mask: Optional[torch.BoolTensor] = None, # Currently unused, the mask of an image
                interpolate: bool = False
                ):
        """
        image_pixels.shape = [batch_size, num_channels, height, width]
        """
        interpolate = interpolate if interpolate is not None else self.config.interpolate
        batch_size, _, height, width = image_pixels.shape

        embeddings = self.patch_embedding(image_pixels)

        if image_mask is not None:
           pass

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # cls_tokens.shape = [batch_size, 1, hidden_size]
        embeddings = torch.cat((cls_tokens, embeddings), dim=1) # embeddings.shape = [batch_size, num_patches + 1, hidden_size]

        if interpolate:
            embeddings = embeddings + self._interpolate_pos_embedding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding # boardcast and add
        return self.dropout(embeddings)

if __name__ == '__main__':
    image_path = './test.jpg'
    transform = transforms.Compose([
        transforms.Resize(224), # resize to 224 * 224
        transforms.CenterCrop(224), # crop an 24 * 224 image from the center
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
    image = transform(Image.open(image_path))
    print(image.shape)
    # save_image(image, './no_norm.png')
    config = ViTConfig()
    emb = ViTEmbedding(config)
    print(emb(image.unsqueeze(0)).shape)



