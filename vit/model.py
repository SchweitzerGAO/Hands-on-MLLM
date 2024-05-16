import torch 
import torch.nn as nn

from config import ViTConfig
from embedding import ViTEmbedding
from encoder import ViTEncoder
from pooler import ViTPooler

from typing import Optional

class ViTModel(nn.Module):
    """
    The ViT model without a LM head
    Derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L499
    """
    def __init__(self,
                 config: ViTConfig, 
                 use_mask_token: bool = False,
                 add_pool: bool = True, 
                 *args, 
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.config = config

        self.embedding = ViTEmbedding(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.ln_after_encoder = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.pooler =  ViTPooler(config) if add_pool else None
    
    def forward(self, 
                image_pixel: torch.Tensor,
                image_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                interpolate: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):
        """

        image_pixel.shape = [batch_size, num_channels, height, width]
        image_mask.shape = [batch_size, num_patches]
        head_mask.shape = [n_heads] OR [num_layers, n_heads]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        """
        Prepare head mask if needed
        1.0 in head_mask indicate we keep the head
        not used for now in this project
        """
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embedding(image_pixel, 
                                          image_mask, 
                                          interpolate) # embedding_output.shape = [bs, seq_len = num_patches + 1, hidden_size]
        """
        if not return_dict:
            encoder_outputs[0] is the output of last layer, shape = [bs, seq_len, hidden_size]
            encoder_outputs[1] is the output of all layers, each with shape = [bs, n_heads, seq_len, seq_len] if output_hidden_states else None
            encoder_outputs[2] is the attention score of all layers if output_attentions else None
        else:
            encoder_outputs['last_hidden_state'] is the output of last layer, shape = [bs, seq_len, hidden_size]
            encoder_outputs['hidden_states'] is the output of all layers if output_hidden_states else None
            encoder_outputs['attentions'] is the attention score of all layers, each with shape = [bs, n_heads, seq_len, seq_len] if output_attentions else None
        
        """
        encoder_outputs = self.encoder(embedding_output,
                                       head_mask,
                                       output_attentions,
                                       output_hidden_states,
                                       return_dict
                                      )
        
        last_hidden_state = encoder_outputs['last_hidden_state'] if return_dict else encoder_outputs[0]
        last_hidden_state = self.ln_after_encoder(last_hidden_state)
        pooled_output = self.pooler(last_hidden_state) if self.pooler is not None else None

        if not return_dict:
            outputs = (last_hidden_state, pooled_output)
            encoder_outputs = outputs + encoder_outputs[1:]
        else:
            encoder_outputs = dict(
                last_hidden_state=last_hidden_state,
                pooled_output=pooled_output,
                hidden_states=encoder_outputs['hidden_states'],
                attentions=encoder_outputs['attentions']
            )
        return encoder_outputs

class ViTForImageClassification(nn.Module):
    """
    Image classification ViT model
    Derived from: 
    """
    def __init__(self,
                 config: ViTConfig,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        self.model = ViTModel(config, add_pool=False)
        self.lm_head = nn.Linear(config.hidden_size, config.num_labels) # classification head

    
    def forward(self, 
                image_pixels: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                interpolate: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):
        encoder_outputs = self.model(image_pixels,
                                     head_mask=head_mask,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     interpolate=interpolate,
                                     return_dict=return_dict
                                     )
        last_hidden_state = encoder_outputs['last_hidden_state'] if return_dict else encoder_outputs[0]
        cls_hidden_state = last_hidden_state[:, 0, :] # shape = [1, hidden_size]
        logits = self.lm_head(cls_hidden_state) # shape = [1, num_labels]

        if not return_dict:
            outputs = (logits,) + outputs[2:]
        else:
            outputs = dict(
                logits=logits,
                hidden_states=encoder_outputs['hidden_states'],
                attentions=encoder_outputs['attentions']
            )
        return outputs
