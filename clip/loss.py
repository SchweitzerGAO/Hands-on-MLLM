import torch
import torch.nn as nn

"""
CLIP loss
Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L53
"""
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0
