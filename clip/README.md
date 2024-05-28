# CLIP

## Basic gadgets of CLIP

- An image encoder

The `transformers` implementation uses ViT as the image encoder, which I implemented [here](https://github.com/SchweitzerGAO/Hands-on-MLLM/tree/main/vit)

- A text encoder

The `transformers` implementation uses vanilla transformer encoder as the text encoder, which I implemented [here](https://github.com/SchweitzerGAO/Hands-on-MLLM/tree/main/transformer)

## Loss function of CLIP

The core innovation of CLIP is introducing contrastive learning loss to the image-language joint training, which optimizes the encoders by simply calculating the cross entropy loss of  $\argmax(\text{softmax(inner\_product\_similarty}(T, I)))$ and `torch.arange(len(T))` .

$T$ stands for the hidden state in a batched input for the special token `[EOT]` indicating the end of text

$I$ stands for the hidden state in a batched input for the special token `[CLS]` indicating the class of the image. 

The steps for CLIP loss computation:

1. Take the vector $T$ and vector $I$ from the batched input, the shape of $T$ and $I$ is `[batch_size, hidden_size]`

2. calculate $S =TI^\top$ to obtain the inner product. shape of $S$ is `[batch_size, batch_size]`

3. calculate `text_loss = cross_entropy(S, torch.arange(len(S))` and `image_loss = cross_entropy(S.t(), torch.arange(len(S.t())`

4. The final `loss = (text_loss + image_loss) / 2`

The `transformers` code for CLIP loss is as below:

```py
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0
```

## Some questions and possible answers

- Why `[EOT]` for text representation?

- Why decoder framework for text encoder?

The possible answers of these 2 questions can be found here in the paper

![](img/1.PNG)

Personally speaking, the `[EOT]` is treated as the text feature because decoder framework is used. If we use encoder framework(BERT-style) text encoders, I think `[CLS]` will be treated as the text feature
