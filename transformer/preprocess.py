import random
import torch 
import torch.nn as nn
import special_tokens
import hparams
# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/generate_dataset.py

"""
This file contains mask and dataset generation (Transformer as a repeater) functions
"""

def generate_key_padding_mask(tokens: torch.LongTensor):
    return tokens == special_tokens.pad

def generate_causal_mask(tokens: torch.LongTensor):
    '''
    the return value is a diagonal matrix like:
    [[0. -inf -inf ... -inf]
     [0.  0.  -inf ... -inf]
     [0.  0.   0.  ... -inf]
     ...
     [0.  0.   0.  ...   0.]]
    
    shape = [seq_len, seq_len]
    '''
    return torch.triu(torch.full((tokens.shape[-1], tokens.shape[-1]),float('-inf'),dtype=torch.float32),diagonal=1)

def generate_dataset(batch_size, 
                     max_length # The maximum training context, 
                     ):
    src = []
    # 生成batch_size个句子
    for _ in range(batch_size):
        # 随机生成句子长度,长度为1到max_length-2,为<bos>和<eos>留出位置
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇,并在开头和结尾增加<bos>和<eos>
        random_nums = [special_tokens.bos] + [random.randint(1, hparams.vocab_size - 3) for _ in range(random_len)] + [special_tokens.eos]
        # 如果句子长度不足max_length,用<pad>进行填充
        random_nums = random_nums + [special_tokens.pad] * (max_length - random_len - 2)
        src.append(random_nums)

    # 将src转换为LongTensor
    src = torch.LongTensor(src)

    # tgt不要<eos>, 即<eos>不作为预测器输入,不预测之后的token
    # 将src中的<eos>替换为<pad>后,去掉最后一个token,作为tgt
    tgt = torch.where(src != special_tokens.eos, src, special_tokens.pad)[:, :-1]

    # tgt_y不要<bos>,即<bos>不作为预测的标签,只预测之后的token
    # 将src中的<bos>,即第一个token去掉后,作为tgt_y
    tgt_y = src[:, 1:]

    return src, tgt, tgt_y

# Generate rotary matrix for RoPE, only compute once in the model
# Copied from https://github.com/meta-llama/llama/blob/main/llama/model.py#L80
def generate_freq_cis(head_dim: int,
                      max_len:int = hparams.max_len,
                      base: float = 10000.0 # the base of generating rotary angles
                      ):
    # freqs = base ** (-2 * j / head_dim) freqs.shape = [head_dim // 2]

    freqs = torch.pow(base, -(torch.arange(0, head_dim, 2)[:(head_dim // 2)].float()) / head_dim)

    # positional index, idx.shape = [max_len]
    idx = torch.arange(max_len)

    """
    freqs = 
    [[0 * theta_0, 0 * theta_1, ..., 0 * theta_m]
     [1 * theta_0, 1 * theta_1, ..., 1 * theta_m]
     ...
     [n * theta_0, n * theta_1, ..., n * theta_m]]
     n = idx.shape[0], m = freqs.shape[0]
     theta_i = base ** (-2i / head_dim) i \in [0, m]
    """
    freqs = torch.outer(idx, freqs).float()

    '''
    freqs_cis[a, b] = cos(a * theta_b) + sin(a * theta_b) * j, where j is the unit of complex
    '''
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

