import torch 
import torch.nn as nn
import torch.nn.functional as F
import hparams

# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/MyTransformer.py

class TransformerMultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model, # hidden size of the model
                 n_heads, # number of heads
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # assert d_model % n_heads == 0, "d_model must be a multiple of n_heads"
        self.hidden_size = d_model
        self.n_heads = n_heads
        self.head_dim = self.hidden_size // self.n_heads # dimension of each head
        # Linear projective layers of Q K and V
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # will be splitted into head_dim * n_heads
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Linear projective layer of the concatenated output
        self.Wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                padding_mask: torch.Tensor=None,
                causal_mask:torch.Tensor=None
                ):
        """
        q.shape = k.shape = v.shape = [batch_size, seq_len, hidden_size]
        padding_mask.shape = [batch_size, seq_len]
        causal_mask.shape = [seq_len, seq_len]
        """
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        # linear projection of q, k and v, after this q.shape = v.shape = k.shape = [batch_size, seq_len, hidden_size]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # split the head of q, k and v for parallelism, use torch.Tensor.reshape may be safer
        q = q.view(batch_size, -1, self.n_heads, self.head_dim) # q.shape = [batch_size, seq_len, n_heads, head_dim]
        q = q.transpose(1, 2) # q.shape = [batch_size, n_heads, seq_len, head_dim]
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) # same as above
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) # same as above
        k = k.transpose(-2, -1) # k.shape = [batch_size, n_heads, head_dim, seq_len]
        attn_score: torch.Tensor = torch.matmul(q, k) / (self.head_dim ** 0.5) # attn_score.shape = [batch_size, n_heads, seq_len, seq_len]

        # padding mask
        if padding_mask is not None:
            # After this padding_mask.shape = [batch_size, n_heads, seq_len, seq_len]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_heads, seq_len, 1)
            '''
            The document for masked_fill is useless. Referring to https://blog.csdn.net/jianyingyao7658/article/details/103382654, 
            we can know that the padding_mask is of type `Tensor[bool]` and when `padding_mask` is `True`, the element at this position will be set to `value` 
            '''
            attn_score = attn_score.masked_fill(padding_mask, value=float('-inf'))
        if causal_mask is not None:
            # After this causal_mask.shape = [batch_size, n_heads, seq_len, seq_len]
            causal_mask = causal_mask.expand(batch_size, self.n_heads, seq_len, seq_len)
            # add the causal mask
            attn_score = attn_score + causal_mask

        # softmax
        attn_score = F.softmax(attn_score,dim=-1) # do softmax in the lowest dimension

        # 'Retrieve' in the values by matmul with V
        '''
        Before:
        attn_score.shape = [batch_size, n_heads, seq_len, seq_len]
        v.shape = [batch_size, n_heads, seq_len, head_dim]
        After:
        attn_score.shape = [batch_size, n_heads, seq_len, head_dim]
        '''
        attn_score = torch.matmul(attn_score, v)

        # Concat the heads, after this, attn_score.shape = [batch_size, seq_len, hidden_size]
        attn_score = attn_score.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.Wo(attn_score) # The last projection to generalize the knowledge of heads, attn_score = [batch_size, seq_len, hidden_size]


# Identical to LLaMA 2, RoPE is leveraged along with GQA
# Reference 
# 1. https://github.com/meta-llama/llama/blob/main/llama/model.py
# 2. https://mp.weixin.qq.com/s/1kH1Ht58cRfl2kR_KzNilw

class TransformerGroupQueryAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_kv_heads: int, # number pf groups = n_heads // n_kv_heads
                 max_len: int = hparams.max_len,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_len = max_len

        self.n_group = n_heads // n_kv_heads # number of groups
        self.head_dim = self.hidden_size // self.n_heads

        # Linear projective layers of Q K and V
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # will be splitted into head_dim * n_heads
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Linear projective layer of the concatenated output
        self.Wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
    def _reshape_for_broadcast(freqs_cis: torch.Tensor,
                               x: torch.Tensor):
        """
        Reshape freqs_cis to the shape of x
        Copied from https://github.com/meta-llama/llama/blob/main/llama/model.py#L107
        freqs_cis.shape = [seq_len, head_dim // 2]
        x.shape = [batch_size, seq_len, n_heads, head_dim // 2]
        """
        ndim = x.ndim
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # shape = [1, seq_len, 1, head_dim // 2]
        return freqs_cis.view(*shape)


    def _apply_rotary_emb(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          freqs_cis: torch.Tensor):
        """
        Apply rotary embedding
        Copied from https://github.com/meta-llama/llama/blob/main/llama/model.py#L132
        q.shape = k.shape = [batch_size, seq_len, n_heads, head_dim]
        freqs_cis.shape = [seq_len, head_dim]
        """
        batch_size, seq_len, n_heads = q.shape[:3]
        # reshape q and k to complex
        # complex_q.shape = complex_k.shape = [batch_size, seq_len, n_heads, head_dim // 2, 2]
        complex_q = torch.view_as_complex(q.float().reshape(batch_size, seq_len, n_heads, -1, 2))
        complex_k = torch.view_as_complex(k.float().reshape(batch_size, seq_len, n_heads, -1, 2))

        # reshape freqs_cis to broadcast
        freqs_cis = self._reshape_for_broadcast(freqs_cis, complex_q)

        # perform complex multiplication
        # real_q_with_rope.shape = real_k_with_rope.shape = [batch_size, seq_len, n_heads, head_dim]
        real_q_with_rope = torch.view_as_real(complex_q * freqs_cis).flatten(3)
        real_k_with_rope = torch.view_as_real(complex_k * freqs_cis).flatten(3)

        return real_q_with_rope.type_as(q), real_k_with_rope.type_as(k)
        
    

    def forward(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                freqs_cis: torch.Tensor,
                padding_mask: torch.Tensor=None,
                causal_mask:torch.Tensor=None):
        pass




if __name__ == '__main__':
    mha = TransformerMultiHeadAttention(
        d_model=hparams.hidden_size,
        n_heads=hparams.n_heads,
    )
    batch_size = 2
    seq_len = 1024
    q = torch.randn(batch_size, seq_len, hparams.hidden_size)
    k = torch.randn(batch_size, seq_len, hparams.hidden_size)
    v = torch.randn(batch_size, seq_len, hparams.hidden_size)
    print(mha(q, k, v).shape)
        
        
