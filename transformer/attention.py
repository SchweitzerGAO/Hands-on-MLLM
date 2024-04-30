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
        assert d_model % n_heads == 0, "d_model must be a multiple of n_heads"
        self.hidden_size = d_model
        self.n_heads = n_heads
        self.head_dim = self.hidden_size // self.n_heads
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
            # The document for masked_fill is useless
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

        # Concat the heads
        attn_score = attn_score.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.Wo(attn_score) # The last projection to generalize the knowledge of heads




if __name__ == '__main__':
    pass
        
        
