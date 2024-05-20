import torch 
import torch.nn as nn
import hparams

# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/MyTransformer.py

class TransformerEmbedding(nn.Module):
    def __init__(self,
                 vocab_size, # size of vocabulary
                 d_model, # hidden size
                 max_len=hparams.max_len, # maximum training length
                 p_dropout=0.1, # dropout prob
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = d_model
        self.max_len = max_len

        self.pe = self._get_positional_encoding() # positional embedding

        self.embed = nn.Embedding(self.vocab_size, self.hidden_size) # Embedding layer
        self.dropout = nn.Dropout(p=p_dropout) # dropout layer, mentioned in vanilla transformer paper: https://arxiv.org/pdf/1706.03762 section 5.4

    def _get_positional_encoding(self, use_exp=True):
        """
        positional encoding 
        pe[:,0::2] = sin(pos / 10000 ** (2 * i / d_model))
        pe[:,1::2] = cos(pos / 10000 ** (2 * i / d_model))
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pe = torch.zeros(self.max_len, self.hidden_size).to(device)

        # generate 2-D positions
        # [[0],[1],[2],...,[max_len - 1]] shape = [max_len, 1]
        positions = torch.arange(0, self.max_len).unsqueeze(1) # unsqueeze(0) is wrong, error on broadcasting when executing positions * mul_term
        if use_exp:
            # Avoid overflow and underflow ?
            mul_term = torch.exp(torch.arange(0, self.hidden_size, 2) * -(torch.log(torch.tensor(10000.0)) / self.hidden_size))
        else:
            # Direct implementation
            mul_term = torch.pow(torch.tensor(10000.0), -torch.arange(0, self.hidden_size, 2).float() / self.hidden_size) # mul_term.shape = [hidden_size / 2]
        
        pe[:,0::2] = torch.sin(positions * mul_term)
        pe[:,1::2] = torch.cos(positions * mul_term)
        pe = pe.unsqueeze(0) # pe.shape = [1, max_len, hidden_size], convenient for broadcasting
        # pe.requires_grad_(False)
        return pe
        

    def forward(self, tokens: torch.LongTensor):
        """
        tokens.shape = [batch_size, seq_len]
        """
        # embed the tokens
        x = self.embed(tokens) # x.shape = [batch_size, seq_len, hidden_size]
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False) # add positional embedding
        return self.dropout(x) # dropout

class TransformerRotaryEmbedding(nn.Module):
    """
    Transformers-style RoPE implementation
    Reference: https://github.com/YueZhengMeng/MyLlama/blob/master/MyLlama.py#L52 
    and https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L96
    The major differences are:
    1. No complex-field computation
    2. group every d/2 position
    """
    def __init__(self, 
                 head_dim: int = hparams.hidden_size // hparams.n_heads,
                 max_len: int = hparams.max_len,
                 base: int = 10000,
                 device = None, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base
        # base ** (-2i / headdim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.head_dim))

        # 如果一个参数不参与梯度下降,但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        # persistent = False: do not save this field in the state_dict
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, 
                           seq_len: int, 
                           device: torch.device, 
                           dtype: torch.dtype):
        self.max_seq_len_cached = seq_len

        # Generate the positional indices
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.get_default_dtype())

        """
        freqs.shape = [seq_len, head_dim // 2]
        freqs = 
        [[0 * theta_0, 0 * theta_1, ..., 0 * theta_m]
         [1 * theta_0, 1 * theta_1, ..., 1 * theta_m]
         ...
         [n * theta_0, n * theta_1, ..., n * theta_m]]
         n = seq_len, m = head_dim // 2
        theta_i = base ** (-2i / head_dim) i \in [0, m]
        """
        freqs = torch.outer(t, self.inv_freq)

        """
        emb.shape = [seq_len, head_dim]
        embs = 
        [[0 * theta_0, 0 * theta_1, ..., 0 * theta_m, 0 * theta_0, 0 * theta_1, ..., 0 * theta_m]
         [1 * theta_0, 1 * theta_1, ..., 1 * theta_m, 1 * theta_0, 1 * theta_1, ..., 1 * theta_m],
         ...
         [n * theta_0, n * theta_1, ..., n * theta_m, n * theta_0, n * theta_1, ..., n * theta_m]]
         n = seq_len, m = head_dim // 2
        theta_i = base ** (-2i / head_dim) i \in [0, m]
        """
        emb = torch.cat((freqs, freqs),dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def _rotate_half(x: torch.Tensor):
        """
        x.shape = [batch_size, seq_len, n_heads, head_dim]
        returns:
        [-q_{d/2}, -q_{d/2 + 1}, ... -q{d}, q_{0}, q_{1},... q{d/2-1}]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                position_ids: torch.Tensor,
                unsqueeze_dim: int = 1
                ):
        """
        q.shape = [batch_size, seq_len, n_q_heads, head_dim]
        k.shape = [batch_size, seq_len, n_kv_heads, head_dim]
        position_ids is the newly seen length. e.g. 1st time inference seq_len = 10 then position_ids = torch.arange(10), second time seq_len=20, position_ids = torch.arange(10, 20)
        unsqueeze_dim, the dimension to unsqueeze
        """
        seq_len = q.shape[1]
        
        # the input seq_len is longer than the cached one, then update the cache
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=q.device, dtype=q.dtype)
        
        # cos_cached.shape = sin_cached.shape = [seq_len, head_dim]
        cos_cached = self.cos_cached[:seq_len].to(dtype=q.dtype)
        sin_cached = self.sin_cached[:seq_len].to(dtype=q.dtype)

        # cos_utilized.shape = sin_utilized.shape = [len(position_ids), 1, head_dim], to broadcast when performing hardmard product
        cos_utilized = cos_cached[position_ids].unsqueeze(unsqueeze_dim)
        sin_utilized = sin_cached[position_ids].unsqueeze(unsqueeze_dim)

        # The deduction of this is in RoPE.ipynb, source: https://github.com/YueZhengMeng/MyLlama/blob/master/RoPE.ipynb
        q_embedded = (q * cos_utilized) + self._rotate_half(q) * sin_utilized
        k_embedded = (k * cos_utilized) + self._rotate_half(k) * sin_utilized

        return q_embedded, k_embedded



if __name__ == '__main__':
    embed = TransformerEmbedding(
        vocab_size=hparams.vocab_size,
        d_model=hparams.hidden_size,
        max_len=hparams.max_len
    )
    batch_size = 2
    seq_len = 1024
    tokens = torch.randint(0,hparams.vocab_size,(batch_size, seq_len),dtype=torch.int64)
    shape = embed(tokens).shape
    assert shape[0] == batch_size and shape[1] == seq_len and shape[2] == hparams.hidden_size

