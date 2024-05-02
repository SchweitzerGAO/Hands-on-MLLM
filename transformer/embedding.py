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
        pe = torch.zeros(self.max_len, self.hidden_size)

        # generate 2-D positions
        # [[0],[1],[2],...,[max_len - 1]] shape = [max_len, 1]
        positions = torch.arange(0, self.max_len).unsqueeze(1) # unsqueeze(0) is wrong, error on broadcasting when executing positions * mul_term
        if use_exp:
            # Avoid overflow and underflow ?
            mul_term = torch.exp(torch.arange(0, self.hidden_size, 2) * -(torch.log(torch.tensor(10000.0)) / self.hidden_size))
        else:
            # Direct implementation
            mul_term = torch.pow(torch.tensor(10000.0), -torch.arange(0, self.hidden_size, 2) / self.hidden_size) # mul_term.shape = [hidden_size / 2,]
        
        pe[:,0::2] = torch.sin(positions * mul_term)
        pe[:,1::2] = torch.cos(positions * mul_term)
        pe = pe.unsqueeze(0) # pe.shape = [1, max_len, hidden_size], convenient for batchify computing
        pe.requires_grad_(False)
        return pe
        

    def forward(self, tokens: torch.LongTensor):
        """
        tokens.shape = [batch_size, seq_len]
        """
        # embed the tokens
        x = self.embed(tokens) # x.shape = [batch_size, seq_len, hidden_size]
        x = x + self.pe[:, :x.shape[1], :] # add positional embedding
        return self.dropout(x) # dropout

        


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