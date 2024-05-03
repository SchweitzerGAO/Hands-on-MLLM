import random
import numpy as np
import torch 
import torch.nn as nn
import hparams
import special_tokens
# Reference: https://github.com/YueZhengMeng/MyTransformer/blob/master/main.py

from model import Transformer
from preprocess import generate_causal_mask, generate_dataset, generate_key_padding_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(hparams.vocab_size, hparams.hidden_size, hparams.n_heads, hparams.num_layers)

def seed_everything(seed):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(batch_size=hparams.batch_size,
          max_length=hparams.max_context_len,
          epoch=hparams.epoch,
          lr=hparams.lr,
          log_step=hparams.log_step):
    # loss function: cross entropy
    loss_fn = nn.CrossEntropyLoss() # softmax embedded in cross entropy loss function

    # optimizer: Adam, hyper-params identical to vanilla transformer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # lr scheduler: cosine annealing 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=8e-6)

    total_loss = 0
    # start training
    for i in range(epoch):
        model.train()
        src, tgt, tgt_y = generate_dataset(batch_size=batch_size, max_length=max_length)
        causal_mask = generate_causal_mask(tgt) # causal_mask of target
        src_key_padding_mask = generate_key_padding_mask(src)
        tgt_key_padding_mask = generate_key_padding_mask(tgt)

        optimizer.zero_grad() # clear the gradients

        # move the tensors to the same device
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_y = tgt_y.to(device)
        causal_mask = causal_mask.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(device)

        # get the output of model, out.shape = [batch_size, max_length, d_model]
        out = model(src, tgt, causal_mask, src_key_padding_mask, tgt_key_padding_mask)
        out = model.lm_head(out)

        # this loss mask is used to mask the padding tokens and omit the loss calculation
        loss_mask = tgt_key_padding_mask.view(-1) # loss_mask.shape = [batch_size * max_length]

        # reshape the prediction to fit the input of cross entropy
        prediction = out.view(-1, hparams.vocab_size)[loss_mask] # prediction.shape = [batch_size * max_length, vocab_size]
        target = tgt_y.view(-1)[loss_mask] # target.shape = [batch_size * max_length]

        # calculate loss, in this step, an implicit softmax is done and the copy of prediction's shape will be the same as target's
        loss = loss_fn(prediction, target)
        
        # calculate grads
        loss.backward()
        # update model parameters
        optimizer.step()
        # update lr
        scheduler.step()
        
        total_loss += loss
        if i % log_step == 0 and i != 0:
            print("#####################")
            print(f'Step {i}: loss:{total_loss / log_step}')
            evaluate()
            total_loss = 0

def evaluate(max_length=hparams.max_context_len):
    model.eval()
    test_src, _, _ = generate_dataset(batch_size=1, max_length=max_length)
    # No need to generate tgt_key_padding_mask as not predicted
    predict_tgt = torch.LongTensor([[special_tokens.bos]])
    src_key_padding_mask = generate_key_padding_mask(test_src)
    causal_mask = generate_causal_mask(predict_tgt)
    
    # move to same device
    test_src = test_src.to(device)
    predict_tgt = predict_tgt.to(device)
    src_key_padding_mask = src_key_padding_mask.to(device)
    causal_mask = causal_mask.to(device)

    for _ in range(max_length):
        out = model(test_src, predict_tgt, causal_mask, src_key_padding_mask, None)
        out = out[:, -1, :] # equivalent to out[:,-1], consider the last one only, shape = [batch_size, vocab_size]
        out = model.lm_head(out)
        new_token = torch.argmax(out, dim=1) # shape = [batch_size]
        predict_tgt = torch.concat([predict_tgt, new_token.unsqueeze(1)],dim=1) # unsqueeze(0) will also do but less robust

        # re-calculate tgt_causal_mask
        causal_mask = generate_causal_mask(predict_tgt)
        causal_mask = causal_mask.to(device)
        if new_token == special_tokens.eos:
            break
    print(f'src:{test_src}')
    print(f'prediction:{predict_tgt}')



seed_everything(hparams.seed)
train()