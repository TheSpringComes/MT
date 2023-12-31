from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluate import evaluate
from utils import *
from data.data_manager import get_data
from data.data_loader import load_data
from model.LSTM import LSTM_model
from model.Transformer import TransformerDecoder_model, TransformerEncoderDecoder_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Transformer', help='LSTM, Transformer')
parser.add_argument('--epoch', type=int, default=5, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--opt', type=str, default="adam", help='')
parser.add_argument('--load', type=bool, default=False, help='')
parser.add_argument('--layers', type=int, default=6, help='')
args = parser.parse_args()

def validate(model:nn.Module, validloader:DataLoader, device=try_gpu()):
    loss = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)
    valid_loss = []
    for x,y in validloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        l = loss(out.view(-1, out.size(-1)), y.view(-1))
        valid_loss.append(l.item())
    return sum(valid_loss) / len(valid_loss)

def train(model:nn.Module, 
          trainloader:DataLoader, 
          validloader:DataLoader, 
          num_epochs=10, 
          lr=0.001, 
          weight_decay=0.0001,
          device=try_gpu()):
    if args.opt.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args.opt.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.05)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        tqdm_iterator = tqdm(trainloader, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for x,y in  tqdm_iterator:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            l = loss(out.view(-1, out.size(-1)), y.view(-1))
            l.backward()
            optimizer.step()

            train_loss.append(l.item())

            tqdm_iterator.set_postfix(loss=sum(train_loss) / len(train_loss))
    valid_loss = validate(model, validloader, device)
    print('valid loss', valid_loss)
    torch.save(model.state_dict(), 'model/checkpoints_{}.pth'.format(model.name))


if __name__ == '__main__':
    # data
    word_dict, train_tokens, test_tokens, val_tokens = get_data(max_len=100)
    trainloader, validloader = load_data([train_tokens, val_tokens], batch_size=128, method=TOTAL_LENGTH)

    # model
    if args.model.lower() == 'lstm':
        model = LSTM_model(len(word_dict))
    elif args.model.lower() == 'transformer':
        model = TransformerEncoderDecoder_model(len(word_dict), num_layers=args.layers)
    elif args.model.lower() == 'transformer_decoder':
        model = TransformerDecoder_model(len(word_dict), num_layers=args.layers)
    
    print("{} has parameters: {:.5f}M".format(model.name, sum(p.numel() for p in model.parameters())/1000000.0))
    path = 'model/checkpoints_{}.pth'.format(model.name)
    if args.load:
        model.load_state_dict(torch.load(path))
        print("Load model from {}".format(path))
    # train
    for i in range(1):
        train(model, trainloader, validloader, 
          num_epochs=args.epoch, lr=args.lr/(i+1), weight_decay=0, device=try_gpu())

    # test
    print("---------------------Evaluating ... ------------------")
    bleu1_score, bleu4_score = evaluate(model, test_tokens, word_dict, n_bleu=4, method=BEAM_SEARCH, beam_width=5)
    print("method: Beam Search")
    print("Bleu-1 Score: {}".format(bleu1_score))
    print("Bleu-4 Score: {}".format(bleu4_score))