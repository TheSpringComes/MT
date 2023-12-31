from tqdm import tqdm
import torch
import torch.nn as nn

from evaluate import evlauate_gpt
from data.data_manager import get_data_for_gpt
from model.GPT import GPTModel
from model.AdapterGPT import AdapterGPTModel
from model.LoRAGPT import LoRAGPTModel
from model.PrefixGPT import PrefixGPTModel
from utils import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GPT', help='GPT-2')
parser.add_argument('--finetune', type=int, default=1, help='')
parser.add_argument('--k', type=int, default=1, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--epoch', type=int, default=1, help='')
parser.add_argument('--load', type=bool, default=False, help='')
parser.add_argument('--rank', type=int, default=32, help='')  # rank for lora
parser.add_argument('--prefix', type=int, default=20, help='')
args = parser.parse_args()

ZERO_SHOT = 0  # zero shot
NO_FRONZEN = 1  # 不冻结参数
PART_LAYERS_FRONZEN = 2  # 冻结部分层
ALL_LAYERS_FRONZEN = 3  # 除了最后一层，其他层都冻结
LAYER_BY_LAYER_UNFREEZE = 4  # 逐层解冻

ADAPTER = 5  # Adapter Tuning
PREFIX = 6  # PEFT: Prefix Tuning
LORA = 7  # LoRA


def validate(model:nn.Module,
              validdata, 
              device=torch.device('cuda:0')):
    loss = nn.CrossEntropyLoss(reduction='mean')
    model.to(device)
    model.eval()
    valid_loss = 0
    tqdm_iterator = tqdm(validdata, dynamic_ncols=True)
    for sentence in tqdm_iterator:
        out, label = model(sentence)
        l = loss(out.view(-1, out.size(-1)), label.view(-1))
        valid_loss += l.item()
    valid_loss /= len(validdata)
    return valid_loss


def finetune_epoch(model:nn.Module, train_data, valid_data, optimizer, loss, device, epoch, num_epochs):
    model.to(device).train()
    train_loss = []
    tqdm_iterator = tqdm(train_data, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')
    for sentence in tqdm_iterator:
        out, label = model(sentence)

        optimizer.zero_grad()
        l = loss(out.view(-1, out.size(-1)), label.view(-1))
        l.backward()
        optimizer.step()

        train_loss.append(l.item())
        tqdm_iterator.set_postfix(loss=sum(train_loss) / len(train_loss))
    valid_loss = validate(model, valid_data, device)
    print('valid loss', valid_loss)


def finetune(model:GPTModel, 
          train_data,
          valid_data,
          num_epochs=10, 
          lr=0.001, 
          weight_decay=0.0001,
          device=torch.device('cuda:0'), 
          finetune_method=NO_FRONZEN, 
          **kwargs):
    if finetune_method == ZERO_SHOT: return
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss(reduction='mean')
    if finetune_method == NO_FRONZEN:
        pass

    elif finetune_method == (ALL_LAYERS_FRONZEN or LAYER_BY_LAYER_UNFREEZE):  # 如果不是最后一层，就冻结
        for param in model.gpt.parameters():
            if param is not model.gpt.lm_head.weight:
                param.requires_grad = False

    elif finetune_method == PART_LAYERS_FRONZEN:
        k = kwargs['k']
        for param in model.gpt.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.gpt.transformer.wpe.parameters():
            param.requires_grad = False
        for i, layer in enumerate(model.gpt.transformer.h):  # 冻结前边k层transformer block
            if i < k:
                for param in layer.parameters():
                    param.requires_grad = False

    for epoch in range(num_epochs):
        if finetune_method == LAYER_BY_LAYER_UNFREEZE:
            k = kwargs['k']  # 最多解冻k层
            if epoch > 0 and epoch <= k:
                for param in model.gpt.transformer.h[-epoch].parameters():
                    param.requires_grad = True
        finetune_epoch(model, train_data, valid_data, optimizer, loss, device, epoch, num_epochs)
        torch.save(model.state_dict(), 'model/checkpoints_{}_{}.pth'.format(model.name, args.finetune))

    torch.save(model.state_dict(), 'model/checkpoints_{}_{}.pth'.format(model.name, args.finetune))

    


if __name__ == "__main__":
    # load data and load model
    train_stories, val_stories, test_stories = get_data_for_gpt(root="data")
    if args.finetune <= LAYER_BY_LAYER_UNFREEZE:
        model = GPTModel()
    elif args.finetune == ADAPTER:
        model = AdapterGPTModel()
    elif args.finetune == PREFIX:
        model = PrefixGPTModel(prefix_num = args.prefix)
    elif args.finetune == LORA:
        model = LoRAGPTModel(low_rank_size = args.rank)
    path = 'model/checkpoints_{}_{}.pth'.format(model.name, args.finetune)
    if args.finetune > ZERO_SHOT:
        if args.load:model.load_state_dict(torch.load(path))
    print("{} has parameters: {:.5f}M".format(model.name, sum(p.numel() for p in model.parameters())/1000000.0))
    
    # finetune
    device = torch.device('cuda:0')
    finetune(model, train_stories, val_stories, num_epochs=args.epoch, lr=args.lr, weight_decay=0.0001, device=device,
             finetune_method=args.finetune, k=args.k)
    
    # evaluate
    bleu1_score, bleu4_score = evlauate_gpt(model, test_stories, model.tokenizer, n_bleu=4, device=device, 
                                            do_sample=True, top_k=10, max_length=70, top_p=0.95, finetune_method=args.finetune)
    print('bleu1 score', bleu1_score)
    print('bleu4 score', bleu4_score)