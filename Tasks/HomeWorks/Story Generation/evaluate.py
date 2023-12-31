import math
import collections
import torch
import torch.nn as nn
import json
import re
from tqdm import tqdm

from utils import *
from predict import predict
from data.data_manager import get_data, get_word
from model.GPT import GPTModel
import argparse


RECIPROCAL = 0  # 在算BLEU的时候，W_n = 1/n
EXPONENT = 1    # 在算BLEU的时候，W_n = 1/2^n

def bleu(pred_seq, label_seq, k:int=4, method = RECIPROCAL):  # 准确率
    """计算BLEU"""
    # 指数
    pred_tokens, label_tokens = list(pred_seq), list(label_seq)  # 深拷贝
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))  # BP

    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        # P_n = num_matches / (len_pred - n + 1)
        # BP = min(1, math.exp(1 - len_label / len_pred))
        # score = BP
        # score *= math.pow(P_n, 1 / k)
        if method == RECIPROCAL:
            score *= math.pow(num_matches / (len_pred - n + 1), 1 / n)
        elif method == EXPONENT:
            score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def evaluate(model:nn.Module,
                testdataset,
                word_dict,
                n_bleu=4,
                device=try_gpu(),
                method=GREEDY_SEARCH,
                special_tokens=[0, 1, 2, 3, 4],
                **kwargs):
        model.eval()
        model.to(device)
        total_score_1, total_score_n = [], []

        result = {}
        with open("result/result_of_{}_in_method_{}.json".format(model.name, method), "a", encoding="utf-8") as f:
            for x, y in tqdm(testdataset):
                pred_seq  = predict(model, x, word_dict, device, method, special_tokens, if_print=False, **kwargs)
                label_seq = [get_word(word_dict, index) for index in y if index not in special_tokens]
                score_1 = bleu(pred_seq, label_seq, k=1, method=RECIPROCAL)
                score_n = bleu(pred_seq, label_seq, k=n_bleu, method=RECIPROCAL)
                total_score_1.append(score_1)
                total_score_n.append(score_n)
                result["pred"] = ' '.join(pred_seq)
                result["label"] = ' '.join(label_seq)
                result["bleu_1"] = score_1
                result["bleu_{}".format(n_bleu)] = score_n
                json.dump(result, f)
                f.write('\n')
        return sum(total_score_1) / len(total_score_1), sum(total_score_n) / len(total_score_n)


def evlauate_gpt(model:GPTModel,
                testdataset,
                tokenizer,
                n_bleu=4,
                device=try_gpu(),
                do_sample=True,
                top_k=5,
                max_length=99,
                top_p=0.75,
                finetune_method=0,
                **kwargs):
        model.eval()
        model.to(device)
        total_score_1, total_score_n = [], []
        words_re = re.compile(r'\w+|\,|\.|\?|\'|\"|\:|^[0-9]*$')
        
        result = {}
        with open("result/result_of_{}_{}.json".format(model.name, finetune_method), "a", encoding="utf-8") as f:
            for sentence_x, sentence_y in tqdm(testdataset):
                input = tokenizer(sentence_x, return_tensors="pt")
                tokens_tensor = input["input_ids"].to(device)
                attention_mask = input["attention_mask"].to(device)

                pred_seq = model.gpt.generate(
                    tokens_tensor,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    top_k=top_k,
                    max_length=max_length,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
                pred_seq = tokenizer.decode(pred_seq[0], skip_special_tokens=True)
                label_seq = sentence_y
                score_1 = bleu(words_re.findall(pred_seq.lower()), words_re.findall(label_seq.lower()), k=1, method=RECIPROCAL)
                score_n = bleu(words_re.findall(pred_seq.lower()), words_re.findall(label_seq.lower()), k=n_bleu, method=RECIPROCAL)
                total_score_1.append(score_1)
                total_score_n.append(score_n)
                result["pred"] = pred_seq
                result["label"] = label_seq
                result["bleu_1"] = score_1
                result["bleu_{}".format(n_bleu)] = score_n
                json.dump(result, f)
                f.write('\n')
       
        # print("predect: "+  pred_seq)
        # print("label: "  + label_seq)
        return sum(total_score_1) / len(total_score_1), sum(total_score_n) / len(total_score_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Transformer', help='LSTM, Transformer')
    parser.add_argument('--load', type=bool, default=False, help='')
    parser.add_argument('--layers', type=int, default=6, help='')
    args = parser.parse_args()

    word_dict, _, test_tokens, _ = get_data(root="data")
    from model.LSTM import LSTM_model
    from model.Transformer import TransformerDecoder_model, TransformerEncoderDecoder_model
    if args.model.lower() == 'lstm':
        model = LSTM_model(len(word_dict))
    elif args.model.lower() == 'transformer':
        model = TransformerEncoderDecoder_model(len(word_dict), num_layers=args.layers)
    elif args.model.lower() == 'transformer_decoder':
        model = TransformerDecoder_model(len(word_dict), num_layers=args.layers)
    path = 'model/checkpoints_{}.pth'.format(model.name)
    if args.load:
        model.load_state_dict(torch.load(path))
        print("Load model from {}".format(path))

    print("---------------------Evaluating {} ------------------".format(model.name))
    bleu1_score, bleu4_score = evaluate(model, test_tokens, word_dict, n_bleu=4)
    print("Bleu-1 Score: {}".format(bleu1_score))
    print("Bleu-4 Score: {}".format(bleu4_score))