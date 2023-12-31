import torch
import torch.nn as nn
import re
from utils import *
import argparse


def get_word(word_dict, index):
    for key, value in word_dict.items():
        if value == index:
            return key
    return '<unk>'


GREEDY_SEARCH = 0
BEAM_SEARCH = 1
NON_TEMPERATURE_SAMPLE = 2
TEMPERATURE_SAMPLE = 3
TOPK_SAMPLE = 4
TOPP_SAMPLE = 5
m_dict = {
    0:'GREEDY_SEARCH',
    1:'BEAM_SEARCH',
    2:'NON_TEMPERATURE_SAMPLE',
    3:'TEMPERATURE_SAMPLE',
    4:'TOPK_SAMPLE',
    5:'TOPP_SAMPLE'
}


def predict(model:nn.Module, tokens, word_dict, device, method, special_tokens, if_print = True, **kwargs):
    x = tokens
    generated_sentence = []
    if method == BEAM_SEARCH:  # beam_search
        beam_width = kwargs['beam_width']
        beam_list = [(list(x), 1)] * beam_width
        cnt = 0
        while cnt < 70:
            tmp_beam_list = []
            for beam in beam_list:
                tmp_x, tmp_p = beam
                if tmp_x[-1] == word_dict['<eos>']:
                    tmp_beam_list.append(beam)
                    continue
                tmp_x = torch.tensor(tmp_x).long().unsqueeze(0).to(device)
                out = model(tmp_x)
                weights = torch.softmax(out, dim=1).squeeze(0).detach()
                values, indices = torch.topk(weights, k=beam_width)
                for i in range(beam_width):
                    tmp_beam_list.append((tmp_x.squeeze(0).tolist() + [indices[i].item()], tmp_p * values[i].item()))
            beam_list = sorted(tmp_beam_list, key=lambda x: x[1], reverse=True)[:beam_width]
            if beam_list[0][0][-1] == word_dict['<eos>']:
                break
            cnt += 1
        generated_sentence = [get_word(word_dict, index) for index in beam_list[0][0][len(x):] if index not in special_tokens]
        if if_print: print(' '.join(generated_sentence))

    else:
        pre = list(x)
        while (pre[-1] != word_dict['<eos>']) and (len(pre) < 70):
            tmp_x = torch.tensor(pre).long().unsqueeze(0).to(device)
            out = model(tmp_x)  # (batch_szie=1, vocab_size)
            if method == GREEDY_SEARCH:  # 贪心搜索
                _, y_pred = torch.max(out, dim=1)

            elif method == NON_TEMPERATURE_SAMPLE:  # 无温度的采样
                weights = torch.softmax(out, dim=1).squeeze(0).detach()
                y_pred = torch.multinomial(weights, num_samples=1)

            elif method == TEMPERATURE_SAMPLE:  # 有温度的采样
                temperature = kwargs['temperature']
                weights = torch.softmax(out/temperature, dim=1).squeeze(0).detach()
                y_pred = torch.multinomial(weights, num_samples=1)

            elif method == TOPK_SAMPLE:  # top-k采样
                k = kwargs['k']
                values, indices = torch.topk(out , k = k, dim=1)
                weights = torch.softmax(values, dim=1).squeeze(0).detach()
                id = torch.multinomial(weights, num_samples=1)
                y_pred = indices.squeeze(0)[id]

            elif method == TOPP_SAMPLE:  # top-p采样
                p = kwargs['p']
                sorted_out, indices = torch.sort(out, descending=True) # 降序排列
                weights = torch.softmax(sorted_out, dim=1).squeeze(0).detach()
                # 取出累积概率大于p的最小索引
                wh = torch.where(torch.cumsum(weights, dim=0) >= p)[0]
                max_id =  wh[0].item() if len(wh) > 0 else len(weights)-1
                # p采样
                id = torch.multinomial(weights[:max_id+1], num_samples=1)
                y_pred = indices.squeeze(0)[id]

            else:
                raise Exception("No such method")
            
            if (if_print and (y_pred.item() not in special_tokens)): print(get_word(word_dict, y_pred.item()), end=' ')
            pre.append(y_pred.item())
        generated_sentence = [get_word(word_dict, index) for index in pre if index not in special_tokens]
    return generated_sentence


if __name__ == '__main__':
    from data.data_manager import get_data
    from model.LSTM import LSTM_model
    from model.Transformer import TransformerDecoder_model, TransformerEncoderDecoder_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Transformer', help='LSTM, Transformer')
    parser.add_argument('--load', type=bool, default=True, help='')
    parser.add_argument('--layers', type=int, default=6, help='')
    args = parser.parse_args()

    word_dict, _, test_tokens, _ = get_data(root="data")
   
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
    
    
    def test_dataset(model:nn.Module, testdata, word_dict, device=try_gpu(), method=GREEDY_SEARCH, **kwargs):
        special_tokens = [word_dict['<pad>'], word_dict['<bos>'], word_dict['<eos>'], word_dict['<unk>'], word_dict['<sep>']]
        model.cuda()
        model.eval()
        for x,y in testdata:
            # 第一句话
            print('----------------------------Method: {}----------------------------'.format(m_dict[method]))
            print("Start:", end=' ')
            print(' '.join([get_word(word_dict, index) for index in x if index not in special_tokens]))

            # 答案
            print("Answer:", end=' ')
            label_sentence = [get_word(word_dict, index) for index in y if index not in special_tokens]
            print(' '.join(label_sentence))

            # 预测
            print("Predict:", end=' ')
            generated_sentence = predict(model, x, word_dict, device, method, special_tokens, **kwargs)
            print('\n')
            break


    def test_my_sentence(model:nn.Module, word_dict, first_sentence, device=try_gpu(), method=GREEDY_SEARCH, **kwargs):
        model.cuda()
        model.eval()
        first_sentence = re.sub(r'[^<pad>\d\w\s\'\.\,\"\:\?]+', ' ', first_sentence)  # remove special characters
        words_re = re.compile(r'<sep>|\w+|\,|\.|\?|\'|\"|\:|^[0-9]*$')
        first_sentence = words_re.findall(first_sentence.lower())  # tokenize
        first_sentence = [word_dict['<bos>']] + [word_dict[word] if word in word_dict else word_dict['<unk>'] for word in first_sentence] + [word_dict['<sep>']]
        special_tokens = [word_dict['<pad>'], word_dict['<bos>'], word_dict['<eos>'], word_dict['<unk>'], word_dict['<sep>']]
        
        print(' '.join([get_word(word_dict, index) for index in first_sentence if index not in special_tokens]))
        story = first_sentence
        g = predict(model, story, word_dict, device, method, special_tokens, **kwargs)
        print(g)
    
    test_dataset(model.cuda(), test_tokens, word_dict, method=GREEDY_SEARCH)
    test_dataset(model.cuda(), test_tokens, word_dict, method=NON_TEMPERATURE_SAMPLE)
    test_dataset(model.cuda(), test_tokens, word_dict, method=TEMPERATURE_SAMPLE, temperature=0.5)
    test_dataset(model.cuda(), test_tokens, word_dict, method=TOPK_SAMPLE, k=20)
    test_dataset(model.cuda(), test_tokens, word_dict, method=TOPP_SAMPLE, p=0.9)
    test_dataset(model.cuda(), test_tokens, word_dict, method=BEAM_SEARCH, bead_width=5)

    # sent = "hello my name is a kind of things in "
    # test_my_sentence(model.cuda(), word_dict, sent, device=try_gpu(), method=GREEDY_SEARCH)