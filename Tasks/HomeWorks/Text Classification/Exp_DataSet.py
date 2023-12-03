import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
from gensim.models import KeyedVectors
from transformers import BertTokenizer
import jieba

class Dictionary(object):
    """
    存放word -> token的字典 token -> word的列表\n
    存放label -> idx的字典  idx -> label的列表
    """
    def __init__(self, path = './data/tnews_public'):

        self.word2tkn = {"[PAD]": 0}  # 用word索引token
        self.tkn2word = ["[PAD]"]     # 用token索引word

        self.label2idx = {}  # {"100": 0, "101": 1, ...}
        self.idx2label = []  # [[100, "news_story"], [101, "news_culture"], ...]

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]  # 返回 token


class Corpus(object):
    '''
    语料库: 

    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，
    例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, 
                 path = './data/tnews_public', 
                 max_token_per_sent: int = 50,
                 embedding=False  # word2vec 预训练词向量
                 ):
        self.dictionary = Dictionary(path)
        self.num_classes = 0
        self.embedding = embedding
        self.embedding_weight = None

        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, 'train.json'), name='train')
        self.valid = self.tokenize(os.path.join(path, 'dev.json'), name='valid')
        self.test = self.tokenize(os.path.join(path, 'test.json'), True, name='test ')
        print("word dict size: {}".format(len(self.dictionary.word2tkn)))
        print(f"classes: {self.num_classes}")

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。 矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置
        if self.embedding:
            print("loading embedding from word to vector model...")
            model = KeyedVectors.load_word2vec_format('sgns.wiki.word')
            self.embedding_weight = np.zeros((len(self.dictionary.word2tkn), 300))
            for word, idx in self.dictionary.word2tkn.items():
                if word in model:
                    self.embedding_weight[idx] = model[word]
                else:
                    self.embedding_weight[idx] = np.random.normal(size=(300, ))
            self.embedding_weight[0] = np.zeros((300, ))
            self.embedding_weight = torch.tensor(self.embedding_weight, dtype=torch.float32)

        #------------------------------------------------------end------------------------------------------------------#

    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) >= self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False, name=None):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss   = []   # 存放 token list 的 list
        labels = [] # 存放 label 序列
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                #-----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词
                if self.embedding:
                    sent = jieba.lcut(sent, cut_all=True)

                #------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in sent:
                    self.dictionary.add_word(word)

                ids = []  # [token1, token2, ...]
                for word in sent:
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad(ids))
                
                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)  # [0, 1, 2...]
                else:
                    label = json.loads(line)['label']  # [100, 101, 102...]
                    label = self.dictionary.label2idx[label]  # [0, 1, 2...]
                    assert 0 <= label <= 14
                    labels.append(label)

            idss   = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()
        
        self.num_classes = len(self.dictionary.label2idx)
        if name is not None:
            print(f"{name} dataset size: {idss.shape[0]} sentences")
        return TensorDataset(idss, labels)
    

class BERT_Corpus(object):
    """

    使用BERT的tokenizer对数据集进行处理得到的token序列和相应的标签序列

    """
    def __init__(self, path = './data/tnews_public', max_token_per_sent: int = 50):
        self.max_token_per_sent = max_token_per_sent
        self.dictionary = Dictionary(path)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        self.train = self.tokenize(os.path.join(path, 'train.json'), name='train')
        self.valid = self.tokenize(os.path.join(path, 'dev.json'),   name='valid')
        self.test  = self.tokenize(os.path.join(path, 'test.json'),  True, name='test ')

    def tokenize(self, path, test_mode=False, name=None):
        sents   = []   # 存放 token list 的 list
        labels = []   # 存放 label 序列
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                # sentence
                one_data = json.loads(line)
                sent = one_data['sentence']
                sents.append(sent)
                # label
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)  # [0, 1, 2...]
                else:
                    label = json.loads(line)['label']    # [100, 101, 102...]
                    label = self.dictionary.label2idx[label]  # [0, 1, 2...]
                    assert 0 <= label <= 14
                    labels.append(label)

        sents = self.tokenizer(sents, max_length=self.max_token_per_sent,
                              padding=True, truncation=True, return_tensors='pt')
        idss = sents['input_ids']
        masks = sents['attention_mask']
        labels = torch.tensor(np.array(labels)).long()
        if name is not None:
            print(f"{name} dataset size: {idss.shape[0]} sentences")
        return TensorDataset(idss, masks, labels)

        
if __name__ == '__main__':
    # dataset = Corpus('./data/tnews_public', embedding=True)
    # print(dataset.train[0])
    dataset = BERT_Corpus('./data/tnews_public')
    print(dataset.train[0])