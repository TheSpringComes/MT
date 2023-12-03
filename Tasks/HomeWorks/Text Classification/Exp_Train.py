import torch
import torch.nn as nn
import time
import json
import os
from matplotlib import pyplot as plt

from tqdm import tqdm
from torch.utils.data import  DataLoader
from Exp_DataSet import Corpus, BERT_Corpus
from Exp_Model import BiLSTM_model, Transformer_model, Bert_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BiLSTM', help="model")
parser.add_argument("--w2v", type=bool, default=False, help="use word2vector")
parser.add_argument('--multi_layer', type=bool, default=False, help="利用多层隐藏层作为特征")
parser.add_argument('--pooling', type=str, default='first', help="对隐藏层的处理方式")
parser.add_argument("--freeze", type=bool, default=False, help="freeze bert")
parser.add_argument('--distall', type=bool, default=False, help="distall Bert to Transfomer")
args = parser.parse_args()


def train():
    '''
    进行训练
    '''
    print("==>Start training...")
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器                                       
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_valid_acc = 0
    total_loss_list = []
    train_acc_list = [0, ]
    valid_acc_list = [0, ]

    for epoch in range(num_epochs):
        model.train()

        total_loss = []
        total_true = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            if args.model.lower() == 'bert':
                batch_x, batch_mask, batch_y = data[0].to(device), data[1].to(device), data[2].to(device)
                y_hat = model(batch_x, batch_mask)
            else:
                batch_x, batch_y = data[0].to(device), data[1].to(device)  # 选取对应批次数据的输入和标签
                y_hat = model(batch_x)  # 模型预测 (batch_size, num_classes)
            
            loss = loss_function(y_hat, batch_y)

            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数

            y_hat = torch.argmax(y_hat, dim=1)
            
            assert y_hat.shape == batch_y.shape, 'y_hat.shape {} != batch_y.shape {}'.format(y_hat.shape, batch_y.shape)
            assert y_hat.dtype == batch_y.dtype, 'y_hat.dtype {} != batch_y.dtype {}'.format(y_hat.dtype, batch_y.dtype)
            total_true.append(torch.sum(y_hat == batch_y).item())
            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss),
                                      acc=sum(total_true) / (batch_size * len(total_true)))
        
        tqdm_iterator.close()

        train_loss = sum(total_loss) / len(total_loss)
        train_acc = sum(total_true) / (batch_size * len(total_true))
        valid_acc = valid()

        total_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        print(f"epoch: {epoch + 1}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%")

        # 保存最优模型
        if valid_acc > max_valid_acc:
            torch.save(model, os.path.join(output_folder, "model.ckpt"))

    return total_loss_list, train_acc_list, valid_acc_list
        

def valid():
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True):
            if args.model.lower() == 'bert':
                batch_x, batch_mask, batch_y = data[0].to(device), data[1].to(device), data[2].to(device)
                y_hat = model(batch_x, batch_mask)
            else:
                batch_x, batch_y = data[0].to(device), data[1].to(device)  # 选取对应批次数据的输入和标签
                y_hat = model(batch_x)  # 模型预测
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            total_true.append(torch.sum(y_hat == batch_y).item())

        return sum(total_true) / (batch_size * len(total_true))


def predict():
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    test_ids = [] 
    test_pred = []

    model = torch.load(os.path.join(output_folder, "model.ckpt")).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_test, dynamic_ncols=True): 
            if args.model.lower() == 'bert':
                batch_x, batch_mask, batch_y = data[0].to(device), data[1].to(device), data[2].to(device)
                y_hat = model(batch_x, batch_mask)
            else:
                batch_x, batch_y = data[0].to(device), data[1].to(device)  # 选取对应批次数据的输入和标签
                y_hat = model(batch_x)  # 模型预测
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat])

            test_ids += batch_y.tolist()
            test_pred += y_hat.tolist()

    # 写入文件
    with open(os.path.join(output_folder, "predict.json"), "w") as f:
        for idx, label_idx in enumerate(test_pred):
            one_data = {}
            one_data["id"] = test_ids[idx]
            one_data["pred_label_desc"] = dataset.dictionary.idx2label[label_idx][1]
            json_data = json.dumps(one_data)    # 将字典转为json格式的字符串
            f.write(json_data + "\n")
            

def plot_process(loss_list, train_acc, test_acc):
    '''
    绘制训练过程中的 loss 曲线和 accuracy 曲线
    '''
    plt.figure(figsize=(10, 5))
    # loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='loss')
    plt.title('{} loss'.format(args.model))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')
    plt.title('{} accuracy'.format(args.model))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset_folder = './data/tnews_public'
    output_folder = './output'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim=300 if args.w2v else 256  # 每个词向量的维度
    nhead = 8  # transformer 的 head 数
    nlayer = 4  # transformer 的 encoder 层数
    max_token_per_sent = 50 # 每个句子预设的最大 token 数
    batch_size = 16
    num_epochs = 5
    lr = 1e-5 if args.model.lower() == 'bert' else 1e-4
    weight_decay = 5e-4
    #------------------------------------------------------end------------------------------------------------------#
    print("==>loading data...")
    if args.model.lower() == 'bert':  # 支持大小写
        dataset = BERT_Corpus(dataset_folder, max_token_per_sent)
    else:    
        dataset = Corpus(dataset_folder, max_token_per_sent, embedding=args.w2v)
        vocab_size = len(dataset.dictionary.tkn2word)

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 可修改选择的模型以及传入的参数
    print("==>loading model...")
    if args.model == 'BiLSTM':
        model = BiLSTM_model(vocab_size=vocab_size, ntoken=max_token_per_sent, 
                             d_emb=embedding_dim, embedding_weight=dataset.embedding_weight).to(device)
    elif args.model == 'Transformer':
        model = Transformer_model(vocab_size=vocab_size, ntoken=max_token_per_sent, 
                                  nhead=nhead, nlayers=nlayer,
                                  multi_layer=args.multi_layer, pooling_method=args.pooling,
                                  d_emb=embedding_dim, embedding_weight=dataset.embedding_weight).to(device)
    elif args.model.lower() == 'bert':
        model = Bert_model(freeze_bert=args.freeze,
                           multi_layer=args.multi_layer, pooling_method=args.pooling,
                           ).to(device)
    # 输出模型参数量
    print("{} has parameters: {:.5f}M".format(args.model, sum(p.numel() for p in model.parameters())/1000000.0))                          
    #------------------------------------------------------end------------------------------------------------------#
    
    # 进行训练
    loss_list, train_acc, test_acc = train()
    # 对测试集进行预测
    predict()
    plot_process(loss_list, train_acc, test_acc)