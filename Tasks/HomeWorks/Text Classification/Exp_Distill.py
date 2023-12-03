import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
from matplotlib import pyplot as plt

from tqdm import tqdm
from torch.utils.data import  DataLoader
from Exp_DataSet import BERT_Corpus
from Exp_Model import Transformer_model, Bert_model


def pre_train_teacher(pre_epoch=1):
    '''
    对预训练的 Bert 模型进行训练
    '''
    print("==>Start pre-training teacher...")
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()                                    
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=teacher_lr, weight_decay=weight_decay)

    for epoch in range(pre_epoch):
        teacher_model.train()
        total_loss = []
        total_true = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{pre_epoch}')

        for data in tqdm_iterator:
            batch_x, batch_mask, batch_y = data[0].to(device), data[1].to(device), data[2].to(device)
            y_hat,_ = teacher_model(batch_x, batch_mask)
            loss = loss_function(y_hat, batch_y)

            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数

            y_hat = torch.argmax(y_hat, dim=1)
            total_true.append(torch.sum(y_hat == batch_y).item())
            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss), acc=sum(total_true) / (batch_size * len(total_true)))

        tqdm_iterator.close()
        

def distill_loss(student_output, teacher_output, labels, student_feature, teacher_feature, T=5.0, alpha=0.5, beta=0.5):
    """
    loss 由3部分构成: 
        1. student 对于标签的hard loss, 使用交叉熵: -\sum_i label_i * log(y_i)
        2. student 对于教师的soft loss, 使用KL散度: D(S||T) = \sum_i S_i * log(S_i/T_i)
        3. student 学习老师特征表示的loss, 使用余弦相似度: 1 - cos(T, S)

    loss =  l_hard + a * l_soft +  β * l_feature

    """
    l_hard = nn.CrossEntropyLoss()(student_output, labels)
    l_soft = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_output, dim=-1), F.softmax(teacher_output / T, dim=-1))
    if student_feature.shape == teacher_feature.shape:  # 仅当维度相同时才计算
        l_feature = 1 - F.cosine_similarity(student_feature, teacher_feature).mean()
    else:
        l_feature = 0
    return l_hard + alpha * l_soft + beta * l_feature

def distill(freeze_teacher = False):
    '''
    进行训练
    '''
    print("==>Start distaill...")
    # 设置优化器                                       
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=teacher_lr, weight_decay=weight_decay)
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=student_lr, weight_decay=weight_decay)

    max_valid_acc = 0
    total_loss_list = []
    train_acc_list = [0, ]
    valid_acc_list = [0, ]

    for epoch in range(num_epochs):
        teacher_model.train()
        student_model.train()

        T_total_loss, S_total_loss = [], []
        T_total_true, S_total_true = [], []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            batch_x, batch_mask, batch_y = data[0].to(device), data[1].to(device), data[2].to(device)
            teacher_output, teacher_feature = teacher_model(batch_x, batch_mask)  # Bert
            student_output, student_feature = student_model(batch_x)  # Transformer

            teacher_loss = nn.CrossEntropyLoss()(teacher_output, batch_y)
            if not freeze_teacher:
                teacher_optimizer.zero_grad()   # 梯度清零
                teacher_loss.backward()         # 计算梯度
                teacher_optimizer.step()        # 更新参数

            teacher_output, teacher_feature = teacher_output.detach(), teacher_feature.detach()
            student_loss = distill_loss(student_output, teacher_output, batch_y, student_feature, teacher_feature, T=temperature, alpha=alpha, beta=beta)
            student_optimizer.zero_grad()   # 梯度清零
            student_loss.backward()         # 计算梯度
            student_optimizer.step()        # 更新参数


            T_hat = torch.argmax(teacher_output, dim=1)
            S_hat = torch.argmax(student_output, dim=1)
            
            assert T_hat.shape == batch_y.shape, 'y_hat.shape {} != batch_y.shape {}'.format(T_hat.shape, batch_y.shape)
            assert T_hat.dtype == batch_y.dtype, 'y_hat.shape {} != batch_y.shape {}'.format(T_hat.dtype, batch_y.dtype)
            assert S_hat.shape == batch_y.shape, 'y_hat.shape {} != batch_y.shape {}'.format(S_hat.shape, batch_y.shape)
            assert S_hat.dtype == batch_y.dtype, 'y_hat.shape {} != batch_y.shape {}'.format(S_hat.dtype, batch_y.dtype)
            T_total_true.append(torch.sum(T_hat == batch_y).item())
            S_total_true.append(torch.sum(S_hat == batch_y).item())
            T_total_loss.append(teacher_loss.item())
            S_total_loss.append(student_loss.item())
            

            tqdm_iterator.set_postfix(Teacher_loss=sum(T_total_loss) / len(T_total_loss), 
                                      Student_loss = sum(S_total_loss) / len(S_total_loss),
                                      Teacher_acc =sum(T_total_true) / (batch_size * len(T_total_true)),
                                      Student_acc =sum(S_total_true) / (batch_size * len(S_total_true)))
        
        tqdm_iterator.close()

        train_loss = sum(S_total_loss) / len(S_total_loss)
        train_acc = sum(S_total_true) / (batch_size * len(S_total_true))
        valid_acc = student_valid()

        total_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        print(f"epoch: {epoch + 1}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%")

        # 保存最优模型
        if valid_acc > max_valid_acc:
            torch.save(student_model, os.path.join(output_folder, "model.ckpt"))

    return total_loss_list, train_acc_list, valid_acc_list
        

def student_valid():
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []
    student_model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True):
            batch_x, batch_mask, batch_y = data[0].to(device), data[1].to(device), data[2].to(device)
            y_hat,_ = student_model(batch_x)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
            total_true.append(torch.sum(y_hat == batch_y).item())

        return sum(total_true) / (batch_size * len(total_true))


def student_predict():
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    test_ids = [] 
    test_pred = []

    model = torch.load(os.path.join(output_folder, "model.ckpt")).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_test, dynamic_ncols=True): 
            batch_x, _, batch_y = data[0].to(device), data[1].to(device), data[2].to(device)
            y_hat,_ = model(batch_x)
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
    plt.title('Student loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')
    plt.title('Student accuracy')
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
    nhead = 8  # transformer 的 head 数
    nlayer = 4  # transformer 的 encoder 层数
    max_token_per_sent = 30 # 每个句子预设的最大 token 数
    batch_size = 16
    num_epochs = 5
    teacher_lr = 1e-5
    student_lr = 1e-4
    weight_decay = 5e-5
    alpha = 0.8
    beta = 0.02
    temperature = 5.0  # 温度
    #------------------------------------------------------end------------------------------------------------------#
    print("==>loading data...")
    dataset = BERT_Corpus(dataset_folder, max_token_per_sent)

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 可修改选择的模型以及传入的参数
    print("==>loading model...")
    teacher_model = Bert_model(freeze_bert=False, return_feature=True).to(device)
    teacher_embed_weight = teacher_model.feature.embeddings.word_embeddings.weight.clone().detach().requires_grad_(True)
    student_model = Transformer_model(vocab_size=teacher_embed_weight.shape[0],
                                      ntoken=max_token_per_sent, 
                                      nhead=nhead, 
                                      nlayers=nlayer,
                                      # d_emb=teacher_embed_weight.shape[1], # 768 
                                      d_emb = 256,
                                      embedding_weight=None,
                                      return_feature=True
                                      ).to(device)


    # 输出模型参数量
    print("Teacher model parameters: {:.5f}M".format(sum(p.numel() for p in teacher_model.parameters())/1000000.0))
    print("Student mdoel parameters: {:.5f}M".format(sum(p.numel() for p in student_model.parameters())/1000000.0))                          
    #------------------------------------------------------end------------------------------------------------------#
    
    # 进行训练
    pre_train_teacher(pre_epoch=1)
    loss_list, train_acc, test_acc = distill()
    for param in teacher_model.parameters(): param.requires_grad = False# 冻住 teacher 的参数
    loss_list, train_acc, test_acc = distill(True)

    # 对测试集进行预测
    student_predict()
    plot_process(loss_list, train_acc, test_acc)