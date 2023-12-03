import random
import jieba


pos_fname = './data/pos.txt'
neg_fname = './data/neg.txt'
stopwords_fname = "./data/stopwords.txt"

pos_sentences = open(pos_fname, encoding='utf-8').read().strip().split('\n')
neg_sentences = open(neg_fname, encoding='utf-8').read().strip().split('\n')
stopwords = open(stopwords_fname, encoding='utf-8').read().strip().split('\n')

def remove_stopwords(_words):
    cnt = 0
    for _ in range(len(_words)):
        if _words[cnt] in stopwords:
            _words.pop(cnt)
        else:
            cnt += 1
    return _words

def load_list_and_dict(pos_sentences, neg_sentences):
    # 得到句子的列表
    random.shuffle(pos_sentences)
    random.shuffle(neg_sentences)

    # 划分训练集和测试集
    TRAIN_SPILT = 0.8
    pos_length = int(len(pos_sentences) * TRAIN_SPILT)
    pos_sent_train = pos_sentences[:pos_length]
    pos_sent_test  = pos_sentences[pos_length:]
    neg_length = int(len(neg_sentences) * TRAIN_SPILT)
    neg_sent_train = neg_sentences[:neg_length]
    neg_sent_test  = neg_sentences[neg_length:]

    # 词汇字典
    pos_word_dict = {}
    neg_word_dict = {}

    print("Building Dict...")
    print("{} sentences in total".format(pos_length+neg_length))
    for sent in pos_sent_train:
        words = remove_stopwords(jieba.lcut(sent, cut_all=True))
        for word in words:
            if word in pos_word_dict.keys():
                pos_word_dict[word] = pos_word_dict[word]+1
            else:
                pos_word_dict[word] = 1

    for sent in neg_sent_train:
        words = remove_stopwords(jieba.lcut(sent, cut_all=True))
        for word in words:
            if word in neg_word_dict.keys():
                neg_word_dict[word] = neg_word_dict[word]+1
            else:
                neg_word_dict[word] = 1
    print("Finished Build")

    lists = [pos_sent_train, pos_sent_test, neg_sent_train, neg_sent_test]
    dicts = [pos_word_dict, neg_word_dict]
    return lists, dicts


def Bayes(lists, dicts):
    pos_sent_train, pos_sent_test, neg_sent_train, neg_sent_test = lists
    pos_word_dict, neg_word_dict = dicts

    # P(pos) P(neg)
    P_pos = len(pos_sent_train)/(len(pos_sent_train)+len(neg_sent_train))
    P_neg = 1-P_pos


    # pos词的总数与neg词的总数
    pos_total_len = 0
    neg_total_len = 0
    for key in pos_word_dict:
        pos_total_len += pos_word_dict[key]
    for key in neg_word_dict:
        neg_total_len += neg_word_dict[key]


    # Bayes
    rights = 0
    print("Calculating Probability...")
    for sent in pos_sent_test:
        prob_pos = P_pos  # P(pos)
        prob_neg = P_neg  # P(neg)
        words = remove_stopwords(jieba.lcut(sent, cut_all=True))
        for word in words:
            if word in pos_word_dict:
                prob_pos *= pos_word_dict[word]/pos_total_len  # P(w|pos)
            else:
                prob_pos *= 1/pos_total_len                    # P(w|pos)
            if word in neg_word_dict:
                prob_neg *= neg_word_dict[word]/neg_total_len  # P(w|neg)
            else:
                prob_neg *= 1/neg_total_len                    # P(w|neg)
        if prob_pos>=prob_neg:
            rights += 1

    for sent in neg_sent_test:
        prob_pos = P_pos  # P(pos)
        prob_neg = P_neg  # P(neg)
        words = remove_stopwords(jieba.lcut(sent, cut_all=True))
        for word in words:
            if word in pos_word_dict:
                prob_pos *= pos_word_dict[word]/pos_total_len  # P(w|pos)
            else:
                prob_pos *= 1/pos_total_len                    # P(w|pos)
            if word in neg_word_dict:
                prob_neg *= neg_word_dict[word]/neg_total_len  # P(w|neg)
            else:
                prob_neg *= 1/neg_total_len                    # P(w|neg)
        if prob_pos<prob_neg:
            rights += 1
    print("Calculation Finished")

    accuracy = rights/((len(pos_sent_test)+len(neg_sent_test)))
    return accuracy


if __name__ == "__main__":
    lists, dicts = load_list_and_dict(pos_sentences, neg_sentences)
    accuracy = Bayes(lists, dicts)
    print("准确率: {:.2f}%".format(accuracy*100))