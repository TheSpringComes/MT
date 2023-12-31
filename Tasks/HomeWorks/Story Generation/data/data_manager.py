import pandas as pd
import re
import torch
import os

 
def build_dict(text):
    word_dict = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3, '<sep>': 4}
    for story in text:
        for word in story:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict


def get_word(word_dict, index):
    for key, value in word_dict.items():
        if value == index:
            return key
    return '<unk>'


def word2token(stories, max_len, word_dict, padding=True, sep = False):
    tokens = []
    for story in stories:
        story = [word_dict['<bos>']] + [word_dict[word] if word in word_dict else word_dict['<unk>'] for word in story] + [word_dict['<eos>']]
        if sep:
            story = story[:-1]
        if padding:
            if len(story) > max_len:
                story = story[:max_len]
            else:
                story = story + [word_dict['<pad>']] * (max_len-len(story))
        tokens.append(story)
    return tokens


def get_stories(path, is_test=False):
    words_re = re.compile(r'<sep>|\w+|\,|\.|\?|\'|\"|\:|^[0-9]*$')
    df = pd.read_csv(path).to_numpy()[:, 1:]
    title = df[:, 0]
    sents = df[:, 1:]
    if is_test:
        stories_x = [story[0] for story in sents]
        stories_y = [story[1]+' '+story[2]+' '+story[3]+' '+story[4] for story in sents]
        stories_x = [re.sub(r'[^<sep>\d\w\s\'\.\,\"\:\?]+', ' ', story) for story in stories_x]
        stories_x = [words_re.findall(story.lower()) for story in stories_x]
        stories_y = [re.sub(r'[^<sep>\d\w\s\'\.\,\"\:\?]+', ' ', story) for story in stories_y]
        stories_y = [words_re.findall(story.lower()) for story in stories_y]
        return stories_x, stories_y
    else:
        stories =  [story[0]+' '+story[1]+' '+story[2]+' '+story[3]+' '+story[4] for story in sents]
        stories = [re.sub(r'[^<sep>\d\w\s\'\.\,\"\:\?]+', ' ', story) for story in stories]  # remove special characters
        stories = [words_re.findall(story.lower()) for story in stories]  # tokenize
        return stories


def get_data(max_len=72, lower=True, root = ""):
    train_path = 'story_genaration_dataset/ROCStories_train.csv'
    test_path  = 'story_genaration_dataset/ROCStories_test.csv'
    val_path   = 'story_genaration_dataset/ROCStories_val.csv'
    train_stories = get_stories(os.path.join(root, train_path))
    val_stories   = get_stories(os.path.join(root, val_path))
    test_storie_x, test_storie_y  = get_stories(os.path.join(root, test_path), is_test=True)

    word_dict = build_dict(train_stories+test_storie_x+test_storie_y+val_stories)
    print('vocab size:', len(word_dict))

    train_tokens = torch.tensor(word2token(train_stories, max_len, word_dict), dtype=torch.long)
    val_tokens   = torch.tensor(word2token(val_stories,   max_len, word_dict), dtype=torch.long)
    test_tokens_x  = word2token(test_storie_x,  max_len, word_dict, padding=False, sep = True)
    test_tokens_y  = word2token(test_storie_y,  max_len, word_dict, padding=False)
    test_tokens = [(test_tokens_x[i], test_tokens_y[i]) for i in range(len(test_tokens_x))]
    return word_dict, train_tokens, test_tokens, val_tokens


def get_stories_for_gpt(path, sep = True):
    df = pd.read_csv(path).to_numpy()[:, 1:]
    title = df[:, 0]
    sents = df[:, 1:]
    if sep: 
        stories_x = [story[0] for story in sents]
        stories_y = [story[1]+' '+story[2]+' '+story[3]+' '+story[4] for story in sents]
        return stories_x, stories_y
    else: 
        stories =  [story[0]+' '+story[1]+' '+story[2]+' '+story[3]+' '+story[4] for story in sents]
        return stories


def get_data_for_gpt(root = ""):
    train_path = 'story_genaration_dataset/ROCStories_train.csv'
    test_path  = 'story_genaration_dataset/ROCStories_test.csv'
    val_path   = 'story_genaration_dataset/ROCStories_val.csv'
    train_stories = get_stories_for_gpt(os.path.join(root, train_path), sep=False)
    val_stories   = get_stories_for_gpt(os.path.join(root, val_path),   sep=False)
    test_stories_x, test_stories_y  = get_stories_for_gpt(os.path.join(root, test_path),  sep=True)
    return train_stories, val_stories, [(x,y) for x,y in zip(test_stories_x, test_stories_y)]


if __name__ =='__main__':
    word_dict, train_tokens, test_tokens, val_tokens = get_data()
    print("length:", end="")
    print(len(test_tokens))
    print(test_tokens[1])
    print(train_tokens[0])

    # train_stories, val_stories, test_stories = get_data_for_gpt()
    # print(len(train_stories))
    # print(train_stories[0])
    # print(len(train_stories[0]))




