import torch
from torch.utils.data import DataLoader

FIXED_LENGTH = 0
TOTAL_LENGTH = 1

def load_data(tokens, batch_size:int = 4, method = FIXED_LENGTH, **kwargs):
    train_tokens, val_tokens = tokens
    train_data, val_data = [], []

    if method == FIXED_LENGTH:
        steps = kwargs['steps']
        for story in train_tokens:
            for i in range(0, len(story)-steps-1, steps):
                if torch.sum(story[i:i+steps]) > 0:
                    train_data.append((story[i:i+steps], story[i+1:i+steps+1]))
        for story in val_tokens:
            for i in range(0, len(story)-steps-1, steps):
                if torch.sum(story[i:i+steps]) > 0:
                    val_data.append((story[i:i+steps], story[i+1:i+steps+1]))

    elif method == TOTAL_LENGTH:
        for story in train_tokens:
            train_data.append((story[:-1], story[1:]))
        for story in val_tokens:
            val_data.append((story[:-1], story[1:]))
            
    
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print('train data:', len(train_data))
    print('val data:', len(val_data))
    return trainloader, validloader



if __name__ == '__main__':
    from data_manager import get_data
    word_dict, train_tokens, test_tokens, val_tokens = get_data()
    trainloader, validloader = load_data((train_tokens, val_tokens), batch_size=4, method=TOTAL_LENGTH, max_len=72)
    for i, (x, y) in enumerate(trainloader):
        print(i, x.shape, y.shape)
        if i == 10:
            break
    for i, (x, y) in enumerate(validloader):
        print(i, x.shape, y.shape)
        if i == 10:
            break