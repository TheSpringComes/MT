import torch


# train.py
def try_gpu():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# data_manager.py
FIXED_LENGTH = 0
TOTAL_LENGTH = 1

# evaluate.py
RECIPROCAL = 0  # 在算BLEU的时候，W_n = 1/n
EXPONENT = 1    # 在算BLEU的时候，W_n = 1/2^n


# predict.py
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


# finetune.py
ZERO_SHOT = 0  # zero shot
NO_FRONZEN = 1  # 不冻结参数
PART_LAYERS_FRONZEN = 2  # 冻结部分层
ALL_LAYERS_FRONZEN = 3  # 除了最后一层，其他层都冻结
LAYER_BY_LAYER_UNFREEZE = 4  # 逐层解冻

ADAPTER = 5  # Adapter Tuning
PREFIX = 6  # PEFT: Prefix Tuning
LORA = 7  # LoRA


