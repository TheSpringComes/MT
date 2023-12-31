import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class GPTModel(torch.nn.Module):
    def __init__(self):
        super(GPTModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.name = 'gpt2'
    
    def forward(self, sentences):
        tokens = self.tokenizer(sentences, return_tensors="pt")

        labels = tokens["input_ids"][:, 1:].contiguous().to(self.gpt.device)
        input_ids = tokens["input_ids"][:, :-1].contiguous().to(self.gpt.device)
        attention_mask = tokens["attention_mask"][:, :-1].contiguous().to(self.gpt.device)
        
        output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)[0]  # (batch_size, seq_len-1, vocab_size)
        
        return output, labels


if __name__ == "__main__":
    model = GPTModel()
    model.to(torch.device('cuda:0'))
    x = "I am a student."
    out, label = model(x)
    print(out.shape, label.shape)


    # torch.Size([1, 5, 50257]) torch.Size([1, 5])