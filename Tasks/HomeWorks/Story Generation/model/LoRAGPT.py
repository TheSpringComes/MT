from model.PEFT import LoRA
from model.GPT import GPTModel

import torch
from  torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from typing import Optional, Tuple, Union


class myGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, low_rank_size = 32):
        super(myGPT2Attention, self).__init__(config, is_cross_attention, layer_idx)
        if self.is_cross_attention:
            self.lora_c_attn = LoRA(2 * config.n_embd, config.n_embd, 2*low_rank_size)
            self.lora_q_attn = LoRA(config.n_embd, config.n_embd, low_rank_size)
        else:
            self.lora_c_attn = LoRA(3 * config.n_embd, config.n_embd, 3*low_rank_size)
        self.lora_c_proj = LoRA(config.n_embd, config.n_embd, low_rank_size)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            lora_query, lora_key, lora_value = self.lora_c_attn(hidden_states).split(self.split_size, dim=2)  # lora
            query = query + lora_query  # lora
            key   = key   + lora_key    # lora
            value = value + lora_value  # lora

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        lora_attn_output = self.lora_c_proj(attn_output)  # lora
        attn_output = self.c_proj(attn_output) + lora_attn_output  # lora
        
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
        

class myGPT2MLP(GPT2MLP):
    def __init__(self, intermediate_size, config, low_rank_size = 32):
        super(myGPT2MLP, self).__init__(intermediate_size, config)
        embed_dim = config.hidden_size
        self.lora_c_fc = LoRA(intermediate_size, embed_dim, low_rank_size)
        self.lora_c_proj = LoRA(embed_dim, intermediate_size, low_rank_size)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states) + self.lora_c_fc(hidden_states)  # lora
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states) + self.lora_c_proj(hidden_states)  # lora
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class myGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None, low_rank_size = 32):
        super(myGPT2Block, self).__init__(config, layer_idx)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.attn = myGPT2Attention(config, layer_idx=layer_idx, low_rank_size = low_rank_size)
        self.mlp = myGPT2MLP(inner_dim, config, low_rank_size = low_rank_size)


class LoRAGPTModel(GPTModel):
    def __init__(self, low_rank_size = 32):
        super(LoRAGPTModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_config = GPT2Config.from_pretrained('gpt2')
        self.gpt.transformer.h = nn.ModuleList(
            [myGPT2Block(self.gpt_config, layer_idx=i, low_rank_size = low_rank_size) for i in range(self.gpt_config.num_hidden_layers)])
        self.name = 'LoRA_gpt2'

        # freeze the parameters of gpt2 except the adapter
        for param in self.gpt.parameters():param.requires_grad = False
        for i in range(len(self.gpt.transformer.h)):
            for param in self.gpt.transformer.h[i].attn.lora_c_attn.parameters():param.requires_grad = True
            for param in self.gpt.transformer.h[i].attn.lora_c_proj.parameters():param.requires_grad = True
            for param in self.gpt.transformer.h[i].mlp.lora_c_fc.parameters():param.requires_grad = True
            for param in self.gpt.transformer.h[i].mlp.lora_c_proj.parameters():param.requires_grad = True


    def forward(self, sentences):
        tokens = self.tokenizer(sentences, return_tensors="pt")

        labels = tokens["input_ids"][:, 1:].contiguous().to(self.gpt.device)
        input_ids = tokens["input_ids"][:, :-1].contiguous().to(self.gpt.device)
        attention_mask = tokens["attention_mask"][:, :-1].contiguous().to(self.gpt.device)
        
        output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)[0]  # (batch_size, seq_len-1, vocab_size)
        
        return output, labels


if __name__ == "__main__":
    model = LoRAGPTModel(low_rank_size = 32)
    print("{} has parameters: {:.5f}M".format(model.name, sum(p.numel() for p in model.parameters())/1000000.0))
    model.to(torch.device('cuda:0'))
    
    x = "hello world, my name is zhangsan"
    output, labels = model(x)
    print(output.shape)
    print(labels.shape)