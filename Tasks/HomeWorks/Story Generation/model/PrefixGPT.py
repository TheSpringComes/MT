from model.PEFT import PrefixTuning
from model.GPT import GPTModel

import torch
from  torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention
from typing import Optional, Tuple, Union


class myGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, prefix_num=20):
        super(myGPT2Attention, self).__init__(config, is_cross_attention, layer_idx)
        self.prefix_num = prefix_num
        self.prefix = PrefixTuning(prefix_num=prefix_num)
 
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

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
        
        # prefix
        key = self.prefix(key)
        value = self.prefix(value)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = key[:, :, self.prefix_num:, :]      # remove prefix attention
            value = value[:, :, self.prefix_num:, :]  # remove prefix attention
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # add prefix attention mask
        if attention_mask is not None:
            size = attention_mask.size()
            prefix_attention_mask = torch.zeros(size=(size[0], size[1], size[2], key.size(-2)-size[3])).to(attention_mask.device)
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=-1)

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
        


class myGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None, prefix_num=20):
        super(myGPT2Block, self).__init__(config, layer_idx)
        self.attn = myGPT2Attention(config, layer_idx=None, prefix_num=prefix_num)


class PrefixGPTModel(GPTModel):
    def __init__(self, prefix_num=20):
        super(PrefixGPTModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_config = GPT2Config.from_pretrained('gpt2')
        self.gpt.transformer.h = nn.ModuleList([myGPT2Block(
            self.gpt_config, layer_idx=i, prefix_num=prefix_num) for i in range(self.gpt_config.num_hidden_layers)])
        self.name = 'Prefix_gpt2'

        # freeze the parameters of gpt2 except the adapter
        for param in self.gpt.parameters():param.requires_grad = False
        for i in range(len(self.gpt.transformer.h)):
            for param in self.gpt.transformer.h[i].attn.prefix.parameters():param.requires_grad = True


    def forward(self, sentences):
        tokens = self.tokenizer(sentences, return_tensors="pt")

        labels = tokens["input_ids"][:, 1:].contiguous().to(self.gpt.device)
        input_ids = tokens["input_ids"][:, :-1].contiguous().to(self.gpt.device)
        attention_mask = tokens["attention_mask"][:, :-1].contiguous().to(self.gpt.device)
        
        output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)[0]  # (batch_size, seq_len-1, vocab_size)
        
        return output, labels


if __name__ == "__main__":
    model = PrefixGPTModel(prefix_num = 20)
    print("{} has parameters: {:.5f}M".format(model.name, sum(p.numel() for p in model.parameters())/1000000.0))
    model.to(torch.device('cuda:0'))
    
    x = "hello world, my name is zhangsan"
    output, labels = model(x)
    print(output.shape)
    print(labels.shape)