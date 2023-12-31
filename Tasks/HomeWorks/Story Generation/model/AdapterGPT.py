from model.PEFT import Adapter
from model.GPT import GPTModel

import torch
from  torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from typing import Optional, Tuple, Union


class myGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super(myGPT2Block, self).__init__(config)
        self.adapter_sa = Adapter(config.n_embd, 128)  # adapter after self-attention
        self.adapter_ca = Adapter(config.n_embd, 128)  # adapter after cross-attention
        self.adapter_ff = Adapter(config.n_embd, 128)  # adapter after feed-forward

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
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection

        attn_output = self.adapter_sa(attn_output)  # add adapter_1

        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection

            attn_output = self.adapter_ca(attn_output)  # add adapter_2

            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection

        feed_forward_hidden_states = self.adapter_ff(feed_forward_hidden_states)  # add adapter_3

        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class AdapterGPTModel(GPTModel):
    def __init__(self):
        super(AdapterGPTModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_config = GPT2Config.from_pretrained('gpt2')
        self.gpt.transformer.h = nn.ModuleList([myGPT2Block(self.gpt_config, layer_idx=i) for i in range(self.gpt_config.num_hidden_layers)])
        self.name = 'adapter_gpt2'

        # freeze the parameters of gpt2 except the adapter
        for param in self.gpt.parameters():param.requires_grad = False
        for i in range(len(self.gpt.transformer.h)):
            for param in self.gpt.transformer.h[i].adapter_sa.parameters():param.requires_grad = True
            for param in self.gpt.transformer.h[i].adapter_ca.parameters():param.requires_grad = True
            for param in self.gpt.transformer.h[i].adapter_ff.parameters():param.requires_grad = True

    def forward(self, sentences):
        tokens = self.tokenizer(sentences, return_tensors="pt")

        labels = tokens["input_ids"][:, 1:].contiguous().to(self.gpt.device)
        input_ids = tokens["input_ids"][:, :-1].contiguous().to(self.gpt.device)
        attention_mask = tokens["attention_mask"][:, :-1].contiguous().to(self.gpt.device)
        
        output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)[0]  # (batch_size, seq_len-1, vocab_size)
        
        return output, labels


if __name__ == "__main__":
    model = AdapterGPTModel()
    print("{} has parameters: {:.5f}M".format(model.name, sum(p.numel() for p in model.parameters())/1000000.0))
    model.to(torch.device('cuda:0'))
    
    x = "hello world, my name is zhangsan"
    output, labels = model(x)
    print(output.shape)
    print(labels.shape)