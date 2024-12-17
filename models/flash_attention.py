import math
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from flash_attn.flash_attn_interface import flash_attn_func


class FlashGPT2Attention(GPT2Attention):
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        bsz, slen, hidden_size = hidden_states.size()
        assert hidden_size == self.embed_dim

        # Project hidden states to query, key, and value
        qkv_proj = self.c_attn(hidden_states)
        query, key, value = qkv_proj.split(self.embed_dim, dim=2)

        # Ensure tensors are in fp16
        query = query.view(bsz, slen, self.num_heads, self.head_dim).to(torch.float16)
        key = key.view(bsz, slen, self.num_heads, self.head_dim).to(torch.float16)
        value = value.view(bsz, slen, self.num_heads, self.head_dim).to(torch.float16)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
        present = (key, value) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)

        # Call FlashAttention
        attn_output = flash_attn_func(
            query,
            key,
            value,
            causal=True,
            softmax_scale=scale,
        )

        # Reshape output
        attn_output = attn_output.reshape(bsz, slen, hidden_size)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present