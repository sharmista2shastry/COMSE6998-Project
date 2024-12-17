from transformers import GPT2LMHeadModel
from .flash_attention import FlashGPT2Attention
from .flash_attention_with_alibi import FlashGPT2AttentionWithALiBi


def load_standard_gpt2(device='cpu'):
    return GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()


def load_flash_attention_gpt2(device):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for block in model.transformer.h:
        block.attn = FlashGPT2Attention(
            config=model.config,
            is_cross_attention=block.attn.is_cross_attention,
            layer_idx=block.attn.layer_idx
        ).to(device)
    return model


def load_flash_attention_with_alibi_gpt2(device):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for block in model.transformer.h:
        block.attn = FlashGPT2AttentionWithALiBi(
            config=model.config,
            is_cross_attention=block.attn.is_cross_attention,
            layer_idx=block.attn.layer_idx
        ).to(device)
    return model