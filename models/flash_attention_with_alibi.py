import math
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from utils.alibi_utils import get_alibi_slopes


class FlashGPT2AttentionWithALiBi(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        """
        Flash Attention with ALiBi (Attention with Linear Biases).

        Args:
            config: GPT2 configuration object.
            is_cross_attention: Whether this layer is for cross-attention.
            layer_idx: Layer index (if needed for custom configurations).
        """
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

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
        assert hidden_size == self.embed_dim, "Hidden size must match embedding dimension."

        # Project hidden states to query, key, and value
        qkv_proj = self.c_attn(hidden_states)
        query, key, value = qkv_proj.split(self.embed_dim, dim=2)

        query = query.view(bsz, slen, self.num_heads, self.head_dim).transpose(1, 2)  # [bsz, num_heads, slen, head_dim]
        key = key.view(bsz, slen, self.num_heads, self.head_dim).transpose(1, 2)      # [bsz, num_heads, slen, head_dim]
        value = value.view(bsz, slen, self.num_heads, self.head_dim).transpose(1, 2)  # [bsz, num_heads, slen, head_dim]

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=2)  # Concatenate along sequence length
            value = torch.cat((past_value, value), dim=2)
        present = (key, value) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)

        # Generate ALiBi slopes and bias
        alibi_slopes = get_alibi_slopes(self.num_heads, device=hidden_states.device).to(hidden_states.dtype)
        alibi = alibi_slopes.view(1, self.num_heads, 1, 1)  # Shape: [1, num_heads, 1, 1]

        # Generate position indices
        position_ids = torch.arange(slen, device=hidden_states.device)  # Shape: [slen]
        relative_positions = position_ids.unsqueeze(0) - position_ids.unsqueeze(-1)  # Shape: [slen, slen]
        relative_positions = relative_positions.to(hidden_states.dtype)

        # Generate ALiBi bias
        alibi_bias = alibi * relative_positions.unsqueeze(0)  # Final shape: [1, num_heads, slen, slen]

        # Compute attention scores
        attn_scores = torch.einsum("bnqd,bnkd->bnqk", query, key) * scale  # Correct einsum

        # Add ALiBi bias
        attn_scores = attn_scores + alibi_bias  # Shape: [bsz, num_heads, slen, slen]

        if attention_mask is not None:
            assert attention_mask.dim() == 4, "Attention mask must be 4D: [bsz, 1, 1, slen]"
            attn_scores += attention_mask

        # Compute attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)  # Apply dropout

        # Compute attention output
        attn_output = torch.einsum("bnqk,bnvd->bnqd", attn_probs, value)  # Shape: [bsz, num_heads, slen, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, slen, hidden_size)  # [bsz, slen, hidden_size]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return (attn_output, present) if use_cache else (attn_output, None)