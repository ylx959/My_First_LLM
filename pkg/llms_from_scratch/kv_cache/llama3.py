# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from .utils import KVCache   # noqa: F401

import os
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken
from tiktoken.load import load_tiktoken_bpe


LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           # 詞彙表大小
    "context_length": 131_072,       # 訓練模型時使用的 context length
    "emb_dim": 2048,                 # Embedding 維度
    "n_heads": 32,                   # Attention head 數
    "n_layers": 16,                  # 層數
    "hidden_dim": 8192,              # FeedForward 中間層的大小
    "n_kv_groups": 8,                # grouped-query attention 的 Key-Value group 數
    "rope_base": 500_000.0,          # RoPE "theta" 的 base
    "dtype": torch.bfloat16,         # 使用較低精度的 dtype 以減少記憶體用量
    "rope_freq": {                   # RoPE 頻率縮放
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,           # 詞彙表大小
    "context_length": 131_072,       # 訓練模型時使用的 context length
    "emb_dim": 3072,                 # Embedding 維度
    "n_heads": 24,                   # Attention head 數
    "n_layers": 28,                  # 層數
    "hidden_dim": 8192,              # FeedForward 中間層的大小
    "n_kv_groups": 8,                # grouped-query attention 的 Key-Value group 數
    "rope_base": 500_000.0,          # RoPE "theta" 的 base
    "dtype": torch.bfloat16,         # 使用較低精度的 dtype 以減少記憶體用量
    "rope_freq": {                   # RoPE 頻率縮放
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}


class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 主要模型參數
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # 用 ModuleList 是因為 Sequential 只能接受一個輸入，而我們需要 `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # 可重複使用的工具
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # 追蹤 KV cache 中目前的位置

    def forward(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        pos_start = self.current_pos if cache is not None else 0
        pos_end = pos_start + num_tokens
        if pos_end > self.cfg["context_length"]:
            raise ValueError(
                f"Sequence length {pos_end} exceeds the model's context length "
                f"of {self.cfg['context_length']} tokens."
            )

        if cache is not None:
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # Shape (1, 1, num_tokens, num_tokens)，可 broadcast 到 batch 和 head
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # attention block 的 shortcut connection
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # 加回原始輸入

        # feed-forward block 的 shortcut connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # 加回原始輸入

        return x, next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, num_heads, num_kv_groups, dtype=None
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        # 套用投影
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 套用 RoPE
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
            next_cache = (keys, values)
        else:
            start_pos = 0  # 重設 RoPE
            keys, values = keys_new, values_new
            next_cache = (keys, values)

        # 擴展 keys 和 values 以符合 head 數量
        # Shape：(b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape：(b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape：(b, num_heads, num_tokens, head_dim)
        # 例如，沿著 dim=1（query groups）做 repeat_interleave 之前：
        #   [K1, K2]
        # repeat_interleave 之後（每個 query group 重複 group_size 次）：
        #   [K1, K1, K2, K2]
        # 如果用一般的 repeat 而不是 repeat_interleave，會得到：
        #   [K1, K2, K1, K2]

        # 計算帶 causal mask 的 scaled dot-product attention（也就是 self-attention）
        # Shape：(b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # 每個 head 的內積

        # 用 mask 填入 attention 分數
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape：(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合併各 head，其中 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可選的投影

        return context_vec, next_cache


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 計算逆頻率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 頻率調整
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # 產生位置 index
    positions = torch.arange(context_length, dtype=dtype)

    # 計算角度
    angles = positions[:, None] * inv_freq[None, :]  # Shape：(context_length, head_dim // 2)

    # 擴展角度以符合 head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape：(context_length, head_dim)

    # 預先計算 sine 和 cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    # x：(batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 把 x 拆成前半和後半
    x1 = x[..., : head_dim // 2]  # 前半段
    x2 = x[..., head_dim // 2:]  # 後半段

    # 調整 sin 和 cos 的形狀
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape：(1, 1, seq_len, head_dim)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # 套用旋轉轉換
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # 套用 cos 和 sin 旋轉之後，使用較低精度是可以的
    return x_rotated.to(dtype=x.dtype)


##########################################
# Tokenizer
##########################################


class Llama3Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        # 從 Meta 的 tokenizer.json 寫死
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False, **kwargs):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        return self.model.decode(ids)


class ChatFormat:

    def __init__(self, tokenizer: Llama3Tokenizer, *,
                 default_system="You are a helpful assistant."):
        self.tok = tokenizer
        self.default_system = default_system

    def _header(self, role):
        """Encode <|start_header_id|>role<|end_header_id|>\n\n"""
        return (
            [self.tok.special["<|start_header_id|>"]]
            + self.tok.encode(role)
            + [self.tok.special["<|end_header_id|>"]]
            + self.tok.encode("\n\n")
        )

    def encode(self, user_message, system_message=None, allowed_special=None):
        sys_msg = system_message if system_message is not None else self.default_system

        ids = [self.tok.special["<|begin_of_text|>"]]

        # system
        ids += self._header("system")
        ids += self.tok.encode(sys_msg, allowed_special=allowed_special)
        ids += [self.tok.special["<|eot_id|>"]]

        # user
        ids += self._header("user")
        ids += self.tok.encode(user_message)
        ids += [self.tok.special["<|eot_id|>"]]

        # assistant header（還沒有內容）
        ids += self._header("assistant")

        return ids

    def decode(self, ids):
        return self.tok.decode(ids)


def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    # 找出第一次出現 "<|end_header_id|>" 的位置
    index = text.find(header_end)

    if index != -1:
        # 回傳 "<|end_header_id|>" 之後的子字串
        return text[index + len(header_end):].strip()  # strip 會去掉前後的空白
    else:
        # 找不到 token 的話就回傳原始文字
        return text


######################################################################
# Llama 3 快速版（著重效率的另一種寫法）
######################################################################

class GroupedQueryAttentionFast(nn.Module):
    """
    Drop-in replacement for GroupedQueryAttention but using PyTorch's
    scaled_dot_product_attention, which uses FlashAttention if run
    on an Ampere GPU (like A100) or newer and uses float16/bfloat16 or lower.
    """
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, cos, sin):
        b, num_tokens, _ = x.shape

        # 投影成 queries、keys、values
        q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 套用 Rotary Positional Embedding
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 把 key/value group 擴展成完整的 head 數
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # 高效率的 scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=True  # 啟用 Flash/FlexAttention kernel
        )

        # 合併各 head 並投影
        attn_output = attn_output.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(attn_output)


class TransformerBlockFast(nn.Module):
    """
    Same as original TransformerBlock but uses
    GroupedQueryAttentionFast instead of GroupedQueryAttention.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttentionFast(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, cos, sin):
        # attention block 的 shortcut connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # 加回原始輸入

        # feed-forward block 的 shortcut connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # 加回原始輸入

        return x


class Llama3ModelFast(nn.Module):
    """
    Same as original Llama3Model but uses TransformerBlockFast
    instead of TransformerBlock, which in turn uses
    GroupedQueryAttentionFast instead of GroupedQueryAttention.
    """
    def __init__(self, cfg):
        super().__init__()

        # 主要模型參數
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # 用 ModuleList 是因為 Sequential 只能接受一個輸入，而我們需要 `x, cos, sin`
            [TransformerBlockFast(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
