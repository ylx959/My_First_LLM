# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 這個檔案收集了到目前為止所有相關的程式碼
# （第 3-4 章）。
# 這個檔案可以當作獨立腳本執行。

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# 第 3 章
#####################################
class MultiHeadAttentionWithSWA(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, sliding_window_size=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 縮小投影維度以符合想要的輸出維度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 合併各 head 輸出的 linear 層
        self.dropout = nn.Dropout(dropout)
        self.sliding_window_size = sliding_window_size

        ####################################################
        # KV cache 相關程式碼
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        ####################################################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape：(b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # 透過加上 `num_heads` 維度，隱含地切分矩陣
        # 展開最後一維：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache 相關
        if use_cache:
            old_cache_k, old_cache_v = self.cache_k, self.cache_v
            old_len = 0 if old_cache_k is None else old_cache_k.size(1)
            if old_cache_k is None:
                combined_k, combined_v = keys_new, values_new
            else:
                combined_k = torch.cat([old_cache_k, keys_new], dim=1)
                combined_v = torch.cat([old_cache_v, values_new], dim=1)

            keys, values = combined_k, combined_v
            if self.sliding_window_size is not None:
                # 分段 prefill 時，最多需要 W-1 個較舊的 key 再加上整個
                # 目前的 chunk（讓 chunk 中最早的 query 保有完整的
                # sliding-window context）
                attn_keep = min(keys.size(1), self.sliding_window_size + num_tokens - 1)
                keys = keys[:, -attn_keep:, :, :]
                values = values[:, -attn_keep:, :, :]

                cache_keep = min(combined_k.size(1), self.sliding_window_size)
                self.cache_k = combined_k[:, -cache_keep:, :, :]
                self.cache_v = combined_v[:, -cache_keep:, :, :]
            else:
                self.cache_k, self.cache_v = combined_k, combined_v

            dropped = combined_k.size(1) - keys.size(1)
            k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped
            q_start_pos_abs = self.ptr_current_pos
        else:
            keys, values = keys_new, values_new
        ####################################################

        # Transpose：(b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 計算帶 causal mask 的 scaled dot-product attention（也就是 self-attention）
        attn_scores = queries @ keys.transpose(2, 3)  # 每個 head 的內積

        ####################################################
        # causal + sliding-window mask
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        # 決定 q 和 k 的絕對位置
        if use_cache:
            q_start = q_start_pos_abs
            k_start = k_start_pos_abs
        else:
            q_start = 0
            k_start = 0
        q_positions = torch.arange(q_start, q_start + num_tokens_Q, device=device, dtype=torch.long)
        k_positions = torch.arange(k_start, k_start + num_tokens_K, device=device, dtype=torch.long)
        # Sliding window 寬度
        W = num_tokens_K + 1 if self.sliding_window_size is None else int(self.sliding_window_size)
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)
        mask_bool = (diff < 0) | (diff >= W)
        if use_cache:
            self.ptr_current_pos += num_tokens_Q
        else:
            self.ptr_current_pos = 0

        # 用 mask 填入 attention 分數
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape：(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合併各 head，其中 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可選的投影

        return context_vec

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# 第 4 章
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionWithSWA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            sliding_window_size=cfg["sliding_window_size"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # attention block 的 shortcut connection
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache 相關
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # 加回原始輸入

        # feed-forward block 的 shortcut connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 加回原始輸入

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache 相關
        blocks = []
        window_stride = cfg["sliding_window_stride"]
        window_size = cfg["sliding_window_size"] if "sliding_window_size" in cfg else None
        for i in range(cfg["n_layers"]):
            blk = TransformerBlock(cfg)
            # K:1 的排程，代表每 K 個 SWA 層後面接 1 個一般層
            K = int(window_stride)
            if K <= 0:
                # 0 => 全部是一般層；負數 => 全部是 SWA
                use_swa = False if K == 0 else True
            else:
                group = K + 1
                use_swa = (i % group) < K
            blk.att.sliding_window_size = window_size if use_swa else None
            blocks.append(blk)
        self.trf_blocks = nn.ModuleList(blocks)

        self.current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache 相關
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache 相關
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    # KV cache 相關
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # 用完整的 prompt 初始化 cache
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for step in range(max_new_tokens):
                # a) 選出 log 機率最高的 token（greedy sampling）
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) 把它接到目前的序列後面
                idx = torch.cat([idx, next_idx], dim=1)
                # c) 只把新的 token 餵給模型
                if step + 1 < max_new_tokens:  # 生成最後一個 token 後就略過 forward pass
                    logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--sliding_window_size", type=int, default=1024, help="Window size for sliding window attention.")
    parser.add_argument("--sliding_window_stride", type=int, default=2, help="K:1 frequency sliding window attention is applied. K=5 means 5 sliding window layers follows by a regular layer.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # 詞彙表大小
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,    # Embedding 維度
        "n_heads": args.n_heads,    # Attention head 數
        "n_layers": args.n_layers,  # 層數
        "drop_rate": 0.0,           # Dropout 比例
        "qkv_bias": False,          # Query-Key-Value 的 bias
        "sliding_window_size": args.sliding_window_size,
        "sliding_window_stride": args.sliding_window_stride
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()  # 關閉 dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
