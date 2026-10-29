# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import warnings

from ..generate import trim_input_tensor  # noqa: F401
from .utils import KVCache
import torch


def generate_text_simple(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.cfg["context_length"]

    with torch.no_grad():
        if use_cache:
            cache = KVCache(n_layers=model.cfg["n_layers"])
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], cache=cache)

            for step in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)
                if step + 1 < max_new_tokens:  # 生成最後一個 token 後就略過 forward pass
                    logits = model(next_idx, cache=cache)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], cache=None)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def generate_text_simple_stream(model, token_ids, max_new_tokens, eos_token_id=None, context_size=None):
    # context_size 是為了向下相容而保留，沒有作用。
    if context_size is not None:
        warnings.warn(
            "context_size is ignored by this streaming generator. Prompts are not "
            "truncated; the model's configured context limit still applies.",
            stacklevel=2,
        )

    model.eval()

    with torch.no_grad():
        cache = KVCache(n_layers=model.cfg["n_layers"])
        model.reset_kv_cache()

        # 用初始 context 預熱 cache
        logits = model(token_ids, cache=cache)

        for step in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)

            # 只把新的 token 餵給模型；歷史交給 cache 處理
            if step + 1 < max_new_tokens:  # 生成最後一個 token 後就略過 forward pass
                logits = model(next_token, cache=cache)
