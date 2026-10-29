# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

# llms_from_scratch 的安裝說明請見：
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with pretrained weights generated in chapter 5.
    This requires that you run the code in chapter 5 first, which generates the necessary model.pth file.
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # 詞彙表大小
        "context_length": 256,  # 縮短的 context length（原本：1024）
        "emb_dim": 768,         # Embedding 維度
        "n_heads": 12,          # Attention head 數
        "n_layers": 12,         # 層數
        "drop_rate": 0.1,       # Dropout 比例
        "qkv_bias": False       # Query-key-value 的 bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path("..") / "01_main-chapter-code" / "model.pth"
    if not model_path.exists():
        print(f"Could not find the {model_path} file. Please run the chapter 5 code (ch05.ipynb) to generate the model.pth file.")
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_124M


# 取得下面 chainlit 函式需要的 tokenizer 和模型檔案
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    token_ids = generate(  # 函式內部已經用了 `with torch.no_grad()`
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # 使用者的文字透過 `message.content` 提供
        max_new_tokens=50,
        context_size=model_config["context_length"],
        top_k=1,
        temperature=0.0
    )

    text = token_ids_to_text(token_ids, tokenizer)

    await chainlit.Message(
        content=f"{text}",  # 把模型回應傳回介面
    ).send()
