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
from llms_from_scratch.ch06 import classify_review


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load finetuned GPT-2 model generated in chapter 6.
    This requires that you run the code in chapter 6 first, which generates the necessary model.pth file.
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 詞彙表大小
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding 維度
        "n_heads": 12,           # Attention head 數
        "n_layers": 12,          # 層數
        "drop_rate": 0.1,        # Dropout 比例
        "qkv_bias": True         # Query-key-value 的 bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path("..") / "01_main-chapter-code" / "review_classifier.pth"
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 6 code"
            " (ch06.ipynb) to generate the review_classifier.pth file."
        )
        sys.exit()

    # 建立模型
    model = GPTModel(GPT_CONFIG_124M)

    # 依照 ch06.ipynb 第 6.5 節把模型改成分類器
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # 接著載入模型權重
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model


# 取得下面 chainlit 函式需要的 tokenizer 和模型檔案
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    user_input = message.content

    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    await chainlit.Message(
        content=f"{label}",  # 把模型回應傳回介面
    ).send()
