# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from llms_from_scratch.ch02 import create_dataloader_v1
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.appendix_d import train_model

import os
import urllib

import tiktoken
import torch
from torch.utils.data import Subset, DataLoader


def test_train(tmp_path):

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # 詞彙表大小
        "context_length": 256,  # 縮短的 context length（原本：1024）
        "emb_dim": 768,         # Embedding 維度
        "n_heads": 12,          # Attention head 數
        "n_layers": 12,         # 層數
        "drop_rate": 0.1,       # Dropout 比例
        "qkv_bias": False       # Query-key-value 的 bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 2,
        "batch_size": 1,
        "weight_decay": 0.1
    }

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # 有需要就下載資料
    ##############################

    file_path = tmp_path / "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # 初始化模型
    ##############################

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)  # nn.Module 類別不需要寫成 model = model.to(device)

    ##############################
    # 設定 dataloader
    ##############################

    # 訓練/驗證比例
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=OTHER_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=OTHER_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # 訓練模型
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_subset = Subset(train_loader.dataset, range(1))
    one_batch_train_loader = DataLoader(train_subset, batch_size=1)
    val_subset = Subset(val_loader.dataset, range(1))
    one_batch_val_loader = DataLoader(val_subset, batch_size=1)

    peak_lr = 0.001  # 書中原本誤設為 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)  # 書中不小心漏掉了 lr 的指派
    tokenizer = tiktoken.get_encoding("gpt2")

    n_epochs = 6
    warmup_steps = 1

    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, one_batch_train_loader, one_batch_val_loader, optimizer, device, n_epochs=n_epochs,
        eval_freq=5, eval_iter=1, start_context="Every effort moves you",
        tokenizer=tokenizer, warmup_steps=warmup_steps,
        initial_lr=1e-5, min_lr=1e-5
    )

    assert round(train_losses[0], 1) == 10.9
    assert round(val_losses[0], 1) == 11.0
    assert train_losses[-1] < train_losses[0]
