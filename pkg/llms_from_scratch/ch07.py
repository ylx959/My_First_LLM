# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import json
import os
import psutil
import requests

import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# 書中原本使用下面這段程式碼
# 不過 urllib 使用較舊的協定設定，
# 對某些使用 VPN 的讀者會造成問題。
# 上面的 `requests` 版本比較穩健
# 在這方面。


# import urllib

# def download_and_load_file(file_path, url):

#     if not os.path.exists(file_path):
#         with urllib.request.urlopen(url) as response:
#             text_data = response.read().decode("utf-8")
#         with open(file_path, "w", encoding="utf-8") as file:
#             file.write(text_data)

#     else:
#         with open(file_path, "r", encoding="utf-8") as file:
#             text_data = file.read()

#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)

#     return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # 預先 tokenize 文字
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # 找出 batch 中最長的序列
    # 並把最大長度加 1，這會多加一個
    # 下面的 padding token
    batch_max_length = max(len(item)+1 for item in batch)

    # padding 並準備輸入
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # 加入 <|endoftext|> token
        new_item += [pad_token_id]
        # 把序列 pad 到 batch_max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # 透過 padded[:-1] 移除多出來的 padding token
        # 這是透過 batch_max_length 的 +1 設定加上的
        # （多出來的 padding token 在後面的程式碼會用到）
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # 把輸入的 list 轉成 tensor 並移到目標裝置
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # 找出 batch 中最長的序列
    batch_max_length = max(len(item)+1 for item in batch)

    # padding 並準備輸入
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 加入 <|endoftext|> token
        new_item += [pad_token_id]
        # 把序列 pad 到 max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # 輸入去掉最後一個 token
        targets = torch.tensor(padded[1:])  # target 往右移 +1
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 把輸入的 list 轉成 tensor 並移到目標裝置
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 找出 batch 中最長的序列
    batch_max_length = max(len(item)+1 for item in batch)

    # padding 並準備輸入和 target
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 加入 <|endoftext|> token
        new_item += [pad_token_id]
        # 把序列 pad 到 max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # 輸入去掉最後一個 token
        targets = torch.tensor(padded[1:])  # target 往右移 +1

        # 新增：把 target 中除了第一個以外的 padding token 都換成 ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 新增：可選擇截斷到最大序列長度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 把輸入和目標的 list 轉成 tensor 並移到目標裝置
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    # 用 dictionary 建立要送出的資料
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # 下面的設定是為了得到確定性的回應
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # 送出 POST 請求
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            response_json = json.loads(line)
            if "message" in response_json:
                response_data += response_json["message"]["content"]

    return response_data


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores
