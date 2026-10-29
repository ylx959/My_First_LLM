# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# 這個檔案收集了到目前為止所有相關的程式碼
# （第 2-4 章）。
# 這個檔案可以當作獨立腳本執行。

import torch


#####################################
# 第 5 章
#####################################
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 加上 batch 維度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 移除 batch 維度
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # for 迴圈和之前一樣：取得 logits，只看最後一個時間步
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 新增：用 top_k sampling 過濾 logits
        if top_k is not None:
            # 只保留 top_k 個值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 新增：套用 temperature 縮放
        if temperature > 0.0:
            logits = logits / temperature

            # 新增（書中沒有）：數值穩定性的小技巧，讓 mps 裝置得到相同的結果
            # softmax 之前先減掉每列的最大值
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # 套用 softmax 得到機率
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 從分佈中取樣
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 其他和之前一樣：取得 logits 值最高的詞彙 index
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # 如果遇到 end-of-sequence token 且有指定 eos_id，就提早停止生成
            break

        # 和之前一樣：把取樣到的 index 接到目前的序列後面
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
