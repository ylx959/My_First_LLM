# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 生成文字用的額外工具與輔助函式（書中未涵蓋）
# 在主章節中

def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    if max_new_tokens > context_len:
        raise ValueError("max_new_tokens must not exceed context_len.")
    # 最後生成的 token 不需要再做一次 forward pass。
    keep_len = max(1, context_len - max(0, max_new_tokens - 1))

    # 如果 prompt 太長，就從左邊截斷到 keep_len
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]

    return input_ids_tensor
