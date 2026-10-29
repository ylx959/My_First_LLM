# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 內部使用的檔案（unit test）

from gpt import main

expected = """
==================================================
                      IN
==================================================

Input text: Hello, I am
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])


==================================================
                      OUT
==================================================

Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]])
Output length: 14
Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous
"""


def test_main(capsys):
    main()
    captured = capsys.readouterr()

    # 統一換行符號，並去掉每行尾端的空白
    normalized_expected = "\n".join(line.rstrip() for line in expected.splitlines())
    normalized_output = "\n".join(line.rstrip() for line in captured.out.splitlines())

    # 比較正規化後的字串
    assert normalized_output == normalized_expected
