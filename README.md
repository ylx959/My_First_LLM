# My First LLM — 學習筆記

這是我跟隨《Build a Large Language Model (From Scratch)》一書，從零開始實作 GPT 類大型語言模型（LLM）的個人學習 Repository。除了參考原書的程式碼與實作流程，我也加入了許多自己認為重要、但原始 Repository 未詳細說明的註解、概念解析與補充內容，透過實際動手與整理筆記，加深對 LLM 底層原理的理解。

> **來源聲明**
> 本 repo 的程式碼與教材內容皆來自 **Sebastian Raschka** 的著作與官方程式碼庫，著作權屬於原作者。
> 我只是為了個人學習而整理（刪除了安裝說明、CI 設定、給讀者的 issue template 等與學習無關的檔案）。
>
> - 書籍：*Build a Large Language Model (From Scratch)*, Sebastian Raschka, Manning, 2024（ISBN 9781633437166）
> - 原始 repo：<https://github.com/rasbt/LLMs-from-scratch>
> - 授權：Apache License 2.0，詳見 [LICENSE.txt](LICENSE.txt)

## 學習進度

| 章節 | 主題 | 主程式 | 狀態 |
|---|---|---|---|
| Ch 1 | Understanding Large Language Models | 無程式碼 | ⬜ |
| Ch 2 | Working with Text Data | [ch02](ch02/01_main-chapter-code) | ⬜ |
| Ch 3 | Coding Attention Mechanisms | [ch03](ch03/01_main-chapter-code) | ⬜ |
| Ch 4 | Implementing a GPT Model from Scratch | [ch04](ch04/01_main-chapter-code) | ⬜ |
| Ch 5 | Pretraining on Unlabeled Data | [ch05](ch05/01_main-chapter-code) | ⬜ |
| Ch 6 | Finetuning for Classification | [ch06](ch06/01_main-chapter-code) | ⬜ |
| Ch 7 | Finetuning to Follow Instructions | [ch07](ch07/01_main-chapter-code) | ⬜ |
| App. A | Introduction to PyTorch | [appendix-A](appendix-A/01_main-chapter-code) | ⬜ |
| App. B | References and Further Reading | [appendix-B](appendix-B) | ⬜ |
| App. C | Exercise Solutions | [appendix-C](appendix-C) | ⬜ |
| App. D | Adding Bells and Whistles to the Training Loop | [appendix-D](appendix-D/01_main-chapter-code) | ⬜ |
| App. E | Parameter-efficient Finetuning with LoRA | [appendix-E](appendix-E/01_main-chapter-code) | ⬜ |

（完成一章就把 ⬜ 改成 ✅）

## Repo 結構

```
chXX/
├── 01_main-chapter-code/   # 主章節 notebook 與程式碼
├── 0X_.../                 # 延伸（bonus）內容
└── README.md               # 該章資料夾說明
appendix-*/                 # 附錄
pkg/llms_from_scratch/      # 各章程式碼整理成的 Python 套件
```

## 環境設定

```bash
# 1. 安裝 uv（macOS / Linux） 選一個
#pip
pip install uv
#homebrew
brew install uv
#official
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 下載 repo 並安裝套件
git clone <這個 repo 的網址>
cd My_First_LLM
uv sync
```

## 執行

```bash
# 方式一：uv run
uv run jupyter lab

# 方式二：activate
source .venv/bin/activate
jupyter lab
```

## 我的筆記

<!-- 在這裡記錄每章的心得、卡關的地方 -->
