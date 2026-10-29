# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os
import time

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

#####################################
# 第 2 章
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 對整段文字做 tokenize
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 用 sliding window 把書切成長度為 max_length 且互相重疊的序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 建立資料集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 建立 dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


#####################################
# 第 3 章
#####################################
class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 拆成 3 個 (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # 合併各 head，其中 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


#####################################
# 第 4 章
#####################################


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # attention block 的 shortcut connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
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

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx 是目前 context 中 index 的 (B, T) 陣列
    for _ in range(max_new_tokens):

        # 如果目前的 context 超過支援的長度就截斷
        # 例如 LLM 只支援 5 個 token，而 context 大小是 10
        # 那就只用最後 5 個 token 作為 context
        idx_cond = idx[:, -context_size:]

        # 取得預測結果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只看最後一個時間步
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # 取得 logits 值最高的詞彙 index
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 把取樣到的 index 接到目前的序列後面
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

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


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # 精簡的輸出格式
    model.train()


def train_model_simple_with_timing(model, train_loader, val_loader, optimizer, device,
                                   num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # 計算累積平均 tokens/sec 用的變數
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA 專用的計時設定
    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # 確保先前的 CUDA 運算都已完成
        t_start.record()          # 開始第一段區間的計時
    else:
        t0 = time.time()          # 開始第一段區間的計時

    # 主要訓練迴圈
    for epoch in range(num_epochs):
        model.train()
        for inp_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Forward 與 backward pass
            loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += inp_batch.numel()

            # 每到評估間隔，就量測經過時間和每秒 token 數
            if global_step % eval_freq == 0:
                # 結束目前區間的計時
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()  # 等待所有 CUDA 運算完成。
                    elapsed = t_start.elapsed_time(t_end) / 1000  # 把毫秒轉成秒
                    t_start.record()  # 重設計時器給下一段區間
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()  # 重設計時器給下一段區間

                # 計算這段區間處理的 token 數
                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / elapsed if elapsed > 0 else 0  # 每秒 token 數

                # 更新累積計數（略過第一次評估區間）
                if global_step:  # 只有在 global_step == 0（第一次評估）時才是 False
                    cumulative_tokens += tokens_interval
                    cumulative_time += elapsed

                # 計算累積平均 tokens/sec（不含第一段區間）
                avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # 評估模型表現（可能會增加額外開銷）
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                print(f"Ep {epoch+1}, Step {global_step:06d}, "
                      f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                      f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}")

        generate_and_print_sample(model, tokenizer, device, start_context)

        # 記憶體統計
        if torch.cuda.is_available():
            device = torch.cuda.current_device()

            allocated = torch.cuda.memory_allocated(device) / 1024**3  # 轉成 GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # 轉成 GB

            print(f"\nAllocated memory: {allocated:.4f} GB")
            print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # 畫出訓練和驗證 loss 隨 epoch 的變化
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 建立第二條 x 軸顯示看過的 token 數
    ax2 = ax1.twiny()  # 建立共用同一條 y 軸的第二條 x 軸
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 用來對齊刻度的隱形圖
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # 調整排版留出空間
    # plt.show()


#####################################
# 主要的函式呼叫
#####################################

def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # Ampere（8.0+）以及更新的架構
            # 保留給 float32 實驗使用；這個設定不影響 bfloat16 的矩陣乘法。
            torch.set_float32_matmul_precision("high")
            print("TF32 enabled for float32 matmuls")
        else:
            print("TF32 is not supported on this GPU.")
    print()

    ##############################
    # 有需要就下載資料
    ##############################

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # 初始化模型
    ##############################

    model = GPTModel(gpt_config)
    model = torch.compile(model)
    model.to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        fused=True
    )

    ##############################
    # 設定 dataloader
    ##############################

    # 訓練/驗證比例
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=4
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=4
    )

    ##############################
    # 訓練模型
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=10,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50304,     # 詞彙表大小
        "context_length": 1024,  # 每個訓練樣本的輸入 token 數
        "emb_dim": 768,          # Embedding 維度
        "n_heads": 12,           # Attention head 數
        "n_layers": 12,          # 層數
        "drop_rate": 0.1,        # Dropout 比例
        "qkv_bias": False        # Query-key-value 的 bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 15,
        "batch_size": 32,
        "weight_decay": 0.1
    }

    ###########################
    # 開始訓練
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # 訓練之後
    ###########################

    # 畫出結果
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # 儲存並載入模型
    #
    # compiled = hasattr(model, "_orig_mod")
    # if compiled:
    #     torch.save(model._orig_mod.state_dict(), "model.pth")
    # else:
    #     torch.save(model.state_dict(), "model.pth")
    #
    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))
