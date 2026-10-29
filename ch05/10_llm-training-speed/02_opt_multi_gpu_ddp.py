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

# 新增的 import（見附錄 A）：
import platform
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# 新增：初始化分散式 process group 的函式（每張 GPU 1 個 process）
# 這讓 process 之間可以互相溝通
# （見附錄 A）：
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # 只有在 torchrun 尚未設定時才設定 MASTER_ADDR 和 MASTER_PORT
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # 初始化 process group
    if platform.system() == "Windows":
        # 關閉 libuv，因為 Windows 版 PyTorch 沒有編譯這項支援
        os.environ["USE_LIBUV"] = "0"
        # Windows 使用者可能需要用 "gloo" 而不是 "nccl" 作為 backend
        # gloo：Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl：NVIDIA Collective Communication Library
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


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


# 新增：改成 shuffle=False 並使用 sampler
# （見附錄 A）：
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, drop_last=True, num_workers=0):
    # 初始化 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 建立資料集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 建立 dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # 新增：設為 False，因為下面用了 DistributedSampler
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        # 新增：把 batch 分配到各 GPU，樣本不重疊：
        sampler=DistributedSampler(dataset)  # 新增
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


def generate_and_print_sample(model, device, start_context):
    model.eval()

    # 新增：為 DDP 修改
    context_size = model.module.pos_emb.weight.shape[0] if isinstance(model, DDP) else model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tiktoken.get_encoding("gpt2")).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tiktoken.get_encoding("gpt2"))
        print(decoded_text.replace("\n", " "))  # 精簡的輸出格式
    model.train()


def train_model_simple_with_timing(model, train_loader, val_loader, optimizer, device,
                                   num_epochs, eval_freq, eval_iter, start_context):
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # 新增：判斷目前的 rank（非分散式時預設為 0）
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

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
        # 新增：為 DistributedSampler 設定 epoch，讓每個 process 都有不同的打亂順序
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

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

                # 計算這段區間內本機處理的 token 數
                local_interval = total_tokens - last_tokens
                last_tokens = total_tokens

                # 彙整所有裝置處理過的 token 數
                local_tensor = torch.tensor([local_interval], device=device, dtype=torch.float)
                global_tensor = local_tensor.clone()
                torch.distributed.all_reduce(global_tensor, op=torch.distributed.ReduceOp.SUM)
                global_interval = global_tensor.item()

                # 這段區間的全域每秒 token 數
                global_tps = global_interval / elapsed if elapsed > 0 else 0

                # 更新累積 token 數（本機），並彙整到全域
                cumulative_tokens += local_interval
                local_cum_tensor = torch.tensor([cumulative_tokens], device=device, dtype=torch.float)
                global_cum_tensor = local_cum_tensor.clone()
                torch.distributed.all_reduce(global_cum_tensor, op=torch.distributed.ReduceOp.SUM)
                global_cumulative_tokens = global_cum_tensor.item()
                cumulative_time += elapsed
                global_avg_tps = global_cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # 評估模型表現（可能會增加額外開銷）
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                # 新增：每張 GPU 只印一次 log（選 rank 0 的 GPU）
                if rank == 0:
                    print(f"Ep {epoch+1}, Step {global_step:06d}, "
                          f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                          f"Step tok/sec: {round(global_tps)}, Global avg tok/sec: {round(global_avg_tps)}")

        # 新增：只有 rank 0 會印出生成的樣本和記憶體用量統計
        if rank == 0 and epoch % 5 == 0:
            generate_and_print_sample(model, device, start_context)

            # 記憶體統計
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # 轉成 GB
                reserved = torch.cuda.memory_reserved(current_device) / 1024**3    # 轉成 GB

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

# 新增：加入 rank 和 world_size
def main(gpt_config, settings, rank, world_size):

    ddp_setup(rank, world_size)  # 新增：初始化 process group
    device = torch.device("cuda", rank)

    torch.manual_seed(123)

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # Ampere（8.0+）以及更新的架構
            # 保留給 float32 實驗使用；這個設定不影響 bfloat16 的矩陣乘法。
            torch.set_float32_matmul_precision("high")

    # 新增：只在 1 張 GPU 上印資訊
    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")

            if capability[0] >= 8:
                print("TF32 enabled for float32 matmuls")
            else:
                print("TF32 is not supported on this GPU.")
        print()

    ##############################
    # 有需要就下載資料
    ##############################

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    # 新增：只下載 1 次
    if rank == 0:
        if not os.path.exists(file_path):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text_data = response.text
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
    # 新增：所有 process 都等 rank 0 完成，使用 GPU index。
    torch.distributed.barrier(device_ids=[device.index])

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # 初始化模型
    ##############################

    model = GPTModel(gpt_config)
    model = torch.compile(model)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    # 新增：用 DDP 包裝模型
    model = DDP(model, device_ids=[rank])
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
        num_workers=4
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        num_workers=4
    )

    ##############################
    # 訓練模型
    ##############################

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
    )

    # 新增：清理分散式 process
    destroy_process_group()

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # 新增：從環境變數取得 rank 和 world size
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

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
        "learning_rate": 5e-4,  # * world_size,  # NEW: Increase learning rate to account for multiple GPUs
        "num_epochs": 50,
        "batch_size": 32,
        "weight_decay": 0.1
    }

    ###########################
    # 開始訓練
    ###########################

    train_losses, val_losses, tokens_seen, model = main(
        GPT_CONFIG_124M, OTHER_SETTINGS,
        rank, world_size  # 新增
    )

    ###########################
    # 訓練之後
    ###########################

    # 新增：只建立 1 張圖
    if rank == 0:
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
