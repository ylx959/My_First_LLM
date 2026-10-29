# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import matplotlib.pyplot as plt
import os
import requests
import torch
import tiktoken


# 從本機檔案匯入
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple


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


def create_muon_optimizers(model, adamw_learning_rate, muon_learning_rate, weight_decay=0.1):
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available. Please update to a more recent PyTorch version.")

    embedding_param_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                embedding_param_names.add(full_name)

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and name not in embedding_param_names:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    optimizers = []
    if muon_params:
        optimizers.append(
            torch.optim.Muon(
                muon_params, lr=muon_learning_rate, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"
            )
        )
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=adamw_learning_rate, weight_decay=weight_decay))
    if not optimizers:
        raise ValueError("No trainable parameters found.")
    return optimizers


def train_model_simple(model, train_loader, val_loader, optimizers, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, max_steps=None):
    # 初始化記錄 loss 和看過 token 數的 list
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # 主要訓練迴圈
    for epoch in range(num_epochs):
        model.train()  # 把模型設為訓練模式

        for input_batch, target_batch in train_loader:
            for optimizer in optimizers:
                optimizer.zero_grad()  # 清掉上一個 batch 迭代的 loss 梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # 計算 loss 的梯度
            for optimizer in optimizers:
                optimizer.step()  # 用 loss 梯度更新模型權重
            tokens_seen += input_batch.numel()
            global_step += 1

            # 可選的評估步驟
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step + 1 >= max_steps:
                break

        # 每個 epoch 結束後印出一段範例文字
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

        if max_steps is not None and global_step + 1 >= max_steps:
            break

    return train_losses, val_losses, track_tokens_seen


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


def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # 有需要就下載資料
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

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
    model.to(device)  # nn.Module 類別不需要寫成 model = model.to(device)
    optimizers = create_muon_optimizers(
        model,
        adamw_learning_rate=settings["learning_rate"],
        muon_learning_rate=settings["muon_learning_rate"],
        weight_decay=settings["weight_decay"]
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
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # 訓練模型
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizers, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer,
        max_steps=settings.get("train_steps")
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="Maximum number of optimizer steps to run. If omitted, trains for all epochs."
    )
    args = parser.parse_args()

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
        "learning_rate": 5e-5,
        "muon_learning_rate": 1e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1,
        "train_steps": args.train_steps,
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
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
