# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from .ch05 import calc_loss_batch, evaluate_model, generate_and_print_sample

import math
import torch


def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, orig_book_version=False):

    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # 從 optimizer 取得最大 learning rate
    peak_lr = optimizer.param_groups[0]["lr"]

    # 計算訓練過程的總迭代次數
    total_training_steps = len(train_loader) * n_epochs

    # 計算 warmup 階段 learning rate 的增量
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # 依目前階段（warmup 或 cosine annealing）調整 learning rate
            if global_step < warmup_steps:
                # 線性 warmup
                lr = initial_lr + global_step * lr_increment
            else:
                # warmup 之後做 cosine annealing
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # 把算好的 learning rate 套用到 optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # 儲存目前的 learning rate

            # 計算 loss 並反向傳播
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # warmup 階段之後套用 gradient clipping，避免梯度爆炸
            if orig_book_version:
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if global_step >= warmup_steps:  # 書中原本用 global_step > warmup_steps，導致 warmup 之後跳過一次 clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            tokens_seen += input_batch.numel()

            # 定期在訓練集和驗證集上評估模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 印出目前的 loss
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # 從模型生成並印出一段樣本來監看進度
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs
