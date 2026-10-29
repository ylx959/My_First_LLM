# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 附錄 A：PyTorch 入門（第 3 部分）

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 新增的 import：
import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# 新增：初始化分散式 process group 的函式（每張 GPU 1 個 process）
# 這讓 process 之間可以互相溝通
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # 執行 rank:0 process 的機器
    # 這裡假設所有 GPU 都在同一台機器上
    os.environ["MASTER_ADDR"] = "localhost"
    # 機器上任何可用的 port
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


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第 1 個隱藏層
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 第 2 個隱藏層
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 輸出層
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # 取消註解這幾行來增加資料集大小，讓這個腳本可以在最多 8 張 GPU 上執行：
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # 新增：設為 False，因為下面用了 DistributedSampler
        pin_memory=True,
        drop_last=True,
        # 新增：把 batch 分配到各 GPU，樣本不重疊：
        sampler=DistributedSampler(train_ds)  # 新增
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# 新增：包裝函式
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # 新增：初始化 process group

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # 新增：用 DDP 包裝模型
    # 核心模型現在可以透過 model.module 存取

    for epoch in range(num_epochs):
        # 新增：設定 sampler，確保每個 epoch 的打亂順序都不同
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:

            features, labels = features.to(rank), labels.to(rank)  # 新增：使用 rank
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss 函式

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 記錄
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # 新增（書中沒有）：
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )
    ####################################################

    destroy_process_group()  # 新增：乾淨地結束分散式模式


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    # 因為資料集很小，這個腳本在超過 2 張 GPU 時可能無法運作
    # 如果 GPU 超過 2 張，請執行 `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)

    # 新增：啟動新的 process
    # 注意 spawn 會自動傳入 rank
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size 會為每張 GPU 啟動一個 process
