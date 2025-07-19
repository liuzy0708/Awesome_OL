import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np


class MLP_OGD(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self._is_initialized = False  # 标记是否已初始化参数
        self.to(device)

    def _init_network(self, input_dim, output_dim):
        """初始化网络结构和优化器"""
        hidden_dim = 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self._is_initialized = True
        self.to(self.device)  # 确保参数在正确的设备上

    def forward(self, x):
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call fit() first.")
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, X, y, batch_size=32, max_epoch=10):
        """训练方法（无进度条版）"""
        # 转换为Tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        # 初始化网络（首次调用时）
        if not self._is_initialized:
            input_dim = X.shape[1]
            output_dim = len(torch.unique(y)) if y.dim() == 1 else y.shape[1]
            self._init_network(input_dim, output_dim)

        # 确保标签是类别索引
        y = y.squeeze() if y.dim() > 1 else y

        # 创建DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)

        for epoch in range(max_epoch):
            self.train()
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:  # 无进度条
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).long()

                outputs = self(batch_x)
                loss = self.criterion(outputs, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            #print(f'Epoch {epoch + 1}, Avg Loss: {epoch_loss / len(dataloader):.4f}')

    def predict(self, X, batch_size=32):
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call fit() first.")

        # 转换为Tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        self.eval()
        pred_list = []
        dataloader = torch.utils.data.DataLoader(X, batch_size=batch_size)

        with torch.no_grad():
            for batch_x in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self(batch_x)
                pred = outputs.argmax(dim=1)
                pred_list.append(pred.cpu())

        return torch.cat(pred_list, dim=0)

    def partial_fit(self, X, y):
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call fit() first.")

        # 转换为Tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        self.train()
        X = X.to(self.device)
        y = y.to(self.device).long()

        outputs = self(X)
        loss = self.criterion(outputs, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()