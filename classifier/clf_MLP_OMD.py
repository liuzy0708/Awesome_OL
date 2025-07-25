import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MLP_OMD(nn.Module):
    def __init__(self, eta=0.05, lambda_ce=0.7, device='cpu'):
        super().__init__()
        self.eta = eta
        self.lambda_ce = lambda_ce
        self.device = torch.device(device)
        self.step = 1
        self._is_initialized = False

    def _init_network(self, input_dim, output_dim):
        self.model = SimpleMLP(input_dim, output_dim).to(self.device)
        self.output_dim = output_dim
        self.pi = torch.ones(output_dim, device=self.device) / output_dim
        self._is_initialized = True

    def forward(self, X):
        return self.model(X)

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        X, y = X.to(self.device), y.to(self.device)

        if not self._is_initialized:
            input_dim = X.shape[1]
            output_dim = len(torch.unique(y)) if y.dim() == 1 else y.shape[1]
            self._init_network(input_dim, 4)

            # 初始化 pi 为真实标签分布
            counts = torch.bincount(y, minlength=self.output_dim).float()
            self.pi = counts / counts.sum()

        self.partial_fit(X, y)

    def partial_fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        X, y = X.to(self.device), y.to(self.device)

        self._mirror_update(X, y)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        logits = self.model(X)
        probs = F.softmax(logits, dim=1)
        return probs.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def _mirror_update(self, X, y):
        self.step += 1
        eta_t = self.eta / math.sqrt(self.step)

        logits = self.model(X)
        probs = F.softmax(logits, dim=1)
        pi_target = self.pi.detach()

        # 加入监督的交叉熵 loss
        ce_loss = F.nll_loss(torch.log(probs + 1e-8), y)
        kl_loss = F.kl_div(torch.log(probs + 1e-8), pi_target, reduction='batchmean')
        loss = self.lambda_ce * ce_loss + (1 - self.lambda_ce) * kl_loss

        self.model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= eta_t * param.grad

            # 更新 pi：用样本平均的预测分布与真实标签差值做梯度
            grad = torch.zeros_like(self.pi)
            for i in range(len(y)):
                pred = probs[i]
                true = F.one_hot(y[i], num_classes=self.output_dim).float()
                grad += pred - true
            grad /= len(y)

            # OMD 指数更新 + 平滑
            new_pi = self.pi * torch.exp(-eta_t * grad)
            new_pi = new_pi + 1e-4  # 避免塌缩
            self.pi = new_pi / new_pi.sum()
