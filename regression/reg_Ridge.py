import numpy as np
from sklearn.preprocessing import StandardScaler


class Ridge:
    def __init__(self, alpha=0.01, learning_rate=0.01, normalize=True):
        """
        Args:
            alpha (float): L2正则化强度
            learning_rate (float): 学习率
            normalize (bool): 是否标准化数据
        """
        self.alpha = alpha  # L2正则化系数
        self.lr = learning_rate  # 学习率
        self.normalize = normalize  # 是否标准化
        self.coef_ = None  # 模型系数
        self.intercept_ = 0.0  # 截距项
        self.scaler = None  # 标准化器
        self._is_fitted = False  # 标记是否已初始化

    def fit(self, X, y):
        """首次全局训练（计算标准化参数并初始化模型）"""
        X, y = np.asarray(X), np.asarray(y)
        self.coef_ = np.zeros(X.shape[1])

        # 初始化标准化器
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # 首次批量训练（模拟多轮迭代增强稳定性）
        for _ in range(10):
            for i in range(X.shape[0]):
                self._update_single_sample(X[i], y[i])

        self._is_fitted = True
        return self

    def partial_fit(self, X, y):
        """增量训练（需在fit之后调用）"""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before partial_fit()")

        X, y = np.asarray(X), np.asarray(y)
        if self.normalize:
            X = self.scaler.transform(X)

        # 单样本或批量更新
        for i in range(X.shape[0]):
            self._update_single_sample(X[i], y[i])
        return self

    def _update_single_sample(self, x_i, y_i):
        """处理单个样本的L2正则化更新"""
        error = y_i - (np.dot(x_i, self.coef_) + self.intercept_)

        # 更新系数（含L2正则化梯度）
        self.coef_ -= self.lr * (-error * x_i + self.alpha * self.coef_)

        # 更新截距（无正则化）
        self.intercept_ -= self.lr * (-error)

    def predict(self, X):
        """预测"""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")

        X = np.asarray(X)
        if self.normalize:
            X = self.scaler.transform(X)
        return np.dot(X, self.coef_) + self.intercept_