import numpy as np
from sklearn.preprocessing import StandardScaler


class Lasso:
    def __init__(self, alpha=0.01, learning_rate=0.01, normalize=True):
        """
        Args:
            alpha (float): L1正则化强度
            learning_rate (float): 学习率
            normalize (bool): 是否标准化数据
        """
        self.alpha = alpha
        self.lr = learning_rate
        self.normalize = normalize

        self.coef_ = None  # 模型系数
        self.intercept_ = 0.0  # 截距
        self.scaler = None  # 标准化器
        self.n_features_ = None
        self._is_fitted = False  # 标记是否已调用fit

    def fit(self, X, y):
        """首次全局训练（计算标准化参数并初始化模型）"""
        X, y = np.asarray(X), np.asarray(y)
        self.n_features_ = X.shape[1]
        self.coef_ = np.zeros(self.n_features_)

        # 初始化标准化器
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # 首次批量训练（可改为多轮迭代）
        for _ in range(10):  # 模拟多轮训练增强稳定性
            for i in range(X.shape[0]):
                self._update_single_sample(X[i], y[i])

        self._is_fitted = True
        return self

    def partial_fit(self, X, y):
        """增量训练（需在fit之后调用）"""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before partial_fit()")

        X, y = np.asarray(X), np.asarray(y)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")

        # 标准化新数据（使用fit时计算的全局参数）
        if self.normalize:
            X = self.scaler.transform(X)

        # 单样本或批量更新
        for i in range(X.shape[0]):
            self._update_single_sample(X[i], y[i])

        return self

    def _update_single_sample(self, x, y):
        """处理单个样本的L1正则化更新"""
        error = y - (np.dot(x, self.coef_) + self.intercept_)

        # 更新截距（无正则化）
        self.intercept_ += self.lr * error

        # 更新系数（带L1正则化）
        for j in range(self.n_features_):
            grad = -error * x[j]

            # L1次梯度处理
            if self.coef_[j] > 0:
                grad += self.alpha
            elif self.coef_[j] < 0:
                grad -= self.alpha
            else:
                if abs(error * x[j]) > self.alpha:
                    grad += self.alpha * np.sign(error * x[j])
                else:
                    grad = 0

            self.coef_[j] -= self.lr * grad

    def predict(self, X):
        """预测"""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")

        X = np.asarray(X)
        if self.normalize:
            X = self.scaler.transform(X)
        return np.dot(X, self.coef_) + self.intercept_