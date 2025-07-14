import numpy as np


class MyOneHotEncoder:
    def __init__(self, max_category=None):
        self.max_category = max_category  # 最大类别值（用于确定编码维度）
        self.categories_ = None  # 存储所有可能的类别

    def fit(self, Y):
        """确定所有可能的类别"""
        Y = np.array(Y).flatten()  # 确保 Y 是一维数组
        self.categories_ = np.unique(Y)
        if self.max_category is None:
            self.max_category = max(self.categories_)
        return self

    def transform(self, Y):
        """将 Y 转换为独热编码"""
        Y = np.array(Y).flatten()  # 确保 Y 是一维数组
        n_samples = len(Y)
        n_categories = self.max_category + 1  # 0~max_category 共 max_category+1 个类别

        # 初始化全 0 矩阵
        one_hot = np.zeros((n_samples, n_categories), dtype=int)

        # 对每个样本，在对应类别位置设为 1
        for i, y in enumerate(Y):
            if y > self.max_category:
                raise ValueError(f"类别 {y} 超出最大类别值 {self.max_category}")
            one_hot[i, y] = 1

        return one_hot

    def fit_transform(self, Y):
        """先 fit 再 transform"""
        self.fit(Y)
        return self.transform(Y)