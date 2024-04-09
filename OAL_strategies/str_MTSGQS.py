import numpy as np
from sklearn import preprocessing

class MTSGQS_strategy:
    def __init__(self, n_class, kappa, gamma, n_capacity):
        self.n_class = n_class
        self.kappa = kappa
        self.gamma = gamma
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.fit_ref = list(range(0, self.n_class))
        self.label_history = []
        self.n_capacity = n_capacity
        self.res_thre = 0.3
        self.X_label_history = []

    def concave_func(self, x, kappa):
        y = x ** (1 / kappa)
        return y

    def normalization(self, para):
        epsilon = 1e-10
        min_value = np.min(para)
        max_value = np.max(para)
        para_mod = (para - min_value) / (max_value - min_value + epsilon)
        para_sum = np.sum(para_mod)
        para_normal = para_mod / para_sum
        return para_normal

    def counter_func(self, class_number, labeled_data_label):
        counter = 0
        for j in range(labeled_data_label.shape[0]):
            if labeled_data_label[j] == class_number:
                counter += 1
        return counter

    def evaluation(self, X, y, clf):
        self.onehotencoder.fit_transform(np.mat(self.fit_ref).T)
        self.para = clf.predict_proba(X)

        if self.para.shape[1] == 1:
            self.para = self.onehotencoder.transform(np.mat(self.para).T)

        para_normal = self.normalization(self.para)

        result = []

        for i in range(para_normal.shape[1]):
            eval_temp = para_normal[0, i] * (
                        self.concave_func(self.label_history.count(i) + 1, self.kappa) - self.concave_func(self.label_history.count(i), self.kappa))
            result.append(eval_temp)

        if sum(result) > self.gamma:
            isLabel = 1
            clf.partial_fit(X, y)
            res = np.linalg.norm(para_normal-y, ord=2, axis=1, keepdims=True) # 漂移检测残差计算
            if res <= self.res_thre:
                if len(self.label_history) > self.n_capacity:
                    X_memory = np.vstack(self.X_label_history[- self.n_capacity:])
                    y_memory = np.array(self.label_history[- self.n_capacity:])
                else:
                    X_memory = np.vstack(self.X_label_history)
                    y_memory = np.array(self.label_history)
                self.label_history = self.label_history + y.tolist()
                self.X_label_history = self.X_label_history + [X]
                clf.fit(X_memory, y_memory)

            else:
                self.label_history = self.label_history + y.tolist()
                self.X_label_history = self.X_label_history + [X]
        else:
            isLabel = 0

        return isLabel, clf

