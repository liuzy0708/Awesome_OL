""" DSA-AI Strategy."""

import scipy.stats
import numpy as np
from sklearn import preprocessing

class DSA_AI_strategy:

    def __init__(self, X_memory_collection, y_memory_collection, d, n_class, kappa, gamma):
        self.memory_space = 30
        self.n_class = n_class
        self.kappa = kappa
        self.Reg_interval = 20
        self.Memory_interval = 100
        self.alpha = 0.10
        self.gamma = gamma
        self.gamma_trv = [self.gamma] * self.n_class
        self.gamma_temp_trv = [0] * self.n_class
        self.gamma_trv_collection = [[]] * self.n_class
        self.gamma_min = 0.15
        self.gamma_collection = []
        self.d = d
        self.para = 0

        self.res = 0

        self.n_annotation = 0

        self.X_label_collection = [[]] * self.n_class
        self.label_count = [0] * self.n_class
        self.X_store_collection = [[]] * self.n_class
        self.y_store_collection = [[]] * self.n_class
        self.X_memory_collection, self.y_memory_collection = self.memory_func(X_memory_collection, y_memory_collection)
        self.ratio = [0] * self.n_class

        self.cold_count = 0
        self.period_count = 0

        self.label_history = []
        self.svf_collection = []
        self.cold_collection = []
        self.period_count_collection = []

        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.fit_ref = list(range(0, self.n_class))

    def normalization(self, para):
        epsilon = 1e-10
        min_value = np.min(para)
        max_value = np.max(para)
        para_mod = (para - min_value) / (max_value - min_value + epsilon)
        para_sum = np.sum(para_mod)
        para_normal = para_mod / para_sum
        return para_normal

    def memory_func(self, X_memory, y_memory):
        X_memory_collection = [[]] * self.n_class
        y_memory_collection = [[]] * self.n_class

        for cls in range(self.n_class):
            for sample in range(y_memory.shape[0]):
                if y_memory[sample] == cls:
                    y_memory_collection[cls] = y_memory_collection[cls] + [cls]
                    X_memory_collection[cls] = X_memory_collection[cls] + [X_memory[sample, :]]
        return X_memory_collection, y_memory_collection

    def JS_divergence(self, P, Q):
        M = ((P + Q) / 2)
        return 0.5 * scipy.stats.entropy(P, M, axis=1) + 0.5 * scipy.stats.entropy(Q, M, axis=1)

    def concave_func(self, x, kappa):
        y = x ** (1 / kappa)
        return y

    def res_cal(self, clf, X, y):
        self.onehotencoder.fit_transform(np.mat(self.fit_ref).T)
        y_pred_prob = clf.predict_proba(X)
        y_pred = clf.predict(X)
        if y_pred_prob.shape[1] != self.n_class:
            y_pred_prob = self.onehotencoder.transform(np.mat(y_pred).T)
        y_one_hot = self.onehotencoder.transform(np.mat(y).T)
        res = self.JS_divergence(y_one_hot, self.normalization(y_pred_prob))
        return res

    def SVF_evaluation(self, X, clf):
        self.onehotencoder.fit_transform(np.mat(self.fit_ref).T)
        self.para = clf.predict_proba(X)
        if self.para.shape[1] == 1:
            self.para = self.onehotencoder.transform(np.mat(self.para).T)
        para_normal = self.normalization(self.para)
        result = []
        for i in range(para_normal.shape[1]):
            eval_temp = para_normal[0, i] * (self.concave_func(self.label_history.count(i) + 1, self.kappa) - self.concave_func(self.label_history.count(i), self.kappa))
            result.append(eval_temp)
        if sum(result) > self.gamma:
            is_SVF = True
        else:
            is_SVF = False
        return is_SVF

    def evaluation(self, X, y, clf):

        if self.SVF_evaluation(X, clf) or self.cold_count == self.Reg_interval:
            self.label_history = self.label_history + [y]

            for i in range(self.n_class):
                if y == i:
                    self.X_label_collection[i] = self.X_label_collection[i] + X.tolist()
                    self.label_count[i] = self.label_count[i] + 1

            clf.partial_fit(X, y)
            self.res = self.res_cal(clf, X, y)

            if self.res > 0.10:

                for i in range(self.n_class):
                    if y == i:
                        gamma_temp = self.gamma - self.alpha / self.kappa * pow(self.label_count[i] + 1, 1 / self.kappa - 1)
                        if gamma_temp > self.gamma_min:
                            self.gamma = gamma_temp
                        else:
                            self.gamma = self.gamma_min

            for i in range(self.n_class):
                if y == i:
                    self.gamma_temp_trv[i] = self.gamma_trv[i] - self.alpha / self.kappa * pow(self.label_count[i] + 1, 1 / self.kappa - 1)
                    if self.gamma_temp_trv[i] > self.gamma_min:
                        self.gamma_trv[i] = self.gamma_temp_trv[i]
                    else:
                        self.gamma_trv[i] = self.gamma_min

            if self.cold_count == self.Reg_interval:

                for j in range(self.n_class):
                    if y == j:
                        if self.gamma > self.gamma_min:
                            self.gamma = (1 - self.res / 2) * self.gamma_trv[j]
                        else:
                            self.gamma = self.gamma_min

                for i in range(self.n_class):
                    self.ratio[i] = self.label_count[i] / sum(self.label_count)

                for i in range(self.n_class):

                    if self.label_count[i] > (self.Memory_interval * self.ratio[i]):
                        self.X_store_collection[i] = self.X_memory_collection[i][-int(self.Memory_interval * self.ratio[i]):]
                    else:
                        if self.label_count[i] == 0:
                            self.X_store_collection[i] = self.X_memory_collection[i][-self.memory_space:]
                        else:
                            self.X_store_collection[i] = self.X_memory_collection[i][-self.memory_space:] + self.X_label_collection[i]

                for i in range(self.n_class):
                    self.y_store_collection[i] = [i] * len(self.X_store_collection[i])

                self.X_store_collection =  np.vstack(self.X_store_collection)
                self.y_store_collection = np.array([item for sublist in self.y_store_collection for item in sublist])

                # Retrain
                clf.fit(self.X_store_collection, self.y_store_collection)

                # Initialization
                self.X_label_collection = [[]] * self.n_class
                self.X_store_collection = [[]] * self.n_class
                self.y_store_collection = [[]] * self.n_class
                self.label_count = [0] * self.n_class
                self.label_history = []
                self.gamma_trv = [0.40] * self.n_class

                self.cold_collection = self.cold_collection + [1]
                self.svf_collection = self.svf_collection + [0]

                self.period_count = 0

            else:
                self.cold_collection = self.cold_collection + [0]
                self.svf_collection = self.svf_collection + [1]

            self.n_annotation += 1

            self.cold_count = 0
            self.period_count += 1

            isLabel = 1

        else:
            self.cold_collection = self.cold_collection + [0]
            self.svf_collection = self.svf_collection + [0]
            self.cold_count += 1
            self.period_count += 1

            isLabel = 0

        for i in range(self.n_class):
            self.gamma_trv_collection[i] = self.gamma_trv_collection[i] + [self.gamma_trv[i]]

        self.gamma_collection = self.gamma_collection + [self.gamma]

        return isLabel, clf