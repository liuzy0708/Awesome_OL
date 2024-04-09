import numpy as np
from sklearn import preprocessing

class USGQS_strategy:
    def __init__(self, n_class, kappa, thre):
        self.n_class = n_class
        self.kappa = kappa
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.fit_ref = list(range(0, self.n_class))
        self._thre = thre
        self.count = 0
        self.N_max = 100
        self.K = 0
        self.N_pass = 0
        self.count = np.zeros(self.n_class)

    def concave_func(self, x, kappa):
        y = x ** (1 / kappa)
        return y

    def softmax_norm(self, array):
        exp_array = np.exp(array)
        sum_exp_array = np.sum(exp_array, axis=1, keepdims=True)
        softmax_array = exp_array / sum_exp_array
        return softmax_array

    def normalization(self, para):
        epsilon = 1e-10
        min_value = np.min(para)
        max_value = np.max(para)
        para_mod = (para - min_value) / (max_value - min_value + epsilon)
        para_sum = np.sum(para_mod)
        para_normal = para_mod / para_sum
        return para_normal

    def calculate_shannon_entropy(self, probabilities):
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy

    def evaluation(self, X, y, clf):
        para_norm = self.softmax_norm(clf.predict_proba(X))
        k_index = np.argmax(para_norm)
        entropy = self.calculate_shannon_entropy(para_norm)

        count_temp = self.count.copy()
        count_temp[k_index] += 1

        m_gain = (np.sum(np.vectorize(self.concave_func)(count_temp, self.kappa)) - np.sum(
            np.vectorize(self.concave_func)(self.count, self.kappa))) * entropy

        if m_gain >= self._thre:
            self.count = count_temp.copy()
            isLabel = 1
            clf.partial_fit(X, y)
        else:
            isLabel = 0
            self.N_pass += 1

        'activate'
        if self.N_pass == self.N_max:
            self.count = np.zeros(self.n_class)
            self.N_pass = 0

        return isLabel, clf
