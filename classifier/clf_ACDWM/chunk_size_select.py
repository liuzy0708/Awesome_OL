from numpy import *
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import f
from scipy.stats import chi2
from .subunderbagging import *
from .check_measure import *
from sklearn.model_selection import train_test_split


class ChunkSizeBase:

    def __init__(self, fix_num, init_num=100):
        self.fix_num = fix_num
        self.init_num = init_num
        self.enough = 0
        self.chunk_data = array([])
        self.chunk_label = array([])
        self.chunk_count = zeros(2)
        self.round = 0

    def update(self, data, label):
        if self.enough == 1:
            self.chunk_data = array([])
            self.chunk_label = array([])
            self.enough = 0
            self.chunk_count = zeros(2)

        if label == 1:
            self.chunk_count[1] += 1
        else:
            self.chunk_count[0] += 1
        if len(label) > 0:
            self.chunk_data = r_[self.chunk_data.reshape(-1, data.size), data.reshape(1, -1)]
            self.chunk_label = r_[self.chunk_label, label]

        self.check_condition()

    def get_enough(self):
        return self.enough

    def get_chunk(self):
        return self.chunk_data, self.chunk_label


class ChunkSizeSelect(ChunkSizeBase):

    def __init__(self, chunk_min=100, min_min=5, P=250, T=100, Q=1000, nt=10, delta=0.05, init_num=100, k_mode=2,
                 mute=1):

        self.chunk_min = chunk_min
        self.min_min = min_min
        self.P = P
        self.T = T
        self.Q = Q
        self.nt = nt
        self.alpha = delta
        self.init_num = init_num
        self.k_mode = k_mode
        self.mute = mute

        self.chunk_count = zeros(2)
        self.chunk_data = array([])
        self.chunk_label = array([])
        self.var_0 = []
        self.var_1 = []
        self.round = 0
        self.data_count = 0
        self.test_data = []
        self.enough = 0
        self.store_chunk_data = []
        self.store_chunk_label = []
        self.min_class = 0

    def update(self, data, label):

        if self.enough == 1:
            self.enough = 0
            self.store_chunk_data = self.chunk_data
            self.store_chunk_label = self.chunk_label
            self.chunk_data = array([])
            self.chunk_label = array([])

        if label == 1:
            self.chunk_count[1] += 1
        else:
            self.chunk_count[0] += 1
        # print('self.chunk_data:', self.chunk_data)
        # print('data:', data)
        # print('shape of self.chunk_data = ', self.chunk_data.shape)
        # print('shape of data = ', data.shape)
        self.chunk_data = r_[self.chunk_data.reshape(-1, data.size), data.reshape(1, -1)]
        self.chunk_label = r_[self.chunk_label, label]

        self.check_condition()

    def check_condition(self):
        self.data_count += 1

        if sum(self.chunk_label == 1) > sum(self.chunk_label == -1):
            self.min_class = -1
        else:
            self.min_class = 1

        if self.round == 0 and min(self.chunk_count) > 0 and sum(self.chunk_count) >= self.init_num:
            self.test_data = self.chunk_data[random.permutation(self.chunk_label.size)[:self.nt]]
            self.store_chunk_data = self.chunk_data
            self.store_chunk_label = self.chunk_label
            self.chunk_count = zeros(2)
            self.enough = 1
            self.round += 1

        elif min(self.chunk_count) >= self.min_min and sum(self.chunk_count) >= self.chunk_min:

            self.chunk_count = zeros(2)
            model = SubUnderBagging(Q=self.Q, T=self.T, k_mode=self.k_mode)

            if len(self.var_0) == 0:
                model.train(self.chunk_data, self.chunk_label)
                pred_result = model.predict(self.test_data[-self.nt:], self.P)
                self.var_0 = var(pred_result, 0)
                self.store_chunk_data = self.chunk_data
                self.store_chunk_label = self.chunk_label
                self.chunk_data = array([])
                self.chunk_label = array([])
            else:
                model.train(r_[self.store_chunk_data, self.chunk_data], r_[self.store_chunk_label, self.chunk_label])
                pred_result = model.predict(self.test_data[-self.nt:], self.P)
                self.var_1 = var(pred_result, 0)
                p = self.check_significance()
                if not self.mute:
                    print('v0: %f / v1: %f' % (mean(self.var_0), mean(self.var_1)))
                    print(p)

                if p < self.alpha:
                    if not self.mute:
                        print('Add more samples')
                    self.var_0 = self.var_1
                    self.store_chunk_data = r_[self.store_chunk_data, self.chunk_data]
                    self.store_chunk_label = r_[self.store_chunk_label, self.chunk_label]
                    self.chunk_data = array([])
                    self.chunk_label = array([])
                else:
                    if not self.mute:
                        print('Enough samples')
                        print('---------------------------------------------------------')
                    self.test_data = r_[self.test_data, self.store_chunk_data[self.store_chunk_label == self.min_class]]
                    model = SubUnderBagging(Q=self.Q, T=self.T, k_mode=self.k_mode)
                    model.train(self.chunk_data, self.chunk_label)
                    pred_result = model.predict(self.test_data[-self.nt:], self.P)
                    self.var_0 = var(pred_result, 0)
                    self.enough = 1

            self.round += 1

    def check_significance(self, n=100):
        f_p = []
        for i in range(self.nt):
            if self.var_1[i] != 0:
                f_p.append(1 - f.cdf(self.var_0[i] / self.var_1[i], n - 1, n - 1))
        f_p = [x for x in f_p if x != 0]
        K = -2 * sum(log(array(f_p)))
        chi2_p_value = 1 - chi2.cdf(K, 2 * len(f_p))

        return chi2_p_value

    def get_chunk(self):
        return self.store_chunk_data, self.store_chunk_label

    def get_chunk_2(self):
        # print('store', self.store_chunk_data)
        # print('chunk', self.chunk_data)
        # print('store', self.store_chunk_label)
        # print('chunk', self.chunk_label)
        # print(self.chunk_label.shape)
        if len(self.chunk_label) > 0:
            return r_[self.store_chunk_data, self.chunk_data], r_[self.store_chunk_label, self.chunk_label]
        else:
            return self.store_chunk_data, self.store_chunk_label

class FixMinorityChunkSizeSelect(ChunkSizeBase):

    def check_condition(self):
        if (self.round == 0 and min(self.chunk_count) > 0 and sum(self.chunk_count) >= self.init_num) or \
                (min(self.chunk_count) == self.fix_num):
            self.enough = 1
            self.round += 1


class FixChunkSizeSelect(ChunkSizeBase):

    def check_condition(self):

        if sum(self.chunk_count) >= self.fix_num:
            if min(self.chunk_count) == 0:
                self.chunk_count = zeros(2)
            else:
                self.enough = 1


class ADWIN(ChunkSizeBase):

    def __init__(self, delta=0.05, max_num=1000, init_num=100):

        self.enough = 0
        self.chunk_data = array([])
        self.chunk_label = array([])
        self.chunk_count = zeros(2)
        self.delta = delta
        self.max_num = max_num
        self.init_num = init_num
        self.round = 0

    def check_condition(self):

        if min(self.chunk_count) > 0:
            n = self.chunk_label.size
            if n >= self.max_num or (self.round == 0 and sum(self.chunk_count >= self.init_num)):
                self.enough = 1
                self.round += 1
            else:
                norm_data = (self.chunk_data - self.chunk_data.min(axis=0)) / (
                        self.chunk_data.max(axis=0) - self.chunk_data.min(axis=0))
                for i in range(n - 1):
                    mu_diff = abs(mean(norm_data[:i + 1], 0) - mean(norm_data[i + 1:], 0))
                    m = 1 / (1 / (i + 1) + 1 / (n - i - 1))
                    eps_cut = sqrt(1 / (2 * m) * log(4 / (self.delta / n)))
                    if max(mu_diff) > eps_cut:
                        self.enough = 1
                        self.round += 1
                        break


class PERM(ChunkSizeBase):

    def __init__(self, P=100, delta=0.05, m=100, max_num=1000, init_num=100):

        self.enough = 0
        self.chunk_data = array([])
        self.chunk_label = array([])
        self.chunk_count = zeros(2)
        self.P = P
        self.delta = delta
        self.m = m
        self.max_num = max_num
        self.init_num = init_num
        self.store_chunk_data = array([])
        self.store_chunk_label = array([])
        self.round = 0

    def update(self, data, label):

        if self.enough == 1:
            self.enough = 0
            self.chunk_count = zeros(2)
            self.store_chunk_data = self.chunk_data
            self.store_chunk_label = self.chunk_label
            self.chunk_data = array([])
            self.chunk_label = array([])

        if label == 1:
            self.chunk_count[1] += 1
        else:
            self.chunk_count[0] += 1
        self.chunk_data = r_[self.chunk_data.reshape(-1, data.size), data.reshape(1, -1)]
        self.chunk_label = r_[self.chunk_label, label]

        self.check_condition()

    def check_condition(self):

        if self.round == 0 and min(self.chunk_count) > 1 and sum(self.chunk_count) >= self.init_num:
            self.enough = 1
            self.round += 1
            self.store_chunk_data = self.chunk_data
            self.store_chunk_label = self.chunk_label
        elif sum(self.chunk_count) >= self.m and min(self.chunk_count) > 1:
            self.chunk_count = zeros(2)
            if self.round == 1:
                self.store_chunk_data = self.chunk_data
                self.store_chunk_label = self.chunk_label
                self.chunk_data = array([])
                self.chunk_label = array([])
            else:
                if self.detect() or sum(self.store_chunk_label.size) >= self.max_num:
                    print('enough')
                    self.enough = 1
                else:
                    print('not enough')
                    self.store_chunk_data = r_[self.store_chunk_data, self.chunk_data]
                    self.store_chunk_label = r_[self.store_chunk_label, self.chunk_label]
                    self.chunk_data = array([])
                    self.chunk_label = array([])

            self.round += 1

    def detect(self):

        m1 = self.store_chunk_label.size
        m2 = self.chunk_label.size
        model_ord = self.train(self.store_chunk_data, self.store_chunk_label)
        loss_ord = self.predict(model_ord, self.chunk_data, self.chunk_label)

        all_data = r_[self.store_chunk_data, self.chunk_data]
        all_label = r_[self.store_chunk_label, self.chunk_label]
        loss_perm = zeros(self.P)

        if sum(all_label == 1) < sum(all_label == -1):
            min_class = 1
        else:
            min_class = -1

        for i in range(self.P):
            X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size=m2 / (m2 + m1),
                                                                stratify=all_label)

            if sum(y_train == min_class) == 0:
                temp_train = X_train[0]
                X_train[0] = X_test[nonzero(y_test == min_class)[0][0]]
                X_test[nonzero(y_test == min_class)[0][0]] = temp_train
                y_train[0] = min_class
                y_test[nonzero(y_test == min_class)[0][0]] = -min_class

            model_perm = self.train(X_train, y_train)
            loss_perm[i] = self.predict(model_perm, X_test, y_test)

        test_value = (1 + sum(loss_ord < loss_perm)) / (self.P + 1)
        if test_value < self.delta:
            return True
        else:
            return False

    @staticmethod
    def train(data, label):

        model = SubUnderBagging(Q=100, T=100)
        model.train(data, label)

        return model

    @staticmethod
    def predict(model, data, label):

        pred_result = model.predict(data, P=1)
        pred_result = sign(pred_result - 0.5)
        loss = 1 - gm_measure(pred_result, label)

        return loss

    def get_chunk(self):
        return self.store_chunk_data, self.store_chunk_label

    def get_chunk_2(self):
        return r_[self.store_chunk_data, self.chunk_data], r_[self.store_chunk_label, self.chunk_label]
