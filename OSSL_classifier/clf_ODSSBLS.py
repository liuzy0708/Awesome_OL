import numpy as np
from sklearn import preprocessing
from numpy import random
import scipy.stats
from scipy.spatial.distance import cdist
import time

# np.random.seed(0)
# random.seed(0)

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / self._std
    def transform(self, testdata):
        return (testdata - self._mean) / self._std


class node_generator:
    def __init__(self, whiten=False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten

    def sigmoid(self, data):
        return 1.0 / (1 + np.exp(-data))

    def linear(self, data):
        return data

    def tanh(self, data):
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    def relu(self, data):
        return np.maximum(data, 0)

    def orth(self, W):

        for i in range(0, W.shape[1]):
            w = np.mat(W[:, i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:, j].copy()).T

                w_sum += (w.T.dot(wj))[0, 0] * wj  # [0,0]就是求矩阵相乘的一元数
            w -= w_sum
            w = w / np.sqrt(w.T.dot(w))
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        # np.random.seed(0)
        # random.seed(0)
        for i in range(times):
            W = 2 * random.random(size=shape) - 1
            if self.whiten == True:
                W = self.orth(W)
            b = 2 * random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, nonlinear):  # 将特征结点和增强结点构建起来
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.nonlinear = {'linear': self.linear,
                          'sigmoid': self.sigmoid,
                          'tanh': self.tanh,
                          'relu': self.relu
                          }[nonlinear]
        nodes = self.nonlinear(data.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i]) + self.blist[i])))
        return nodes

    def transform(self, testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

    def update(self, otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb


class ODSSBLS:
    def __init__(self,
                 Nf=10,
                 Ne=10,
                 N1=10,
                 N2=10,
                 map_function='linear',
                 enhence_function='linear',
                 reg=0.001,
                 gamma=0.01,
                 n_anchor=30,
                 n_class=3):

        self._Nf = Nf
        self._Ne = Ne
        self._map_function = map_function
        self._enhence_function = enhence_function
        self._reg = reg
        self._gamma = gamma
        self._N1 = N1
        self._N2 = N2
        self.n_anchor = n_anchor
        self.n_class = n_class

        self.W = 0
        self.pesuedoinverse = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator(whiten=True)
        self.local_mapgeneratorlist = []
        self.local_enhgeneratorlist = []

        self.kappa = 2
        self.count = 0
        self.K = 0

        self.X_K = 0
        self.y_K= 0

    def softmax_norm(self, array):
        exp_array = np.exp(array)
        sum_exp_array = np.sum(exp_array, axis=1, keepdims=True)
        softmax_array = exp_array / sum_exp_array
        return softmax_array

    def cal_S(self, X, K):
        sigma = 5
        k = 5

        n, _ = X.shape
        d, _ = K.shape

        # 计算欧几里德距离矩阵
        distance_matrix = cdist(X, K, metric='euclidean')

        # 根据条件计算 s_ij
        s_matrix = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                if j in np.argsort(distance_matrix[i])[:k]:
                    s_matrix[i, j] = np.exp(-np.linalg.norm(X[i] - K[j]) ** 2 / (2 * sigma ** 2))

        row_sums = s_matrix.sum(axis=1) + 1e-10  # 在外层循环之外计算row_sums
        s_matrix_normalized = s_matrix / row_sums[:, np.newaxis]

        return s_matrix_normalized

    def get_K(self, X_pt, y_pt):
        X_K = []
        y_K = []
        for i in range(self.n_class):
            indices_with_label = np.where(y_pt == i)[0][:int(self.n_anchor / 3)]
            selected_data_label = X_pt[indices_with_label]
            selected_labels = y_pt[indices_with_label]
            X_K.append(selected_data_label)
            y_K.append(selected_labels)
        X_K = np.concatenate(X_K, axis=0)
        y_K = np.concatenate(y_K, axis=0)
        return X_K, y_K

    def fit(self, X, Y):
        X_enc = self.normalscaler.fit_transform(X)
        Y_enc = self.onehotencoder.fit_transform(np.mat(Y).T)
        Z = self.mapping_generator.generator_nodes(X_enc, self._Nf, self._N1, self._map_function)
        H = self.enhence_generator.generator_nodes(Z, self._Ne, self._N2, self._enhence_function)
        self.A = np.column_stack((Z, H))

        self.X_K, self.y_K = self.get_K(X_pt=X, y_pt=Y)

        r, w = self.A.T.dot(self.A).shape
        self.C = np.eye(X.shape[0])
        self.S = self.cal_S(X, self.X_K)

        # Calculate anchor
        K_enc = self.normalscaler.transform(self.X_K)
        self.K = self.transform(K_enc)

        self.G = self.A.T.dot(self.C).dot(self.A) + self._reg * np.eye(r) #+ self._gamma * (self.A - self.S.dot(self.K)).T.dot(self.A - self.S.dot(self.K))
        liuzeyi = self._gamma * (self.A - self.S.dot(self.K)).T.dot(self.A - self.S.dot(self.K))
        self.Omega = self.A.T.dot(self.C).dot(Y_enc)
        self.G_inv = np.linalg.inv(self.G)
        self.W = self.G_inv.dot(self.Omega)


    def pinv(self, A):
        return np.mat(self._reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def predict(self, testdata):
        logit = self.predict_proba(testdata)
        return self.decode(self.softmax_norm(logit))

    def predict_proba(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        org_prediction = test_inputdata.dot(self.W)
        return self.softmax_norm(org_prediction)

    def update_K(self, X_K, y_K, X, y_pred):
        indices_to_replace = np.where(y_K == y_pred)[0]
        index_to_remove = indices_to_replace[-1]
        X_K_new = np.delete(X_K, index_to_remove, axis=0)
        X_K_new = np.insert(X_K_new, index_to_remove, X, axis=0)
        return X_K_new

    def norm(self, x):  # 标准化一个矩阵
        col_sums = x.sum(axis=0)
        normalized_x = x / col_sums
        return normalized_x

    def transform(self, data):
        mappingdata = self.mapping_generator.transform(data)
        enhencedata = self.enhence_generator.transform(mappingdata)
        inputdata = np.column_stack((mappingdata, enhencedata))
        for elem1, elem2 in zip(self.local_mapgeneratorlist, self.local_enhgeneratorlist):
            inputdata = np.column_stack((inputdata, elem1.transform(data)))
            inputdata = np.column_stack((inputdata, elem2.transform(mappingdata)))
        return inputdata

    def partial_fit(self, X_at, Y_at, label_flag=1):
        X_at_enc = self.normalscaler.transform(X_at)
        A_at = self.transform(X_at_enc)

        if label_flag == 1:
            theta = 1  # 默认为有标注
        else:
            theta = 0

        # Calculate anchor
        X_K_new = self.update_K(self.X_K, self.y_K, X_at, self.predict(X_at))
        X_K_new_enc = self.normalscaler.transform(X_K_new)
        K_at = self.transform(X_K_new_enc) - self.K

        Y_at_enc = self.onehotencoder.transform(np.mat(Y_at).T)
        S_at = self.cal_S(X_at, X_K_new)

        P_at = A_at - S_at.dot(self.K)
        U = self._gamma * (P_at - S_at.dot(K_at)).T
        V = P_at - S_at.dot(K_at)

        Psi_inv = self.G_inv - theta * (self.G_inv.dot(A_at.T).dot(A_at).dot(self.G_inv)) / (1 + A_at.dot(self.G_inv).dot(A_at.T))
        self.G_inv = Psi_inv - (Psi_inv.dot(U).dot(V).dot(Psi_inv)) / (1 + V.dot(Psi_inv).dot(U))

        self.W = self.G_inv.dot(self.Omega + A_at.T.dot(Y_at_enc))

        self.K = K_at + self.K
        self.Omega = self.Omega + theta * A_at.T.dot(Y_at_enc)
