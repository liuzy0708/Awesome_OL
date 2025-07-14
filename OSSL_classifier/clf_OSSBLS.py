""" OSSBLS classifier."""

import numpy as np
from numpy import random
from scipy.spatial.distance import cdist

from OSSL_classifier.MyOneHotEncoder import MyOneHotEncoder


class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / (self._std + 1e-6)

    def transform(self, testdata):
        return (testdata - self._mean) / (self._std + 1e-6)


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

                w_sum += (w.T.dot(wj))[0, 0] * wj
            w -= w_sum
            w = w / np.sqrt(w.T.dot(w))
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        for i in range(times):
            W = 2 * random.random(size=shape) - 1
            if self.whiten == True:
                W = self.orth(W)
            b = 2 * random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, nonlinear):
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


class OSSBLS:
    def __init__(self, Nf, Ne, N1, N2, map_function, enhence_function, reg, gamma, n_anchor):
        self.anchor = n_anchor
        self.K = 0  # anchor set
        self._Nf = Nf
        self._Ne = Ne
        self._map_function = map_function
        self._enhence_function = enhence_function
        self._reg = reg
        self._gamma = gamma
        self._N1 = N1
        self._N2 = N2
        self.N_pass = 0
        self.W = 0
        self.pesuedoinverse = 0
        self.normalscaler = scaler()
        self.normalscaler_K = scaler()
        self.onehotencoder = MyOneHotEncoder()
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator(whiten=True)
        self.local_mapgeneratorlist = []
        self.local_enhgeneratorlist = []

        self.kappa = 1.5
        self.K = 0

    def norm(self, x):
        col_sums = x.sum(axis=0)
        normalized_x = x / col_sums
        return normalized_x

    def cal_S(self, X, K):
        sigma = 5
        k = 5

        n, _ = X.shape
        d, _ = K.shape

        distance_matrix = cdist(X, K, metric='euclidean')

        s_matrix = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                if j in np.argsort(distance_matrix[i])[:k]:
                    s_matrix[i, j] = np.exp(-np.linalg.norm(X[i] - K[j]) ** 2 / (2 * sigma ** 2))

        row_sums = s_matrix.sum(axis=1) + 1e-10
        s_matrix_normalized = s_matrix / row_sums[:, np.newaxis]

        return s_matrix_normalized

    def fit(self, X, Y):
        X_enc = self.normalscaler.fit_transform(X)
        Y_enc = self.onehotencoder.fit_transform(Y)
        Z = self.mapping_generator.generator_nodes(X_enc, self._Nf, self._N1, self._map_function)
        H = self.enhence_generator.generator_nodes(Z, self._Ne, self._N2, self._enhence_function)
        self.A = np.column_stack((Z, H))

        # Calculate anchor
        rand_arr = np.arange(X.shape[0])[:self.anchor]
        self.X_K = X[rand_arr]
        self.K = self.A[rand_arr]

        r, w = self.A.T.dot(self.A).shape
        self.C = np.eye(X.shape[0])
        self.S = self.cal_S(X, self.X_K)

        self.G = self.A.T.dot(self.C).dot(self.A) + self._reg * np.eye(r) + self._gamma * (
                self.A - self.S.dot(self.K)).T.dot(self.A - self.S.dot(self.K))
        self.Omega = self.A.T.dot(self.C).dot(Y_enc)
        self.G_inv = np.linalg.inv(self.G)
        self.W = self.G_inv.dot(self.Omega)


    def softmax_norm(self, array):
        exp_array = np.exp(array)
        sum_exp_array = np.sum(exp_array, axis=1, keepdims=True)
        softmax_array = exp_array / sum_exp_array
        return softmax_array

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
        Y_at = Y_at.ravel()
        Y_at_enc = self.onehotencoder.transform(Y_at)
        S_at = self.cal_S(X_at, self.X_K)

        if label_flag == 1:
            C_at = np.eye(len(X_at))
        else:
            C_at = np.zeros((len(X_at), len(X_at)))

        U_ts = A_at.T * C_at
        V_ts = A_at
        U_tu = self._gamma * (A_at - S_at.dot(self.K)).T
        V_tu = A_at - S_at.dot(self.K)
        Psi_inv = self.G_inv - (self.G_inv.dot(U_ts)).dot(
            np.linalg.inv(np.eye(V_ts.shape[0]) + V_ts.dot(self.G_inv).dot(U_ts))).dot(V_ts).dot(self.G_inv)
        self.G_inv = Psi_inv - (Psi_inv.dot(U_tu)).dot(
            np.linalg.inv(np.eye(V_tu.shape[0]) + V_tu.dot(Psi_inv).dot(U_tu))).dot(V_tu).dot(Psi_inv)
        self.Omega = self.Omega + A_at.T.dot(C_at).dot(Y_at_enc)

        self.W = self.G_inv.dot(self.Omega + A_at.T.dot(Y_at_enc))
