from sklearn import preprocessing
from numpy import random
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
import numpy as np


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


class create_Graph:
    def __init__(self, data):
        self.X = data
        self.k = 10
        self.normalized = False
        self.degree = 1


    def adjacency(self):
        n_sample, n_D = self.X.shape
        dist = squareform(pdist(self.X, metric='euclidean'))
        topk = np.argsort(dist, axis=-1)[:, 1:self.k + 1]
        A = np.zeros((n_sample, n_sample))
        np.put_along_axis(A, indices=topk, values=1, axis=-1)
        return A + (A != A.T) * A.T

    def laplacian(self):
        A = self.adjacency()
        sparse_A = sparse.csr_matrix(A)
        laplacian_A = sparse.csgraph.laplacian(sparse_A, normed=self.normalized)
        if self.degree > 1:
            laplacian_A = laplacian_A ** self.degree
        return laplacian_A.A


class ISSBLS:
    def __init__(self, Nf, Ne, N1, N2, map_function, enhence_function, reg):
        self._Nf = Nf
        self._Ne = Ne
        self._N1 = N1
        self._N2 = N2
        self._map_function = map_function
        self._enhence_function = enhence_function
        self._reg = reg

        self.normalscaler = scaler()
        self.normalscaler_K = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator(whiten=True)
        self.local_mapgeneratorlist = []
        self.local_enhgeneratorlist = []

        self.W = []
        self.K = []
        self.P = []

    def fit(self, X, Y):
        C_off = np.eye(len(X))
        X_enc = self.normalscaler.fit_transform(X)
        Y_enc = self.onehotencoder.fit_transform(np.mat(Y).T)
        Z = self.mapping_generator.generator_nodes(X_enc, self._Nf, self._N1, self._map_function)
        H = self.enhence_generator.generator_nodes(Z, self._Ne, self._N2, self._enhence_function)
        self.A = np.column_stack((Z, H))

        # 计算离线集的拉普拉斯矩阵L_off
        G_off = create_Graph(X_enc)
        L_off = G_off.laplacian()

        r, w = self.A.T.dot(self.A).shape # r = w = nEnhance + n_mapping * n_feature

        self.K = self.A.T.dot(C_off).dot(self.A) + np.eye(r) + self._reg * self.A.T.dot(L_off).dot(self.A)
        self.P = np.linalg.inv(self.K)
        self.W = self.P.dot(self.A.T).dot(C_off).dot(Y_enc)

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def predict(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        return self.decode(test_inputdata.dot(self.W))

    def predict_proba(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        org_prediction = test_inputdata.dot(self.W)
        return org_prediction

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
        Y_at_enc = self.onehotencoder.transform(np.mat(Y_at).T)

        if label_flag == 1:
            C_at = np.eye(len(X_at))  # 默认为有标注
        else:
            C_at = np.zeros((len(X_at), len(X_at)))

        # 计算在线新到来数据的拉普拉斯矩阵L_on
        G_on = create_Graph(X_at_enc)
        L_on = G_on.laplacian()

        self.P = self.P - self.P.dot(A_at.T).dot(np.linalg.inv(
            np.eye(len(X_at), len(X_at)) + (C_at + self._reg * L_on).dot(A_at).dot(self.P).dot(A_at.T)).dot(
            (C_at + self._reg * L_on).dot(A_at).dot(self.P)))
        self.W = self.W + self.P.dot(A_at.T).dot(C_at.dot(Y_at_enc) - (C_at + self._reg * L_on).dot(A_at).dot(self.W))
