""" BLS-SMW classifier."""

import numpy as np
from sklearn import preprocessing
from numpy import random

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

class BLS:
    def __init__(self,
                 Nf=10,
                 Ne=10,
                 N1=10,
                 N2=10,
                 map_function='sigmoid',
                 enhence_function='sigmoid',
                 reg=0.001,
                 n_class=0):

        self._Nf = Nf
        self._Ne = Ne
        self._map_function = map_function

        self._enhence_function = enhence_function

        self._reg = reg

        self._N1 = N1

        self._N2 = N2

        self._n_class = n_class

        self.W = 0
        self.pesuedoinverse = 0
        self.K = 0
        self.P = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = node_generator()
        self.enhence_generator = node_generator(whiten=True)
        self.local_mapgeneratorlist = []
        self.local_enhgeneratorlist = []

    def fit(self, oridata, orilabel):
        data = self.normalscaler.fit_transform(oridata)

        if self._n_class == 0:
            label = self.onehotencoder.fit_transform(np.mat(orilabel).T)
        else:
            label = np.eye(self._n_class)[orilabel]

        mappingdata = self.mapping_generator.generator_nodes(data, self._Nf, self._N1, self._map_function)

        enhencedata = self.enhence_generator.generator_nodes(mappingdata, self._Ne, self._N2, self._enhence_function)
        inputdata = np.column_stack((mappingdata, enhencedata))

        r, w = inputdata.T.dot(inputdata).shape
        self.pesuedoinverse = np.linalg.inv(inputdata.T.dot(inputdata) + self._reg * np.eye(r))
        self.W = (self.pesuedoinverse.dot(inputdata.T)).dot(label)
        self.K = inputdata.T.dot(inputdata) + self._reg * np.eye(r)
        self.P = np.linalg.inv(self.K)

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

    def partial_fit(self, extratraindata, extratrainlabel):

        xdata = self.normalscaler.transform(extratraindata)
        xdata = self.transform(xdata)

        if self._n_class == 0:
            xlabel = self.onehotencoder.transform(np.mat(extratrainlabel).T)
        else:
            xlabel = np.eye(self._n_class)[extratrainlabel]

        temp = (xdata.dot(self.P)).dot(xdata.T)
        r, w = temp.shape

        self.P = self.P - (((self.P.dot(xdata.T)).dot(np.linalg.inv(np.eye(r) + temp))).dot(xdata)).dot(self.P)
        self.W = self.W + (self.P.dot(xdata.T)).dot(xlabel - xdata.dot(self.W))














