import time

import numpy as np
from sklearn import preprocessing
from numpy import random
import random
import copy
import math
import shap
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN
import scipy.stats
from scipy.special import expit


# np.random.seed(0)
# random.seed(0)

class scaler:
    """
    Class to scale data using mean and standard deviation.

    Attributes:
        _mean (float): Mean value calculated during fitting.
        _std (float): Standard deviation calculated during fitting.
    """

    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        """
        Fit and transform the input data.

        Parameters:
            traindata (numpy.ndarray): Training data.

        Returns:
            numpy.ndarray: Transformed data.
        """
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / (self._std + 1e-6)

    def transform(self, testdata):
        """
        Transform the input test data.

        Parameters:
            testdata (numpy.ndarray): Test data.

        Returns:
            numpy.ndarray: Transformed test data.
        """
        return (testdata - self._mean) / (self._std + 1e-16)


class node_generator:
    """
    Class to generate nodes for the mapping and enhancement functions.

    Attributes:
        Wlist (list): List to store weight matrices.
        blist (list): List to store bias vectors.
        nonlinear (int): Type of nonlinear activation function.
        whiten (bool): Flag to indicate whether whitening should be applied.
    """

    def __init__(self, whiten=False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten

    def sigmoid(self, data):
        """
        Sigmoid activation function.

        Parameters:
            data (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data after applying the sigmoid function.
        """
        return 1.0 / (1 + np.exp(-data))

    def linear(self, data):
        """
        Linear activation function.

        Parameters:
            data (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data.
        """
        return data

    def tanh(self, data):
        """
        Hyperbolic tangent activation function.

        Parameters:
            data (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data after applying the tanh function.
        """
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    def relu(self, data):
        """
        Rectified Linear Unit (ReLU) activation function.

        Parameters:
            data (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data after applying the ReLU function.
        """
        return np.maximum(data, 0)

    def orth(self, W):
        """
        Orthogonalization function for weight matrix.

        Parameters:
            W (numpy.ndarray): Weight matrix.

        Returns:
            numpy.ndarray: Orthogonalized weight matrix.
        """
        for i in range(0, W.shape[1]):
            w = np.mat(W[:, i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:, j].copy()).T

                w_sum += (w.T.dot(wj))[0, 0] * wj  # [0,0] is the product of the matrices
            w -= w_sum
            w = w / (np.sqrt(w.T.dot(w)) + 1e-10)
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        for i in range(times):
            W = 2 * np.random.random(size=shape) - 1
            if self.whiten == True:
                W = self.orth(W)
            b = 2 * np.random.random() - 1
            yield (W, b)

    def generator_nodes_feature(self, data, times, batchsize, nonlinear, clf_XAI, y):
        self.Wlist = []
        final_indices_list = []
        for idx in range(times):

            try:
                rnd_idx = random.sample(range(data.shape[0]), 5)

                X_train_subset = data[rnd_idx]
                y_ratio = np.sum(y, axis=0) / np.sum(np.sum(y, axis=0))
                explainer = shap.KernelExplainer(clf_XAI.predict_proba, X_train_subset)
                explained_value = explainer.shap_values(X_train_subset, nsamples=20)
                explained_value_weighted = np.sum(
                    np.sum([array * weight for array, weight in zip(explained_value, y_ratio)], axis=0), axis=0)

                non_zero_indices = np.where(abs(explained_value_weighted) >= 1e-5)[0]

                if non_zero_indices.shape[0] == 0:
                    indices = range(data.shape[1])
                    normalized_weight = np.random.uniform(-1, 1, size=(data.shape[1],))
                else:
                    indices = non_zero_indices
                    min_value = np.min(explained_value_weighted[non_zero_indices])
                    max_value = np.max(explained_value_weighted[non_zero_indices])
                    scaled_explained_value = -1 + 2 * (explained_value_weighted[non_zero_indices] - min_value) / (
                                max_value - min_value)
                    normalized_weight = 2 * (expit(scaled_explained_value) - 0.5)

                final_indices_list.append(indices)
                self.Wlist.append(normalized_weight)

            except Exception as e:
                indices = range(data.shape[1])
                normalized_weight = np.random.uniform(-1, 1, size=(data.shape[1],))
                final_indices_list.append(indices)
                self.Wlist.append(normalized_weight)
                continue

        self.blist = [elem[1] for elem in self.generator((len(final_indices_list), batchsize), times)]

        self.nonlinear = {'linear': self.linear,
                          'sigmoid': self.sigmoid,
                          'tanh': self.tanh,
                          'relu': self.relu
                          }[nonlinear]
        nodes = self.nonlinear(data[:, final_indices_list[0]].dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack(
                (nodes, self.nonlinear(data[:, final_indices_list[i]].dot(self.Wlist[i]) + self.blist[i])))
        return nodes, final_indices_list

    def generator_nodes_enhance(self, data, times, batchsize, nonlinear):
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
        # liuzeyi = testdata.dot(self.Wlist[0])
        # wuyurong = self.nonlinear(liuzeyi)
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

    def transform_feature(self, testdata, final_indices_list):

        testnodes = self.nonlinear(testdata[:, final_indices_list[0]].dot(self.Wlist[0]) + self.blist[0])
        # testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack(
                (testnodes, self.nonlinear(testdata[:, final_indices_list[i]].dot(self.Wlist[i]) + self.blist[i])))
            # testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

    def update(self, otherW, otherb):
        """
        Update the generator with additional weight matrices and bias vectors.

        Parameters:
            otherW (list): List of additional weight matrices.
            otherb (list): List of additional bias vectors.
        """
        self.Wlist += otherW
        self.blist += otherb


# class Data_Loader:
#     def __init__(self, oridata, orilabel, max_data_pairs):
#
#         # Store paired data and labels
#         self.data_pairs = list(zip(oridata, orilabel))
#         self.max_data_pairs = max_data_pairs
#
#     def add_data(self, data, label):
#
#         # Append new data pair to the queue
#         self.data_pairs.append((data, label))
#         # If the queue exceeds the maximum size, remove the oldest data pair
#         if len(self.data_pairs) > self.max_data_pairs:
#             self.data_pairs.pop(0)
#
#     def get_data_loader(self):
#         random.shuffle(self.data_pairs)
#         self.data_pairs = self.data_pairs[:self.max_data_pairs]
#
#     def get_all_data_labels(self):
#
#         all_data, all_labels = zip(*self.data_pairs)
#         return np.vstack(all_data), np.array(all_labels)

# class Data_Loader:
#     def __init__(self, n_class, max_data_class, max_data_class_low):
#         # Initialize data structure to hold data and labels for each class
#         self.data_per_class = [[] for _ in range(n_class)]
#         self.data_list = []
#         self.max_data_class = max_data_class
#         self.max_data_class_low = max_data_class_low
#
#     def add_data(self, data, labels):
#         # Iterate over each sample and its label
#         # for i, label in enumerate(labels):
#         #     # Convert label to integer if necessary
#         #     label = int(label)
#         #     # Add data and label to corresponding class list based on label
#         for i, label in enumerate(labels):
#             # Convert label to integer if necessary
#             label = int(label)
#             # Add data and label to corresponding class list based on label
#             self.data_list.append((data[i], label))
#             if len(self.data_list) > self.max_data_class:
#                 self.data_list.pop(0)
#             self.data_per_class[label].append((data[i], label))
#             if len(self.data_per_class[label]) > self.max_data_class_low:
#                 self.data_per_class[label].pop(0)
#
#     # def get_data_loader(self):
#     #     data_loader = []
#     #     for class_data in self.data_per_class:
#     #         data_loader.extend(class_data)
#     #
#     #     return data_loader
#
#     # def get_all_data_labels(self):
#     #     # Get all data and labels for each class
#     #     all_data = []
#     #     all_labels = []
#     #     for class_data in self.data_per_class:
#     #         data, labels = zip(*class_data)
#     #         all_data.extend(data)
#     #         all_labels.extend(labels)
#     #     all_data = np.vstack(all_data)
#     #     all_labels = np.array(all_labels)
#     #     return all_data, all_labels
#
#     def get_all_data_labels(self):
#         # Get all data and labels
#         all_data = [data for data, _ in self.data_list]
#         all_labels = [label for _, label in self.data_list]
#         all_data = np.vstack(all_data)
#         all_labels = np.array(all_labels)
#         return all_data, all_labels

class Data_Loader:
    def __init__(self, n_class, max_data_class, max_data_class_low):
        # Initialize data structure to hold data and labels for each class
        self.data_per_class = [[] for _ in range(n_class)]
        self.data_list = []
        self.max_data_class = max_data_class
        self.max_data_class_low = max_data_class_low

    def add_data(self, data, labels):
        # Iterate over each sample and its label
        # print('labels', labels)
        for i, label in enumerate(labels):
            # Convert label to integer if necessary
            label = int(label)
            # Add data and label to data_list
            self.data_list.append((data[i], label))
            if len(self.data_list) > self.max_data_class:
                self.data_list.pop(0)
            # Add data and label to corresponding class list based on label
            self.data_per_class[label].append((data[i], label))
            if len(self.data_per_class[label]) > self.max_data_class_low:
                self.data_per_class[label].pop(0)

    def get_all_data_labels(self):
        # Combine data_list and data_per_class data and labels into one list
        all_data = [data for data, _ in self.data_list]
        all_labels = [label for _, label in self.data_list]

        for class_data in self.data_per_class:
            data_per_class, labels_per_class = zip(*class_data)
            all_data.extend(data_per_class)
            all_labels.extend(labels_per_class)

        all_data = np.vstack(all_data)
        all_labels = np.array(all_labels)

        # print(all_data.shape)

        return all_data, all_labels


class BLS:
    def __init__(self,
                 Nf=10,
                 Ne=10,
                 N1=10,
                 N2=10,
                 map_function='sigmoid',
                 enhance_function='sigmoid',
                 reg=0.01,
                 n_class=0):

        self._Nf = Nf
        self._Ne = Ne
        self._map_function = map_function
        self._enhance_function = enhance_function
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
        self.enhance_generator = node_generator(whiten=True)
        self.local_mapgeneratorlist = []
        self.local_enhgeneratorlist = []

    def fit(self, oridata, orilabel):
        data = self.normalscaler.fit_transform(oridata)

        if self._n_class == 0:
            label = self.onehotencoder.fit_transform(np.mat(orilabel).T)
        else:
            label = np.eye(self._n_class)[orilabel]

        mappingdata = self.mapping_generator.generator_nodes(data, self._Nf, self._N1, self._map_function)
        enhancedata = self.enhance_generator.generator_nodes(mappingdata, self._Ne, self._N2, self._enhance_function)
        inputdata = np.column_stack((mappingdata, enhancedata))

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
        enhancedata = self.enhance_generator.transform(mappingdata)
        inputdata = np.column_stack((mappingdata, enhancedata))
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


class DGWBLS:

    def __init__(self,
                 Nf=10,
                 Ne=10,
                 N1=10,
                 N2=10,
                 map_function='sigmoid',
                 enhance_function='sigmoid',
                 reg=0.01,
                 theta=0.01,
                 gamma=0.001,
                 n_class=0):

        self._Nf = Nf
        self._Ne = Ne
        self._map_function = map_function
        self._enhance_function = enhance_function
        self._reg = reg
        self._N1 = N1
        self._N2 = N2
        self._n_class = n_class

        self.acc_pre = 0.80
        self.adwin_theta = ADWIN(delta=theta)
        self.adwin_gamma = ADWIN(delta=gamma)

        self.W = 0
        self.Lambda = 0
        self.pesuedoinverse = 0
        self.K = 0
        self.P = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.mapping_generator = node_generator()
        self.enhance_generator = node_generator()
        self.local_mapgeneratorlist = []
        self.local_enhgeneratorlist = []

    def fit(self, oridata, orilabel, Lambda):

        data = self.normalscaler.fit_transform(oridata)
        if self._n_class == 0:
            label = self.onehotencoder.fit_transform(np.mat(orilabel).T)
        else:
            orilabel = [int(i) for i in orilabel]
            label = np.eye(self._n_class)[orilabel]

        self.clf_XAI = BLS(Nf=10,
                           Ne=10,
                           N1=10,
                           N2=10,
                           map_function='sigmoid',
                           enhance_function='sigmoid',
                           reg=0.01,
                           n_class=self._n_class)
        self.clf_XAI.fit(data, orilabel)

        mappingdata, self.final_indices_list = self.mapping_generator.generator_nodes_feature(data, self._Nf, self._N1,
                                                                                              self._map_function,
                                                                                              self.clf_XAI, label)
        # mappingdata = self.mapping_generator.generator_nodes_enhance(data, self._Nf, self._N1, self._map_function)
        enhancedata = self.enhance_generator.generator_nodes_enhance(mappingdata, self._Ne, self._N2,
                                                                     self._enhance_function)
        inputdata = np.column_stack((mappingdata, enhancedata))

        r, w = inputdata.T.dot(inputdata).shape
        self.pesuedoinverse = np.linalg.inv(inputdata.T.dot(Lambda).dot(inputdata) + self._reg * np.eye(r))
        self.W = self.pesuedoinverse.dot(inputdata.T).dot(Lambda).dot(label)
        self.K = inputdata.T.dot(Lambda).dot(inputdata) + self._reg * np.eye(r)
        self.P = np.linalg.inv(self.K)

    def softmax_norm(self, array):
        exp_array = np.exp(array)
        sum_exp_array = np.sum(exp_array, axis=1, keepdims=True)
        softmax_array = exp_array / sum_exp_array
        return softmax_array

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def predict(self, testdata):
        logit = self.predict_proba(testdata)
        y_pred = self.decode(self.softmax_norm(logit))
        return y_pred

    def predict_proba(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        org_prediction = test_inputdata.dot(self.W)
        return self.softmax_norm(org_prediction)

    def transform(self, data):
        mappingdata = self.mapping_generator.transform_feature(data, self.final_indices_list)
        enhancedata = self.enhance_generator.transform(mappingdata)
        inputdata = np.column_stack((mappingdata, enhancedata))
        for elem1, elem2 in zip(self.local_mapgeneratorlist, self.local_enhgeneratorlist):
            inputdata = np.column_stack((inputdata, elem1.transform(data)))
            inputdata = np.column_stack((inputdata, elem2.transform(mappingdata)))
        return inputdata

    def partial_fit(self, extratraindata, extratrainlabel, extraLambda):
        xdata = self.normalscaler.transform(extratraindata)
        xdata = self.transform(xdata)
        if self._n_class == 0:
            xlabel = self.onehotencoder.transform(np.mat(extratrainlabel).T)
        else:
            extratrainlabel = int(extratrainlabel)
            xlabel = np.eye(self._n_class)[extratrainlabel]

        temp = (xdata.dot(self.P)).dot(xdata.T)

        self.P = self.P - (
            ((self.P.dot(xdata.T)).dot(np.linalg.inv(np.linalg.inv(extraLambda) + temp))).dot(xdata)).dot(self.P)
        self.W = self.W + (self.P.dot(xdata.T)).dot(extraLambda).dot(xlabel - xdata.dot(self.W))

        # self.clf_XAI.partial_fit(self.normalscaler.transform(extratraindata), extratrainlabel)


class DGEBLS_turbo:
    def __init__(self,
                 n_base_learner=20,
                 vartheta=0.80,
                 theta=1e-5,
                 gamma=1e-4,
                 tau=0.40,
                 max_data_pairs=10,
                 max_data_pairs_low=5,
                 n_class=0,
                 k=0.7):

        self._n_base_learner = n_base_learner
        self._vartheta = vartheta
        self._theta = theta
        self._gamma = gamma
        self._tau = tau
        self._max_data_pairs = max_data_pairs
        self._n_class = n_class
        self.count = 0
        self.bl_list = []
        self.alpha_list = []
        self.Acc_list = []
        self.dl_list = []
        self.n_size = 2
        self.Low = 0
        self.High = 0
        self.is_add_BL = False
        self.pre_test = False
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self._max_data_pairs_low = max_data_pairs_low
        self.k = k

    def _initialize_base_learners(self):
        self.bl_list = [DGWBLS(
            Nf=10,
            Ne=np.random.randint(5, 15),
            # Ne=10,
            N1=1,
            N2=np.random.randint(5, 15),
            # N2=10,
            map_function='sigmoid',
            enhance_function='sigmoid',
            reg=0.01,
            theta=self._theta,
            gamma=self._gamma,
            n_class=self._n_class) for _ in range(self._n_base_learner)]

    def update_data_loader(self, X, y, dl):
        for data, label in zip(X, y):
            dl.add_data(data, label)

    def predict_proba(self, X):

        if not self.bl_list:
            raise ValueError("The model is not fitted yet. Call fit() before predict_proba().")

        # Initialize an array to store the weighted predictions
        if self._n_class == 0:
            raise ValueError("The number of classes are not setted.")
        else:
            weighted_predictions = np.zeros((X.shape[0], self._n_class))

        proba_list = []
        # Make predictions with each base learner and weight them
        for bl, alpha in zip(self.bl_list, self.alpha_list):
            proba = bl.predict_proba(X)
            proba_list.append(proba)
            weighted_predictions += alpha * proba
            # print(bl.predict(X))
            # print(sum(1 for pred, true_label in zip(bl.predict(X), true_temp) if pred == true_label) / X.shape[0])
        return weighted_predictions

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def fit(self, oridata, orilabel):

        self._initialize_base_learners()

        idx = 0
        for bl in self.bl_list:
            bl.fit(oridata, orilabel, np.eye(oridata.shape[0]))
            # dl = Data_Loader(oridata[-self._max_data_pairs:, :], orilabel[-self._max_data_pairs:], self._max_data_pairs)
            dl = Data_Loader(self._n_class, self._max_data_pairs, self._max_data_pairs_low)
            dl.add_data(oridata, orilabel)
            # dl.get_data_loader()
            self.dl_list.append(dl)
            idx += 1

        self.alpha_list = [1 / self._n_base_learner] * self._n_base_learner

    def partial_fit(self, Xt, yt):

        self.Acc_list = []

        for idx in range(len(self.bl_list)):

            bl = copy.deepcopy(self.bl_list[idx])
            dl = copy.deepcopy(self.dl_list[idx])
            for sp, lb in zip(Xt, yt):

                sp_inv = sp.reshape(1, -1)

                y_pred_sp = bl.predict(sp_inv)
                if y_pred_sp[0] != lb:
                    bl.adwin_gamma.add_element(0)
                    bl.partial_fit(sp_inv, lb, np.eye(1))
                else:
                    bl.adwin_gamma.add_element(1)
                    dl.add_data(sp, [lb])

                if bl.adwin_gamma.detected_change():
                    print("drift!")
                    new_data, new_label = dl.get_all_data_labels()
                    print(new_label)
                    new_bl = DGWBLS(
                                Nf=10,
                                Ne=np.random.randint(5, 15),
                                N1=1,
                                N2=np.random.randint(5, 15),
                                map_function='sigmoid',
                                enhance_function='sigmoid',
                                reg=0.01,
                                theta=self._theta,
                                gamma=self._gamma,
                                n_class=self._n_class)

                    new_bl.fit(new_data, new_label, np.eye(new_data.shape[0]))
                    bl = copy.deepcopy(new_bl)

            Acc_update_fit = sum(1 for pred, true_label in zip(bl.predict(Xt), yt) if pred == true_label) / yt.shape[0]

            # print("Acc_update_fit", Acc_update_fit)

            self.bl_list[idx] = copy.deepcopy(bl)
            self.dl_list[idx] = copy.deepcopy(dl)
            self.Acc_list.append((Acc_update_fit, self.bl_list[idx], self.dl_list[idx]))

        t2 = time.time()

        self.update_alpha_list()

    def update_alpha_list(self):
        """
        Updates the alpha_list based on the updated Acc_list with WOWA operator.
        """
        if not self.Acc_list:
            raise ValueError("Acc_list is empty. Run partial_fit before updating alpha_list.")

        self.Acc_list = [(a, b, c, i) for (a, b, c), i in zip(self.Acc_list, range(self._n_base_learner))]

        temp_alpha_list = []
        new_alpha_list = [0] * self._n_base_learner

        sorted_acc_list = sorted(self.Acc_list, key=lambda x: x[0], reverse=True)
        sorted_order = [item[3] for item in sorted_acc_list]

        min_acc = min(sorted_acc_list, key=lambda x: x[0])[0] - 1e-3
        # print(sorted_acc_list)
        acc_difference_list = [item[0] - min_acc for item in sorted_acc_list]
        # print(acc_difference_list)
        sum_acc_difference = sum(acc_difference_list)
        normalized_acc_difference_list = [diff / sum_acc_difference for diff in acc_difference_list]
        # print(normalized_acc_difference_list)

        for t in range(1, len(self.Acc_list) + 1):  # Range should start from 1
            numerator_p = sum(normalized_acc_difference_list[:t])
            numerator_p_minus_1 = sum(
                normalized_acc_difference_list[:t - 1])  # Subtract one from t for previous elements
            alpha_p = self.RIM_func(numerator_p) - self.RIM_func(numerator_p_minus_1)

            temp_alpha_list.append(alpha_p)

        # for t in range(len(sorted_order)):
        #     numerator_p = (t + 1) / (self._n_base_learner)
        #     numerator_p_minus_1 = (t) / (self._n_base_learner)
        #     alpha_p = self.RIM_func(numerator_p) - self.RIM_func(numerator_p_minus_1)
        #
        #     temp_alpha_list.append(alpha_p)

        for idx in range(len(self.Acc_list)):
            new_alpha_list[sorted_order[idx]] = temp_alpha_list[idx]

        self.Acc_list = [[acc, bl, dl] for acc, bl, dl, _ in self.Acc_list]

        # Update alpha_list
        self.alpha_list = new_alpha_list

    def RIM_func(self, x):
        """
        Regular Increasing Monotone (RIM) Q function implementation with x as the parameter.
        Assumes a simple identity function here.

        Parameters:
        - x: float
          Input value for the RIM Q function.

        Returns:
        - float
          Result of the RIM Q function.
        """
        return math.pow(x, self.k)