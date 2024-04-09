#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Project : OLIDS
# @File : OLIDS.py
# @Time : 2020/9/12 11:00
# @Author : www.mlzhilu.com
# @Version : 1.0
# @Descriptions : here put the descriptions about this file

# here put the import lib

from .toolbox import *
from . import preprocess
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import copy
np.seterr(all="ignore")


class OLI2DS:
    def __init__(self, C=0.0100000, Lambda=30, B=1, theta=8, gama=0, sparse=0, mode="capricious"):
        self.C = C
        self.Lambda = Lambda
        self.B = B
        self.mode = mode
        self.theta = theta
        self.gama = gama
        self.sparse = sparse
        self.count_sample = 0
        self.column_mean = 0
        self.column_std = 0

    def updateInstanceMetadata(self, x, y_value):
        x_t = x
        getXt = x_t.get
        getVar = self.instance_variance_vec.get
        getAvg = self.instance_average_vec.get
        getCount = self.instance_count_vec.get
        for key in x_t.keys():
            if key in self.instance_variance_vec.keys():
                if getCount(key) == 1:
                    tmp = np.var(np.array([getAvg(key), getXt(key)]))
                    self.instance_variance_vec[key] = tmp if tmp else 0
                    self.instance_average_vec[key] = np.mean(np.array([getAvg(key), getXt(key)]))
                    self.instance_count_vec[key] = getCount(key) + 1
                else:
                    self.instance_count_vec[key] = getCount(key) + 1
                    self.instance_variance_vec[key] = (getVar(key) * (getAvg(key) - 2)) / (getAvg(key) - 2) + np.power(
                        (getXt(key) - getAvg(key)), 2) / getCount(key)
                    self.instance_average_vec[key] = getAvg(key) + (getXt(key) - getAvg(key)) / getCount(key)
            else:
                self.instance_count_vec[key] = 1
                self.instance_average_vec[key] = getXt(key) / getCount(key)
                self.instance_variance_vec[key] = 0
        label = y_value
        # print('label_dic', label)
        self.label_dic[int(label)] += 1

    def set_metadata(self):
        x_0 = self.X[0]
        self.instance_count_vec = {i: 1 for i in x_0.keys()}
        self.instance_average_vec = x_0.copy()
        self.instance_variance_vec = {i: 0 for i in x_0.keys()}
        self.label_dic = {1: 1, -1: 1}

    def set_classifier(self):
        self.weights = {key: 0 for key in self.X[0].keys()}
        self.current_weights = {key: 0 for key in self.X[0].keys()}

    def reWeights(self, w_share, w_new):
        getVar = self.instance_variance_vec.get
        # print('w_new:', w_new)
        h_s = [getVar(key) for key in w_share.keys()]
        h_n = [getVar(key) for key in w_new.keys()]
        sumS = None if all(element is None for element in h_s) else sum(filter(lambda x: x is not None, h_s))
        if sumS==None:
            sumS=0
        sumN = None if all(element is None for element in h_n) else sum(filter(lambda x: x is not None, h_n))
        if sumN==None:
            sumN=0
        total_h = sumS + sumN
        # print('sumS:', sumS)
        # print('sumN:', sumN)
        # print('total_h:', total_h)
        s = sumS / total_h if bool(total_h) else 1
        return s, 1 - s

    def parameter_set(self, xs, xn, ps, pn, loss):
        inner_product = dotDict(xs, xs) * ps * ps + dotDict(xn, xn) * pn * pn
        return np.minimum(self.C, loss / np.where(inner_product == 0, 1, inner_product))

    def get_informativeness_vector(self):
        var = [self.instance_variance_vec[k] for k in sorted(self.weights.keys())]
        total = np.sum(var)
        return np.array([v / total if total != 0 else 1 for v in var])

    def sparsity_step(self):
        numpyWeights = dict2NumpyArray(self.weights)
        normWeights = np.linalg.norm(dict2NumpyArray(self.weights), ord=1)
        if normWeights != 0:
            info_vector = self.get_informativeness_vector()
            projected = np.multiply(np.minimum(1, self.Lambda / np.linalg.norm(
                numpyWeights * info_vector, ord=1)), numpyWeights)
            self.weights = self.truncate(projected)
            self.current_weights = {key: self.weights[key] for key in
                                    findCommonKeys(self.current_weights, self.weights)}

    def truncate(self, projected):
        sortedWeightsKeys = sorted(self.weights.keys())
        projectedLength = len(projected)
        if np.linalg.norm(projected, ord=0) > self.B * projectedLength:
            remaining = int(np.maximum(1, np.floor(self.B * projectedLength)))
            projected = NumpyArray2Dict(projected, sortedWeightsKeys)
            sort_projected = sorted(projected.items(), key=lambda x: x[1])
            for item in sort_projected[:(projectedLength - remaining)]:
                projected[item[0]] = 0
            return projected
        else:
            projected = NumpyArray2Dict(projected, sortedWeightsKeys)
            return projected

    def updateWeights(self, weights, x, tao):
        return {key: weights[key] + tao * x[key] for key in list(weights.keys())}

    def classifier(self, x):
        weights_new = dict((k, self.weights[k]) if k in self.weights.keys() else (k, 0) for k in
                           findDifferentKeys(x, self.current_weights))
        # print('self.current_weights:', self.current_weights)

        share_key = findCommonKeys(x, self.current_weights)
        # print('share_key:', share_key)
        # print('self.current_weights:', self.current_weights)
        weights_share = subsetDictionary(self.current_weights, share_key)
        # print('weights_share:', weights_share)
        # print('weights_new:', weights_share)
        ps, pn = self.reWeights(weights_share, weights_new)
        pres = np.sum([x[k] * weights_share[k] for k in weights_share.keys()])

        pren = np.sum([x[k] * weights_new[k] for k in weights_new.keys()])
        y_pre = pres * ps + pren * pn
        return weights_share, weights_new, y_pre, ps, pn

    def Gmean(self, tn, tp, fn, fp):
        if (tp + fn) * (tn + fp) == 0:
            return 0
        else:
            return np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

    def getShuffledData(self, data):  # generate data for cross validation
        copydata = copy.deepcopy(data)
        np.random.shuffle(copydata)
        dataset = preprocess.NumpyArrary2Dict(copydata)
        if self.mode == 'trapezoidal':
            dataset = preprocess.removeDataTrapezoidal(dataset)
            X, Y = [], []
            for row in dataset:
                Y.append(row['class_label'])
                del row['class_label']
                for key in list(row.keys()):
                    if row.get(key) == 0:
                        row.pop(key)
                X.append(row)
        if self.mode == 'capricious':
            dataset = preprocess.removeRandomData(dataset) # 随机移除特征
            X, Y = [], []
            for row in dataset:
                Y.append(row['class_label'])
                del row['class_label']
                for key in list(row.keys()):
                    if row.get(key) == 0:
                        row.pop(key)
                X.append(row)
        return X, Y

    def fit(self, x_train, y_train):
        for i in range(len(y_train)):
            if y_train[i] == 0:
                y_train[i] = -1
        self.column_means = np.mean(x_train, axis=0)  # 计算每列的均值
        self.column_std = np.std(x_train, axis=0)  # 计算每列的标准差
        x_train_normalized = (x_train - self.column_means) / (self.column_std + 1e-6)
        self.count_sample += len(y_train)
        data = np.column_stack((x_train_normalized, y_train))
        self.X, self.y = self.getShuffledData(data)
        self.set_classifier()
        self.set_metadata()
        self.train_error, self.train_loss, self.train_acc = 0, 0, 0
        self.train_error_vector, self.train_loss_vector, self.train_acc_vector = [], [], []
        self.truth_label = []
        self.pre_label = []
        for t in range(0, len(self.y)):
            # print('t',t)
            row = self.X[t]
            if t:
                self.updateInstanceMetadata(row, self.y[t])
            weights_share, weights_new, y_pre, ps, pn = self.classifier(row)
            y_hat = np.sign(y_pre)
            self.truth_label.append(self.y[t])
            self.pre_label.append(-self.y[t] if y_hat == 0 else y_hat)  # 标签空间从[-1, 1]变成[0, 1]
            if len(row) == 0:
                self.train_error_vector.append(self.train_error / (t + 1))
                self.train_loss_vector.append(self.train_loss / (t + 1))
                self.train_acc_vector.append(1 - self.train_error / (t + 1))
                continue
            if y_hat != self.y[t]:
                self.train_error += 1
            # update classifier
            y_t = self.y[t]
            I = 1 if y_t > 0 else 0
            posNum = self.label_dic[1]
            negNum = self.label_dic[-1]
            alpha = 1 / ((posNum / negNum) ** I + (negNum / posNum) ** (1 - I))
            loss = (np.maximum(0, (1 - y_t * y_pre)))
            row_share_vector = subsetDictionary(row, findCommonKeys(row, weights_share))
            row_new_vector = subsetDictionary(row, findCommonKeys(row, weights_new))
            param = alpha * self.theta
            tao = self.parameter_set(row_share_vector, row_new_vector, ps, pn, loss)
            weights_share = self.updateWeights(weights_share, row, tao * y_t * ps * param)
            weights_new = self.updateWeights(weights_new, row, tao * y_t * pn * param)
            self.current_weights = dict()
            self.current_weights.update(weights_share)
            self.current_weights.update(weights_new)
            self.weights.update(self.current_weights)
            if self.sparse: self.sparsity_step()
            self.train_error_vector.append(self.train_error / (t + 1))
            self.train_loss += loss
            self.train_loss_vector.append(self.train_loss / (t + 1))
            self.train_acc = 1 - (self.train_error / (t + 1))
            self.train_acc_vector.append(self.train_acc)
        print('train over')
    def predict(self, x):
        x_normalized = (x - self.column_means) / (self.column_std + 1e-6)
        y_hat = []
        for i in range(x_normalized.shape[0]):
            dictx = [{k: v for k, v in enumerate(row)} for row in np.array([x_normalized[i,:]])]
            # print(dictx)
            weights_share, weights_new, y_pre, ps, pn = self.classifier(dictx[0])
            # print(y_pre)
            y_hat.append(int(np.sign(y_pre)))
        for i in range(len(y_hat)):
            if y_hat[i] == -1:
                y_hat[i] = 0
        return y_hat

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, x):
        y_proba = []
        x_normalized = (x - self.column_means) / self.column_std
        for i in range(x_normalized.shape[0]):
            dictx = [{k: v for k, v in enumerate(row)} for row in np.array([x_normalized[i, :]])]
            weights_share, weights_new, y_pre, ps, pn = self.classifier(dictx[0])
            y_proba.append([1 - self.sigmoid(y_pre), self.sigmoid(y_pre)])

        return np.array(y_proba)
    def partial_fit(self, x, y):
        x_normalized = (x - self.column_means) / (self.column_std + 1e-6)
        self.count_sample += len(y)
        for i in range(x_normalized.shape[0]):
            dictx = [{k: v for k, v in enumerate(row)} for row in np.array([x_normalized[i,:]])]
            row = dictx[0]
            if y[i] == 0:
                y_value = -1
            else:
                y_value = 1


            self.updateInstanceMetadata(row, y_value)
            weights_share, weights_new, y_pre, ps, pn = self.classifier(row)
            y_hat = np.sign(y_pre)
            self.truth_label.append(y_value)
            self.pre_label.append(-y_value if y_hat == 0 else y_hat)  # 标签空间从[-1, 1]变成[0, 1]
            # if len(row) == 0:
            #     self.train_error_vector.append(self.train_error / (self.count_sample + 1))
            #     self.train_loss_vector.append(self.train_loss / (self.count_sample + 1))
            #     self.train_acc_vector.append(1 - self.train_error / (self.count_sample + 1))
            #     continue
            if y_hat != y_value:
                self.train_error += 1
            # update classifier
            y_t = y_value
            I = 1 if y_t > 0 else 0
            posNum = self.label_dic[1]
            negNum = self.label_dic[-1]
            alpha = 1 / ((posNum / negNum) ** I + (negNum / posNum) ** (1 - I))
            loss = (np.maximum(0, (1 - y_t * y_pre)))
            row_share_vector = subsetDictionary(row, findCommonKeys(row, weights_share))
            row_new_vector = subsetDictionary(row, findCommonKeys(row, weights_new))
            param = alpha * self.theta
            tao = self.parameter_set(row_share_vector, row_new_vector, ps, pn, loss)
            weights_share = self.updateWeights(weights_share, row, tao * y_t * ps * param)
            weights_new = self.updateWeights(weights_new, row, tao * y_t * pn * param)
            self.current_weights = dict()
            self.current_weights.update(weights_share)
            self.current_weights.update(weights_new)
            self.weights.update(self.current_weights)
            if self.sparse: self.sparsity_step()
            self.train_error_vector.append(self.train_error / (self.count_sample + 1))
            self.train_loss += loss
            self.train_loss_vector.append(self.train_loss / (self.count_sample + 1))
            self.train_acc = 1 - (self.train_error / (self.count_sample + 1))
            self.train_acc_vector.append(self.train_acc)

    def run(self):
        np.random.seed(p.random_seed)


        self.getShuffledData()
        self.set_classifier()
        self.set_metadata()
        train_error, train_loss, train_acc = 0, 0, 0
        train_error_vector, train_loss_vector, train_acc_vector = [], [], []
        truth_label = []
        pre_label = []
        F1 = []
        G_mean = []
        for t in range(0, len(self.y)):
            print(t)
            row = self.X[t]
            if t:
                self.updateInstanceMetadata(t)
            weights_share, weights_new, y_pre, ps, pn = self.classifier(row)
            y_hat = np.sign(y_pre)
            truth_label.append(self.y[t])
            pre_label.append(-self.y[t] if y_hat == 0 else y_hat)  # 标签空间从[-1, 1]变成[0, 1]
            tn, fp, fn, tp = confusion_matrix(truth_label, pre_label).ravel()
            g_t = self.Gmean(tn, tp, fn, fp)
            G_mean.append(g_t)
            f1_t = f1_score(truth_label, pre_label)
            F1.append(f1_t)
            if len(row) == 0:
                train_error_vector.append(train_error / (t + 1))
                train_loss_vector.append(train_loss / (t + 1))
                train_acc_vector.append(1 - train_error / (t + 1))
                continue
            if y_hat != self.y[t]:
                train_error += 1
            # update classifier
            y_t = self.y[t]
            I = 1 if y_t > 0 else 0
            posNum = self.label_dic[1]
            negNum = self.label_dic[-1]
            alpha = 1 / ((posNum / negNum) ** I + (negNum / posNum) ** (1 - I))
            loss = (np.maximum(0, (1 - y_t * y_pre)))
            row_share_vector = subsetDictionary(row, findCommonKeys(row, weights_share))
            row_new_vector = subsetDictionary(row, findCommonKeys(row, weights_new))
            param = alpha * self.theta
            tao = self.parameter_set(row_share_vector, row_new_vector, ps, pn, loss)
            weights_share = self.updateWeights(weights_share, row, tao * y_t * ps * param)
            weights_new = self.updateWeights(weights_new, row, tao * y_t * pn * param)
            self.current_weights = dict()
            self.current_weights.update(weights_share)
            self.current_weights.update(weights_new)
            self.weights.update(self.current_weights)
            if self.sparse: self.sparsity_step()
            train_error_vector.append(train_error / (t + 1))
            train_loss += loss
            train_loss_vector.append(train_loss / (t + 1))
            train_acc = 1 - (train_error / (t + 1))
            train_acc_vector.append(train_acc)



