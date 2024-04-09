from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import SEAGenerator, HyperplaneGenerator, WaveformGenerator, FileStream, DataStream
import numpy as np
import copy
import random
from scipy.stats import entropy
import pandas as pd
import csv
import time
from sklearn.metrics import f1_score
class ROALE_DI_strategy(object):
    def __init__(self, x_train, y_train, label_ratio = 0.05, chunk_size = 150, L = 3,  step = 0.01, theta = 0.005, D = 10, sigma_imbalance = 0.01, clf = NaiveBayes()):
        self.label_ratio = label_ratio
        self.I = chunk_size # chunk size
        self.L = L #number of the classes
        self.s = step # adjustment step
        self.D = D # the number of classifiers
        self.theta = theta
        self.theta_m = theta / 2
        self.sigma_imbalance = sigma_imbalance
        self.DCIR = [0.0 for i in range(self.L)]
        self.A = [0 * i for i in range(self.I)] # x of instances
        self.A_y = [0 * i for i in range(self.I)] # y of instances, sketch when labeled
        self.E = [] # Dynamic classifier ensemble
        self.U_x = [[] for i in range(self.L)]  # instances buffer
        self.U_y = [[] for i in range(self.L)]  # instances buffer
        self.sigma_set = [0.0 * i for i in range(self.L) ] # sigma of the classes
        self.w_d = [] #weights
        self.H = [] # is correspending to E
        self.p = 0 # counter of processed instances
        self.i = 0 # indicator of the current position in circular array
        self.k = 0 # indicator of the created dynamic classifier
        self.labeled_instances = []

        #self.n_annotation = x_train.shape[0]
        self.n_annotation = 0
        clf.fit(x_train, y_train)
        self.C_stable = copy.deepcopy(clf)
        self.E.append(copy.deepcopy(clf))
        self.w_d.append(1 / self.D)
        Hn = [0 for i in range(self.L)]
        # print(R_n_y)
        for i in range(x_train.shape[0]):
            Hn[int(y_train[i])] += 1
        self.H.append(Hn)
        for i in range(len(y_train)):
            self.U_x[int(y_train[i])].append(np.array([x_train[i, :]]))
            self.U_y[int(y_train[i])].append(y_train[i])
            if len(self.U_x[int(y_train[i])]) > self.I * self.label_ratio / self.L:
                self.U_x[int(y_train[i])].pop(0)
                self.U_y[int(y_train[i])].pop(0)

    # def normalization(self, para):
    #     Min = np.min(para)
    #     Max = np.max(para)
    #     para_mod = np.zeros((1, para.shape[1]))
    #     para_normal = np.zeros((1, para.shape[1]))
    #
    #     for i in range(para.shape[1]):
    #         para_mod[0, i] = ((para[0, i]) - Min) / (Max - Min)
    #     para_sum = np.sum(para_mod)
    #     # print(para_sum)
    #     for i in range(para.shape[1]):
    #         para_normal[0, i] = para_mod[0, i] / para_sum
    #
    #     return para_normal
    def normalization(self, para):
        sum_para = np.sum(para[0])
        para_normal = para / sum_para

        return para_normal

    def PredictResult_single(self, x, y, i):
        # print(self.E)
        prob = self.E[i].predict_proba(x)
        # print('prob before_ROALE = {}'.format(prob))
        prob = self.normalization(prob)

        y_predict = 0
        max_prob = -1
        # print('prob AFTER_ROALE = {}'.format(prob))
        for i in range(len(prob[0])):
            if prob[0][i] > max_prob:
                max_prob = prob[0][i]
                y_predict = i
        if y_predict == y:
            return True
        else:
            return False

    def PredictResult(self, x, y):
        prob_1 = self.C_stable.predict_proba(x)
        prob_1 = 0.5 * self.normalization(prob_1)
        # print('prob_1={}'.format(prob_1))
        prob_2 = self.w_d[0] * self.E[0].predict_proba(x)
        # prob_2 = self.normalization(prob_2)

        for i in range(1, len(self.E)):
            prob_2 += self.w_d[i] * self.E[i].predict_proba(x)
        prob_2 = self.normalization(prob_2)

        prob_2 *= 0.5
        # print('prob_2={}'.format(prob_2))
        y_predict = 0
        max_prob = -1
        # print(prob_1.shape)
        # print(prob_2.shape)
        prob = prob_1 + prob_2
        # print('prob = {}'.format(prob))
        for i in range(len(prob[0])):
            if prob[0][i] > max_prob:
                max_prob = prob[0][i]
                y_predict = i
        return [y_predict]
        # if y_predict == y:
        #     return True
        # else:
        #     return False

    # def Predict_prob(self, x):
    #
    #     prob = self.w_d[0] * self.E[0].predict_proba(x)
    #     print('prob_before', prob)
    #     prob = self.normalization(prob)
    #     print('prob_after', prob)
    #     for i in range(1, len(self.E)):
    #         prob += self.w_d[i] * self.normalization(self.E[i].predict_proba(x))
    #     print('prob', prob)
    #     # prob = prob / np.sum(prob[0])
    #     # # print('prob', prob)
    #     return prob
    def Predict_prob(self, x):

        prob_1 = self.C_stable.predict_proba(x)
        # print('prob_1before = {}'.format(prob_1))
        prob_1 = 0.5 * self.normalization(prob_1)
        # print('prob_1after = {}'.format(prob_1))
        prob_2 = self.w_d[0] * self.E[0].predict_proba(x)
        # print('0={}'.format(self.E[0].predict_proba(x)))

        for i in range(1, len(self.E)):
            # print('{}={}'.format(i, self.E[i].predict_proba(x)))
            prob_2 += self.w_d[i] * self.E[i].predict_proba(x)
        # print('prob_2before = {}'.format(prob_2))
        prob_2 = self.normalization(prob_2)
        # print('prob_2after = {}'.format(prob_2))

        prob_2 *= 0.5



        prob = prob_1 + prob_2

        return prob

    def predict(self, x):
        prob_1 = self.C_stable.predict_proba(x)
        prob_1 = 0.5 * self.normalization(prob_1)
        # print('prob_1={}'.format(prob_1))
        prob_2 = self.w_d[0] * self.E[0].predict_proba(x)
        # prob_2 = self.normalization(prob_2)

        for i in range(1, len(self.E)):
            prob_2 += self.w_d[i] * self.E[i].predict_proba(x)
        prob_2 = self.normalization(prob_2)

        prob_2 *= 0.5
        # print('prob_2={}'.format(prob_2))
        y_predict = 0
        max_prob = -1
        # print(prob_1.shape)
        # print(prob_2.shape)
        prob = prob_1 + prob_2
        # print('prob = {}'.format(prob))
        for i in range(len(prob[0])):
            if prob[0][i] > max_prob:
                max_prob = prob[0][i]
                y_predict = i
        return [y_predict]
    def ReinforcementWeightAdjustment(self, x, y):

        if self.DCIR[int(y[0])] < 1 / self.L:
            for i in range(len(self.E)):
                # print('i = {}'.format(i))
                if self.PredictResult_single(x, y, i) == True:
                    self.w_d[i] = self.w_d[i] * (1 + 1 / self.D)
                else:
                    self.w_d[i] = self.w_d[i] * (1 - 1 / self.D)

    def CreatNewBaseClassifier(self, labeling_ratio, is_first, clf):
        batch_size = int(labeling_ratio * self.I)
        sample_list = [i for i in range(len(self.A))]
        sample_list = random.sample(sample_list, batch_size)
        R_n_x = [self.A[i] for i in sample_list]
        # print(self.A_y)
        # print(R_n_x)
        # print(self.labeled_instances)
        #########################################
        if not is_first:
            for i in R_n_x:
                decision = False
                for j in self.labeled_instances:

                    flag = (i == j)[0].all()
                    if flag == True:
                        decision = True
                        break
                if decision == False:
                    self.n_annotation += 1

            self.labeled_instances = []

        #########################################



        R_n_y = [self.A_y[i] for i in sample_list]
        row_Rn = len(R_n_x)
        Hn = [0 for i in range(self.L)]
        # print(R_n_y)
        for i in range(len(R_n_x)):
            # print('R_n_y', R_n_y)
            Hn[int(R_n_y[i][0])] += 1
        if not is_first:
            for i in range(int(self.I * labeling_ratio)):
                xi = R_n_x[i]
                yi = R_n_y[i]
                if self.PredictResult(xi, yi) == True:
                    self.theta_m = self.theta_m * (1 - self.s)
                else:
                    self.ReinforcementWeightAdjustment(xi, yi)
        C_new = NaiveBayes()
        # C_new = DES_ICD(window_size=self.I, max_classifier=20)
        if is_first == False:
            for i in range(self.L):
                if Hn[i] < self.I * labeling_ratio / self.L:
                    R_n_x = R_n_x + self.U_x[i][int(len(self.U_x[i]) - self.I * labeling_ratio / self.L + Hn[i]) : len(self.U_x[i])]
                    R_n_y = R_n_y + self.U_y[i][int(len(self.U_y[i]) - self.I * labeling_ratio / self.L + Hn[i]) : len(self.U_y[i])]

        R_n_x_nparray = R_n_x[0]
        for i in range(1, len(R_n_x)):
            R_n_x_nparray = np.row_stack((R_n_x_nparray, R_n_x[i]))
        R_n_y_nparray = R_n_y[0]
        for i in range(1, len(R_n_y)):
            R_n_y_nparray = np.append(R_n_y_nparray, R_n_y[i])

        C_new.fit(R_n_x_nparray, R_n_y_nparray)
        clf.partial_fit(R_n_x_nparray, R_n_y_nparray)
        # print('creat={}'.format(R_n_y_nparray))
        R_n_x = R_n_x[0 : row_Rn]
        R_n_y = R_n_y[0 : row_Rn]

        for i in range(len(R_n_y)):
            self.U_x[int(R_n_y[i][0])].append(R_n_x[i])
            self.U_y[int(R_n_y[i][0])].append(R_n_y[i])
            if len(self.U_x[int(R_n_y[i][0])]) > self.I * labeling_ratio / self.L:
                self.U_x[int(R_n_y[i][0])].pop(0)
                self.U_y[int(R_n_y[i][0])].pop(0)


        return C_new, Hn, clf

    def UncertaintyStrategy(self, x):
        y_predict = self.Predict_prob(x)
        # print(y_predict)
        # margin_x = np.fabs(y_predict[0][0] - y_predict[0][1])

        # ###################### modify ##################
        # margin_x = max(y_predict[0]) - min(y_predict[0])



        ###################### modify ##################
        margin_x = 1 - entropy(y_predict[0], base=self.L)

        # max_p = max(y_predict[0])
        # max_2_p = np.sort(y_predict[0])[-2]
        # margin_x = max_p - max_2_p


        # print('magin_x = {}'.format(margin_x))
        ####################modify ######################



        # print('margin_x= {}'.format(margin_x))
        # print(y_predict)
        if margin_x <= self.theta_m:
            self.theta_m = self.theta_m * (1 - self.s)
            return True
        else:
            # ################add#####################
            # self.theta_m = self.theta_m * (1 + self.s)
            # ################add######################
            return False

    def ImbalanceStratrgy(self, x):
        y_predict = self.Predict_prob(x)
        index = 0
        max_prob = -1
        for i in range(len(y_predict[0])):
            if y_predict[0][i] > max_prob:
                max_prob = y_predict[0][i]
                index = i

        kesi = np.random.uniform(0, 1)
        if kesi < self.sigma_set[index]:

            return True
        else:
            return False

    def DealInstance(self, x_new, y_new):
        labeling = self.UncertaintyStrategy(x_new)
        if labeling:
            # get yi
            self.labeled_instances.append(x_new)
            self.U_x[int(y_new[0])].append(x_new)
            self.U_y[int(y_new[0])].append(y_new)
            if len(self.U_x[int(y_new[0])]) > self.I * self.label_ratio / self.L:
                self.U_x[int(y_new[0])].pop(0)
                self.U_y[int(y_new[0])].pop(0)
            ################# add ##############################
            self.C_stable.partial_fit(x_new, y_new)
            ################# add ##############################
            for classifier in self.E:
                classifier.partial_fit(x_new, y_new)
            if self.PredictResult(x_new, y_new) == True:
                self.theta_m = self.theta_m * (1 - self.s)
            # else:
            #     ##################add#####################
            #     self.theta_m = self.theta_m * (1 + self.s)
            #     #################add#####################
        else:
            labeling = self.ImbalanceStratrgy(x_new)
            if labeling:
                #get yi
                self.labeled_instances.append(x_new)
                # print('y_new', y_new)
                self.U_x[int(y_new[0])].append(x_new)
                self.U_y[int(y_new[0])].append(y_new)
                if len(self.U_x[int(y_new[0])]) > self.I * self.label_ratio / self.L:
                    self.U_x[int(y_new[0])].pop(0)
                    self.U_y[int(y_new[0])].pop(0)
                ################# add ##############################
                self.C_stable.partial_fit(x_new, y_new)
                ################# add ##############################
                for classifier in self.E:
                    classifier.partial_fit(x_new, y_new)
                if self.PredictResult(x_new, y_new) == False:
                    self.ReinforcementWeightAdjustment(x_new, y_new)

        self.A[self.i] = x_new
        self.A_y[self.i] = y_new

        return labeling



    def evaluation(self, x_new, y_new, clf):

        self.p += 1

        self.i = (self.p - 1) % self.I
        is_Label = self.DealInstance(x_new, y_new)
        self.i = (self.i + 1) % self.I
        if self.i == 0:
            self.k += 1
            self.theta_m = self.theta * 2 / self.L
            C_new, H_cnew, clf = self.CreatNewBaseClassifier(self.label_ratio, False, clf)
            if self.k > self.D:
                # c_new --> c_min
                weight = np.inf
                index = 0
                for i in range(len(self.E)):
                    if self.w_d[i] < weight:
                        weight = self.w_d[i]
                        index = i
                self.E[index] = copy.deepcopy(C_new)
                self.w_d = [i * (1 - 1 / self.D) for i in self.w_d]
                self.w_d[index] = 1 / self.D
                self.H[index] = H_cnew
            else:
                self.E.append(copy.deepcopy(C_new))
                self.H.append(H_cnew)
                #print(self.w_d)
                self.w_d = [i * (1 - 1 / self.D) for i in self.w_d]
                #print(self.w_d)
                self.w_d.append(1 - 1 / self.D)

            # update imbalance ratio
            sum_down = 0
            for i in range(self.L):
                for j in range(len(self.E)):
                    sum_down += self.H[j][i] * self.w_d[j]
            for i in range(len(self.DCIR)):
                sum_up = 0
                for j in range(len(self.E)):
                    sum_up += self.H[j][i] * self.w_d[j]
                self.DCIR[i] = sum_up / sum_down

            for i in range(self.L):
                if self.DCIR[i] == 0:
                    self.sigma_set[i] = 1
                else:
                    self.sigma_set[i] = max(self.sigma_imbalance / self.DCIR[i] / self.L, self.sigma_imbalance)

        if is_Label == True:
            self.n_annotation += 1
            clf.partial_fit(x_new, y_new)

        return is_Label, clf

