""" OALE Strategy."""

from skmultiflow.bayes import NaiveBayes
import numpy as np
import copy
import random

class OALE_strategy(object):
    def __init__(self, x_train, y_train, r=0.05, I=150, L=2, s=0.01, theta=0.4, sigma = 0.01, D=10):
        self.I = I  # chunk size
        self.L = L  # number of the classes
        self.s = s  # adjustment step
        self.D = D  # the number of classifiers
        self.p = 0 # counter for processed instances
        self.k = 0 # counter for created dynamic classifiers
        self.theta = theta
        self.theta_m = theta * 2 / L # set θm for UncertaintyStrategy
        self.r = r
        self.sigma = sigma
        self.w = []
        self.E = []
        self.U_x = [[] for i in range(self.L)]  # instances buffer
        self.U_y = [[] for i in range(self.L)]  # instances buffer
        for i in range(len(y_train)):
            self.U_x[int(y_train[i])].append(np.array([x_train[i, :]]))
            self.U_y[int(y_train[i])].append(y_train[i])
        self.A = [0 for i in range(self.I)]
        self.A_y = [0 for i in range(self.I)]
        self.i = 0
        self.n_annotation = 0
        clf = NaiveBayes()
        clf.fit(x_train, y_train)
        self.k += 1
        self.C_stable = copy.deepcopy(clf)
        self.E.append(copy.deepcopy(clf))
        self.w.append(1 / self.D)



    def predict(self, x):
        proba = self.predict_proba(x)
        y_predict = np.argmax(proba[0])
        return [y_predict]

    def predict_proba(self, x):

        proba_1 = 0.5 * self.C_stable.predict_proba(x)
        proba_2 = self.w[0] * self.E[0].predict_proba(x)
        for i in range(1, len(self.E)):
            proba_2 += self.w[i] * self.E[i].predict_proba(x)
        proba = proba_1 + proba_2
        return proba

    def UncertaintyStrategy(self, x):
        y_predict = self.predict_proba(x)
        max_p = max(y_predict[0])
        max_2_p = np.sort(y_predict[0])[-2]
        margin_x = max_p - max_2_p

        if margin_x <= self.theta_m:
            self.theta_m = self.theta_m * (1 - self.s)
            return True
        else:
            return False

    def RandomStrategy(self, x):
        if random.random() <= self.sigma:
            return True
        else:
            return False

    def evaluation(self, x_new, y_new):
        islabel = False
        labeling = self.UncertaintyStrategy(x_new)
        if labeling:

           islabel = True
           self.n_annotation += 1
           self.C_stable.partial_fit(x_new, y_new)
           for classifier in self.E:
               classifier.partial_fit(x_new, y_new)
        else:
            labeling = self.RandomStrategy(x_new)
            if labeling:
                islabel = True
                self.n_annotation += 1

                self.C_stable.partial_fit(x_new, y_new)
                for classifier in self.E:
                    classifier.partial_fit(x_new, y_new)
        self.A[self.i] = x_new[0]
        self.A_y[self.i] = y_new[0]
        self.i = (self.i + 1) % self.I
        if self.i == 0:
            self.k += 1
            self.n_annotation += int(self.I * self.r)
            random_row_indices = np.random.choice(np.array(self.A).shape[0], size=int(self.I * self.r), replace=False)
            x_create = np.array(self.A)[random_row_indices, :]
            y_create = np.array(self.A_y)[random_row_indices]
            for i in range(self.L):
                for j in range(3):
                    x_create = np.row_stack((x_create, self.U_x[i][j]))
                    y_create = np.append(y_create, self.U_y[i][j])
            clf_new = NaiveBayes()
            clf_new.fit(x_create, y_create)
            self.theta_m = self.theta * 2 / self.L
            for i in range(len(self.w)):
                self.w[i] = self.w[i] * (1 - 1 / self.D)
            self.E.append(copy.deepcopy(clf_new))
            self.w.append(1 / self.D)
            if len(self.w) > self.D:
                self.E.pop(0)
                self.w.pop(0)



