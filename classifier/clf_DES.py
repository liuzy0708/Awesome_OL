import copy as cp
import numpy as np
import random
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes
from imblearn.over_sampling import SMOTE
import math


class Knn:

    def __init__(self):
        self.data = []
        self.dic = {}


    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def get_neighbers(self, x, knn):
        self.distance = np.zeros(knn)
        N, D = self.X.shape
        neighbers = np.zeros((knn, D))
        neighbers_y = np.zeros(knn)
        for i in range(len(self.X)):
            self.dic[i] = math.sqrt( math.fsum( ( (a-b)**2 for a, b in zip(x, self.X[i])) ) )

        L = sorted(self.dic.items(),key=lambda item:item[1],reverse=False)
        L = L[:knn]
        for i in range(knn):
            self.distance[i] = L[i][1]
            neighbers[i] = self.X[L[i][0]]
            neighbers_y[i] = self.Y[L[i][0]]
        return neighbers, neighbers_y

    def get_distance(self):
        return self.distance

    def get_average_dis(self):
        sum = 0
        for i in range(len(self.distance)):
            sum = sum + self.distance[i]
        ave = sum/len(self.distance)
        return ave

class k_calculator():
    def __init__(self):
        self.best_k = 0

    def __average(self, distance, k):
        sum = 0
        for i in range(k):
            sum = sum + distance[i]
        ave = sum / k
        return ave


    def get_k(self, Dphi, nei,neiy, minclss, distance):
        min_dif = 1
        nei = list(nei)
        distance = list(distance)
        best_k = None
        for i in range(len(neiy))[::-1]:
            if neiy[i] != minclss:
                nei.pop(i)
                distance.pop(i)

        nei = np.array(nei)
        distance = np.array(distance)

        if len(nei)==0:
            return None, None
        else:
            for i in range(len(nei)):
                if i == 0:
                    continue
                dif = abs((self.__average(distance=distance, k=i) / Dphi) - 1.0)
                if dif < min_dif:
                    min_dif = dif
                    best_k = i
            nei = nei[:best_k, :]
            return best_k, nei

def class_size(blocky):
    size = np.array([0, 0])
    for i in range(len(blocky)):
        if blocky[i] == 0:
            size[0] = size[0] + 1
        else:
            size[1] = size[1] + 1
    return size

def get_min_data(B, By, minclass):
    N, D = B.shape
    B_size = class_size(By)
    B_min =  np.zeros((B_size[minclass], D))

    j = 0
    for i in range(N):
        if By[i] == minclass:
            B_min[j] = B[i]
            j = j + 1

    return B_min



def AKS(refB, refBy, newB, newBy):
    N, D = newB.shape

    newB_size = class_size(newBy)
    #需要生成的少数类样本数量
    new_num = 0

    if newB_size[0] > newB_size[1] :
        minclass = 1
        new_num = newB_size[0] - newB_size[1]
    elif newB_size[1] > newB_size[0]:
        minclass = 0
        new_num = newB_size[1] - newB_size[0]
    elif newB_size[0] == newB_size[1]:
        return newB, newBy

    #获得参考集中少数类
    refB_min = get_min_data(B=refB, By=refBy, minclass=minclass)
    #如果参考集中没有少数类，则不采样
    if len(refB_min) == 0:
        return newB, newBy

    #获得新数据块中每个少数类与近邻
    newB_min = get_min_data(B=newB, By=newBy, minclass=minclass)
    # 如果新数据块中没有少数类，则不采样
    if len(newB_min) == 0:
        return newB, newBy

    refB_knn = Knn()
    refB_min_y = np.zeros(len(refB_min))
    for i in range(len(refB_min_y)):
        refB_min_y[i] = minclass
    newB_min_y = np.zeros(len(newB_min))
    for i in range(len(newB_min_y)):
        newB_min_y[i] = minclass

    refB_knn.fit(refB_min,refB_min_y)
    if len(refB_min) < 5:
        k_n = len(refB_min)
    else: k_n = 5
    nei_list = []
    for i in range(len(newB_min)):
        refB_knn.get_neighbers(x=newB_min[i], knn=k_n)
        Dphi = refB_knn.get_average_dis()
        newB_knn = Knn()
        newB_knn.fit(X=newB, Y=newBy)
        xnei, xneiy = newB_knn.get_neighbers(x=newB_min[i], knn=7)
        nei_dis = newB_knn.get_distance()
        k_cal = k_calculator()
        best_k, nei = k_cal.get_k(Dphi=Dphi, nei=xnei, neiy=xneiy, minclss=minclass, distance=nei_dis)
        if best_k == None:
            nei_list.append(newB_min[i])
        else:nei_list.append(nei)


    coulp = []
    for i in range(len(newB_min)):
        if len(nei_list[i]) == 1:
            continue
        else:
            for j in range(len(nei_list[i])):
                coulp.append([newB_min[i], nei_list[i][j]])


    new_x_list = []
    for i in range(new_num):
        ran_x = random.randint(0, len(coulp)-1)
        some_big_number = 10000000.
        alpha = random.randint(0, some_big_number) / some_big_number
        new_x = coulp[ran_x][0] + alpha * (coulp[ran_x][1]-coulp[ran_x][0])
        new_x_list.append(new_x)

    RnewB = np.zeros((N+new_num, D))
    RnewB_y = np.zeros(N+new_num)
    for i in range(N+new_num):
        if i < N:
            RnewB[i] = newB[i]
            RnewB_y[i] = newBy[i]
        else:
            RnewB[i] = new_x_list[i-N]
            RnewB_y[i] = minclass
    return RnewB, RnewB_y




class DES_ICD(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_classifier=NaiveBayes(), window_size=200, max_classifier=20):
        self.base_classifier = base_classifier
        self.window_size = window_size
        self.max_classifier = max_classifier

        self.ensemble = []
        self.X_block = None
        self.Y_block = None
        self.sample_wight = None
        self.i = -1
        self.refB = []
        self.refBy = []
        self.expert_size = []

    def fit(self, X, y, classes=None, sample_weight=None):
        if len(set(y)) > 2 :
            raise Exception("DES-ICD只能处理二分类问题")
        if X.shape[0] < self.window_size:
            raise Exception("fit的样本数应该大于window_size")
        self.partial_fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):

        N, D = X.shape

        if self.i < 0:
            self.X_block = np.zeros((self.window_size, D))
            self.Y_block = np.zeros(self.window_size)
            self.i = 0

        for n in range(N):
            self.X_block[self.i] = X[n]
            self.Y_block[self.i] = y[n]

            self.i = self.i + 1
            if self.i == self.window_size:
                if len(self.ensemble) <= 0:
                    classifier = cp.deepcopy(self.base_classifier)
                    classifier.partial_fit(X=self.X_block, y=self.Y_block.astype(int))
                    self.ensemble.append(classifier)
                else:
                    X_block_res, Y_block_res = AKS(refB=self.refB, refBy=self.refBy, newB=self.X_block, newBy=self.Y_block)
                    if len(self.ensemble) >= self.max_classifier:
                        self.ensemble.pop(0)
                    if len(self.ensemble) > 0:
                        for i in range(len(self.ensemble)):
                            self.ensemble[i].partial_fit(X=X_block_res, y=Y_block_res.astype(int))
                    classifier = cp.deepcopy(self.base_classifier)
                    classifier.partial_fit(X=X_block_res, y=Y_block_res.astype(int))
                    self.ensemble.append(classifier)

                self.i = 0
                self.refB = cp.deepcopy(self.X_block)
                self.refBy = cp.deepcopy(self.Y_block)

        return self

    def predict_proba(self, X):
        N, D = X.shape
        votes = np.zeros(N)
        experts = []
        testX = X

        if len(self.ensemble) <= 0:
            return np.array([[0, 1]])

        for n in range(N):
            x = testX[n]

            find_knn = Knn()
            find_knn.fit(self.X_block, self.Y_block)
            neighbors, neighbors_y = find_knn.get_neighbers(x=x, knn=5)

            ref_size = class_size(self.refBy)
            if ref_size[0] < ref_size[1]:
                minclass = 0
            else:
                minclass = 1
            if 1 in neighbors_y and 0 in neighbors_y:
                min_neighbors = []
                for i in range(len(neighbors_y)):
                    if neighbors_y[i] == minclass:
                        min_neighbors.append(neighbors[i])
                neighbors = np.array(min_neighbors)
                if minclass == 0:
                    neighbors_y = np.zeros(len(neighbors))
                else:
                    neighbors_y = np.ones(len(neighbors))

            score = []
            for i in range(len(self.ensemble)):
                score.append(self.ensemble[i].score(X=neighbors, y=neighbors_y))
            for i in range(len(self.ensemble)):
                if score[i] == max(score):
                    experts.append(self.ensemble[i])


            if len(experts) > 0:
                self.expert_size.append(len(experts))
                for C in experts:
                    votes[n] = votes[n] + (1. / len(experts)) * C.predict([x])
            else:
                self.expert_size.append(len(self.ensemble))
                for C in self.ensemble:
                    votes[n] = votes[n] +(1. / len(self.ensemble)) * C.predict([x])
            experts = []
        return np.array([[1-votes[0], votes[0]]])

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):

            votes = self.predict_proba(np.array([X[i, :]]))
            y_pred.append(int((votes[0][1] >= 0.5) * 1.))
        return y_pred

    def reset(self):
        self.ensemble = []
        self.i = -1
        self.X_block = None
        self.Y_block = None
        self.expert_size = []
        return self

    def experts_size(self):
        return self.expert_size
