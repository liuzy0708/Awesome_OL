from numpy import *
import math
from sklearn.naive_bayes import GaussianNB


class UnderBagging:

    def __init__(self, T=11, r=1.0, sampling_class=0, pos_weight=[], neg_weight=[],
                 replace=False, auto_T=False, auto_r=False):
        # sampling_class is 0 for undersampling the majority class

        self.T = T
        self.r = r
        self.sampling_class = sampling_class
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.replace = replace
        self.auto_T = auto_T
        self.auto_r = auto_r
        self.model = list()

    def train(self, data, label):

        data_num = label.size
        neg_num = sum(label == -1)
        pos_num = sum(label == 1)
        neg_idx = nonzero(label == -1)[0]
        pos_idx = nonzero(label == 1)[0]

        if len(self.pos_weight) == 0:
            self.pos_weight = ones(pos_num) / pos_num
        if len(self.neg_weight) == 0:
            self.neg_weight = ones(neg_num) / neg_num

        if (neg_num > pos_num and self.sampling_class == 0) or self.sampling_class == -1:
            if self.auto_r:
                neg_sampling_num = math.ceil(neg_num / self.T)
            else:
                neg_sampling_num = math.ceil(pos_num / self.r)
            pos_sampling_num = pos_num

        else:
            if self.auto_r:
                pos_sampling_num = math.ceil(pos_num / self.T)
            else:
                pos_sampling_num = math.ceil(neg_num / self.r)
            neg_sampling_num = neg_num

        if self.auto_T:
            T = int(maximum(math.ceil(maximum(pos_num, neg_num) / minimum(pos_num, neg_num) * self.r), self.T))
            if T % 2 == 0:
                T += 1
        else:
            T = self.T

        for j in range(T):

            if neg_num != 0 and pos_num != 0:

                all_pos_idx = pos_idx
                random.shuffle(all_pos_idx)
                all_neg_idx = neg_idx
                random.shuffle(all_neg_idx)

                if self.replace:
                    sampling_idx = r_[all_neg_idx[random.choice(neg_num, neg_sampling_num, p=self.neg_weight)],
                                      all_pos_idx[random.choice(pos_num, pos_sampling_num, p=self.pos_weight)]]
                else:
                    sampling_idx = r_[all_neg_idx[:neg_sampling_num], all_pos_idx[:pos_sampling_num]]

                sampling_data = data[sampling_idx]
                sampling_label = label[sampling_idx]

                # self.model.append(NaiveBayes())
                # self.model.append(DecisionTreeClassifier())
                self.model.append(GaussianNB())
                # for i in range(len(sampling_label)):
                #     if sampling_label[i] == -1:
                #         sampling_label[i] = 0
                self.model[j] = self.model[j].fit(sampling_data, sampling_label)
                # for i in range(len(sampling_label)):
                #     if sampling_label[i] == 0:
                #         sampling_label[i] = -1

            else:
                self.model.append([])

    def predict(self, test_data):
        # print('underbagging_test_data', test_data)
        test_num = test_data.shape[0]
        temp_result = zeros([len(self.model), test_num])

        for i in range(len(self.model)):
            if self.model[i] != []:
                temp_result[i, :] = self.model[i].predict(test_data)
                # print('slf.model[i].predict', self.model[i].predict(test_data))
            else:
                temp_result[i, :] = zeros(test_num)

        pred_result = mean(temp_result, 0)
        # print('pred_result', pred_result)
        # if pred_result[0] == 0:
        #     pred_result[0] = -1
        return pred_result

    def predict_proba(self, test_data):
        proba = [[0, 0]]
        for i in range(len(self.model)):
            if self.model[i] != []:
                base_predict_proba = self.model[i].predict_proba(test_data)
                proba[0][0] += base_predict_proba[0][0]
                proba[0][1] += base_predict_proba[0][1]
                # print('self.model[i].predict_proba', self.model[i].predict_proba(test_data))
                # temp_result[i, :] = self.model[i].predict(test_data)
                # print('slf.model[i].predict', self.model[i].predict(test_data))
            else:
                print('underbagging_self.model==[]')


        sum_proba = sum(proba[0])
        proba[0][0] /= sum_proba
        proba[0][1] /= sum_proba

        # print('underbagging_proba', proba)
        return proba
