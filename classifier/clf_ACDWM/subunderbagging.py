from numpy import *
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


class SubUnderBagging:

    def __init__(self, Q=1000, T=100, k_mode=2):
        self.Q = Q
        self.T = T
        self.k_mode = k_mode

        self.model = list()

    def train(self, data, label):
        data_num = label.size
        neg_num = sum(label == -1)
        pos_num = sum(label == 1)
        neg_idx = nonzero(label == -1)[0]
        pos_idx = nonzero(label == 1)[0]

        # k = int(sqrt(min(neg_num, pos_num)))
        k = int(sqrt(data_num))

        for j in range(self.Q):

            all_pos_idx = pos_idx
            random.shuffle(all_pos_idx)
            all_neg_idx = neg_idx
            random.shuffle(all_neg_idx)
            all_idx = array(range(data_num))
            random.shuffle(all_idx)

            # compare k and class size
            if self.k_mode == 1:
                if k / 2 < min(pos_num, neg_num):
                    sampling_idx = r_[all_neg_idx[:int(k / 2)], all_pos_idx[:int(k / 2)]]
                else:
                    if neg_num > pos_num:
                        sampling_idx = r_[all_neg_idx[:k - pos_num], all_pos_idx]
                    else:
                        sampling_idx = r_[all_neg_idx, all_pos_idx[:k - neg_num]]
            elif self.k_mode == 2:
                if neg_num > pos_num:
                    sampling_idx = r_[all_neg_idx[:k], all_pos_idx]
                else:
                    sampling_idx = r_[all_neg_idx, all_pos_idx[:k]]

            sampling_data = data[sampling_idx]
            sampling_label = label[sampling_idx]

            self.model.append(DecisionTreeClassifier(max_depth=1))
            self.model[j] = self.model[j].fit(sampling_data, sampling_label)

    def predict(self, test_data, P):
        test_num = test_data.shape[0]
        temp_result = zeros([P, self.T, test_num])
        all_pred = zeros([self.Q, test_num])

        for i_Q in range(self.Q):
            all_pred[i_Q, :] = self.model[i_Q].predict_proba(test_data)[:, 1]

        for i_P in range(P):
            rand_idx = random.permutation(self.Q)
            temp_result[i_P, :, :] = all_pred[rand_idx[:self.T], :]

        pred_result = mean(temp_result, 1)

        return pred_result
