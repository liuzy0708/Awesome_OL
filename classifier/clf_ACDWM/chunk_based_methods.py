from .check_measure import *
from .subunderbagging import *
from sklearn.metrics import f1_score
import numpy as np
import abc


class ChunkBase:

    def __init__(self):

        self.ensemble = list()
        self.chunk_count = 0
        self.train_count = 0
        self.w = array([])
        self.buf_data = array([])
        self.buf_label = array([])

    def _predict_base(self, test_data, prob_output=False):
        # print('chunk_based_test_data', test_data)
        # print('len(self.ensemble)', len(self.ensemble))
        if len(self.ensemble) == 0:
            pred = zeros(test_data.shape[0])
        else:
            pred = zeros([test_data.shape[0], len(self.ensemble)])
            for i in range(len(self.ensemble)):
                if prob_output:
                    pred[:, i] = self.ensemble[i].predict_proba(test_data)[:, 1]
                else:
                    pred[:, i] = self.ensemble[i].predict(test_data)

        return pred

    def predict_proba(self, x):
        proba = [[0, 0]]
        if len(self.ensemble) == 0:
            print('chunk_based_methods_self.ensemble == []')
            # proba = zeros(x.shape[0])
        else:
            for i in range(len(self.ensemble)):
                base_predict_proba = self.ensemble[i].predict_proba(x)
                proba[0][0] += base_predict_proba[0][0]
                proba[0][1] += base_predict_proba[0][1]
            sum_proba = sum(proba[0])
            proba[0][0] /= sum_proba
            proba[0][1] /= sum_proba
        proba = np.array(proba)
        # print('chunk_based_methods_proba', proba)
        return proba


    @abc.abstractmethod
    def _update_chunk(self, data, label):
        pass

    def update(self, single_data, single_label):

        pred = self.predict(single_data.reshape(1, -1))

        if self.buf_label.size < self.chunk_size:
            self.buf_data = r_[self.buf_data.reshape(-1, single_data.shape[0]), single_data.reshape(1, -1)]
            self.buf_label = r_[self.buf_label, single_label]
            self.train_count += 1

        if self.buf_label.size == self.chunk_size or self.train_count == self.data_num:
            print('Data ' + str(self.train_count) + ' / ' + str(self.data_num))
            self._update_chunk(self.buf_data, self.buf_label)
            self.buf_data = array([])
            self.buf_label = array([])

        return pred

    def update_chunk(self, data, label):

        pred = self.predict(data)
        self._update_chunk(data, label)

        return pred

    def predict(self, test_data):
        y_pred = []
        for i in range(test_data.shape[0]):

            all_pred = sign(self._predict_base(np.array([test_data[i, :]])))
            if len(self.w) != 0:
                pred = sign(dot(all_pred, self.w))
            else:
                pred = all_pred
            if pred == -1:
                pred = 0
            if pred == 1:
                pred = 1
            y_pred.append(pred)
        return y_pred

    def calculate_err(self, all_pred, label):

        ensemble_size = all_pred.shape[1]
        err = zeros(ensemble_size)
        for i in range(ensemble_size):
            if self.err_func == 'gm':
                err[i] = 1 - gm_measure(all_pred[:, i], label)

            elif self.err_fun == 'f1':
                err[i] = 1 - f1_score(label, all_pred[:, i])

        return err

