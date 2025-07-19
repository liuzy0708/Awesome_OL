""" ACDWM classifier"""
import numpy as np

from classifier.clf_ACDWM_subfile.underbagging import *
from classifier.clf_ACDWM_subfile.chunk_size_select import *
from classifier.clf_ACDWM_subfile.check_measure import *
from classifier.clf_ACDWM_subfile.chunk_based_methods import ChunkBase


class ACDWM(ChunkBase):

    def __init__(self, chunk_size=1000, theta=0.1, err_func='gm', r=1, max_ensemble_size = 10):
        ChunkBase.__init__(self)
        self.acss = ChunkSizeSelect()
        self.chunk_size = chunk_size
        self.theta = theta
        self.err_func = err_func
        self.r = r
        self.max_ensmeble_size = max_ensemble_size

        self.ensemble_size_record = array([])

    def fit(self, x_train, y_train):
        self._update_chunk(x_train, y_train)
    def _update_chunk(self, data, label):
        for i in range(len(label)):
            if label[i] == 0:
                label[i] = -1
        model = UnderBagging(r=self.r, auto_T=True)
        # model = NaiveBayes()
        model.train(data, label)
        # model.fit(data, label)
        self.ensemble.append(model)
        if len(self.ensemble) > self.max_ensmeble_size:
            del  self.ensemble[0]
        self.chunk_count += 1
        self.w = append(self.w, 1)
        if len(self.w) > self.max_ensmeble_size:
            self.w = np.delete(self.w, 0)
        all_pred = sign(self._predict_base(data))

        if self.chunk_count > 1:
            pred = dot(all_pred[:, :-1], self.w[:-1])
        else:
            pred = zeros_like(label)
        # print('pred=========', pred)
        pred = sign(pred)
        # print('sign_pred=======', pred)
        err = self.calculate_err(all_pred, label)
        self.w = (1 - err) * self.w

        remove_idx = nonzero(self.w < self.theta)[0]
        if len(remove_idx) != 0:
            for index in sorted(remove_idx, reverse=True):
                del self.ensemble[index]
            self.w = delete(self.w, remove_idx)
            self.chunk_count -= remove_idx.size

        self.ensemble_size_record = r_[self.ensemble_size_record, len(self.ensemble)]

        return pred

    def partial_fit(self, x, y):
        for i in range(x.shape[0]):
            # print(i)
            if y[i] == 0:
                y[i] = -1
            # print('x, y', np.array([x]), [int(y[i])])
            self.acss.update(np.array([x]), [int(y[i])])
            if self.acss.get_enough() == 1:
                chunk_data, chunk_label = self.acss.get_chunk()
                self._update_chunk(chunk_data, chunk_label)
