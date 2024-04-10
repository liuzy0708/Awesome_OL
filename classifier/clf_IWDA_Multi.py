""" IWDA-Multi classifier"""

import pandas as pd

from clf_IWDA_subfile.Environment import *
from sklearn.naive_bayes import GaussianNB
import copy
class IWDA_multi(object):
    def __init__(self, old_to_use=300, update_wm=500, whiten=False):
        self.weight_model_multi_1 = WeightModel(likelihood_model=MultivariateNormal(), reweighting_function=multiple_reweighting)
        self.lgbm_multi_rw_1 = ErrorDriftModel(weight_model=self.weight_model_multi_1, ml_model=GaussianNB(),
                                          name="multi_rw_oas_error",
                                          drift_detector=HDDM_W(),
                                          old_to_use=old_to_use, update_wm=update_wm, whiten=whiten)
        self.X_all = []
        self.y_all = []
    def fit(self, X_train, y_train):
        self.n_classes = len(set(y_train))
        self.n_features = X_train.shape[1]
        self.X_all = pd.DataFrame(X_train, columns=['x{}'.format(i) for i in range(1, self.n_features+1)])
        self.X_all["batch"] = 0
        self.y_all = pd.DataFrame(y_train, columns=["label"])
        self.lgbm_multi_rw_1.initial_fit(self.X_all, self.y_all)

    def predict_proba(self, x):
        x = pd.DataFrame(x, columns=['x{}'.format(i) for i in range(1, self.n_features+1)])
        x["batch"] = 0
        return self.lgbm_multi_rw_1.predict_proba(x)
    def partial_fit(self, X, y):
        X = pd.DataFrame(X, columns=['x{}'.format(i) for i in range(1, self.n_features+1)])
        X["batch"] = 0
        y = pd.DataFrame(y, columns=["label"])
        error = (y != np.argmax(self.lgbm_multi_rw_1.predict_proba(X)[0]))
        self.X_all = pd.concat((self.X_all, X), ignore_index=True)
        self.y_all = pd.concat((self.y_all, y), ignore_index=True)
        self.lgbm_multi_rw_1.partial_fit(copy.deepcopy(self.X_all), copy.deepcopy(self.y_all), error)
    def predict(self, x):
        return [np.argmax(self.predict_proba(x)[0])]