import numpy as np

class US_var_strategy():
    def __init__(self, theta):
        self.theta = theta
        self.var = 0.01

    def evaluation(self, X, y, clf):
        proba = clf.predict_proba(X)
        max_values = np.max(proba, axis=1)
        min_values = np.min(proba, axis=1)
        diff_values = max_values - min_values

        if diff_values < self.theta:
            isLabel = 1
            clf.partial_fit(X, y)
            self.theta = self.theta * (1 - self.var)
        else:
            isLabel = 0
            self.theta = self.theta * (1 + self.var)
        return isLabel, clf