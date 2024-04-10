""" US_fix Strategy."""

import numpy as np

class US_fix_strategy():
    def __init__(self, theta):
        self.theta = theta

    def evaluation(self, X, y, clf):
        proba = clf.predict_proba(X)
        max_values = np.max(proba, axis=1)
        min_values = np.min(proba, axis=1)
        diff_values = max_values - min_values

        if diff_values < self.theta:
            isLabel = 1
            clf.partial_fit(X, y)
        else:
            isLabel = 0
        return isLabel, clf