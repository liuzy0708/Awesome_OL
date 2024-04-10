""" RS Strategy."""

import random

class RS_strategy():
    def __init__(self, label_ratio):
        self.label_count = 0
        self.label_ratio = label_ratio
    def evaluation(self, X, y, clf):
        if random.random() < self.label_ratio:
            isLabel = 1
            clf.partial_fit(X, y)
        else:
            isLabel = 0
        return isLabel, clf
