import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Image, Video, display


class plot_tool():
    def __init__(self, n_class, n_round, n_size, method, pred_file_name, true_file_name, plot_interval, std_alpha, linewidth):
        self.n_class = n_class
        self.n_round = n_round
        self.method = method
        self.n_size = n_size
        self.true_file_name = true_file_name
        self.pred_file_name = pred_file_name
        self.plot_interval = plot_interval
        self.linewidth = linewidth
        self.x_shift = 0
        self.std_alpha = std_alpha
        self.result_method = None
        self.result_true = None

    def read_pred_result(self):
        self.result_method = np.array(pd.read_csv(f'./Prediction_{self.pred_file_name}.csv', header=None))

    def read_true_result(self):
        self.result_true = np.array(pd.read_csv(f'./True_{self.true_file_name}.csv', header=None))

    def prequential(self, y_true, y_pred, interval):
        n_all = len(y_true)
        correct = 0
        total = 0
        acc = []

        for i in range(0, n_all, interval):
            y_t = y_true[i:i+interval]
            y_p = y_pred[i:i+interval]
            for yt, yp in zip(y_t, y_p):
                if yt == yp:
                    correct += 1
            total += len(y_t)
            acc.append(correct / total)
        return copy.deepcopy(acc)