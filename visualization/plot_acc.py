""" plot for accuracy."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.metrics import accuracy_score

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
        result_method = np.array(pd.read_csv('./Prediction_%s.csv' % self.pred_file_name, header=None))
        self.result_method = result_method

    def read_true_result(self):
        result_true = np.array(pd.read_csv('./True_%s.csv' % self.true_file_name, header=None))
        self.result_true = result_true

    def update(self, score_prev, round, n_new, interval):
        score = (n_new - 1) * interval * score_prev + accuracy_score(self.result_method[round, (n_new - 1)  * interval :n_new * interval], self.result_true[round, (n_new - 1) * interval: n_new * interval]) * interval
        n_new = n_new * interval
        metric_new = score / n_new
        return metric_new

    def prequential(self, y_true, y_pred, interval):
        n_all_samples = len(y_true)
        n_now_samples = 0
        n_correct = 0
        prequential_acc = []

        if interval >= n_all_samples:
            round = 1
        else:
            if n_all_samples % interval == 0:
                round = int(n_all_samples / interval)
            else:
                round = int(n_all_samples / interval) + 1

        for plot_interval in range(round):
            y_true_per = y_true[plot_interval * interval: min(int(plot_interval + 1) * interval, n_all_samples + 1)]
            y_pred_per = y_pred[plot_interval * interval: min(int(plot_interval + 1) * interval, n_all_samples + 1)]
            for i in range(len(y_true_per)):
                if y_true_per[i] == y_pred_per[i]:
                    n_correct += 1
            n_now_samples += len(y_true_per)
            acc_now = n_correct / n_now_samples
            prequential_acc.append(acc_now)
        return copy.deepcopy(prequential_acc)



    def plot_learning_curve(self, std_area, color, interval):
        self.read_pred_result()
        self.read_true_result()

        plot_size = math.ceil(self.n_size / interval)
        acc_method = np.zeros((self.n_round, plot_size))
        for round in range(self.n_round):
            acc_method[round] = [i for i in self.prequential(self.result_method[round], self.result_true[round], interval)]
        x_axis = np.arange(plot_size, step=1)
        std_points = np.std(acc_method, axis=0)
        means_points = np.mean(acc_method, axis=0)

        # plt.ylim(-0.2, 1.05)
        # plt.xlim(-10, 1050)

        plt.plot(x_axis + self.x_shift, means_points, label="%s" % self.method, color=color, linewidth=self.linewidth)
        plt.xlabel("Instances")
        plt.ylabel("Accuracy")
        # plt.title('')

        if std_area:
            plt.fill_between(np.arange(x_axis.shape[0]) + self.x_shift, means_points - std_points,
                             means_points + std_points, interpolate=True, alpha=self.std_alpha, color=color)
        return plt
