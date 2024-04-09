import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import copy
from skmultiflow.utils.data_structures import ConfusionMatrix
import warnings

warnings.filterwarnings("ignore")

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

    def prequential(self, y_true, y_pred, interval):
        confusion_matrix = ConfusionMatrix(n_targets=len(set(y_true)))
        n_all_samples = len(y_true)
        prequential_macro_f1 = []

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
            for true_label, predicted_label in zip(y_true_per, y_pred_per):
                confusion_matrix.update(int(true_label), int(predicted_label))
            # 计算每个类别的TP、FP、FN
            tp = np.diag(confusion_matrix.confusion_matrix)
            fp = confusion_matrix.confusion_matrix.sum(axis=0) - tp
            fn = confusion_matrix.confusion_matrix.sum(axis=1) - tp

            # 计算每个类别的精确度和召回率，处理分母为零的情况
            precision = np.where(tp + fp == 0, 0, tp / (tp + fp))
            recall = np.where(tp + fn == 0, 0, tp / (tp + fn))

            # 计算每个类别的F1分数，处理分母为零的情况
            f1_scores = np.where(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall))

            # 计算macro-F1
            macro_f1 = f1_scores.mean()
            prequential_macro_f1.append(macro_f1)
        return copy.deepcopy(prequential_macro_f1)
    def plot_learning_curve(self, std_area, color, interval):
        self.read_pred_result()
        self.read_true_result()

        plot_size = int(self.n_size / interval)
        f1_method = np.zeros((self.n_round, plot_size))
        for round in range(self.n_round):
            f1_method[round] = [i for i in self.prequential(self.result_true[round], self.result_method[round], interval)]
        x_axis = np.arange(plot_size, step=1)
        std_points = np.std(f1_method, axis=0)
        means_points = np.mean(f1_method, axis=0)

        plt.ylim(-0.2, 1.05)
        # plt.xlim(-10, 1050)
        plt.plot(x_axis + self.x_shift, means_points, label="%s" % self.method, color=color, linewidth=self.linewidth)

        plt.xlabel("Instances")
        plt.ylabel("macro-F1")
        # plt.title('macro-F1')

        if std_area:
            plt.fill_between(np.arange(x_axis.shape[0]) + self.x_shift, means_points - std_points,
                             means_points + std_points, interpolate=True, alpha=self.std_alpha, color=color)
        return plt
