import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from skmultiflow.utils.data_structures import ConfusionMatrix
import warnings
from matplotlib.animation import FuncAnimation
from IPython.display import Image, Video, display

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
            rounds = 1
        else:
            if n_all_samples % interval == 0:
                rounds = int(n_all_samples / interval)
            else:
                rounds = int(n_all_samples / interval) + 1

        for plot_i in range(rounds):
            y_true_per = y_true[plot_i * interval: min(int(plot_i + 1) * interval, n_all_samples)]
            y_pred_per = y_pred[plot_i * interval: min(int(plot_i + 1) * interval, n_all_samples)]
            for true_label, predicted_label in zip(y_true_per, y_pred_per):
                confusion_matrix.update(int(true_label), int(predicted_label))

            tp = np.diag(confusion_matrix.confusion_matrix)
            fp = confusion_matrix.confusion_matrix.sum(axis=0) - tp
            fn = confusion_matrix.confusion_matrix.sum(axis=1) - tp

            precision = np.where(tp + fp == 0, 0, tp / (tp + fp))
            recall = np.where(tp + fn == 0, 0, tp / (tp + fn))

            f1_scores = np.where(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall))

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
        x_axis = np.arange(plot_size)
        std_points = np.std(f1_method, axis=0)
        means_points = np.mean(f1_method, axis=0)

        plt.plot(x_axis + self.x_shift, means_points, label="%s" % self.method, color=color, linewidth=self.linewidth)

        plt.xlabel("Instances")
        plt.ylabel("macro-F1")

        if std_area:
            plt.fill_between(np.arange(x_axis.shape[0]) + self.x_shift, means_points - std_points,
                             means_points + std_points, interpolate=True, alpha=self.std_alpha, color=color)
        return plt


    def animate_learning_curve(self, std_area=True, color='blue', interval=100, frame_interval=300, max_frames=None,
                               save_as=None):
        """
        动态绘制macro-F1学习曲线

        参数：
        std_area: 是否显示标准差阴影区域
        color: 曲线颜色
        interval: 计算F1的时间间隔（样本数）
        frame_interval: 动画每帧间隔(ms)
        max_frames: 最大帧数限制（默认全部）
        save_as: 保存文件名（如果不为None，将保存为gif或mp4）

        返回：
        matplotlib.animation.FuncAnimation实例
        """
        self.read_pred_result()
        self.read_true_result()

        plot_size = int(self.n_size / interval)
        f1_method = np.zeros((self.n_round, plot_size))
        for r in range(self.n_round):
            f1_method[r] = self.prequential(self.result_true[r], self.result_method[r], interval)

        std_points = np.std(f1_method, axis=0)
        means_points = np.mean(f1_method, axis=0)
        x_axis = np.arange(plot_size) * interval

        if max_frames is not None:
            plot_size = min(plot_size, max_frames)

        fig, ax = plt.subplots()
        line, = ax.plot([], [], color=color, label=self.method, linewidth=self.linewidth)
        ax.set_xlim(0, self.n_size)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Instances")
        ax.set_ylabel("macro-F1")
        ax.legend()

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            x = x_axis[:frame + 1] + self.x_shift
            y = means_points[:frame + 1]
            line.set_data(x, y)
            ax.collections.clear()
            if std_area:
                ax.fill_between(x, y - std_points[:frame + 1], y + std_points[:frame + 1], color=color, alpha=self.std_alpha)
            return line,

        ani = FuncAnimation(fig, update, frames=plot_size, init_func=init, blit=False,
                            interval=frame_interval, cache_frame_data=False)

        if save_as is not None:
            if save_as.endswith('.gif'):
                ani.save(save_as, writer='pillow')

        return ani

    def show_in_notebook(self, path, filetype='gif'):
        if filetype == 'gif':
            display(Image(filename=path))
        elif filetype == 'mp4':
            display(Video(filename=path, embed=True))
