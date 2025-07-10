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

    def animate_learning_curve(self, std_area=True, color='black', interval=100, frame_interval=300, max_frames=None,
                               save_as='gif'):
        self.read_pred_result()
        self.read_true_result()

        fill = None  # 添加这个变量用于缓存阴影对象

        plot_size = math.ceil(self.n_size / interval)
        acc_method = np.zeros((self.n_round, plot_size))
        for r in range(self.n_round):
            acc_method[r] = self.prequential(self.result_method[r], self.result_true[r], interval)

        std_points = np.std(acc_method, axis=0)
        means_points = np.mean(acc_method, axis=0)
        # 修改这里：x_axis 乘以 interval 来反映实际样本数量
        x_axis = np.arange(plot_size) * interval

        if max_frames is not None:
            plot_size = min(plot_size, max_frames)

        fig, ax = plt.subplots()
        line, = ax.plot([], [], color=color, label=self.method, linewidth=self.linewidth)
        # 修改这里：xlim 上限乘以 interval
        ax.set_xlim(0, self.n_size)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Instances")
        ax.set_ylabel("Accuracy")

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            nonlocal fill
            x = x_axis[:frame + 1] + self.x_shift
            y = means_points[:frame + 1]
            line.set_data(x, y)

            if fill:
                fill.remove()  # 删除上一帧的阴影

            if std_area:
                fill = ax.fill_between(
                    x, y - std_points[:frame + 1], y + std_points[:frame + 1],
                    color=color, alpha=self.std_alpha
                )

            return line,

        ani = FuncAnimation(fig, update, frames=plot_size, init_func=init, blit=False, interval=frame_interval,cache_frame_data=False)
        plt.legend()
        return ani

    def save_final_curve_as_pdf(self, filename, interval, color='black'):
        self.read_pred_result()
        self.read_true_result()

        x_axis = np.arange(int(self.n_size / interval)) * interval
        acc_data = np.zeros((self.n_round, len(x_axis)))
        for r in range(self.n_round):
            acc_data[r] = self.prequential(self.result_method[r], self.result_true[r], interval)

        mean_curve = np.mean(acc_data, axis=0)
        std_curve = np.std(acc_data, axis=0)

        fig, ax = plt.subplots()
        ax.plot(x_axis, mean_curve, color=color, label='Accuracy', linewidth=self.linewidth)
        ax.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=self.std_alpha, color=color)
        ax.set_xlim(0, self.n_size)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Instances")
        ax.set_ylabel("Accuracy")
        ax.legend()
        plt.savefig(filename, format='pdf')
        plt.close()

    def show_in_notebook(self, path, filetype='gif'):
        if filetype == 'gif':
            display(Image(filename=path))
        elif filetype == 'mp4':
            display(Video(filename=path, embed=True))
