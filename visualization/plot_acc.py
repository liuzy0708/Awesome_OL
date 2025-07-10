import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.metrics import accuracy_score
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

    def animate_learning_curve(self, std_area=True, color='blue', interval=100, frame_interval=300, max_frames=None,
                               save_as='gif'):
        self.read_pred_result()
        self.read_true_result()

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
            x = x_axis[:frame + 1] + self.x_shift
            y = means_points[:frame + 1]
            line.set_data(x, y)
            ax.collections.clear()
            if std_area:
                ax.fill_between(x, y - std_points[:frame + 1], y + std_points[:frame + 1], color=color,
                                alpha=self.std_alpha)
            return line,

        ani = FuncAnimation(fig, update, frames=plot_size, init_func=init, blit=False, interval=frame_interval,cache_frame_data=False)
        plt.legend()
        return ani

    def show_in_notebook(self, path, filetype='gif'):
        if filetype == 'gif':
            display(Image(filename=path))
        elif filetype == 'mp4':
            display(Video(filename=path, embed=True))
