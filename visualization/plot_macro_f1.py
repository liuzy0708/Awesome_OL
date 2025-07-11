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

    def save_and_show_avg_confusion_matrix(self, filename_prefix):
        """
        输出 n_round 轮的平均混淆矩阵（以整数显示），并显示+保存为 CSV 和 PDF
        """
        from IPython.display import display
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from skmultiflow.utils.data_structures import ConfusionMatrix

        self.read_pred_result()
        self.read_true_result()

        matrix_sum = np.zeros((self.n_class, self.n_class), dtype=np.float64)

        # 累加所有轮的混淆矩阵
        for round_idx in range(self.n_round):
            cm = ConfusionMatrix(n_targets=self.n_class)
            y_true = self.result_true[round_idx]
            y_pred = self.result_method[round_idx]

            for true_label, pred_label in zip(y_true, y_pred):
                cm.update(int(true_label), int(pred_label))

            matrix_sum += cm.confusion_matrix

        # 求平均混淆矩阵，并四舍五入为整数
        avg_matrix = np.round(matrix_sum / self.n_round).astype(int)

        # 构造 DataFrame
        df = pd.DataFrame(avg_matrix,
                          index=[f'True_{i}' for i in range(self.n_class)],
                          columns=[f'Pred_{i}' for i in range(self.n_class)])

        # 保存 CSV
        df.to_csv(f"{filename_prefix}_avg.csv")

        # 显示表格
        display(df)

        # 画热力图 + 显示
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Average Confusion Matrix (across all rounds)")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()

        # 显示图像
        display(fig)

        # 保存 PDF
        plt.savefig(f"{filename_prefix}_avg.pdf")
        plt.close()

