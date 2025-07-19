import warnings
import os
import time
from enum import Enum

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Tools.utils import *
from Tools.log_config import logger

warnings.filterwarnings("ignore")


class OL_Regression:
    def __init__(self, max_samples=1000, n_round=1, n_pt=100, dataset_name="Waveform",
                 reg_name_list=None, chunk_size=1, framework="OL_Regression", stream=None):
        # 初始化参数
        self.max_samples = max_samples
        self.n_round = n_round
        self.n_pt = n_pt
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size
        self.framework = framework
        self.stream = stream

        # 日志记录
        logger.info(f"Initializing OL_Regression with: max_samples={max_samples}, "
                    f"n_round={n_round}, dataset={dataset_name}, chunk_size={chunk_size}")

        # 回归模型列表处理
        if reg_name_list is None:
            self.reg_name_list = ["Lasso"]
        else:
            parsed_list = [name.strip() for name in reg_name_list.split(',')]
            self.reg_name_list = validate_regression_names_OL(parsed_list)

        self.num_method = len(self.reg_name_list)

        # 初始化评估指标存储结构
        self.metrics = {
            'MSE': np.zeros((self.num_method, self.n_round)),
            'MAE': np.zeros((self.num_method, self.n_round)),
            'R2': np.zeros((self.num_method, self.n_round)),
            'RMSE': np.zeros((self.num_method, self.n_round)),
            'Time': np.zeros((self.num_method, self.n_round))
        }

        # 结果目录初始化
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.result_path = os.path.join(script_dir, "Results")

        try:
            os.makedirs(self.result_path, exist_ok=True)
            logger.info(f"Results will be saved to: {self.result_path}")
        except Exception as e:
            logger.error(f"Failed to create results directory: {e}")
            raise

        self.directory_path = os.path.join(
            self.result_path,
            f"Results_{self.dataset_name}_{self.framework}_{self.n_pt}_{self.chunk_size}_{self.max_samples}"
        )

        # 尝试创建目录
        try:
            os.makedirs(self.directory_path, exist_ok=True)
            logger.info(f"Successfully created directory: {self.directory_path}")
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            raise

    def run(self):
        """执行回归实验的主流程"""
        for method_idx, reg_name in enumerate(self.reg_name_list):
            logger.info(f"Starting evaluation for {reg_name}")

            for round in range(self.n_round):
                round_start = time.time()

                # 初始化数据流和模型
                stream = get_regression_stream(self.dataset_name)
                X_pt, y_pt = get_pt(stream=stream, n_pt=self.n_pt)
                reg = self._init_regressor(reg_name, X_pt, y_pt)

                # 实时评估变量
                y_true = np.zeros(self.max_samples)
                y_pred = np.zeros(self.max_samples)
                sample_count = 0

                # 增量学习循环
                while sample_count < self.max_samples and stream.has_more_samples():
                    X, y = stream.next_sample(self.chunk_size)

                    # 预测并更新模型
                    current_pred = reg.predict(X)
                    if hasattr(reg, 'partial_fit'):
                        reg.partial_fit(X, y)
                    else:
                        logger.warning(f"{reg_name} does not support partial_fit, using batch mode")
                        reg.fit(X, y)

                    # 存储结果
                    batch_size = len(y)
                    y_true[sample_count:sample_count + batch_size] = y.ravel()
                    y_pred[sample_count:sample_count + batch_size] = current_pred.ravel()
                    sample_count += batch_size

                # 计算本轮指标
                valid_idx = np.where(y_true != 0)[0]  # 过滤未填充位置
                self._update_round_metrics(
                    method_idx, round,
                    y_true[valid_idx], y_pred[valid_idx],
                    time.time() - round_start
                )

                # 保存原始数据
                self._save_round_results(reg_name, y_true[:sample_count], y_pred[:sample_count])

            # 打印方法汇总结果
            #self._save_summary_metrics()
            self._print_method_summary(reg_name, method_idx)

    def _init_regressor(self, name, X_pt, y_pt):
        """初始化回归器实例"""
        if name == "KNN":
            knn = KNN()
            return knn.fit(X_pt, y_pt)
        elif name == "Lasso":
            from regression.reg_Lasso import Lasso
            lasso = Lasso()
            return lasso.fit(X_pt, y_pt)
        elif name == "HoeffdingTree":
            from skmultiflow.trees import HoeffdingTreeRegressor
            tree = HoeffdingTreeRegressor()
            return tree.fit(X_pt, y_pt)
        elif name == "Ridge":
            from regression.reg_Ridge import Ridge
            ridge = Ridge()
            return ridge.fit(X_pt, y_pt)
        elif name == "Linear":
            from regression.reg_Linear import Linear
            linear = Linear()
            return linear.fit(X_pt, y_pt)
        else:
            raise ValueError(f"Unsupported regressor: {name}")

    def _update_round_metrics(self, method_idx, round, y_true, y_pred, elapsed_time):
        """更新当前轮的评估指标"""
        self.metrics['MSE'][method_idx, round] = mean_squared_error(y_true, y_pred)
        self.metrics['MAE'][method_idx, round] = mean_absolute_error(y_true, y_pred)
        self.metrics['R2'][method_idx, round] = r2_score(y_true, y_pred)
        self.metrics['RMSE'][method_idx, round] = np.sqrt(self.metrics['MSE'][method_idx, round])
        self.metrics['Time'][method_idx, round] = elapsed_time

    def _save_round_results(self, reg_name, y_true, y_pred):
        """保存更丰富的预测结果到 CSV 文件"""
        import pandas as pd

        # 创建 DataFrame 存储结果
        results_df = pd.DataFrame({
            "y_true": y_true,  # 真实值
            "y_pred": y_pred,  # 预测值
            "absolute_error": np.abs(y_true - y_pred),  # 绝对误差
            "squared_error": (y_true - y_pred) ** 2,  # 平方误差
        })

        # 计算额外指标（可选）
        results_df["percentage_error"] = (results_df["absolute_error"] / (results_df["y_true"] + 1e-10)) * 100

        # 保存到 CSV
        result_csv_path = os.path.join(
            self.directory_path,
            f"Detailed_Results_{reg_name}.csv"
        )
        results_df.to_csv(result_csv_path, index=False)
        logger.info(f"Saved detailed results to {result_csv_path}")
        self.plot_true_vs_pred(y_true, y_pred, reg_name)

    def _save_summary_metrics(self):
        """保存所有回归方法的汇总指标"""
        summary_data = []
        for method_idx, reg_name in enumerate(self.reg_name_list):
            for round in range(self.n_round):
                summary_data.append({
                    "regressor": reg_name,
                    "round": round + 1,
                    "MSE": self.metrics['MSE'][method_idx, round],
                    "MAE": self.metrics['MAE'][method_idx, round],
                    "R2": self.metrics['R2'][method_idx, round],
                    "RMSE": self.metrics['RMSE'][method_idx, round],
                })

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(self.directory_path, "Summary_Metrics.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved summary metrics to {summary_csv_path}")

    def _print_method_summary(self, reg_name, method_idx):
        """打印方法评估结果摘要"""
        print(f"\n=== {reg_name} ===")
        for metric in ['MSE', 'MAE', 'R2', 'RMSE']:
            mean_val = np.mean(self.metrics[metric][method_idx])
            std_val = np.std(self.metrics[metric][method_idx])
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

        avg_time = np.mean(self.metrics['Time'][method_idx])
        print(f"Average Time: {avg_time:.4f}s per round")

    def plot_true_vs_pred(self, y_true, y_pred, reg_name):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, label=reg_name)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Perfect Prediction")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{reg_name}: True vs Predicted Values")
        plt.legend()
        plt.grid(True)

        result_png_path = os.path.join(
            self.directory_path,
            f"Visualization_Results_{reg_name}.png"
        )

        plt.savefig(result_png_path)  # 保存图片
        plt.show()


def get_regression_stream(name):
    """获取回归数据流"""
    if name == "RegressionGenerator":
        from skmultiflow.data import RegressionGenerator
        return RegressionGenerator(random_state=42)
    else:
        raise ValueError(f"Unknown dataset: {name}")


class RegressorEnum(Enum):
    """支持的回归器枚举"""
    KNN = "KNN"
    Lasso = "Lasso"
    Ridge = "Ridge"
    Linear = "Linear"
    HoeffdingTree = "HoeffdingTree"


def validate_regression_names_OL(input_list):
    """验证回归器名称有效性"""
    valid_names = [reg.value for reg in RegressorEnum]
    for name in input_list:
        if name not in valid_names:
            raise ValueError(f"Invalid regressor name: '{name}'. Valid options are: {valid_names}")
    return input_list


# 示例用法
if __name__ == "__main__":
    experiment = OL_Regression(
        reg_name_list="Linear",
        dataset_name="Waveform",
        max_samples=400,
        n_round=3
    )
    experiment.run()