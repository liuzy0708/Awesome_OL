import os
import math
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Image, display
from visualization import plot_acc, plot_macro_f1
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from log_config import logger

class plot_comparison:
    def __init__(self, dataset, n_class, n_round, n_pt, max_samples, interval, chunk_size, filename_list, framework,need_matrix):
        project_root = Path(__file__).parent.parent
        target_dir = project_root / "Results" / f"Results_{dataset}_{framework}_{n_pt}_{chunk_size}_{max_samples}"
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)

        logger.info(f"Saving results to: {target_dir}")
        print("Saving results to:", target_dir)
        std_alpha = 0.2

        all_acc_tools = []
        all_f1_tools = []
        all_labels = []
        all_confusion_matrices = []

        for idx, filename in enumerate(filename_list):
            logger.info(f"[{idx+1}/{len(filename_list)}] Processing {filename}...")
            print(f"[{idx + 1}/{len(filename_list)}] Processing {filename}...")

            # === Accuracy ===
            plot_analyzer_acc = plot_acc.plot_tool(
                pred_file_name=filename,
                true_file_name=filename,
                n_class=n_class,
                n_round=n_round,
                n_size=max_samples,
                linewidth=1.5,
                method=filename,
                plot_interval=interval,
                std_alpha=std_alpha
            )

            all_acc_tools.append(plot_analyzer_acc)


            # === macro-F1 + confusion matrix ===
            plot_analyzer_f1 = plot_macro_f1.plot_tool(
                pred_file_name=filename,
                true_file_name=filename,
                n_class=n_class,
                n_round=n_round,
                n_size=max_samples,
                linewidth=1.5,
                method=filename,
                plot_interval=interval,
                std_alpha=std_alpha
            )

            all_f1_tools.append(plot_analyzer_f1)
            all_labels.append(filename)

            if need_matrix:
                cm_df = plot_analyzer_f1.save_and_show_avg_confusion_matrix(
                    filename_prefix=f"ConfMatrix_{dataset}_{filename}")
                all_confusion_matrices.append((filename, cm_df))

        if need_matrix and all_confusion_matrices:
            import seaborn as sns
            fig_cols = 2  # 每行展示几个模型
            fig_rows = math.ceil(len(all_confusion_matrices) / fig_cols)
            fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 6, fig_rows * 5))

            # 展平 axes 数组，适配任何形状
            axes = np.array(axes).reshape(-1)

            for idx, (label, df) in enumerate(all_confusion_matrices):
                ax = axes[idx]
                sns.heatmap(df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_title(f"{label}", fontsize=14)
                ax.set_xlabel("Predicted", fontsize=12)
                ax.set_ylabel("True", fontsize=12)

            # 隐藏多余的子图（如果模型数量不是 2 的倍数）
            for j in range(len(all_confusion_matrices), len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            combined_path = f"ConfMatrix_{dataset}_all_models.png"
            plt.savefig(combined_path, dpi=300)
            display(Image(filename=combined_path))
            logger.info(f"Saved combined confusion matrix figure: {combined_path}")
            plt.close()


        if len(all_acc_tools) >= 1 and len(all_f1_tools) >= 1:
            logger.info("[ALL] Generating combined Accuracy + macro-F1 animation in subplots...")
            print("\n[ALL] Generating combined Accuracy + macro-F1 animation in subplots...")
            self.animate_multi_metric_curve(
                acc_tools=all_acc_tools,
                f1_tools=all_f1_tools,
                labels=all_labels,
                interval=10,
                frame_interval=100,
                save_path=f"Results_combined_{dataset}_all_models.gif"
            )
            display(Image(filename=f"Results_combined_{dataset}_all_models.gif"))

    def animate_multi_metric_curve(self, acc_tools, f1_tools, labels, interval, frame_interval, save_path):
        assert len(acc_tools) == len(f1_tools), "Accuracy 和 F1 工具数量不一致"

        custom_colors = [
            '#E8D3C0',  # SU-NB (浅米色)
            '#D89C7A',  # NB-Cog (橙棕色)
            '#D6C38B',  # ACDWM-Cog (浅卡其色)
            '#849B91',  # OLI2DS-Cog (灰绿色)
            '#C2CEDC',  # IWDA_PL-Cog (淡蓝色)
            '#686789',  # CPSSDS (灰蓝色)
            '#AB545A',  # DES-Cog (酒红色)
            '#9A7549',  # IWDA_Multi-Cog (棕褐色)
            '#B0B1B6',  # CADM+ (银灰色)
            '#7D7465'  # 其他未标注曲线 (深灰褐色)
        ]
        colors = [custom_colors[i % len(custom_colors)] for i in range(len(acc_tools))]

        plot_size = math.ceil(acc_tools[0].n_size / interval)
        x_axis = np.arange(plot_size) * interval

        acc_means, acc_stds, f1_means, f1_stds = [], [], [], []
        for acc_tool, f1_tool in zip(acc_tools, f1_tools):
            # Accuracy
            acc_tool.read_pred_result()
            acc_tool.read_true_result()
            acc_data = np.zeros((acc_tool.n_round, plot_size))
            for r in range(acc_tool.n_round):
                acc_data[r] = acc_tool.prequential(acc_tool.result_true[r], acc_tool.result_method[r], interval)
            acc_means.append(np.mean(acc_data, axis=0))
            acc_stds.append(np.std(acc_data, axis=0))

            # F1
            f1_tool.read_pred_result()
            f1_tool.read_true_result()
            f1_data = np.zeros((f1_tool.n_round, plot_size))
            for r in range(f1_tool.n_round):
                f1_data[r] = f1_tool.prequential(f1_tool.result_true[r], f1_tool.result_method[r], interval)
            f1_means.append(np.mean(f1_data, axis=0))
            f1_stds.append(np.std(f1_data, axis=0))

        # 子图布局
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        for ax, ylabel in zip([ax1, ax2], ['Accuracy', 'macro-F1']):
            ax.set_xlim(0, acc_tools[0].n_size)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(ylabel, fontsize=16)
            ax.tick_params(axis='both', labelsize=14)

        ax2.set_xlabel("Instances", fontsize=16)

        # 初始化线条和填充
        acc_lines, acc_fills = [], []
        f1_lines, f1_fills = [], []

        for i in range(len(acc_tools)):
            acc_line, = ax1.plot([], [], color=colors[i], label=labels[i], linewidth=2)
            acc_lines.append(acc_line)
            acc_fills.append(None)

            f1_line, = ax2.plot([], [], color=colors[i], label=labels[i], linewidth=2)
            f1_lines.append(f1_line)
            f1_fills.append(None)

        ax1.legend(fontsize=14)
        ax2.legend(fontsize=14)

        def init():
            for line in acc_lines + f1_lines:
                line.set_data([], [])
            return acc_lines + f1_lines

        def update(frame):
            for i in range(len(acc_tools)):
                x = x_axis[:frame + 1]

                # 更新 Accuracy
                y_acc = acc_means[i][:frame + 1]
                acc_lines[i].set_data(x, y_acc)
                if acc_fills[i]:
                    acc_fills[i].remove()
                acc_fills[i] = ax1.fill_between(x, y_acc - acc_stds[i][:frame + 1], y_acc + acc_stds[i][:frame + 1],
                                                color=colors[i], alpha=acc_tools[i].std_alpha)

                # 更新 macro-F1
                y_f1 = f1_means[i][:frame + 1]
                f1_lines[i].set_data(x, y_f1)
                if f1_fills[i]:
                    f1_fills[i].remove()
                f1_fills[i] = ax2.fill_between(x, y_f1 - f1_stds[i][:frame + 1], y_f1 + f1_stds[i][:frame + 1],
                                               color=colors[i], alpha=f1_tools[i].std_alpha)

            return acc_lines + f1_lines

        ani = FuncAnimation(fig, update, frames=plot_size, init_func=init,
                            blit=False, interval=frame_interval, cache_frame_data=False)
        ani.save(save_path, writer='pillow')
        plt.close()
        logger.info(f"Saved combined animation: {save_path}")
