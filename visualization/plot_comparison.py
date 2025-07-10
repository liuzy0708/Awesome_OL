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
    def __init__(self, dataset, n_class, n_round, n_pt, max_samples, interval, chunk_size, filename_list, framework):
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

            ani_acc = plot_analyzer_acc.animate_learning_curve(
                std_area=True,
                color=None,
                interval=10,
                frame_interval=100,
                max_frames=None,
                save_as='gif'
            )

            gif_acc = f"Results_acc_{dataset}_{filename}.gif"
            pdf_acc = f"Results_acc_{dataset}_{filename}.pdf"
            ani_acc.save(gif_acc, writer='pillow')
            plot_analyzer_acc.show_in_notebook(gif_acc, filetype='gif')
            plot_analyzer_acc.save_final_curve_as_pdf(pdf_acc, interval, color='black')
            plt.close()
            logger.info(f"Saved accuracy gif: {gif_acc}")
            logger.info(f"Saved accuracy pdf: {pdf_acc}")

            all_acc_tools.append(plot_analyzer_acc)
            all_labels.append(filename)

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

            ani_f1 = plot_analyzer_f1.animate_learning_curve(
                std_area=True,
                color=None,
                interval=10,
                frame_interval=10,
                max_frames=None,
                save_as='gif'
            )

            gif_f1 = f"Results_f1_{dataset}_{filename}.gif"
            pdf_f1 = f"Results_f1_{dataset}_{filename}.pdf"
            ani_f1.save(gif_f1, writer='pillow')
            plot_analyzer_f1.show_in_notebook(gif_f1, filetype='gif')
            plot_analyzer_f1.save_final_curve_as_pdf(pdf_f1, interval, color='black')
            plot_analyzer_f1.save_and_show_avg_confusion_matrix(filename_prefix=f"ConfMatrix_{dataset}_{filename}")
            plt.close()
            logger.info(f"Saved macro-F1 gif: {gif_f1}")
            logger.info(f"Saved macro-F1 pdf: {pdf_f1}")

            all_f1_tools.append(plot_analyzer_f1)

        if len(all_acc_tools) > 1:
            # === 多模型 Accuracy 联合动图 ===
            logger.info("[ALL] Generating combined multi-model Accuracy animation...")
            print("\n[ALL] Generating combined multi-model Accuracy animation...")
            self.animate_multi_model_curve(
                tools=all_acc_tools,
                labels=all_labels,
                interval=10,
                frame_interval=100,
                save_path=f"Results_acc_{dataset}_all_models.gif",
                ylabel='Accuracy'
            )
            display(Image(filename=f"Results_acc_{dataset}_all_models.gif"))
            logger.info("Combined Accuracy animation saved.")

        if len(all_f1_tools) > 1:
            # === 多模型 macro-F1 联合动图 ===
            logger.info("[ALL] Generating combined multi-model macro-F1 animation...")
            print("\n[ALL] Generating combined multi-model macro-F1 animation...")
            self.animate_multi_model_curve(
                tools=all_f1_tools,
                labels=all_labels,
                interval=10,
                frame_interval=100,
                save_path=f"Results_f1_{dataset}_all_models.gif",
                ylabel='macro-F1'
            )
            display(Image(filename=f"Results_f1_{dataset}_all_models.gif"))
            logger.info("Combined macro-F1 animation saved.")

    def animate_multi_model_curve(self, tools, labels, interval, frame_interval, save_path, ylabel):
        cmap = cm.get_cmap('tab20', len(tools))
        colors = [mcolors.to_hex(cmap(i)) for i in range(len(tools))]

        plot_size = math.ceil(tools[0].n_size / interval)
        x_axis = np.arange(plot_size) * interval

        all_means, all_stds = [], []
        for tool in tools:
            tool.read_pred_result()
            tool.read_true_result()
            acc_data = np.zeros((tool.n_round, plot_size))
            for r in range(tool.n_round):
                acc_data[r] = tool.prequential(tool.result_true[r], tool.result_method[r], interval)
            all_means.append(np.mean(acc_data, axis=0))
            all_stds.append(np.std(acc_data, axis=0))

        fig, ax = plt.subplots()
        ax.set_xlim(0, tools[0].n_size)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Instances")
        ax.set_ylabel(ylabel)

        lines = []
        fills = []

        for i in range(len(tools)):
            line, = ax.plot([], [], color=colors[i], label=labels[i], linewidth=2)
            lines.append(line)
            fills.append(None)

        plt.legend()

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i in range(len(tools)):
                x = x_axis[:frame + 1]
                y = all_means[i][:frame + 1]
                lines[i].set_data(x, y)

                if fills[i]:
                    fills[i].remove()
                fills[i] = ax.fill_between(
                    x,
                    y - all_stds[i][:frame + 1],
                    y + all_stds[i][:frame + 1],
                    color=colors[i],
                    alpha=tools[i].std_alpha
                )
            return lines

        ani = FuncAnimation(fig, update, frames=plot_size, init_func=init,
                            blit=False, interval=frame_interval, cache_frame_data=False)
        ani.save(save_path, writer='pillow')
        plt.close()
        logger.info(f"Saved combined animation: {save_path}")
