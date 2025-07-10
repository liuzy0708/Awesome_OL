import os
from pathlib import Path
from matplotlib import pyplot as plt
from visualization import plot_acc, plot_macro_f1  # 确保你的plot_acc和plot_f1模块都在visualization里

class plot_comparison:
    def __init__(self, dataset, n_class, n_round, n_pt, max_samples, interval, chunk_size, filename_list, framework):
        project_root = Path(__file__).parent.parent
        target_dir = project_root / "Results" / f"Results_{dataset}_{framework}_{n_pt}_{chunk_size}_{max_samples}"
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)

        print("Saving results to:", target_dir)
        std_alpha = 0.2
        colors = ['#E8D3C0', '#D89C7A', '#D6C38B', '#849B91', '#C2CEDC', '#686789', '#AB545A', '#9A7549', '#B0B1B6', '#7D7465']

        # 先生成Accuracy动画
        for idx, filename in enumerate(filename_list):
            print(f"[ACC][{idx+1}/{len(filename_list)}] Processing {filename}...")

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
                color=colors[idx % len(colors)],
                interval=10,  # 抽稀数据点，提高速度
                frame_interval=10,  # 每帧 400ms，播放更慢
                max_frames=None,  # 限制最大帧数，加速生成
                save_as='gif'  # 可改为 'mp4'
            )

            outfile_acc = f"Results_acc_{dataset}_{filename}.gif"
            ani_acc.save(outfile_acc, writer='pillow')
            plot_analyzer_acc.show_in_notebook(outfile_acc, filetype='gif')
            plt.close()

        # 再生成Macro-F1动画
        for idx, filename in enumerate(filename_list):
            print(f"[F1][{idx+1}/{len(filename_list)}] Processing {filename}...")

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
                color=colors[idx % len(colors)],
                interval=10,  # 抽稀数据点，提高速度
                frame_interval=10,  # 每帧 400ms，播放更慢
                max_frames=None,  # 限制最大帧数，加速生成
                save_as='gif'  # 可改为 'mp4'
            )

            outfile_f1 = f"Results_f1_{dataset}_{filename}.gif"
            ani_f1.save(outfile_f1, writer='pillow')
            plot_analyzer_f1.show_in_notebook(outfile_f1, filetype='gif')
            plt.close()
