import os
from pathlib import Path

from matplotlib import pyplot as plt

from visualization import plot_acc  # 确保你的 plot_tool 文件在此模块中

class plot_comparison:
    def __init__(self, dataset, n_class, n_round, n_pt, max_samples, interval, chunk_size, filename_list, framework):
        project_root = Path(__file__).parent.parent
        target_dir = project_root / "Results" / f"Results_{dataset}_{framework}_{n_pt}_{chunk_size}_{max_samples}"
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)

        print("Saving results to:", target_dir)
        std_alpha = 0.2
        colors = ['#E8D3C0', '#D89C7A', '#D6C38B', '#849B91', '#C2CEDC', '#686789', '#AB545A', '#9A7549', '#B0B1B6', '#7D7465']

        for idx, filename in enumerate(filename_list):
            print(f"[{idx+1}/{len(filename_list)}] Processing {filename}...")

            plot_analyzer = plot_acc.plot_tool(
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

            # ✅ 调参优化
            ani = plot_analyzer.animate_learning_curve(
                std_area=True,
                color=colors[idx % len(colors)],
                interval=10,          # 抽稀数据点，提高速度
                frame_interval=10,    # 每帧 400ms，播放更慢
                max_frames=None,         # 限制最大帧数，加速生成
                save_as='gif'          # 可改为 'mp4'
            )

            # ✅ 文件名
            outfile = f"Results_acc_{dataset}_{filename}.gif"

            # ✅ 保存动画
            ani.save(outfile, writer='pillow')

            # ✅ Notebook 中显示
            plot_analyzer.show_in_notebook(outfile, filetype='gif')

            # ✅ 释放内存
            plt.close()
