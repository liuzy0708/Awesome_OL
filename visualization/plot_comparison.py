import os
from pathlib import Path
from matplotlib import pyplot as plt
from visualization import plot_acc, plot_macro_f1  # plot_f1 改名为 plot_macro_f1 以示清晰

class plot_comparison:
    def __init__(self, dataset, n_class, n_round, n_pt, max_samples, interval, chunk_size, filename_list, framework):
        project_root = Path(__file__).parent.parent
        target_dir = project_root / "Results" / f"Results_{dataset}_{framework}_{n_pt}_{chunk_size}_{max_samples}"
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)

        print("Saving results to:", target_dir)
        std_alpha = 0.2
        colors = ['#E8D3C0', '#D89C7A', '#D6C38B', '#849B91', '#C2CEDC', '#686789',
                  '#AB545A', '#9A7549', '#B0B1B6', '#7D7465']

        for idx, filename in enumerate(filename_list):
            color = colors[idx % len(colors)]
            print(f"[{idx+1}/{len(filename_list)}] Processing {filename}...")

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
                color=color,
                interval=10,
                frame_interval=10,
                max_frames=None,
                save_as='gif'
            )

            gif_acc = f"Results_acc_{dataset}_{filename}.gif"
            pdf_acc = f"Results_acc_{dataset}_{filename}.pdf"
            ani_acc.save(gif_acc, writer='pillow')
            plot_analyzer_acc.show_in_notebook(gif_acc, filetype='gif')
            plot_analyzer_acc.save_final_curve_as_pdf(pdf_acc, interval, color=color)
            plt.close()

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
                color=color,
                interval=10,
                frame_interval=10,
                max_frames=None,
                save_as='gif'
            )

            gif_f1 = f"Results_f1_{dataset}_{filename}.gif"
            pdf_f1 = f"Results_f1_{dataset}_{filename}.pdf"
            ani_f1.save(gif_f1, writer='pillow')
            plot_analyzer_f1.show_in_notebook(gif_f1, filetype='gif')
            plot_analyzer_f1.save_final_curve_as_pdf(pdf_f1, interval, color=color)
            plt.close()

            print(f"    → Saving + displaying confusion matrix for {filename}")
            #plot_analyzer_f1.save_and_show_confusion_matrix(filename_prefix=f"ConfMatrix_{dataset}_{filename}")
            plot_analyzer_f1.save_and_show_avg_confusion_matrix(filename_prefix=f"ConfMatrix_{dataset}_{filename}")
            plt.close()

