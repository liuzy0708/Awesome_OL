""" Demo of plots. """

import os
from visualization import plot_acc, plot_macro_f1

class plot_comparison:
    def __init__(self, dataset, n_class, n_round, max_samples, interval, filename_list):
        # 获取父目录的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        # 更改工作目录为父目录
        os.chdir(parent_dir)
        os.chdir('./Results/Results_{}_100_{}/'.format(dataset, max_samples))
        saving_path = './Results/Results_{}_100_{}/'.format(dataset, max_samples)
        std_alpha = 0.2

        colors = ['#E8D3C0', '#D89C7A', '#D6C38B', '#849B91', '#C2CEDC', '#686789', '#AB545A', '#9A7549', '#B0B1B6', '#7D7465']

        # filename_list = filename_list + []
        import matplotlib.pyplot as plt
        fig_acc = plt.figure(figsize=(9, 4))
        for idx in range(len(filename_list)):
            filename = filename_list[idx]
            plot_analyzer = plot_acc.plot_tool(pred_file_name=filename, true_file_name=filename, n_class=n_class, n_round=n_round, n_size=max_samples, linewidth=1.5, method="%s" % (filename), plot_interval=interval, std_alpha=std_alpha)
            plt = plot_analyzer.plot_learning_curve(std_area=True, color=colors[idx], interval=interval)
        plt.legend(fancybox=True, framealpha=0.5, loc='lower right', fontsize=9, ncol=2)
        plt.savefig('Results_acc_{}.pdf'.format(dataset))


        fig_f1 = plt.figure(figsize=(9, 4))
        for idx in range(len(filename_list)):
            filename = filename_list[idx]
            plot_analyzer = plot_macro_f1.plot_tool(pred_file_name=filename, true_file_name=filename, n_class=n_class, n_round=n_round, n_size=max_samples, linewidth=1.5, method="%s" % (filename), plot_interval=interval, std_alpha=std_alpha)
            plt = plot_analyzer.plot_learning_curve(std_area=True, color=colors[idx], interval=interval)
        plt.legend(fancybox=True, framealpha=0.5, loc='lower right', fontsize=9, ncol=2)
        plt.savefig('Results_macro_F1_{}.pdf'.format(dataset))
        plt.show()
