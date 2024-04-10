""" Demo of plots. """

import matplotlib.pyplot as plt
import os
from visualization import plot_acc, plot_macro_f1

saving_path = '..'
n_class = 2
n_round = 2
max_samples = 1000
std_alpha = 0.2
interval = 10

colors = ['#A3142E', '#d9541a', '#edb021', '#7d2e8f', '#78ab30', '#3274D9', '#FF5733', '#29A19C', '#C1440E', '#7F8C8D']
clf_name_list = ["clf_ARF", "clf_BLS"]
str_name_list = ["DSA_AI_str", "CogDQS_str"]
clf_name_list_form = ["ARF", "BLS"]
str_name_list_form = ["DSA_AI", "CogDQS"]

fig_acc = plt.figure(figsize=(9, 4))
for n_str in range(len(str_name_list)):
    for n_clf in range(len(clf_name_list)):
        filename = clf_name_list[n_clf] + '_' +  str_name_list[n_str]
        plot_analyzer = plot_acc.plot_tool(pred_file_name=filename, true_file_name=filename, n_class=n_class, n_round=n_round, n_size=max_samples, method="%s+%s" % (str_name_list_form[n_str], clf_name_list_form[n_clf]), plot_interval=interval, std_alpha=std_alpha)
        plt = plot_analyzer.plot_learning_curve(std_area=True, color=colors[int((n_str + 1) * (n_clf + 1))], interval=interval)
plt.legend(fancybox=True, framealpha=0.5, loc='lower right', fontsize=9, ncol=2)
plt.savefig(os.path.join(saving_path, 'Results_acc.pdf'))


fig_f1 = plt.figure(figsize=(9, 4))
for n_str in range(len(str_name_list)):
    for n_clf in range(len(clf_name_list)):
        filename = clf_name_list[n_clf] + '_' +  str_name_list[n_str]
        plot_analyzer = plot_macro_f1.plot_tool(pred_file_name=filename, true_file_name=filename, n_class=n_class, n_round=n_round, n_size=max_samples, method="%s+%s" % (str_name_list_form[n_str], clf_name_list_form[n_clf]), plot_interval=interval, std_alpha=std_alpha)
        plt = plot_analyzer.plot_learning_curve(std_area=True, color=colors[int((n_str + 1) * (n_clf + 1))], interval=interval)
plt.legend(fancybox=True, framealpha=0.5, loc='lower right', fontsize=9, ncol=2)
plt.savefig(os.path.join(saving_path, 'Results_macro_F1.pdf'))
plt.show()
