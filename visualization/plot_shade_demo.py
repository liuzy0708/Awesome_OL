""" Demo for shade plot"""

import matplotlib.pyplot as plt

def plot_accuracy_curves():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    alpha_reg = 0.25

    # fig = plt.figure(figsize=(7, 4))

    datasets = [
        ("OSELM", [0.395, 0.663, 0.734, 0.736, 0.733, 0.746, 0.746, 0.759]),
        ("OB", [0.389, 0.64, 0.687, 0.791, 0.859, 0.871, 0.894, 0.914]),
        ("%HT", [0.653, 0.666, 0.602, 0.711, 0.734, 0.756, 0.755, 0.772]),
        ("EFDT", [0.653, 0.666, 0.602, 0.711, 0.748, 0.848, 0.918, 0.897]),
        ("ARF", [0.596, 0.687, 0.696, 0.823, 0.879, 0.911, 0.934, 0.955])
    ]

    vals = [
        ("OSELM", [0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003]),
        ("OB", [0.003, 0.003, 0.003, 0.003, 0.024, 0.003, 0.003, 0.003]),
        ("%HT", [0.003, 0.014, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003]),
        ("EFDT", [0.003, 0.003, 0.012, 0.003, 0.003, 0.003, 0.003, 0.003]),
        ("ARF", [0.003, 0.003, 0.003, 0.003, 0.003, 0.021, 0.003, 0.003]),
    ]

    colors = ['#A3142E', '#d9541a', '#edb021', '#7d2e8f', '#78ab30']

    for idx, (label, data) in enumerate(vals):
        dataset_idx = next((idx for idx, (dataset_label, _) in enumerate(datasets) if dataset_label == label), None)
        if dataset_idx is not None:
            mean_data = datasets[dataset_idx][1]
            y_upper = [mean + std for mean, std in zip(mean_data, data)]
            y_lower = [mean - std for mean, std in zip(mean_data, data)]

            plt.plot(x, mean_data, label=label)
            plt.fill_between(x, y_upper, y_lower, facecolor=colors[idx], edgecolor=colors[idx], alpha=alpha_reg)

    plt.hlines(0.959, xmin=1, xmax=8, colors='black', linestyles='dashed', label="DSLS Line")

    plt.legend(loc="lower right", fontsize=8)
    plt.xlabel('Probability Parameter of Bernoulli Variable')
    plt.ylabel('Average Accuracy')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['0.01%', '0.05%', '0.1%', '0.5%', '1%', '2%', '3%', '5%'])

    plt.show()

plot_accuracy_curves()
