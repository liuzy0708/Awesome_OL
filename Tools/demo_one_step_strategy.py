import csv
import numpy as np
import warnings
import os
import time
from sklearn.metrics import accuracy_score, f1_score
from Tools.utils import para_init
from Tools.utils import get_stream, get_pt
from visualization.plot_comparison import plot_comparison
warnings.filterwarnings("ignore")

#settings
max_samples = 100  # The range of tested stream
n_round = 3   #Number of run round
n_pt = 100    #Number of train samples
n_ratio_max = 1  #Annotation ratio
chunk_size = 1
framework = "ONE-STEP"
theta = 0.15  #Parameter for US
dataset_name = "Waveform"
method_name_list = ["ROALE_DI"]
num_method = len(method_name_list)

acc_list = [[[] for _ in range(n_round)] for _ in range(num_method)]
f1_list = [[[] for _ in range(n_round)] for _ in range(num_method)]

result_path = "../Results/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
#Result Record
directory_path = "./Results/Results_%s_%s_%d_%d_%d/" % (dataset_name, framework, n_pt, chunk_size, max_samples)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

for n_method in range(len(method_name_list)):

    method_name = method_name_list[n_method]
    n_annotation_list = []
    y_pred_all = []
    y_true_all = []
    t1 = time.time()
    print("method:{}".format(method_name_list[n_method]))
    for round in range(n_round):
        print('round:', round)

        y_pred_list = []
        y_true_list = []
        'stream initialization'
        stream = get_stream(dataset_name)
        X_pt_source, y_pt_source = get_pt(stream=stream, n_pt=n_pt)
        para_method = para_init(X_pt_source, y_pt_source, n_class=stream.n_classes)
        method = para_method.get_method(method_name_list[n_method])

        # Setup Hyper-parameters
        count = 0
        n_annotation = 0

        # Train the classifier with the samples provided by the data stream
        while count < max_samples and stream.has_more_samples():
            count += chunk_size


            X, y = stream.next_sample(chunk_size)

            y_pred = method.predict(X)
            for i in range(len(y_pred)):
                y_pred_list.append(y_pred[i])
                y_true_list.append(y[i])
            method.evaluation(X, y)

        n_annotation_list.append(method.n_annotation / max_samples)
        acc_list[n_method][round] = accuracy_score(y_true_list, y_pred_list)
        f1_list[n_method][round] = f1_score(y_true_list, y_pred_list, labels=list(range(0, stream.n_classes)), average='macro')

        y_pred_all = y_pred_all + y_pred_list
        y_true_all = y_true_all + y_true_list

    t2 = time.time()
    # print(y_pred_all)
    result_pred = np.array(y_pred_all).reshape(n_round, max_samples)
    result_true = np.array(y_true_all).reshape(n_round, max_samples)

    result_pred_name = './Results/Results_%s_%s_%d_%d_%d/Prediction_%s.csv' % (dataset_name, framework, n_pt, chunk_size, max_samples, method_name_list[n_method])
    result_true_name = './Results/Results_%s_%s_%d_%d_%d/True_%s.csv' % (dataset_name, framework, n_pt, chunk_size, max_samples, method_name_list[n_method])

    with open(result_pred_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result_pred)
    with open(result_true_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result_true)

    print("\nAccuracy %s: %.3f ± %.3f" % (method_name_list[n_method], np.mean(acc_list[n_method]), np.std(acc_list[n_method])))
    print("macro-F1 %s: %.3f ± %.3f" % (method_name_list[n_method], np.mean(f1_list[n_method]), np.std(f1_list[n_method])))
    print("Annotation Rate %s: %.4f" % (method_name_list[n_method], np.mean(n_annotation_list)))
    print("Average Time %s: %.4f s\n" % (method_name_list[n_method], (t2 - t1) / n_round))

plot_comparison(dataset=dataset_name, n_class=stream.n_classes, n_round=n_round, max_samples=max_samples, interval=1, chunk_size=chunk_size, filename_list=method_name_list, n_pt=n_pt, framework=framework)


