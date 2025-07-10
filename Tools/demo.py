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
max_samples = 1000  # The range of tested stream
n_round = 3   #Number of run round
n_pt = 100    #Number of train samples
chunk_size = 1
dataset_name = "Waveform"
clf_name_list = ["BLS"]  # You can replace the supervised classifier you want to test
num_method = len(clf_name_list)
framework = "OL"
acc_list = [[[] for _ in range(n_round)] for _ in range(num_method)]
f1_list = [[[] for _ in range(n_round)] for _ in range(num_method)]

result_path = "../Results/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
#Result Record
directory_path = "./Results/Results_%s_%s_%d_%d_%d/" % (dataset_name, framework, n_pt, chunk_size, max_samples)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

for n_clf in range(len(clf_name_list)):
    clf_name = clf_name_list[n_clf]
    y_pred_all = []
    y_true_all = []
    t1 = time.time()
    print("method:{}".format(clf_name_list[n_clf]))
    
    for round in range(n_round):
        print('round:', round)

        y_pred_list = []
        y_true_list = []
        'stream initialization'
        stream = get_stream(dataset_name)
        X_pt_source, y_pt_source = get_pt(stream=stream, n_pt=n_pt)
        para_clf = para_init(n_class=stream.n_classes, X_pt_source=X_pt_source, y_pt_source=y_pt_source)
        clf = para_clf.get_clf("clf_" + clf_name_list[n_clf])

        # Setup counter
        count = 0

        clf.fit(X_pt_source, y_pt_source)

        # Train the classifier with the samples provided by the data stream
        while count < max_samples and stream.has_more_samples():
            count += chunk_size
            X, y = stream.next_sample(chunk_size)
            
            # 预测
            y_pred = clf.predict(X)
            for i in range(len(y_pred)):
                y_pred_list.append(y_pred[i])
                y_true_list.append(y[i])
            
            # 全监督学习 - 直接用真实标签更新模型
            clf.partial_fit(X, y)

        acc_list[n_clf][round] = accuracy_score(y_true_list, y_pred_list)
        f1_list[n_clf][round] = f1_score(y_true_list, y_pred_list, labels=list(range(0, stream.n_classes)), average='macro')

        y_pred_all = y_pred_all + y_pred_list
        y_true_all = y_true_all + y_true_list

    t2 = time.time()
    
    result_pred = np.array(y_pred_all).reshape(n_round, max_samples)
    result_true = np.array(y_true_all).reshape(n_round, max_samples)

    result_pred_name = './Results/Results_%s_%s_%d_%d_%d/Prediction_%s.csv' % (dataset_name, framework, n_pt, chunk_size, max_samples, clf_name_list[n_clf])
    result_true_name = './Results/Results_%s_%s_%d_%d_%d/True_%s.csv' % (dataset_name, framework, n_pt, chunk_size, max_samples, clf_name_list[n_clf])

    with open(result_pred_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result_pred)
    with open(result_true_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result_true)

    print("\nAccuracy %s: %.3f ± %.3f" % (clf_name_list[n_clf], np.mean(acc_list[n_clf]), np.std(acc_list[n_clf])))
    print("macro-F1 %s: %.3f ± %.3f" % (clf_name_list[n_clf], np.mean(f1_list[n_clf]), np.std(f1_list[n_clf])))
    print("Average Time %s: %.4f s\n" % (clf_name_list[n_clf], (t2 - t1) / n_round))

plot_comparison(dataset=dataset_name, n_class=stream.n_classes, n_round=n_round, max_samples=max_samples, interval=1, chunk_size=1, filename_list=clf_name_list, n_pt=n_pt, framework=framework)


