import csv
import copy
import numpy as np
import warnings
import os
import time
from visualization import plot_acc, plot_macro_f1
from sklearn.metrics import accuracy_score, f1_score
from utils import para_init
from utils import get_stream, get_pt
from visualization.plot_comparison import plot_comparison
warnings.filterwarnings("ignore")

#settings
max_samples = 100  # The range of tested stream
n_round = 3   #Number of run round
n_pt = 100    #Number of train samples
n_ratio_max = 0.30  #Annotation ratio
chunk_size = 20
query_size = int(chunk_size * n_ratio_max)
dataset_name = "SEA"
clf_name_list = ["BLS"]
str_name_list = ["DMI_DD"]

num_str = len(str_name_list)
num_clf = len(clf_name_list)

acc_list = [[[[] for _ in range(n_round)] for _ in range(num_clf)] for _ in range(num_str)]
f1_list = [[[[] for _ in range(n_round)] for _ in range(num_clf)] for _ in range(num_str)]

result_path = "./Results/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

#Result Record
directory_path = "./Results/Results_%s_%d_%d_%d/" % (dataset_name, n_pt, chunk_size, max_samples)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


for n_clf in range(len(clf_name_list)):

    clf_name = clf_name_list[n_clf]
    para_clf = para_init()

    for n_str in range(len(str_name_list)):

        n_annotation_list = []
        y_pred_all = []
        y_true_all = []
        t1 = time.time()
        print("{} + {}".format(clf_name_list[n_clf], str_name_list[n_str]))

        for round in range(n_round):
            print('round:', round)
            clf = para_clf.get_clf("clf_" + clf_name)
            y_pred_list = []
            y_true_list = []

            'stream initialization'
            stream = get_stream(dataset_name)
            X_pt_source, y_pt_source = get_pt(stream=stream, n_pt=n_pt)
            para_str = para_init(n_class=stream.n_classes, X_pt_source=X_pt_source, y_pt_source=y_pt_source, n_ratio_max=n_ratio_max, clf=clf, query_size=query_size, chunk_size=chunk_size)

            str_name = str_name_list[n_str]
            str = para_str.get_str(str_name)

            # Setup Hyper-parameters
            count = 0
            n_annotation = 0

            clf.fit(X_pt_source, y_pt_source)

            # Train the classifier with the samples provided by the data stream
            while count < max_samples and stream.has_more_samples():

                count += chunk_size
                X, y = stream.next_sample(chunk_size)
                y_pred = clf.predict(X)

                y_pred_list = y_pred_list + y_pred.tolist()
                y_true_list = y_true_list + y.tolist()

                clf = str.evaluation(X, y, clf)

                n_annotation += query_size

                print("n_annotation", n_annotation)
                print("count", count)

            n_annotation_list.append(n_annotation / max_samples)
            acc_list[n_str][n_clf][round] = accuracy_score(y_true_list, y_pred_list)
            f1_list[n_str][n_clf][round] = f1_score(y_true_list, y_pred_list, labels=list(range(0, stream.n_classes)), average='macro')

            y_pred_all = y_pred_all + y_pred_list
            y_true_all = y_true_all + y_true_list

        t2 = time.time()

        result_pred = np.array(y_pred_all).reshape(n_round, max_samples)
        result_true = np.array(y_true_all).reshape(n_round, max_samples)

        result_pred_name = './Results/Results_%s_%d_%d_%d/Prediction_%s_%s.csv' % (dataset_name, n_pt, chunk_size, max_samples, clf_name_list[n_clf], str_name_list[n_str])
        result_true_name = './Results/Results_%s_%d_%d_%d/True_%s_%s.csv' % (dataset_name, n_pt, chunk_size, max_samples, clf_name_list[n_clf], str_name_list[n_str])

        with open(result_pred_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result_pred)
        with open(result_true_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result_true)

        print("\nAccuracy %s + %s: %.3f ± %.3f" % (clf_name_list[n_clf], str_name_list[n_str], np.mean(acc_list[n_str][n_clf]), np.std(acc_list[n_str][n_clf])))
        print("macro-F1 %s + %s: %.3f ± %.3f" % (clf_name_list[n_clf], str_name_list[n_str], np.mean(f1_list[n_str][n_clf]), np.std(f1_list[n_str][n_clf])))
        print("Annotation Rate %s + %s: %.4f" % (clf_name_list[n_clf], str_name_list[n_str], np.mean(n_annotation_list)))
        print("Average Time %s + %s: %.4f s\n" % (clf_name_list[n_clf], str_name_list[n_str], (t2 - t1) / n_round))


filename_list = []
for n_str in range(len(str_name_list)):
    for n_clf in range(len(clf_name_list)):
        filename_list = filename_list + [clf_name_list[n_clf] + '_' + str_name_list[n_str]]
plot_comparison(dataset=dataset_name, n_class=stream.n_classes, n_round=n_round, max_samples=max_samples, interval=1, chunk_size=chunk_size, filename_list=filename_list, n_pt=n_pt)


