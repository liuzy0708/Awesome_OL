# Imports
import csv
import copy
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

from visualization import plot_acc, plot_macro_f1
from sklearn.metrics import accuracy_score, f1_score
from utils import para_init
from utils import get_stream, get_pt

warnings.filterwarnings("ignore")

max_samples = 30000  # The range of tested stream
n_class = 3
n_round = 3
n_ratio_max = 0.15

init_directory = os.getcwd()

for n_pt in [30, 300, 3000]:

    stream_pt = get_stream("Jiaolong")  # 2 class
    X_pt_source, y_pt_source = get_pt(stream=stream_pt, n_pt=n_pt, n_class=n_class)

    for n_anchor in [3, 5, 10]:

        for theta in [0.10, 0.15, 0.20]:

            acc_list = [[] for _ in range(n_round)]
            f1_list = [[] for _ in range(n_round)]

            result = [[] for _ in range(n_round)]
            result_pred = [[] for _ in range(n_round)]
            isLabel_collection = [[] for _ in range(n_round)]


            for round in range(n_round):


                # X_pt_source, y_pt_source = stream.next_sample(n_pt)

                para = para_init(n_class=n_class, X_pt_source=X_pt_source, y_pt_source=y_pt_source, n_ratio_max=n_ratio_max, n_anchor=n_anchor, theta=theta)

                'stream initialization'
                stream = get_stream("Jiaolong")
                X_pt, y_pt = copy.deepcopy(X_pt_source), copy.deepcopy(y_pt_source)

                'clf and str inialization'
                clf = para.get_clf("clf_ODSSBLS")
                str = para.get_str("USGQS_str")

                # Setup Hyper-parameters
                y_pred = 0
                count = 0

                #Pretrain
                clf.fit(X_pt, y_pt)

                # Train the classifier with the samples provided by the data stream
                while count < max_samples and stream.has_more_samples():

                    X, y = stream.next_sample()

                    y_pred = clf.predict(X)

                    result[round] = result[round] + y.tolist()
                    result_pred[round] = result_pred[round] + [y_pred[0]]

                    if int(count * n_ratio_max) >= sum(isLabel_collection[round]):
                        isLabel, clf = str.evaluation(X, y, clf)
                        if isLabel == 0 and clf.__class__.__name__ in ['OSSBLS', 'ODSSBLS', "ISSBLS", "SOSELM"]:
                            clf.partial_fit(X, y, label_flag=0)

                        isLabel_collection[round] = isLabel_collection[round] + [isLabel]

                    count += 1

                acc_list[round] = accuracy_score(result[round], result_pred[round])
                f1_list[round] = f1_score(result[round], result_pred[round], labels=list(range(0, n_class)), average='macro')

            #print((isLabel_collection))
            print("\nAccuracy %d + %d + %f: %.3f $\pm$ %.3f" % (n_pt, n_anchor, theta, np.mean(acc_list), np.std(acc_list)))
            print("macro-F1 %d + %d + %f: %.3f $\pm$ %.3f" % (n_pt, n_anchor, theta, np.mean(f1_list), np.std(f1_list)))
            print("Annotation Rate %d + %d + %f: %.4f \n" % (n_pt, n_anchor, theta, np.sum(np.sum(isLabel_collection)) / (max_samples * n_round)))


            #Result Record
            directory_path = "./Results_para/"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            os.chdir(directory_path)


            result_pred_name = 'Prediction_%d_%d_%f.csv' % (n_pt, n_anchor, theta)
            result_true_name = 'True_%d_%d_%f.csv' % (n_pt, n_anchor, theta)
            with open(result_pred_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                for sublist in result_pred:
                    writer.writerow(sublist)
            with open(result_true_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                for sublist in result:
                    writer.writerow(sublist)

            os.chdir(os.path.dirname(os.getcwd()))