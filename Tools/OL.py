import warnings
import os
import time
from enum import Enum
from sklearn.metrics import accuracy_score, f1_score

from Tools.utils import *
from visualization.plot_comparison import plot_comparison
from log_config import logger
warnings.filterwarnings("ignore")


class OL:
    def __init__(self, max_samples=1000, n_round=3, n_pt=100, dataset_name="Waveform",
                 clf_name_list=None, chunk_size=1, framework="OL",stream=None):
        self.max_samples = max_samples
        logger.info(f"Max samples: {max_samples}")
        self.n_round = n_round
        logger.info(f"n_round: {n_round}")
        self.n_pt = n_pt
        logger.info(f"n_pt: {n_pt}")
        self.dataset_name = dataset_name
        logger.info(f"dataset_name: {dataset_name}")
        self.chunk_size = chunk_size
        logger.info(f"chunk_size: {chunk_size}")
        self.framework = framework
        logger.info(f"framework: {framework}")
        self.stream = stream
        logger.info(f"stream: {stream}")

        if clf_name_list is None:
            self.clf_name_list = ["BLS"]
        else:
            parsed_list = [name.strip() for name in clf_name_list.split(',')]
            self.clf_name_list = validate_classifier_names_OL(parsed_list)
        logger.info(f"clf_name_list: {self.clf_name_list}")

        self.num_method = len(self.clf_name_list)
        logger.info(f"num_method: {self.num_method}")
        self.acc_list = [[[] for _ in range(n_round)] for _ in range(self.num_method)]
        self.f1_list = [[[] for _ in range(n_round)] for _ in range(self.num_method)]

        self.result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results")
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.directory_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"Results","Results_%s_%s_%d_%d_%d" % (
            self.dataset_name, self.framework,
            self.n_pt, self.chunk_size, self.max_samples
        ))

        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
        logger.info(f"directory_path: {self.directory_path}")

    def run(self):
        for n_clf in range(len(self.clf_name_list)):
            clf_name = self.clf_name_list[n_clf]
            y_pred_all = []
            y_true_all = []
            t1 = time.time()
            print("method:{}".format(clf_name))

            for round in range(self.n_round):
                print('round:', round + 1)

                y_pred_list = []
                y_true_list = []
                # stream initialization
                self.stream = get_stream(self.dataset_name)
                X_pt_source, y_pt_source = get_pt(stream=self.stream, n_pt=self.n_pt)
                para_clf = para_init(n_class=self.stream.n_classes, X_pt_source=X_pt_source, y_pt_source=y_pt_source)
                clf = para_clf.get_clf("clf_" + self.clf_name_list[n_clf])

                # Setup counter
                count = 0

                clf.fit(X_pt_source, y_pt_source)

                # Train the classifier with the samples provided by the data stream
                while count < self.max_samples and self.stream.has_more_samples():
                    count += self.chunk_size
                    X, y = self.stream.next_sample(self.chunk_size)

                    # Predict
                    y_pred = clf.predict(X)
                    for i in range(len(y_pred)):
                        y_pred_list.append(y_pred[i])
                        y_true_list.append(y[i])

                    # Fully supervised learning - update model with true labels
                    clf.partial_fit(X, y)

                self.acc_list[n_clf][round] = accuracy_score(y_true_list, y_pred_list)
                self.f1_list[n_clf][round] = f1_score(y_true_list, y_pred_list,
                                                      labels=list(range(0, self.stream.n_classes)), average='macro')

                y_pred_all += y_pred_list
                y_true_all += y_true_list

            t2 = time.time()

            result_pred = np.array(y_pred_all).reshape(self.n_round, self.max_samples)
            result_true = np.array(y_true_all).reshape(self.n_round, self.max_samples)

            result_pred_name = os.path.join(
                self.directory_path,
                "Prediction_%s.csv" % self.clf_name_list[n_clf]
            )

            result_true_name = os.path.join(
                self.directory_path,
                "True_%s.csv" % self.clf_name_list[n_clf]
            )

            with open(result_pred_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_pred)
            with open(result_true_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(result_true)

            print("\nAccuracy %s: %.3f ± %.3f" % (self.clf_name_list[n_clf],
                                                  np.mean(self.acc_list[n_clf]), np.std(self.acc_list[n_clf])))
            print("macro-F1 %s: %.3f ± %.3f" % (self.clf_name_list[n_clf],
                                                np.mean(self.f1_list[n_clf]), np.std(self.f1_list[n_clf])))
            print("Average Time %s: %.4f s\n" % (self.clf_name_list[n_clf], (t2 - t1) / self.n_round))

    def show(self,need_matrix):
        plot_comparison(dataset=self.dataset_name, n_class=self.stream.n_classes, n_round=self.n_round,
                            max_samples=self.max_samples, interval=1, chunk_size=self.chunk_size,
                            filename_list=self.clf_name_list, n_pt=self.n_pt, framework=self.framework, need_matrix=need_matrix)



class ClassifierEnum(Enum):
    ARF = "ARF"
    LB = "LB"
    OB = "OB"
    OBADWIN = "OBADWIN"
    DWM = "DWM"
    OOB = "OOB"
    SRP = "SRP"
    AdaC2 = "AdaC2"
    QRBLS = "QRBLS"
    BLS = "BLS"
    OSSBLS = "OSSBLS"
    ISSBLS = "ISSBLS"
    SOSELM = "SOSELM"
    NB = "NB"
    DES = "DES"
    DES_5 = "DES_5"
    ACDWM = "ACDWM"
    OLI2DS = "OLI2DS"
    MLP_OGD = "MLP_OGD"
    MLP_OMD = "MLP_OMD"


def validate_classifier_names_OL(input_list):
    valid_names = [clf.value for clf in ClassifierEnum]
    for name in input_list:
        if name not in valid_names:
            raise ValueError(f"Invalid classifier name: '{name}'. Valid options are: {valid_names}")
    return input_list
