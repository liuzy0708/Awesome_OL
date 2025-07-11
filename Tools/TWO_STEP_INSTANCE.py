import warnings
import os
import time
from enum import Enum
from sklearn.metrics import accuracy_score, f1_score
from Tools.utils import *
from visualization.plot_comparison import plot_comparison
from log_config import logger
warnings.filterwarnings("ignore")

class Two_Step_Instance:
    def __init__(self, max_samples=1000, n_round=3, n_pt=100,
                 dataset_name="Waveform", clf_name_list=None, str_name_list="RS"):
        self.stream = None
        self.max_samples = max_samples
        logger.info(f"max_samples: {max_samples}")
        self.n_round = n_round
        logger.info(f"n_round: {n_round}")
        self.n_pt = n_pt
        logger.info(f"n_pt: {n_pt}")
        self.n_ratio_max = 1
        logger.info(f"n_ratio_max: {self.n_ratio_max}")
        self.chunk_size = 1
        logger.info(f"chunk_size: {self.chunk_size}")
        self.dataset_name = dataset_name
        logger.info(f"dataset_name: {self.dataset_name}")

        if clf_name_list is None:
            self.clf_name_list = ['BLS']
        else:
            parsed_list = [name.strip() for name in clf_name_list.split(',')]
            self.clf_name_list = validate_classifier_names_TWO_STEP_INSTANCE(parsed_list)
        logger.info(f"clf_name_list: {self.clf_name_list}")

        if str_name_list is None:
            self.str_name_list = ['DMI_DD']
        else:
            parsed_list = [name.strip() for name in str_name_list.split(',')]
            self.str_name_list = validate_strategies_names_TWO_STEP_INSTANCE(parsed_list)
        logger.info(f"str_name_list: {self.str_name_list}")

        self.framework = "TWO-STEP-INSTANCE"
        logger.info(f"framework: {self.framework}")

        self.num_str = len(str_name_list)
        logger.info(f"num_str: {self.num_str}")
        self.num_clf = len(clf_name_list)
        logger.info(f"num_clf: {self.num_clf}")

        self.acc_list = [[[[] for _ in range(n_round)] for _ in range(self.num_clf)] for _ in range(self.num_str)]
        self.f1_list = [[[[] for _ in range(n_round)] for _ in range(self.num_clf)] for _ in range(self.num_str)]

        self.result_path = "./Results/"
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        #logger.info(f"result_path: {self.result_path}")

        self.directory_path = "./Results/Results_%s_%s_%d_%d_%d/" % (
            self.dataset_name, self.framework, self.n_pt, self.chunk_size, self.max_samples
        )
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)
        logger.info(f"directory_path: {self.directory_path}")

    def run(self):
        for n_clf in range(len(self.clf_name_list)):

            clf_name = self.clf_name_list[n_clf]
            para_clf = para_init()

            for n_str in range(len(self.str_name_list)):

                n_annotation_list = []
                y_pred_all = []
                y_true_all = []
                t1 = time.time()
                print("{} + {}".format(self.clf_name_list[n_clf], self.str_name_list[n_str]))

                for round in range(self.n_round):
                    print('round:', round)
                    clf = para_clf.get_clf("clf_" + clf_name)
                    y_pred_list = []
                    y_true_list = []

                    # stream initialization
                    self.stream = get_stream(self.dataset_name)
                    X_pt_source, y_pt_source = get_pt(stream=self.stream, n_pt=self.n_pt)
                    para_str = para_init(n_class=self.stream.n_classes, X_pt_source=X_pt_source,
                                         y_pt_source=y_pt_source, n_ratio_max=self.n_ratio_max)

                    str_name = self.str_name_list[n_str]
                    str = para_str.get_str(str_name)

                    # Setup Hyper-parameters
                    count = 0
                    n_annotation = 0

                    clf.fit(X_pt_source, y_pt_source)

                    while count < self.max_samples and self.stream.has_more_samples():
                        count += self.chunk_size
                        X, y = self.stream.next_sample(self.chunk_size)
                        y_pred = clf.predict(X)
                        for i in range(len(y_pred)):
                            y_pred_list.append(y_pred[i])
                            y_true_list.append(y[i])

                        if self.n_ratio_max >= (n_annotation / count):
                            isLabel, clf = str.evaluation(X, y, clf)
                            if isLabel == 1:
                                n_annotation += 1
                            elif isLabel == 0 and clf.__class__.__name__ in ['OSSBLS', 'ISSBLS', 'SOSELM']:
                                clf.partial_fit(X, y, label_flag=isLabel)

                    n_annotation_list.append(n_annotation / self.max_samples)
                    self.acc_list[n_str][n_clf][round] = accuracy_score(y_true_list, y_pred_list)
                    self.f1_list[n_str][n_clf][round] = f1_score(
                        y_true_list, y_pred_list,
                        labels=list(range(0, self.stream.n_classes)),
                        average='macro'
                    )

                    y_pred_all += y_pred_list
                    y_true_all += y_true_list

                t2 = time.time()

                result_pred = np.array(y_pred_all).reshape(self.n_round, self.max_samples)
                result_true = np.array(y_true_all).reshape(self.n_round, self.max_samples)

                result_pred_name = os.path.join(
                    self.directory_path,
                    "Prediction_%s_%s.csv" % (self.clf_name_list[n_clf], self.str_name_list[n_str])
                )
                result_true_name = os.path.join(
                    self.directory_path,
                    "True_%s_%s.csv" % (self.clf_name_list[n_clf], self.str_name_list[n_str])
                )

                with open(result_pred_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(result_pred)
                with open(result_true_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(result_true)

                print("\nAccuracy %s + %s: %.3f ± %.3f" % (
                    self.clf_name_list[n_clf], self.str_name_list[n_str],
                    np.mean(self.acc_list[n_str][n_clf]), np.std(self.acc_list[n_str][n_clf])))
                print("macro-F1 %s + %s: %.3f ± %.3f" % (
                    self.clf_name_list[n_clf], self.str_name_list[n_str],
                    np.mean(self.f1_list[n_str][n_clf]), np.std(self.f1_list[n_str][n_clf])))
                print("Annotation Rate %s + %s: %.4f" % (
                    self.clf_name_list[n_clf], self.str_name_list[n_str], np.mean(n_annotation_list)))
                print("Average Time %s + %s: %.4f s\n" % (
                    self.clf_name_list[n_clf], self.str_name_list[n_str], (t2 - t1) / self.n_round))


    def show(self,need_matrix):
        # plot
        filename_list = []
        for n_str in range(len(self.str_name_list)):
            for n_clf in range(len(self.clf_name_list)):
                filename_list.append(self.clf_name_list[n_clf] + '_' + self.str_name_list[n_str])

        stream = get_stream(self.dataset_name)  # Re-acquire for class count
        plot_comparison(dataset=self.dataset_name, n_class=stream.n_classes, n_round=self.n_round,
                            max_samples=self.max_samples, interval=1, chunk_size=self.chunk_size,
                            filename_list=filename_list, n_pt=self.n_pt, framework=self.framework, need_matrix=need_matrix)

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


def validate_classifier_names_TWO_STEP_INSTANCE(input_list):
    valid_names = [clf.value for clf in ClassifierEnum]
    for name in input_list:
        if name not in valid_names:
            raise ValueError(f"Invalid classifier name: '{name}'. Valid options are: {valid_names}")
    return input_list


class StrategyEnum(Enum):
    DSA_AI = "DSA_AI"
    Supervised = "Supervised"
    MTSGQS = "MTSGQS"
    US_fix = "US_fix"
    US_var = "US_var"
    CogDQS = "CogDQS"
    RS = "RS"
    DMI_DD = "DMI_DD"


def validate_strategies_names_TWO_STEP_INSTANCE(input_list):
    valid_names = [clf.value for clf in StrategyEnum]
    for name in input_list:
        if name not in valid_names:
            raise ValueError(f"Invalid classifier name: '{name}'. Valid options are: {valid_names}")
    return input_list