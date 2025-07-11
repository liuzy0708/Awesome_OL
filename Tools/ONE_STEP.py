import os
import time
import warnings
from enum import Enum
from sklearn.metrics import accuracy_score, f1_score
from Tools.utils import *
from visualization.plot_comparison import plot_comparison
from log_config import logger

warnings.filterwarnings("ignore")

class One_Step:
    def __init__(self,
                 max_samples=1000,
                 n_round=3,
                 n_pt=100,
                 dataset_name="Waveform",
                 method_name_list=None,
                 chunk_size=1,
                 framework="ONE-STEP",
                 theta=0.15):

        self.max_samples = max_samples
        logger.info(f"max_samples: {max_samples}")
        self.n_round = n_round
        logger.info(f"n_round: {n_round}")
        self.n_pt = n_pt
        logger.info(f"n_pt: {n_pt}")
        self.chunk_size = chunk_size
        logger.info(f"chunk_size: {chunk_size}")
        self.framework = framework
        logger.info(f"framework: {framework}")
        self.dataset_name = dataset_name
        logger.info(f"dataset_name: {dataset_name}")
        self.theta = theta
        logger.info(f"theta: {theta}")

        if method_name_list is None:
            self.method_name_list = ["OALE"]
        else:
            parsed_list = [name.strip() for name in method_name_list.split(',')]
            self.method_name_list = validate_classifier_names_ONE_STEP(parsed_list)
        logger.info(f"method_name_list: {self.method_name_list}")

        self.num_method = len(self.method_name_list)
        logger.info(f"num_method: {self.num_method}")
        self.acc_list = [[[] for _ in range(n_round)] for _ in range(self.num_method)]
        self.f1_list = [[[] for _ in range(n_round)] for _ in range(self.num_method)]

        self.result_path = "../Results/"
        os.makedirs(self.result_path, exist_ok=True)
        #logger.info(f"result_path: {self.result_path}")

        self.directory_path = f"./Results/Results_{self.dataset_name}_{self.framework}_{self.n_pt}_{self.chunk_size}_{self.max_samples}/"
        os.makedirs(self.directory_path, exist_ok=True)
        logger.info(f"directory_path: {self.directory_path}")

    def run(self):
        for n_method, method_name in enumerate(self.method_name_list):
            print(f"Method: {method_name}")
            n_annotation_list = []
            y_pred_all = []
            y_true_all = []
            t1 = time.time()

            for round_id in range(self.n_round):
                print(f"  Round {round_id + 1}/{self.n_round}")
                y_pred_list = []
                y_true_list = []

                self.stream = get_stream(self.dataset_name)
                X_pt_source, y_pt_source = get_pt(stream=self.stream, n_pt=self.n_pt)
                para_method = para_init(X_pt_source, y_pt_source, n_class=self.stream.n_classes)
                method = para_method.get_method(method_name)

                count = 0

                while count < self.max_samples and self.stream.has_more_samples():
                    count += self.chunk_size
                    X, y = self.stream.next_sample(self.chunk_size)

                    y_pred = method.predict(X)
                    y_pred_list.extend(y_pred)
                    y_true_list.extend(y)

                    method.evaluation(X, y)

                # 保存单轮结果
                acc = accuracy_score(y_true_list, y_pred_list)
                f1 = f1_score(y_true_list, y_pred_list, labels=list(range(self.stream.n_classes)), average='macro')
                ann_ratio = method.n_annotation / self.max_samples

                self.acc_list[n_method][round_id] = acc
                self.f1_list[n_method][round_id] = f1
                n_annotation_list.append(ann_ratio)

                y_pred_all += y_pred_list
                y_true_all += y_true_list

            t2 = time.time()

            result_pred = np.array(y_pred_all).reshape(self.n_round, self.max_samples)
            result_true = np.array(y_true_all).reshape(self.n_round, self.max_samples)

            pred_path = os.path.join(self.directory_path, f"Prediction_{method_name}.csv")
            true_path = os.path.join(self.directory_path, f"True_{method_name}.csv")

            with open(pred_path, mode='w', newline='') as f:
                csv.writer(f).writerows(result_pred)
            with open(true_path, mode='w', newline='') as f:
                csv.writer(f).writerows(result_true)

            print("\nFinal Results for", method_name)
            print(f"Accuracy     : {np.mean(self.acc_list[n_method]):.3f} ± {np.std(self.acc_list[n_method]):.3f}")
            print(f"Macro-F1     : {np.mean(self.f1_list[n_method]):.3f} ± {np.std(self.f1_list[n_method]):.3f}")
            print(f"Annotation % : {np.mean(n_annotation_list):.4f}")
            print(f"Avg Time     : {(t2 - t1) / self.n_round:.4f} s\n")

    def show(self,need_matrix):
        plot_comparison(dataset=self.dataset_name, n_class=self.stream.n_classes, n_round=self.n_round,
                            max_samples=self.max_samples, interval=1, chunk_size=self.chunk_size,
                            filename_list=self.method_name_list, n_pt=self.n_pt, framework=self.framework, need_matrix=need_matrix)


class ClassifierEnum(Enum):
    ROALE_DI = "ROALE_DI"
    OALE = "OALE"

def validate_classifier_names_ONE_STEP(input_list):
    valid_names = [clf.value for clf in ClassifierEnum]
    for name in input_list:
        if name not in valid_names:
            raise ValueError(f"Invalid classifier name: '{name}'. Valid options are: {valid_names}")
    return input_list