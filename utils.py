from skmultiflow.data import DataStream, WaveformGenerator, SEAGenerator, HyperplaneGenerator
import pandas as pd
import numpy as np
import csv

from legacy.str_DMGT import DMGT_strategy
from OAL_strategies.str_MTSGQS import MTSGQS_strategy
from OAL_strategies.str_DSA_AI import DSA_AI_strategy
from OAL_strategies.str_US_fix import US_fix_strategy
from OAL_strategies.str_US_var import US_var_strategy
from OAL_strategies.str_CogDQS import CogDQS_strategy
from OAL_strategies.str_ROALE_DI import ROALE_DI_strategy
from OAL_strategies.str_RS import RS_strategy
from OAL_strategies.str_USGQS import USGQS_strategy
from OAL_strategies.str_OALE import OALE_strategy

from classifier.clf_BLS import BLS
from classifier.clf_SRP import SRP
from classifier.clf_DES import DES_ICD
from skmultiflow.bayes import NaiveBayes
from classifier.clf_ACDWM.clf_ACDWM import ACDWM
from classifier.clf_OLI2DS.clf_OLI2DS import OLI2DS
# from classifier.clf_IWDA_Multi.clf_IWDA_Multi import IWDA_multi
# from classifier.clf_IWDA_Multi.clf_IWDA_PL import IWDA_PL
from classifier.clf_DGEBLS import DGEBLS
from classifier.clf_DGEBLS_turbo import DGEBLS_turbo

from OSSL_classifier.clf_OSSBLS import OSSBLS
from OSSL_classifier.clf_ODSSBLS import ODSSBLS
from OSSL_classifier.clf_ISSBLS import ISSBLS
from OSSL_classifier.clf_SOSELM import SOSELM
from classifier.clf_ARF import ARF
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import LeveragingBaggingClassifier, OnlineUnderOverBaggingClassifier, OnlineUnderOverBagging,OzaBaggingClassifier, OzaBaggingADWINClassifier, DynamicWeightedMajorityClassifier
from skmultiflow.meta import OnlineAdaC2Classifier
def get_stream(name):
    if name == "Jiaolong":
        with open('./datasets/Jiaolong_DSMS_V2.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)

            X = []
            Y = []

            for row in csvreader:
                data_row = row[:-1]
                label = row[-1]
                X.append(data_row)
                Y.append(label)

            X = np.array(X, dtype=float)
            Y = np.array(Y, dtype=int)
        stream = DataStream(X, Y)
    elif name == "waveform":
        stream = WaveformGenerator(random_state=1)
    elif name == "SEA":
        stream = SEAGenerator(random_state=1)
    elif name == "hyperplane":
        stream = HyperplaneGenerator(random_state=1)

    else:
        data = pd.read_csv('./datasets/' + name + '.csv')
        data = data.values
        vol, col = data.shape
        X = data[:, 0:col - 1]
        Y = data[:, col - 1]
        Y = np.array([int(i) for i in Y])
        stream = DataStream(X, Y)
    # print('This dataset has {} columns of attributes and {} samples'.format(X.shape[1], X.shape[0]))
    return stream

def get_pt(stream, n_pt, n_class):
    data, labels = stream.next_sample(n_pt)
    return data, labels

class para_init:

    def __init__(self, n_class=2, X_pt_source=np.array([[]]), y_pt_source=np.array([[]]), n_ratio_max=0.2, n_anchor=10, theta=0.2):
        self.n_class = n_class
        self.X_pt_source = X_pt_source
        self.y_pt_source = y_pt_source
        self.n_ratio_max = n_ratio_max
        self.n_anchor = n_anchor
        self.theta = theta

    def get_clf(self, name):
        if name == "clf_ARF":
            return ARF()
        elif name == "clf_LB":
            return LeveragingBaggingClassifier()
        elif name=="clf_OB":
            return OzaBaggingClassifier()
        elif name=="clf_OBADWIN":
            return OzaBaggingADWINClassifier()
        elif name=='clf_DWM':
            return DynamicWeightedMajorityClassifier()
        elif name == "clf_OOB":
            return OnlineUnderOverBaggingClassifier()
        elif name == "clf_SRP":
            return SRP(n_estimators=3, n_class=self.n_class)
        elif name == "clf_AdaC2":
            return OnlineAdaC2Classifier()
        elif name == "clf_IWDA_Multi":
            return IWDA_multi(old_to_use=120, update_wm=150, whiten=False)
        elif name == "clf_IWDA_PL":
            return IWDA_PL(old_to_use=120, update_wm=150, whiten=False)
        elif name == "clf_BLS":
            return BLS(Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=0.001)
        elif name == "clf_OSSBLS":
            return OSSBLS(Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=0.001,
                     gamma=0.005,
                     n_anchor=10)
        elif name == "clf_ODSSBLS":
            return ODSSBLS(Nf=10,
                         Ne=10,
                         N1=10,
                         N2=10,
                         map_function='sigmoid',
                         enhence_function='sigmoid',
                         reg=0.001,
                         gamma=0.001,
                        n_anchor=self.n_anchor,
                        n_class=3)
        elif name == "clf_ISSBLS":
            return ISSBLS(
                        Nf=10,
                        Ne=10,
                        N1=10,
                        N2=10,
                        map_function='sigmoid',
                        enhence_function='sigmoid',
                        reg=0.001)
        elif name == "clf_SOSELM":
            return SOSELM(
                    Ne=20,
                    N2=10,
                    enhence_function='sigmoid',
                    reg=0.001)
        elif name == "clf_NB":
            return NaiveBayes()
        elif name == "clf_DES":
            return DES_ICD(base_classifier=NaiveBayes(), window_size=50, max_classifier=10)
        elif name == "clf_DES_5":
            return DES_ICD(base_classifier=NaiveBayes(), window_size=50, max_classifier=5)
        elif name == "clf_ACDWM":
            return ACDWM(chunk_size=0, max_ensemble_size=10)
        elif name == "clf_OLI2DS":
            return OLI2DS(C=0.0100000, Lambda=30, B=1, theta=8, gama=0, sparse=0, mode="capricious")
        elif name == "clf_DGEBLS":
            return DGEBLS(n_base_learner=20,
                            vartheta=1.00,
                            theta=1e-3,
                            gamma=1e-3,
                            tau=0.50,
                            max_data_pairs=200,
                            n_class=self.n_class,
                            k=0.7)
        elif name == "clf_DGEBLS_turbo":
            return DGEBLS_turbo(n_base_learner=20,
                            vartheta=1.00,
                            theta=1e-3,
                            gamma=1e-3,
                            tau=0.50,
                            max_data_pairs=200,
                            n_class=self.n_class)
        raise ValueError("没有这个分类器")

    def get_str(self, name):
        if name == "DSA_AI_str":
            return DSA_AI_strategy(n_class=self.n_class, X_memory_collection=self.X_pt_source,
                                 y_memory_collection=self.y_pt_source, d=self.X_pt_source.shape[1],
                                 kappa=3, gamma=0.4)
        elif name == "Supervised_str":
            return None
        elif name == "MTSGQS_str":
            return MTSGQS_strategy(n_class=self.n_class, kappa=2, gamma=0.4, n_capacity=100)
        elif name == "DMGT_str":
            return DMGT_strategy(n_class=self.n_class, kappa=2, gamma=0.4)
        elif name == "US_fix_str":
            return US_fix_strategy(theta=0.5)
        elif name == "US_var_str":
            return US_var_strategy(theta=0.5)
        elif name == "CogDQS_str":
            return CogDQS_strategy(B=0.25, n=1, c=3, cw_size=10, window_size=200, s=0.01)
        elif name == "ROALE_DI_str":
            return ROALE_DI_strategy(x_train=self.X_pt_source, y_train=self.y_pt_source, label_ratio=self.n_ratio_max, L=self.n_class, chunk_size=150,
                                         step=0.01, theta=0.005, D=10, sigma_imbalance=0.01)
        elif name == "RS_str":
            return RS_strategy(label_ratio=self.n_ratio_max)
        elif name == "USGQS_str":
            return USGQS_strategy(n_class=self.n_class, kappa=2, thre=self.theta)
        elif name == "OALE_str":
            return OALE_strategy(x_train=self.X_pt_source, y_train=self.y_pt_source, r=0.05, I=200, L=2, s=0.01, theta=0.5, sigma = 0.01, D=10)
        raise ValueError("没有这个策略")

    def clf_init(self):
        clf_ARF = ARF()
        clf_SRP = SRP()
        clf_BLS = BLS(Nf=10,
                      Ne=10,
                      N1=10,
                      N2=10,
                      map_function='sigmoid',
                      enhence_function='sigmoid',
                      reg=0.001)
        clf_OSSBLS = OSSBLS(
                     Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=0.001,
                     gamma=0.05,
                     n_anchor=10)

        clf_ISSBLS = ISSBLS(
                     Nf=10,
                     Ne=10,
                     N1=10,
                     N2=10,
                     map_function='sigmoid',
                     enhence_function='sigmoid',
                     reg=0.001)

        clf_SOSELM = SOSELM(
                     Ne=20,
                     N2=10,
                     enhence_function='sigmoid',
                     reg=0.001)
        clf_ODSSBLS = ODSSBLS(Nf=10,
                Ne=10,
                N1=10,
                N2=10,
                map_function='sigmoid',
                enhence_function='sigmoid',
                reg=0.001,
                gamma=0.005,
                n_anchor=self.n_anchor,
                n_class=3)
        return clf_ARF, clf_SRP, clf_BLS, clf_OSSBLS, clf_ODSSBLS, clf_ISSBLS, clf_SOSELM

    def str_init(self):
        DSA_AI_str = DSA_AI_strategy(n_class=self.n_class, X_memory_collection=self.X_pt_source, y_memory_collection=self.y_pt_source, d=self.X_pt_source.shape[1],
                                     kappa=2, gamma=0.4)
        MTSGQS_str = MTSGQS_strategy(n_class=self.n_class, kappa=2, gamma=0.4, n_capacity=100)
        DMGT_str = DMGT_strategy(n_class=self.n_class, kappa=2, gamma=0.4)
        US_fix_str = US_fix_strategy(theta=0.5)
        US_var_str = US_var_strategy(theta=0.5)
        CogDQS_str = CogDQS_strategy(B=0.25, n=1, c=2, cw_size=10, window_size=200, s=0.01)
        ROALE_DI_str = ROALE_DI_strategy(x_train=self.X_pt_source, y_train=self.y_pt_source, label_ratio=self.n_ratio_max, L=self.n_class, chunk_size=150,
                                         step=0.01, theta=0.005, D=10, sigma_imbalance=0.01)
        RS_str = RS_strategy(label_ratio=self.n_ratio_max)
        USGQS_str = USGQS_strategy(n_class=self.n_class, kappa=2, thre=self.theta)
        return DSA_AI_str, MTSGQS_str, DMGT_str, US_fix_str, US_var_str, CogDQS_str, ROALE_DI_str, RS_str, USGQS_str
