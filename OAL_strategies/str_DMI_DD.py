import numpy as np
import shap
import scipy

class DMI_DD_strategy():
    def __init__(self, n_class, clf, X_pt, y_pt, chunk_size, query_size):
        self.n_class = n_class
        self.X_pt = X_pt
        self.y_pt = y_pt
        self.chunk_size = chunk_size
        self.class_weight = np.ones(n_class) / n_class
        self.count_cls = [] * self.n_class
        self.explainer = self.explainer_init(clf)
        self.ranking_shap_pt = self.calculate_ranking(self.X_pt)
        self.ranking_shap_new = self.ranking_shap_pt
        self.update_explainer = True
        self.gamma_init = 0.60
        self.gamma = self.gamma_init
        self.cold_count = 0
        self.alpha = 0.10
        self.div_list = []
        self.gamma_list = []
        self.query_size = query_size
        self.shap_values_t = 0

    def sum_Jeffery_div(self, ranking_1, ranking_2):
        ranking_1_norm = (ranking_1 + 1) / ranking_1.shape[0]
        ranking_2_norm = (ranking_2 + 1) / ranking_2.shape[0]
        JS_div = scipy.stats.entropy(ranking_1_norm, ranking_2_norm) + scipy.stats.entropy(ranking_2_norm, ranking_1_norm)
        return JS_div

    def explainer_init(self, clf):
        clf.fit(self.X_pt, self.y_pt)
        return shap.KernelExplainer(clf.predict_proba, self.X_pt)

    def calculate_ranking(self, X, nsamples=100):
        self.shap_values_t = self.explainer.shap_values(X, nsamples=nsamples)
        shap_values_t_sum = np.sum(self.shap_values_t, axis=1)
        combined_shap_t = np.dot(shap_values_t_sum.T, self.class_weight)
        return np.argsort(combined_shap_t, axis=0).T

    def evaluation(self, X, y, clf):

        ranking_shap_t = self.calculate_ranking(X)
        div_t = self.sum_Jeffery_div(ranking_shap_t, self.ranking_shap_new)

        if div_t < self.gamma:
            self.update_explainer = False
            update_model = False
            self.cold_count += 1
        else:
            self.update_explainer = True
            update_model = True
            self.cold_count = 0

        self.gamma = self.gamma_init * np.exp(-self.alpha * self.cold_count)
        self.gamma_list = self.gamma_list + [self.gamma]
        self.div_list = self.div_list + [div_t]

        if update_model:
            shap_orig_t_indiv = 0
            for i in range(self.n_class):
                shap_orig_t_indiv = shap_orig_t_indiv + self.shap_values_t[i] * self.class_weight[i]

            ranking_shap_orig_t_indiv = np.argsort(shap_orig_t_indiv, axis=1)
            div_indiv_list = [self.sum_Jeffery_div(ranking_shap_orig_t_indiv[j, :], self.ranking_shap_new) for j in range(self.chunk_size)]
            predicton_indiv_list = [(self.class_weight[clf.predict(X[j, :].reshape(1, -1))]) * np.linalg.norm(clf.predict_proba(X[j, :].reshape(1, -1)), ord=2, axis=None) for j in range(self.chunk_size)]

            query_list = (np.array(div_indiv_list).reshape(1, -1) + np.array(predicton_indiv_list).reshape(1,-1)).argsort().reshape(-1)
            query_set_t = query_list[0:self.query_size].reshape(-1)
            query_batch_t_X = X[query_set_t, :]
            query_batch_t_y = y[query_set_t]

            clf.partial_fit(query_batch_t_X, query_batch_t_y)

        if self.update_explainer:
            self.explainer = self.explainer_init(clf)

        self.ranking_shap_new = ranking_shap_t
        counts = np.bincount(clf.predict(X), minlength=self.n_class)
        self.class_weight = np.array(counts / self.chunk_size)

        return clf