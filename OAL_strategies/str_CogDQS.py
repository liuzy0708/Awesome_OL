""" CogDQS Strategy."""

import numpy as np

class CogDQS_strategy():
    def __init__(self, B=0.25, n=1, c=3, cw_size=10, window_size=200, s=0.01):
        self.B = B # Budget
        self.n = n # the threshold of local density
        self.c = c # the number of classes
        self.cw_size = cw_size
        self.window_size = window_size
        self.s = s


        self.cw_x_s_lambda_tao_f_minDis = {}
        self.i = 0 #the number of samples passed
        self.B_vary = 0
        self.v_hat = 0
        self.label_list = []
        self.u = 1 / self.c + self.B * (1 - 1 / self.c) # the threshold of uncertainty
        self.n_annotation = 0

    def distance(self, xi, xj):
        d_sum = 0
        for i in range(len(xi)):
            d_sum += (xi[i] - xj[i]) ** 2
        return np.sqrt(d_sum)

    def uncertaintySample_VarUn(self, p_xi):
        if p_xi < self.u:
            self.u = self.u * (1 - self.s)
            return 1
        else:
            self.u = self.u * (1 + self.s)
            return 0

    def uncertaintySample_FixUnself(self, p_xi):
        if p_xi < self.u:
            return 1
        else:
            return 0
    def ld(self, xi, sample_count):
        s_xi = 1
        lambda_xi = 0
        tao_xi = sample_count
        f_xi = 1
        new_xi = [xi, s_xi, lambda_xi, tao_xi, f_xi, np.inf]
        Dis_xi = np.inf
        row = len(self.cw_x_s_lambda_tao_f_minDis.keys())
        col = len(new_xi[0])
        ld = 0

        for j in range(row):
            if self.cw_x_s_lambda_tao_f_minDis[j][5] > self.distance(self.cw_x_s_lambda_tao_f_minDis[j][0][0], xi[0]):
                self.cw_x_s_lambda_tao_f_minDis[j][5] = self.distance(self.cw_x_s_lambda_tao_f_minDis[j][0][0], xi[0])
                self.cw_x_s_lambda_tao_f_minDis[j][3] = sample_count
                self.cw_x_s_lambda_tao_f_minDis[j][2] += 1
                ld += 1
            if new_xi[5] > self.distance(self.cw_x_s_lambda_tao_f_minDis[j][0][0], xi[0]):
                new_xi[5] = self.distance(self.cw_x_s_lambda_tao_f_minDis[j][0][0], xi[0])

        for j in range(row):
            self.cw_x_s_lambda_tao_f_minDis[j][4] = 1 / (self.cw_x_s_lambda_tao_f_minDis[j][2] + 1)
            self.cw_x_s_lambda_tao_f_minDis[j][1] = np.exp(- self.cw_x_s_lambda_tao_f_minDis[j][4] * \
                                                      (sample_count - self.cw_x_s_lambda_tao_f_minDis[j][3]))

        smallest_index = 0
        min_s = np.inf
        if row > self.cw_size:
            for j in range(len(self.cw_x_s_lambda_tao_f_minDis)):
                if self.cw_x_s_lambda_tao_f_minDis[j][1] < min_s:
                    min_s = self.cw_x_s_lambda_tao_f_minDis[j][1]
                    smallest_index = j
            del self.cw_x_s_lambda_tao_f_minDis[smallest_index]
            self.cw_x_s_lambda_tao_f_minDis[smallest_index] = new_xi
        else:
            self.cw_x_s_lambda_tao_f_minDis[len(self.cw_x_s_lambda_tao_f_minDis)] = new_xi

        return ld

    def evaluation(self, xi, yi, clf):
        self.i += 1
        labeling = 0
        a = self.ld(xi, self.i)
        if self.B_vary < self.B and a >= self.n:
            p_xi = clf.predict_proba(xi)
            p_xi_max = max(p_xi[0])
            labeling = self.uncertaintySample_VarUn(p_xi_max)
            if labeling:
                self.n_annotation += 1
                clf.partial_fit(xi, yi)
                self.label_list = self.label_list + [1]
                if len(self.label_list) <= self.window_size:
                    v_hat = sum(self.label_list)
                else:
                    self.label_list.pop(0)
                    v_hat = sum(self.label_list)
            else:
                self.label_list = self.label_list + [0]
                if len(self.label_list) <= self.window_size:
                    v_hat = sum(self.label_list)
                else:
                    self.label_list.pop(0)
                    v_hat = sum(self.label_list)
        else:
            self.label_list = self.label_list + [0]
            if len(self.label_list) <= self.window_size:
                v_hat = sum(self.label_list)
            else:
                self.label_list.pop(0)
                v_hat = sum(self.label_list)

        v_hat = v_hat * (self.window_size - 1) / self.window_size + labeling
        self.B_vary = v_hat / self.window_size
        return labeling, clf