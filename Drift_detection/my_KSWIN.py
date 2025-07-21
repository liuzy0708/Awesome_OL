import numpy as np
from scipy import stats

class KSWIN:
    def __init__(self, alpha=0.005, window_size=100, stat_size=30, data=None):
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.__change_detected = False
        self.p_value = 0
        self.n = 0

        if type(data) != np.ndarray or type(data) is None:
            self.window = np.array([])
        else:
            self.window = data

    def add_element(self, input_value):
        self.n += 1
        currentLength = self.window.shape[0]
        if currentLength >= self.window_size:
            self.window = np.delete(self.window,0)
            rnd_window = np.random.choice(self.window[:-self.stat_size], self.stat_size)

            (st, self.p_value) = stats.ks_2samp(rnd_window, self.window[-self.stat_size:],mode="exact")

            if self.p_value <= self.alpha and st > 0.1:
                self.__change_detected = True
                self.window = self.window[-self.stat_size:]
            else:
                self.__change_detected = False
        else:
            self.__change_detected = False

        self.window = np.concatenate([self.window,[input_value]])

    def detected_change(self):
        return self.__change_detected

    def reset(self):
        self.p_value = 0
        self.window = np.array([])
        self.__change_detected = False
