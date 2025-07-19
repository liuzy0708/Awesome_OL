import numpy as np
from scipy import stats
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

class KSWIN(BaseDriftDetector):
    """
    References
    ----------
    .. [1] Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive
       Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,
    """

    def __init__(self, alpha=0.005, window_size=100, stat_size=30, data=None):
        super().__init__()
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.change_detected = False
        self.p_value = 0
        self.n = 0
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.window_size < 0:
            raise ValueError("window_size must be greater than 0")

        if self.window_size < self.stat_size:
            raise ValueError("stat_size must be smaller than window_size")

        if type(data) != np.ndarray or type(data) is None:
            self.window = np.array([])
        else:
            self.window = data

    def add_element(self, input_value):
        """ Add element to sliding window

        Adds an element on top of the sliding window and removes
        the oldest one from the window. Afterwards, the KS-test
        is performed.

        Parameters
        ----------
        input_value: ndarray
            New data sample the sliding window should add.
        """
        self.n += 1
        currentLength = self.window.shape[0]
        if currentLength >= self.window_size:
            self.window = np.delete(self.window,0)
            rnd_window = np.random.choice(self.window[:-self.stat_size], self.stat_size)

            (st, self.p_value) = stats.ks_2samp(rnd_window, self.window[-self.stat_size:],mode="exact")

            if self.p_value <= self.alpha and st > 0.1:
                self.change_detected = True
                self.window = self.window[-self.stat_size:]
            else:
                self.change_detected = False
        else: # Not enough samples in sliding window for a valid test
            self.change_detected = False

        self.window = np.concatenate([self.window,[input_value]])

    def detected_change(self):
        return self.change_detected

    def reset(self):
        self.p_value = 0
        self.window = np.array([])
        self.change_detected = False
