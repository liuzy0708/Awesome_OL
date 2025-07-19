import numpy as np

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class EDDM(BaseDriftDetector):
    """
    References
    ----------
    .. [1] Early Drift Detection Method. Manuel Baena-Garcia, Jose Del Campo-Avila,
       Raúl Fidalgo, Albert Bifet, Ricard Gavalda, Rafael Morales-Bueno. In Fourth
       International Workshop on Knowledge Discovery from Data Streams, 2006.
    """


    FDDM_OUTCONTROL = 0.9
    FDDM_WARNING = 0.95
    FDDM_MIN_NUM_INSTANCES = 30

    def __init__(self):
        super().__init__()
        self.m_num_errors = None
        self.m_min_num_errors = 30
        self.m_n = None
        self.m_d = None
        self.m_lastd = None
        self.m_mean = None
        self.m_std_temp = None
        self.m_m2s_max = None
        self.m_last_level = None
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.m_n = 1
        self.m_num_errors = 0
        self.m_d = 0
        self.m_lastd = 0
        self.m_mean = 0.0
        self.m_std_temp = 0.0
        self.m_m2s_max = 0.0
        self.estimation = 0.0

    def add_element(self, prediction):
        """ Add a new element to the statistics

        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        Returns
        -------
        EDDM
            self

        Notes
        -----
        After calling this method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.

        """

        if self.in_concept_change:
            self.reset()

        self.in_concept_change = False

        self.m_n += 1

        if prediction == 1.0:
            self.in_warning_zone = False
            self.delay = 0
            self.m_num_errors += 1
            self.m_lastd = self.m_d
            self.m_d = self.m_n - 1
            distance = self.m_d - self.m_lastd
            old_mean = self.m_mean
            self.m_mean = self.m_mean + (float(distance) - self.m_mean) / self.m_num_errors   # 均值的递推形式
            self.estimation = self.m_mean
            self.m_std_temp = self.m_std_temp + (distance - self.m_mean) * (distance - old_mean)
            std = np.sqrt(self.m_std_temp / self.m_num_errors)
            m2s = self.m_mean + 2 * std

            if self.m_n < self.FDDM_MIN_NUM_INSTANCES:
                return

            if m2s > self.m_m2s_max:
                self.m_m2s_max = m2s
            else:
                p = m2s / self.m_m2s_max
                if (self.m_num_errors > self.m_min_num_errors) and (p < self.FDDM_OUTCONTROL):
                    self.in_concept_change = True

                elif (self.m_num_errors > self.m_min_num_errors) and (p < self.FDDM_WARNING):
                    self.in_warning_zone = True

                else:
                    self.in_warning_zone = False
