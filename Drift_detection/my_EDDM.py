import numpy as np

class EDDM:
    def __init__(self):
        self.__in_warning_zone = False
        self.__in_concept_change = False

        self.m_num_errors = 0
        self.m_min_num_errors = 30
        self.m_n = 1
        self.m_d = 0
        self.m_lastd = 0
        self.m_mean = 0.0
        self.m_std_temp = 0.0
        self.m_m2s_max = 0.0
        self.estimation = 0.0
        self.m_last_level = None

        self.FDDM_OUTCONTROL = 0.9
        self.FDDM_WARNING = 0.95
        self.FDDM_MIN_NUM_INSTANCES = 30

    def reset(self):
        self.__in_warning_zone = False
        self.__in_concept_change = False

        self.m_n = 1
        self.m_num_errors = 0
        self.m_d = 0
        self.m_lastd = 0
        self.m_mean = 0.0
        self.m_std_temp = 0.0
        self.m_m2s_max = 0.0
        self.estimation = 0.0

    def add_element(self, prediction):
        if self.__in_concept_change:
            self.reset()

        self.__in_concept_change = False

        self.m_n += 1

        if prediction == 1.0:
            self.__in_warning_zone = False
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
                    self.__in_concept_change = True

                elif (self.m_num_errors > self.m_min_num_errors) and (p < self.FDDM_WARNING):
                    self.__in_warning_zone = True

                else:
                    self.__in_warning_zone = False

    def detected_warning_zone(self):
        return self.__in_warning_zone

    def detected_change(self):
        return self.__in_concept_change