import numpy as np

class DDM:
    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0):
        self.__in_warning_zone = False
        self.__in_concept_change = False

        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.estimation = 0.0


    def reset(self):
        self.__in_warning_zone = False
        self.__in_concept_change = False

        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")

    def add_element(self, prediction):
        """
        prediction = 0: correct prediction
        prediction = 1: incorrect prediction
        """

        if self.__in_warning_zone:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        self.sample_count += 1

        self.estimation = self.miss_prob
        self.__in_concept_change = False
        self.__in_warning_zone = False

        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min:
            self.__in_concept_change = True

        elif self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
            self.__in_warning_zone = True

        else:
            self.__in_warning_zone = False


    def detected_warning_zone(self):
        return self.__in_warning_zone

    def detected_change(self):
        return self.__in_concept_change