class PageHinkley:
    def __init__(self, min_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001):
        self.__in_warning_zone = False
        self.__in_concept_change = False
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = 0.0
        self.sample_count = 1
        self.sum = 0.0
        self.estimation = 0.0

    def reset(self):
        self.__in_warning_zone = False
        self.__in_concept_change = False

        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0
        self.estimation = 0.0

    def add_element(self, x):
        if self.__in_concept_change:
            self.reset()

        self.x_mean = self.x_mean + (x - self.x_mean) / float(self.sample_count)
        self.sum = max(0., self.alpha * self.sum + (x - self.x_mean - self.delta))

        self.sample_count += 1

        self.estimation = self.x_mean
        self.__in_concept_change = False
        self.__in_warning_zone = False

        if self.sample_count < self.min_instances:
            return None

        if self.sum > self.threshold:
            self.__in_concept_change = True

    def detected_change(self):
        return self.__in_concept_change