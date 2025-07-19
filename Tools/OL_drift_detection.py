from Tools.utils import *
from Tools.log_config import logger


class Drift_Detector:
    def __init__(self, det_name_list):
        self.det_name_list = det_name_list
        self.datasets = None
        self.stream = None

    def run(self):
        for n_det in range(len(self.det_name_list)):
            det_name = self.det_name_list[n_det]
            det = get_det(det_name)
            logger.info(f"Detection: {det_name}")
            print(f"Detection: {det_name}")

            if det_name == "DDM" or det_name == "EDDM":
                data_stream = np.random.randint(2, size=2000)
                for i in range(999, 1500):
                    data_stream[i] = 0
                for i in range(2000):
                    det.add_element(data_stream[i])
                    if det.detected_warning_zone():
                        print('Warning zone has been detected in data: ' + str(
                            data_stream[i]) + ' - of index: ' + str(i))
                    if det.detected_change():
                        print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))


            elif det_name == "KSWIN":
                stream = SEAGenerator(classification_function=2,
                                      random_state=112, balance_classes=False, noise_percentage=0.28)
                # Store detections
                detections = []
                # Process stream via KSWIN and print detections
                for i in range(1000):
                    data = stream.next_sample(10)
                    batch = data[0][0][0]
                    det.add_element(batch)
                    if det.detected_change():
                        print("\rIteration {}".format(i))
                        print("\r KSWINReject Null Hyptheses")
                        detections.append(i)
                print("Number of detections: " + str(len(detections)))

            elif det_name == "PageHinkley":
                data_stream = np.random.randint(2, size=2000)
                # Changing the data concept from index 999 to 2000
                for i in range(999, 2000):
                    data_stream[i] = np.random.randint(4, high=8)
                # Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
                for i in range(2000):
                    det.add_element(data_stream[i])
                    if det.detected_change():
                        print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

            else:
                raise NotImplementedError


if __name__ == "__main__":
    det_name_list = ["DDM", "EDDM", "KSWIN", "PageHinkley"]
    det_detector = Drift_Detector(det_name_list)
    det_detector.run()