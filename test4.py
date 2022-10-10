# Imports
import numpy as np
from skmultiflow.drift_detection.hddm_w import HDDM_W
hddm_w = HDDM_W()
# Simulating a data stream as a normal distribution of 1's and 0's
data_stream = np.random.randint(2, size=2000)
# Changing the data concept from index 999 to 1500, simulating an
# increase in error rate
for i in range(999, 1500):
    data_stream[i] = 0
np.savetxt("data5.csv", data_stream, delimiter=",")
# Adding stream elements to HDDM_A and verifying if drift occurred
for i in range(2000):
    hddm_w.add_element(data_stream[i])
    if hddm_w.detected_warning_zone():
        print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    if hddm_w.detected_change():
        print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))