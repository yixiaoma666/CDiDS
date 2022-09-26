import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
from generate_uniform_circle import generate_uniform_circle

adwin = ADWIN()

# c1 = np.array(generate_uniform_circle((0, 0), 1, 100))
# c2 = np.array(generate_uniform_circle((2, 2), 1, 100))
# c3 = np.array(generate_uniform_circle((8, 8), 2, 100))

# data_stream = np.concatenate((c1, c2, c3), axis=0)

# np.savetxt("data2.csv", data_stream, delimiter=",")

data_stream = np.array([0] * 200 + [1] * 200 + [0] * 200) 


# data_stream = np.loadtxt("data2.csv", delimiter=",")

for i in range(600):
    adwin.add_element(data_stream[i])
    if adwin.detected_change():
        print(f"{data_stream[i]}\t{i}")
    
