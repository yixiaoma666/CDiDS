from psKC import psKC
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

# path = r"SENNE-master\Datasets\Synthetic.mat"
# all_data = scio.loadmat(path)

# data = all_data["art4"][:, 0]
# label = all_data["art4"][:, 1]

path = r"my_data\nine_direction.mat"
all_data = scio.loadmat(path)["nine_direction"]

data = all_data[:, 0]
label = all_data[:, 1]




test_data = data[0]
