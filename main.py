import numpy as np
from my_IDK import my_IDK
import matplotlib.pyplot as plt

dl = np.loadtxt("norm1d_drift.csv", delimiter=",")
N = dl.shape[0]
plt.scatter(np.arange(N), dl[:, 0], c=dl[:, 1])
plt.show()