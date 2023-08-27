import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(5000, 1)
label = np.zeros((5000, 1))
drift = np.zeros((5000, 1))
label[data>3] = 1
label[data<-3] = 1
print(sum(label))

dld = np.concatenate((data, label, drift), axis=1)

data = np.random.randn(5000, 1) * 3
label = np.zeros((5000, 1))
drift = np.zeros((5000, 1))
drift[0] = 1
label[data>9] = 1
label[data<-9] = 1
print(sum(label))

temp = np.concatenate((data, label, drift), axis=1)
dld = np.concatenate((dld, temp), axis=0)

np.savetxt("norm1d_drift.csv", dld, delimiter=",")


# plt.scatter(np.arange(10000), dld[:, 0], c=dld[:, 1])
# plt.show()