import numpy as np
import matplotlib.pyplot as plt
from my_IDK import my_IDK
from sklearn.metrics import roc_auc_score

dld = np.loadtxt("norm1d_drift.csv", delimiter=",")

data = dld[:10000, :-2].reshape(-1, 1)
label = dld[:10000, -2]

detector = my_IDK(data, 2, 20)
scores = 1 - detector.IDK_score()

print(roc_auc_score(label, scores))

plt.scatter(np.arange(10000), scores, c=label)
plt.show()

