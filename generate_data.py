from random import random
from re import A
from Adwin import Adwin
import numpy as np
import random
from generate_uniform_circle import generate_uniform_circle
import matplotlib.pyplot as plt



data = []
label = []

for _ in range(1000):
    if random.random() < 0.98:
        data.append(np.random.randn(1, 2)[0])
        label.append([0, 1])
    else:
        data.append(np.random.randn(1, 2)[0] * 10)
        label.append([1, 0])

for _ in range(1000):
    if random.random() < 0.98:
        data.append(np.random.randn(1, 2)[0] + np.array([5, 5]))
        label.append([0, 1])
    else:
        data.append(np.random.randn(1, 2)[0] * 10  + np.array([5, 5]))
        label.append([1, 0])

for _ in range(2000):
    r = random.random()
    if r < 0.49:
        data.append(np.random.randn(1, 2)[0] + np.array([5, 10]))
        label.append([0, 1])
    elif 0.49 <= r < 0.50:
        data.append(np.random.randn(1, 2)[0] * 10 + np.array([5, 10]))
        label.append([1, 0])
    elif 0.50 <= r < 0.99:
        data.append(np.random.randn(1, 2)[0] + np.array([10, 5]))
        label.append([0, 1])
    elif 0.99 <= r:
        data.append(np.random.randn(1, 2)[0] * 10 + np.array([10, 5]))
        label.append([1, 0])

data = np.array(data)
label = np.array(label)

for i, _data in enumerate(data):
    if label[i][0] == 0:
        plt.scatter(_data[0], _data[1], c="b")
    else:
        plt.scatter(_data[0], _data[1], c="r")
        
output = np.concatenate((data, label), axis=1)
np.savetxt("data10.csv", output, delimiter=",")
plt.savefig("data10.png")
