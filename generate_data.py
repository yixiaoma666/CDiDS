from random import random
import numpy as np
import random
from generate_uniform_circle import generate_uniform_circle
import matplotlib.pyplot as plt


def gen_ano(direction, center=np.array([0, 0]), distance=10, var=1):
    _direction = random.sample(direction, 1)[0]
    if _direction == "c":
        return (np.random.randn(1, 2)[0]) * var + center
    if _direction == "n":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([0, distance])
    if _direction == "e":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([distance, 0])
    if _direction == "s":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([0, -distance])
    if _direction == "w":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([-distance, 0])
    if _direction == "ne":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([distance, distance])
    if _direction == "se":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([distance, -distance])
    if _direction == "sw":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([-distance, -distance])
    if _direction == "nw":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([-distance, distance])


data = []
label = []

for _ in range(1000):  # 1
    r = random.random()
    if r < 0.02:
        data.append(gen_ano(["w"]))
        label.append((1, 0))
        continue
    data.append(gen_ano(["e"]))
    label.append((0, 1))

for _ in range(1000): # 2
    r = random.random()
    if r < 0.5:
        data.append(gen_ano(["w"]))
        label.append((0, 1))
        continue
    data.append(gen_ano(["e"]))
    label.append((0, 1))

for _ in range(1000):  # 3
    r = random.random()
    if r < 0.02:
        data.append(gen_ano(["e"]))
        label.append((1, 0))
        continue
    data.append(gen_ano(["w"]))
    label.append((0, 1))


for i in range(3000):
    if random.random() < 0.01:
        data[i] = gen_ano(["n"])
        label[i] = (1, 0)
    


data = np.array(data)
label = np.array(label)

for i, _data in enumerate(data):
    if label[i][0] == 0:
        plt.scatter(_data[0], _data[1], c="b")
    else:
        plt.scatter(_data[0], _data[1], c="r")

output = np.concatenate((data, label), axis=1)
np.savetxt("data13.csv", output, delimiter=",")
plt.savefig("data13.png")
# plt.show()