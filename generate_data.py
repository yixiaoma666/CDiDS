from random import random
import numpy as np
import random
from generate_uniform_circle import generate_uniform_circle
import matplotlib.pyplot as plt


def gen_ano(direction, center=np.array([0, 0]), distance=10, var=1):
    _direction = random.sample(direction, 1)[0]
    if _direction == "c":
        return (np.random.randn(1, 2)[0]) * var + center
    elif _direction == "n":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([0, distance])
    elif _direction == "e":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([distance, 0])
    elif _direction == "s":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([0, -distance])
    elif _direction == "w":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([-distance, 0])
    elif _direction == "ne":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([distance, distance])
    elif _direction == "se":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([distance, -distance])
    elif _direction == "sw":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([-distance, -distance])
    elif _direction == "nw":
        return (np.random.randn(1, 2)[0]) * var + center + np.array([-distance, distance])


data = []
label = []

for _ in range(1000):
    if random.random()<0.02:
        data.append((gen_ano(["ne", "se", "sw", "nw"])))
        label.append(1)
    else:
        if random.random()<0.5:
            data.append((gen_ano(["n"])))
            label.append(2)
        else:
            data.append((gen_ano(["s"])))
            label.append(3)

for _ in range(1000):
    if random.random()<0.02:
        data.append((gen_ano(["ne", "se", "sw", "nw"])))
        label.append(1)
    else:
        if random.random()<1/3:
            data.append((gen_ano(["n"])))
            label.append(2)
        elif random.random()<1/2:
            data.append((gen_ano(["s"])))
            label.append(3)
        else:
            data.append((gen_ano(["w"])))
            label.append(4)

for _ in range(1000):
    if random.random()<0.02:
        data.append((gen_ano(["ne", "se", "sw", "nw"])))
        label.append(1)
    else:
        if random.random()<1/4:
            data.append((gen_ano(["n"])))
            label.append(2)
        elif random.random()<1/3:
            data.append((gen_ano(["s"])))
            label.append(3)
        elif random.random()<1/2:
            data.append((gen_ano(["w"])))
            label.append(4)
        else:
            data.append((gen_ano(["e"])))
            label.append(5)


data_np = np.array(data)
label_np = np.array(label)
output = np.concatenate((data_np, label_np.reshape(-1, 1)), axis=1)
np.savetxt("stream9direction.csv", output, delimiter=",")
