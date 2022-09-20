from turtle import color
from generate_uniform_circle import *
from SENNE import SENNE
import matplotlib.pyplot as plt

DS = []
DS.append(generate_uniform_circle((0, 0), 1, 100))
# DS.append(generate_uniform_circle((2, 2), 1, 100))
new_point = (1, 1)

for each in DS:
    for point in each:
        plt.scatter(point[0], point[1], color="red")
plt.scatter(new_point[0], new_point[1], color="blue")

myx = SENNE(DS, 32, 100, 0.6)
print(f"{myx.get_Ni(new_point, 0)}")
plt.show()