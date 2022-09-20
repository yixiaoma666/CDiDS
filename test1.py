from IK_AHC import IK_AHC
from IK_AHC_new import IK_AHC_new
from SENNE import SENNE
import random
import matplotlib.pyplot as plt
import math

from generate_uniform_circle import generate_uniform_circle

DS = []
DS += generate_uniform_circle((0, 0), 1, 100)
DS += generate_uniform_circle((5, 5), 1, 100)
DS += generate_uniform_circle((0, 5), 1, 100)
DS += generate_uniform_circle((5, 0), 1, 100)
random.shuffle(DS)


myx = IK_AHC_new(DS, 2, 1000)
c1, c2, c3, c4 = myx.streaKHC(class_num=4)
# c1, c2, c3, c4= myx.streaKHC()
for each in c1:
    plt.scatter(each[0], each[1], color="red")

for each in c2:
    plt.scatter(each[0], each[1], color="blue")

for each in c3:
    plt.scatter(each[0], each[1], color="green")

for each in c4:
    plt.scatter(each[0], each[1], color="yellow")


plt.show()

# new_point = (1, 1)



