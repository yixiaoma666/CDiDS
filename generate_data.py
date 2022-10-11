from random import random
from Adwin import Adwin
import numpy
import random
from generate_uniform_circle import generate_uniform_circle
import matplotlib.pyplot as plt


lens = 1000
data = []
label = []

for i in range(lens):
    if random.random()<0.98:
        data += generate_uniform_circle((i/100, i/100), 1, 1)
        label.append(0)
    else:
        data += generate_uniform_circle((i/100, i/100), 10, 1)
        label.append(1)

numpy.savetxt("data7.csv", numpy.array([(data[i][0],data[i][1], label[i]) for i in range(lens)]), delimiter=",")

for i in range(lens):
    if label[i] == 0:
        plt.scatter(data[i][0], data[i][1], c="b")
for i in range(lens):
    if label[i] == 1:
        plt.scatter(data[i][0], data[i][1], c="r")
plt.savefig("data7.png")
plt.show()
# lists = numpy.loadtxt("data5.csv", delimiter=",")
# lists = [random.random()/10 for _ in range(1000)] + [random.random()/10+0.9 for _ in range(1000)] + [random.random()/10 for _ in range(1000)]
# numpy.savetxt("data6.csv", lists, delimiter=",")
# lists = numpy.array([0] * 1000 + [1] * 30 + [0] * 1000)

# myx = Adwin(lists).show_fig("output10.png")