from random import random
from Adwin import Adwin
import numpy
import random



# lists = numpy.loadtxt("data5.csv", delimiter=",")
# lists = [random.random()/10 for _ in range(1000)] + [random.random()/10+0.9 for _ in range(1000)] + [random.random()/10 for _ in range(1000)]
# numpy.savetxt("data6.csv", lists, delimiter=",")
lists = numpy.array([0] * 1000 + [1] * 30 + [0] * 1000)

myx = Adwin(lists).show_fig("output10.png")