import random
import numpy

lists = [0] * 1000 + [random.randint(0, 1) for _ in range(100)] + [1] * 1000

numpy.savetxt("data4.csv", lists, delimiter=",")

