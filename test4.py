import math
import matplotlib.pyplot as plt
import random
import numpy

lists = [random.normalvariate(0.8, 0.01) for _ in range(1000)] + \
        [random.normalvariate(0.8 - i / 1000 * 0.6, 0.01) for i in range(1000)] + \
        [random.normalvariate(0.2, 0.01) for _ in range(2000)]


delta = 0.002


numpy.savetxt("data3.csv", lists, delimiter=",")

def partition(mylist: list, left):
    return mylist[:left], mylist[left:]


    

def cal_max_ratio(W):        
    _max = 0
    n = len(W)
    M = 5
    c = 1 + 1 / M

    part_list = list()
    i = 0
    while n - int(c**i) > 1:
        part_list.append(n - int(c**i))
        i += 1
    part_list = list(set(part_list))

    for i in part_list:
        W_0, W_1 = partition(W, i)
        mu_0 = sum(W_0) / len(W_0)
        mu_1 = sum(W_1) / len(W_1)
        n_0 = len(W_0)
        n_1 = len(W_1)
        m = 1 / (1 / n_0 + 1 / n_1)
        delta_ = delta / n
        variance = numpy.var(W)
        epsilon = (math.log(2 / delta_) * 2 / m * variance) ** (1/2) + 2 / (3 * m) * math.log(2 / delta_)
        _max = max(abs(mu_0-mu_1)/epsilon, _max)
    return _max

window = list()
window_len = list()
window_mean = list()



for i in range(len(lists)):
    if len(window)<3 or cal_max_ratio(window) < 1:
        window.append(lists[i])
    while len(window) >=3 and cal_max_ratio(window) >= 1:
        window.pop(0)
    window_len.append(len(window))
    window_mean.append(numpy.mean(window))

# numpy.savetxt("output.csv", numpy.concatenate((numpy.array(lists).reshape(3000, 1), numpy.array(window_len).reshape(3000, 1), numpy.array(window_mean).reshape(3000, 1)), 1), delimiter=",")
    
x_axis_data = [i for i in range(1, len(lists) + 1)]

fig, ax1 = plt.subplots()

ax1.plot(x_axis_data, lists, color="red")
ax1.plot(x_axis_data, window_mean, color="green")
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.plot(x_axis_data, window_len, color="blue")


plt.show()
# plt.savefig("output.png")