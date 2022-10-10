# 此方法暂且可用，待完善


import math
import matplotlib.pyplot as plt
import numpy


class Adwin:
    def __init__(self, _lists, _delta=0.002, _M=2, _window_threshold=10, show_fig=False) -> None:
        self.lists = _lists
        self.delta = _delta
        self.M = _M
        self.window_threshold = _window_threshold



    def min_max_norm(self):
        self.lists = (self.lists - numpy.min(self.lists)) / \
            (numpy.max(self.lists) - numpy.min(self.lists))

    @staticmethod
    def partition(mylist: list, left):
        return mylist[:left], mylist[left:]

    def cal_max_ratio(self, W):
        _max = 0
        n = len(W)
        c = 1 + 1 / self.M

        part_list = list()
        i = 0
        while n - int(c**i) > 1:
            part_list.append(n - int(c**i))
            i += 1
        part_list = list(set(part_list))

        for i in part_list:
            W_0, W_1 = self.partition(W, i)
            mu_0 = sum(W_0) / len(W_0)
            mu_1 = sum(W_1) / len(W_1)
            n_0 = len(W_0)
            n_1 = len(W_1)
            m = 1 / (1 / n_0 + 1 / n_1)
            delta_ = self.delta / n
            variance = numpy.var(W)
            epsilon = (math.log(2 / delta_) * 2 / m *
                       variance) ** (1/2) + 2 / (3 * m) * math.log(2 / delta_)
            _max = max(abs(mu_0-mu_1)/epsilon, _max)
        return _max

    def detect_drift(self):
        window = list()
        window_len = list()
        window_mean = list()
        output_list = list()

        for i in range(len(self.lists)):
            if len(window) < 3 or self.cal_max_ratio(window) < 1:
                window.append(self.lists[i])
            while len(window) >= 3 and self.cal_max_ratio(window) >= 1:
                window.pop(0)
            window_len.append(len(window))
            window_mean.append(numpy.mean(window))

        max_point, min_point = -1, -1
        check_max = True
        temp_i = 0
        for i in range(len(self.lists)):
            if check_max:
                if window_len[i-1] >= max(window_len[i], window_len[i-2]):
                    max_point = window_len[i-1]
                    temp_i = i-1
                    check_max = False
            else:
                if window_len[i-1] <= min(window_len[i], window_len[i-2]):
                    min_point = window_len[i-1]
                    if max_point - min_point > self.window_threshold:
                        output_list.append((temp_i, self.lists[i], max_point - min_point))
                    check_max = True
        output_list.pop(0)
        output_list.sort(key=lambda x: x[2], reverse=True)
        self.window_len = window_len
        self.window_mean = window_mean
        return output_list

    def show_fig(self, fname):
        self.detect_drift()
        x_axis_data = [i for i in range(1, len(self.lists) + 1)]
        fig, ax1 = plt.subplots()

        ax1.plot(x_axis_data, self.lists, color="red")
        ax1.plot(x_axis_data, self.window_mean, color="green")
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        ax2.plot(x_axis_data, self.window_len, color="blue")

        # plt.show()
        plt.savefig(fname)




# lists = [0]*1000 + [1]*1000 + [0]*1000


# delta = 0.002


# lists = numpy.loadtxt("data5.csv", delimiter=",")
# lists = (lists - numpy.min(lists))/(numpy.max(lists) - numpy.min(lists))





# numpy.savetxt("output.csv", numpy.concatenate((numpy.array(lists).reshape(3000, 1), numpy.array(window_len).reshape(3000, 1), numpy.array(window_mean).reshape(3000, 1)), 1), delimiter=",")

# x_axis_data = [i for i in range(1, len(lists) + 1)]

# fig, ax1 = plt.subplots()

# ax1.plot(x_axis_data, lists, color="red")
# ax1.plot(x_axis_data, window_mean, color="green")
# ax1.set_ylim(0, 1)

# ax2 = ax1.twinx()
# ax2.plot(x_axis_data, window_len, color="blue")

# output_list.pop(0)
# output_list.sort(key=lambda x: x[2], reverse=True)
# for each in output_list:
#     print(each)
# plt.show()
# # plt.savefig("output1.png")
