import copy
import math
import random
from typing import List, Set, Tuple
from Hypersphere import Hypersphere


class SENNE:
    def __init__(self, _dataset: List[List[Tuple[float]]], _psi, _p, _t) -> None:
        """
        :param _dataset: 数据集
        :param _psi: IK参数
        :param _p: IK参数
        :param _t: 归类阈值

        归类阈值越大，归类越严格
        """
        self.dataset = _dataset
        self.psi = _psi
        self.p = _p
        self.t = _t

        self.k = len(self.dataset)
        self.hypersphere_list = list()
        self.get_isolation_hypersphere()

    def get_class_isolation_hypersphere(self, temp_class: List[Tuple[float]]):
        """
        对一个分类构造Hypersphere
        """
        output = list()
        for _ in range(self.p):
            temp_hyperspheres = list()
            samples = random.sample(temp_class, self.psi)
            for each in samples:
                temp_hyperspheres.append(Hypersphere(
                    each, self.get_nearest(each, samples)))
            output.append(temp_hyperspheres)
        return output

    def get_isolation_hypersphere(self):
        """
        对所有分类构造Hyperspheres
        """
        for each in self.dataset:
            self.hypersphere_list.append(
                self.get_class_isolation_hypersphere(each))

    def get_nearest(self, point, data_list):
        """
        :return: 返回data_list中距离point最近且不为point的点，
        """
        nearest_point = 0
        min_dis = math.inf
        for each in data_list:
            if point == each:
                continue
            temp_distance = self.get_distance(each, point)
            if temp_distance < min_dis:
                nearest_point = each
                min_dis = temp_distance
        return min_dis

    def get_distance(self, x, y):
        """
        :return: 返回x和y的距离
        """
        if len(x) != len(y):
            raise "DimError"
        S = 0
        for i in range(len(x)):
            S += (x[i] - y[i]) ** 2
        S = S ** (1/2)
        return S

    def get_i_Ni(self, x, i, k):
        """
        :return: 返回N_i^{(k)}(x)
        """
        temp_B = self.hypersphere_list[k][i]
        cnn_dis = math.inf
        cnn = 0
        flag = False
        for each in temp_B:
            if each.isIn(x) and each.radius < cnn_dis:
                flag = True
                cnn_dis = each.radius
                cnn = each.center
        if not flag:
            return 1
        return max(0, 1-self.get_nearest(cnn, self.dataset[k])/self.get_distance(x, cnn))

    def get_Ni(self, x, k):
        """
        :return: N^{(k)}(x)
        """
        output = 0
        for i in range(self.p):
            output += self.get_i_Ni(x, i, k)
        output /= self.p
        return output

    def get_Pi(self, x, k):
        """
        :return: P^{(k)}(x)
        """
        output = 0
        for i in range(self.p):
            if self.get_i_Ni(x, i, k) < self.t:
                output += 1
        output /= self.p
        return output

    def f(self, x):
        """
        :return: f(x)
        """
        NC_flag = True
        for k in range(self.k):
            if self.get_Ni(x, k) < self.t:
                NC_flag = False
        if NC_flag:
            return "NC"
        temp_k = 0
        min_P = 0
        for k in range(self.k):
            temp_Pi = self.get_Pi(x, k)
            if temp_Pi > min_P:
                temp_k = k
                min_p = temp_Pi
        return temp_k

    def classify(self):
        output = list()
        temp_set_list = []
        for each_p in self.hypersphere_list[0]:
            for each_psi in each_p:
                temp_set = {each_psi.center}
                for data in self.dataset[0]:
                    if each_psi.isIn(data):
                        temp_set.add(data)
                temp_set_list.append(temp_set)
        output = self.compose_one(temp_set_list)

        return list(map(list, output))

    def compose_one(self, data_list: List[Set]):
        """
        将data_list中交不为空的集合取并集
        """
        output = copy.deepcopy(data_list)
        i = 0
        while True:
            if i == len(output):
                break
            flag = False
            for j in range(i+1, len(output)):
                if len(output[i] & output[j]) != 0:
                    output[j] |= output[i]
                    flag = True
                    break
            if flag:
                output.pop(i)
            else:
                i += 1
        return output
