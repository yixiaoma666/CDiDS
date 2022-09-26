import math
from typing import Tuple

import numpy as np
from Hypersphere import Hypersphere
import random


class Isolation_Kernel:
    def __init__(self, data: list, psi, t) -> None:
        self.data = data
        self.psi = psi
        self.t = t
        self.size = len(self.data)
        self.hypersphere_list: list[Hypersphere] = list()
        self.get_hyperspheres()

    def get_hyperspheres(self):
        for _ in range(self.t):
            samp = random.sample(self.data, self.psi)
            for each in samp:
                self.hypersphere_list.append(Hypersphere(center=each,
                                                         radius=self.get_nearest_dis(each, samp)))

    def get_point_feature_map(self, point):
        output = [0] * (self.psi * self.t)
        for _t in range(self.t):
            temp = 0
            min_dis = np.inf
            for _psi in range(self.psi):
                if self.hypersphere_list[self.psi * _t + _psi].isIn(point):
                    if self.distance(self.hypersphere_list[self.psi * _t + _psi].center, point) < min_dis:
                        min_dis = self.distance(
                            self.hypersphere_list[self.psi * _t + _psi].center, point)
                        temp = _psi
            output[self.psi * _t + temp] = 1
        return output

    def get_list_feature_map(self, point_list):
        map_list = [self.get_point_feature_map(each) for each in point_list]
        output = [0] * (self.psi * self.t)
        for each1 in map_list:
            for i in range(len(each1)):
                output[i] += each1[i]
        lens = len(point_list)
        return list(map(lambda x: x/lens, output))

    def distance(self, x: Tuple, y: Tuple):
        S = 0
        for i in range(len(x)):
            S += (x[i] - y[i]) ** 2
        S = S ** (1 / 2)
        return S

    def get_nearest_dis(self, point, data_list: list):
        min_dis = math.inf
        for each in data_list:
            if point == each:
                continue
            min_dis = min(min_dis, self.distance(point, each))
        return min_dis

    def list_mult(self, list1, list2):
        return sum([list1[i] * list2[i] for i in range(len(list1))])

    def list_add(self, list1, list2, num=1):
        n = len(list1)
        return [(list1[i] + list2[i]) / num for i in range(n)]

    def point_point_kernel(self, point1, point2):
        feature_map1 = self.get_point_feature_map(point1)
        feature_map2 = self.get_point_feature_map(point2)
        S = self.list_mult(feature_map1, feature_map2)
        return S/self.t

    def point_list_kernel(self, point, point_list):
        feature_map1 = self.get_point_feature_map(point)
        feature_map2 = self.get_list_feature_map(point_list)
        S = self.list_mult(feature_map1, feature_map2)
        return S/self.t

    def list_list_kernel(self, list1, list2):
        feature_map1 = self.get_list_feature_map(list1)
        feature_map2 = self.get_list_feature_map(list2)
        S = self.list_mult(feature_map1, feature_map2)
        return S/self.t

    def new_get_point_feature_map(self, point):
        output = list()
        for _t in range(self.t):
            for _psi in range(self.psi):
                if self.hypersphere_list[self.psi * _t + _psi].isIn(point):
                    output.append(1)
                else:
                    output.append(0)
        return output
    
    def new_get_list_feature_map(self, point_list):
        map_list = [self.new_get_point_feature_map(each) for each in point_list]
        output = [0] * (self.psi * self.t)
        for each1 in map_list:
            for i in range(len(each1)):
                output[i] += each1[i]
        lens = len(point_list)
        return list(map(lambda x: x/lens, output))


