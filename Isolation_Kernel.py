import math
import time
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from Hypersphere import Hypersphere
import random


class Isolation_Kernel:
    def __init__(self, data: list, psi, t) -> None:
        self.data = data
        self.psi = psi
        self.t = t
        self.size = len(self.data)
        self.hypersphere_list: list[Hypersphere] = list()
        self.list_feature_map = self.get_list_feature_map(self.data)
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
        return csr_matrix(output).transpose()

    def get_list_feature_map(self, point_list):
        map_list = [self.get_point_feature_map(each) for each in point_list]
        output = csr_matrix(np.zeros((self.psi * self.t, 1)))
        for each in map_list:
            output += each
        lens = len(point_list)
        return output / lens

    def distance(self, x: Tuple, y: Tuple):
        return sum([(x[i] - y[i]) ** 2 for i in range(len(x))]) ** 0.5

    def get_nearest_dis(self, point, data_list: list):
        min_dis = math.inf
        for each in data_list:
            if point == each:
                continue
            min_dis = min(min_dis, self.distance(point, each))
        return min_dis

    def point_point_kernel(self, point1, point2):
        feature_map1 = self.get_point_feature_map(point1)
        feature_map2 = self.get_point_feature_map(point2)
        S = feature_map1.transpose() * feature_map2
        return S[0, 0] / self.t

    def point_list_kernel(self, point, point_list):
        feature_map1 = self.get_point_feature_map(point)
        feature_map2 = self.get_list_feature_map(point_list)
        S = feature_map1.transpose() * feature_map2
        return S[0, 0] / self.t

    def list_list_kernel(self, list1, list2):
        feature_map1 = self.get_list_feature_map(list1)
        feature_map2 = self.get_list_feature_map(list2)
        S = feature_map1.transpose() * feature_map2
        return S[0, 0] / self.t

    def inner(self, x:csr_matrix, y:csr_matrix):
        a = x.transpose() * y
        return a[0, 0]

    def similarity(self, point):
        point_map = self.get_point_feature_map(point)
        output = point_map.transpose() * self.list_feature_map
        return output[0, 0]

