import numpy as np
from Isolation_Kernel import Isolation_Kernel
from BiTree import BiTree, Node
from typing import List
import random
import math
import matplotlib.pyplot as plt


class IK_AHC(Isolation_Kernel):
    def __init__(self, data: List, psi, t) -> None:
        super().__init__(data, psi, t)

    def streaKHC(self, data=None):
        if data is None:
            data = self.data
        map_data = list(map(self.get_point_feature_map, data))
        map2data = {tuple(map_data[i]) : data[i] for i in range(len(map_data))}
        AHC_tree = BiTree(Node([map_data[0], 1]))
        for each in map_data[1:]:
            self.grow_tree(AHC_tree, each)

        c1 = AHC_tree.get_node_child(AHC_tree.root.lchild)
        temp1 = [tuple(c1[i][0]) for i in range(len(c1))]
        points1 = [map2data[temp1[i]]for i in range(len(temp1))]

        c2 = AHC_tree.get_node_child(AHC_tree.root.rchild)
        temp2 = [tuple(c2[i][0]) for i in range(len(c2))]
        points2 = [map2data[temp2[i]]for i in range(len(temp2))]

        # c3 = AHC_tree.get_node_child(AHC_tree.root.rchild)
        # temp3 = [tuple(c3[i][0]) for i in range(len(c3))]
        # points3 = [map2data[temp3[i]]for i in range(len(temp3))]

        # c4 = AHC_tree.get_node_child(AHC_tree.root.rchild.rchild)
        # temp4 = [tuple(c4[i][0]) for i in range(len(c4))]
        # points4 = [map2data[temp4[i]]for i in range(len(temp4))]

        return points1, points2

    def grow_tree(self, tree: BiTree, new_node):
        p = tree.root
        while p.lchild is not None and p.rchild is not None:

            p.data[0] = [p.data[0][i]*p.data[1] for i in range(len(p.data[0]))]
            p.data[0] = self.list_add(p.data[0], new_node)
            p.data[0] = [p.data[0][i]/(p.data[1] + 1)
                         for i in range(len(p.data[0]))]
            p.data[1] += 1

            l_kernel = self.list_mult(p.lchild.data[0], new_node)/(self.list_mult(p.lchild.data[0], p.lchild.data[0])**(1/2)*self.list_mult(new_node, new_node)**(1/2))
            r_kernel = self.list_mult(p.rchild.data[0], new_node)/(self.list_mult(p.rchild.data[0], p.rchild.data[0])**(1/2)*self.list_mult(new_node, new_node)**(1/2))
            if l_kernel >= r_kernel:
                p = p.lchild
            else:
                p = p.rchild

        p.lchild = Node(p.data)
        p.rchild = Node([new_node, 1])
        p.data = [self.list_add(p.lchild.data[0], p.rchild.data[0], p.lchild.data[1] +
                                p.rchild.data[1]), p.lchild.data[1] + p.rchild.data[1]]


DS = list()
class1 = list()
class2 = list()
TS = list()
ts1 = list()
ts2 = list()


c1, r1 = (0, 0), 1
c2, r2 = (2, 2), 1
c3, r3 = (5, 5), 2
c4, r4 = (0, 5), 2

DS = list()
class1 = list()
class2 = list()
class3 = list()
class4 = list()
for _ in range(100):
    r = random.random() * r1
    theta = random.random() * 2 * math.pi
    class1.append((c1[0] + r * math.cos(theta), c1[1] + r * math.sin(theta)))
for _ in range(100):
    r = random.random() * r2
    theta = random.random() * 2 * math.pi
    class2.append((c2[0] + r * math.cos(theta), c2[1] + r * math.sin(theta)))
for _ in range(100):
    r = random.random() * r3
    theta = random.random() * 2 * math.pi
    class3.append((c3[0] + r * math.cos(theta), c3[1] + r * math.sin(theta)))
for _ in range(100):
    r = random.random() * r4
    theta = random.random() * 2 * math.pi
    class4.append((c4[0] + r * math.cos(theta), c4[1] + r * math.sin(theta)))


for _ in range(10):
    r = random.random() * r1
    theta = random.random() * 2 * math.pi
    ts1.append((c1[0] + r * math.cos(theta), c1[1] + r * math.sin(theta)))
for _ in range(10):
    r = random.random() * r2
    theta = random.random() * 2 * math.pi
    ts2.append((c2[0] + r * math.cos(theta), c2[1] + r * math.sin(theta)))

DS += class1
# DS += class2
# DS += class3
DS += class4
random.shuffle(DS)

TS += ts1
TS += ts2

myx = IK_AHC(DS, 2, 300)
a = myx.streaKHC()
for each in a[0]:
    plt.scatter(each[0],each[1],c="r")
for each in a[1]:
    plt.scatter(each[0],each[1],c="b")
# for each in a[2]:
#     plt.scatter(each[0],each[1],c="g")
# for each in a[3]:
#     plt.scatter(each[0],each[1],c="y")
plt.show()

# print("===============================")
# print(a.rchild.root[1])
# pass

