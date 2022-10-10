import numpy as np
from Isolation_Kernel import Isolation_Kernel
from BiTree import BiTree, Node
import random
import math
import matplotlib.pyplot as plt


class IK_AHC(Isolation_Kernel):
    def __init__(self, data: list, psi, t) -> None:
        super().__init__(data, psi, t)

    def streaKHC(self, data=None):
        if data is None:
            data = self.data
        map_data = list(map(self.get_point_feature_map, data))
        map2data = {tuple(map_data[i]) : data[i] for i in range(len(map_data))}
        print(len(map2data))
        AHC_tree = BiTree(Node([map_data[0], 1]))
        for each in map_data[1:]:
            self.grow_tree(AHC_tree, each)

        class_num = 4
        output = [AHC_tree.root]
        while len(output) < class_num:
            min_kernel = min(list(map(lambda x:x.data[0], output)))
            for each in output:
                if each.data[0] == min_kernel:
                    output.remove(each)
                    output.append(each.lchild)
                    output.append(each.rchild)
        for i in range(class_num):
            output.append([])
            for each in AHC_tree.get_node_child(output[i]):
                output[-1].append(map2data[tuple(each[0])])
        for _ in range(class_num):
            output.pop(0)
        return output




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


