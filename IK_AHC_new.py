from Isolation_Kernel import Isolation_Kernel
from BiTree import BiTree, Node
import copy
import time


class IK_AHC_new(Isolation_Kernel):
    def __init__(self, data: list, psi, t) -> None:
        super().__init__(data, psi, t)

    def streaKHC(self, data=None, class_num=2):
        if data is None:
            data = self.data
        map_data = list(map(self.get_point_feature_map, data))
        AHC_tree = BiTree(Node([[data[0]], map_data[0], 1]))
        for each in data[1:]:
            self.grow_tree(AHC_tree, each)

        output = [AHC_tree.root]
        while len(output) < class_num:
            min_kernel = min(list(map(lambda x:(self.inner(x.lchild.data[1], x.rchild.data[1]))/((self.inner(x.lchild.data[1], x.lchild.data[1]))**0.5*(self.inner(x.rchild.data[1], x.rchild.data[1]))**0.5), output)))
            for each in output:
                if self.inner(each.lchild.data[1],each.rchild.data[1])/(self.inner(each.lchild.data[1],each.lchild.data[1])**0.5*self.inner(each.rchild.data[1],each.rchild.data[1])**0.5) == min_kernel:
                    output.remove(each)
                    output.append(each.lchild)
                    output.append(each.rchild)
                    break
        for i in range(class_num):
            output[i] = output[i].data[0]
        return output

    def grow_tree(self, tree: BiTree, new_node):
        p = tree.root
        map_new_node = self.get_point_feature_map(new_node)

        while p.lchild is not None and p.rchild is not None:

            p.data[0].append(new_node)
            p.data[1] = p.data[1]*p.data[2]/(p.data[2]+1) + map_new_node/(p.data[2]+1)
            p.data[2] += 1

            l_kernel = self.inner(p.lchild.data[1], map_new_node)/(self.inner(p.lchild.data[1],p.lchild.data[1])**0.5*self.inner(map_new_node,map_new_node)**0.5)
            r_kernel = self.inner(p.rchild.data[1], map_new_node)/(self.inner(p.rchild.data[1],p.rchild.data[1])**0.5*self.inner(map_new_node,map_new_node)**0.5)
            if l_kernel > r_kernel:
                p = p.lchild
            else:
                p = p.rchild
        p.lchild = Node(copy.deepcopy(p.data))
        p.rchild = Node(copy.deepcopy([[new_node], map_new_node, 1]))
        p.data[0] = p.lchild.data[0] + p.rchild.data[0]
        p.data[1] = (p.lchild.data[1] + p.rchild.data[1]) / 2
        p.data[2] = 2
