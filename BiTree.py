class Node:
    def __init__(self, _data) -> None:
        self.data = _data
        self.lchild = None
        self.rchild = None

class BiTree:
    def __init__(self, _p:Node) -> None:
        self.root = _p

    def set_lchild(self, x:Node):
        p = self.root
        if p.lchild is None:
            p.lchild = x
            return

    def set_rchild(self, x:Node):
        p = self.root
        if p.rchild is None:
            p.rchild = x
            return        
    
    def get_node_child(self, p:Node):
        output = []
        if p.lchild is None and p.rchild is None:
            output.append(p.data)
            return output
        output += self.get_node_child(p.lchild)
        output += self.get_node_child(p.rchild)
        return output

    def get_child(self):
        return self.get_node_child(self.root)