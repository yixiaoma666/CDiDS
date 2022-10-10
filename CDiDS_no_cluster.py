from Isolation_Kernel import Isolation_Kernel


class CDiDS:
    def __init__(self, _data, _psi=2 , _t=100, _memory=200) -> None:
        """
        :param _data: 数据
        :param _memory: 内存限制
        ---
        :return: abc
        """
        self.data = _data
        self.psi = _psi
        self.t = _t
        self.memory = _memory
        self.window = []

    def detect_drift(self):
        
        pass

    def make_kernel(self):
        self.kernel = Isolation_Kernel(self.data, self.psi, self.t)
        pass


