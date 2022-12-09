import numpy as np

class BaseLineModel:
    def __init__(self,
                 _data: np.ndarray) -> None:
        self.data = _data
        self.mean, self.var = self._get_model()

    def _get_model(self):
        mean = self.data.mean(0)
        var = (self.data.var(0) ** 0.5).mean()
        return mean, var

    def get_average_threshold(self):
        S = 0
        for each in self.data:
            S += self.kappa(each)
        return S / self.size

    def kappa(self,
              point: np.ndarray):
        return self._phi(point, self.mean, self.var)

    @staticmethod
    def _phi(x: np.ndarray,
            mu: np.ndarray,
            sigma: float) -> float:
        dis = np.sum(((x - mu) ** 2)) ** 0.5
        return np.exp(-dis**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    @property
    def size(self):
        return self.data.shape[0]

if __name__ == "__main__":
    test_data = np.random.randn(100, 2)
    myx = BaseLineModel(test_data)
    a = myx.get_average_threshold()
    # a = myx.kappa(np.array([3, 0]))
    print(a)
    k = 1 / 40