from typing import Tuple


class Hypersphere:
    def __init__(self, center: Tuple[float], radius: float) -> None:
        self.center = center
        self.radius = radius
        self.dim = len(self.center)

    def isIn(self, point: Tuple[float]) -> bool:
        if len(point) != len(self.center):
            raise "point not match dim"
        S = 0
        for i in range(self.dim):
            S += (point[i] - self.center[i]) ** 2
        S = S ** (1/2)
        if S <= self.radius:
            return True
        else:
            return False

