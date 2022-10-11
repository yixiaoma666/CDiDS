import numpy as np
from scipy.sparse import csr_array
from scipy.spatial.distance import cdist, pdist, squareform
from generate_uniform_circle import generate_uniform_circle
from Hypersphere import Hypersphere

class IDK:
    def __init__(self,
                 _data: np.ndarray,
                 _psi: int,
                 _t: int) -> None:
        """生成data的IDK

        Args:
        ------
            _data (np.array): data
            _psi (int): psi
            _t (int): t

        """
        # self.data = self.normalization(_data)
        self.data = _data
        self.psi = _psi
        self.t = _t
        self.size = self.data.shape[0]
        self.dim = self.data.shape[1]
        self.hyperspheres_center_list = np.zeros((1, self.dim))
        self.hyperspheres_radius_list = np.zeros((1, 1))
        self.list_feature_map = csr_array(np.zeros((self.psi * self.t, 1)))
        self.get_map()

    def get_map(self):
        for _t in range(self.t):
            sample_num = np.random.choice(a=self.size,
                                           size=self.psi,
                                           replace=False)
            sample_data = self.data[sample_num]
            dist_mat = cdist(XA=self.data,
                             XB=sample_data,
                             metric="euclidean")
            temp_hypersphere_center, temp_hypersphere_radius = self.get_hyperspheres(dist_mat, sample_num)
            for i in range(self.size):
                nearest_sample_point_index = np.argmin(self.data[i])
                if self.is_in(temp_hypersphere_center[nearest_sample_point_index],
                              temp_hypersphere_radius[nearest_sample_point_index],
                              tuple(self.data[i, :])):
                    self.list_feature_map[_t*self.psi + nearest_sample_point_index, 0] += 1
            self.hyperspheres_center_list = np.concatenate((self.hyperspheres_center_list, np.array(temp_hypersphere_center)), axis=0)
            self.hyperspheres_radius_list = np.concatenate((self.hyperspheres_radius_list, np.array(temp_hypersphere_radius).reshape(self.psi, 1)), axis=0)
        self.hyperspheres_center_list = np.delete(self.hyperspheres_center_list, 0, 0)
        self.hyperspheres_radius_list = np.delete(self.hyperspheres_radius_list, 0, 0)
        self.list_feature_map /= (self.size * self.t ** 0.5)
        self.list_feature_map /= (self.list_feature_map.transpose().dot(self.list_feature_map)[0, 0])**0.5


    def get_hyperspheres(self, dist_mat, sample_num):
        dist_mat = dist_mat[sample_num, :]
        center_output = []
        radius_output = []
        for i in range(dist_mat.shape[0]):
            dist_mat[i, i] = np.inf
            center_output.append(tuple(self.data[sample_num[i],:]))
            radius_output.append(np.min(dist_mat[i, :]))
            dist_mat[i, i] = 0
        return center_output, radius_output

    @staticmethod
    def normalization(x: np.ndarray):
        _max = np.max(x)
        _min = np.min(x)
        return (x - _min) / (_max - _min)

    def kappa(self, point):
        feature_map = csr_array(np.zeros((self.psi * self.t, 1)))
        for _t in range(self.t):
            dist_mat = cdist(point, self.hyperspheres_center_list[_t * self.psi:(_t+1)*self.psi, :])
            nearest_sample_point_index = np.argmin(dist_mat[0, :])
            if self.is_in(self.hyperspheres_center_list[_t * self.psi + nearest_sample_point_index],
                          self.hyperspheres_radius_list[_t * self.psi + nearest_sample_point_index],
                          point):
                feature_map[_t * self.psi + nearest_sample_point_index, 0] += 1
        feature_map /= (self.t ** 0.5)
        output = feature_map.transpose().dot(self.list_feature_map)[0, 0]
        return output


    @staticmethod
    def is_in(x, radius, y):
        return np.sum((np.array(x)-np.array(y))**2) < radius ** 2



def main():
    for epoch in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        S = 0
        for _ in range(100):
            # test_data = np.array(generate_uniform_circle((0, 0), 1, 100))
            test_data = np.random.randn(100, 2)
            myx = IDK(test_data, epoch, 100)
            output = myx.kappa(np.array([[3, 0]]))
            # print(i, output)
            S += output
        S /= 100
        print(f"$\psi={epoch},t=100,x=\mu+2\sigma,\kappa={S:.5f}$")
if __name__ == "__main__":
    main()