import numpy as np
from scipy.sparse import csr_array
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class IDK:
    def __init__(self,
                 _data: np.ndarray,
                 _psi: int,
                 _t: int = 100) -> None:
        """生成data的IDK

        Args:
        ------
            _data (np.array): data
            _psi (int): psi
            _t (int): t

        """
        # self.data = self._normalization(_data)
        self.data = _data
        self.psi = _psi
        self.t = _t
        self.size = self.data.shape[0]
        self.dim = self.data.shape[1]
        self.hyperspheres_center_list = np.zeros((1, self.dim))
        self.hyperspheres_radius_list = np.zeros((1, 1))
        self.list_feature_map = csr_array(np.zeros((self.psi * self.t, 1)))
        self._get_map()

    def _get_map(self):
        for _t in range(self.t):
            sample_num = np.random.choice(a=self.size,
                                          size=self.psi,
                                          replace=False)
            sample_num = [1000, 1500]
            sample_data = self.data[sample_num]
            dist_mat = cdist(XA=self.data,
                             XB=sample_data,
                             metric="euclidean")
            temp_hypersphere_center, temp_hypersphere_radius = self._get_hyperspheres(
                dist_mat, sample_num)
            for i in range(self.size):
                nearest_sample_point_index = np.argmin(self.data[i])
                if self._is_in(temp_hypersphere_center[nearest_sample_point_index],
                               temp_hypersphere_radius[nearest_sample_point_index],
                               tuple(self.data[i, :])):
                    self.list_feature_map[_t*self.psi +
                                          nearest_sample_point_index, 0] = 1
            self.hyperspheres_center_list = np.concatenate(
                (self.hyperspheres_center_list, np.array(temp_hypersphere_center)), axis=0)
            self.hyperspheres_radius_list = np.concatenate((self.hyperspheres_radius_list, np.array(
                temp_hypersphere_radius).reshape(self.psi, 1)), axis=0)
        self.hyperspheres_center_list = np.delete(
            self.hyperspheres_center_list, 0, 0)
        self.hyperspheres_radius_list = np.delete(
            self.hyperspheres_radius_list, 0, 0)
        self.list_feature_map /= (self.size)

    def _get_hyperspheres(self, dist_mat: np.ndarray, sample_num):
        dist_mat = dist_mat[sample_num, :]
        center_output = []
        radius_output = []
        for i in range(dist_mat.shape[0]):
            dist_mat[i, i] = np.inf
            center_output.append(tuple(self.data[sample_num[i], :]))
            radius_output.append(np.min(dist_mat[i, :]))
            dist_mat[i, i] = 0
        return center_output, radius_output

    @staticmethod
    def _normalization(x: np.ndarray) -> np.ndarray:
        _max = np.max(x)
        _min = np.min(x)
        return (x - _min) / (_max - _min)

    def kappa(self, point):
        feature_map = csr_array(np.zeros((self.psi * self.t, 1)))
        for _t in range(self.t):
            dist_mat = cdist(
                [point], self.hyperspheres_center_list[_t * self.psi:(_t+1)*self.psi, :])
            nearest_sample_point_index = np.argmin(dist_mat[0, :])
            if self._is_in(self.hyperspheres_center_list[_t * self.psi + nearest_sample_point_index],
                           self.hyperspheres_radius_list[_t *
                                                         self.psi + nearest_sample_point_index],
                           point):
                feature_map[_t * self.psi + nearest_sample_point_index, 0] += 1
        output = feature_map.transpose().dot(self.list_feature_map)[0, 0]
        if (feature_map.transpose().dot(feature_map))[0, 0] == 0:
            return 0
        output /= ((self.list_feature_map.transpose().dot(self.list_feature_map))[0, 0]**0.5 *
                   (feature_map.transpose().dot(feature_map))[0, 0]**0.5)
        return output

    @staticmethod
    def _is_in(x, radius, y):
        return np.sum((np.array(x)-np.array(y))**2) < radius ** 2

    def get_average_thershold(self) -> float:
        output = 0
        for each in self.data:
            output += self.kappa(each)
        output /= self.size
        return output

    def get_score(self):
        output = []
        for _data in self.data:
            output.append(self.kappa(_data))
        return output


def main():
    test_data = np.loadtxt("data8.csv", delimiter=",")[:, :2]
    label = np.loadtxt("data8.csv", delimiter=",")[:, 2] + 2
    myx = IDK(test_data, 2)
    predict = myx.get_score()

    fpr, tpr, thersholds = roc_curve(label, predict, pos_label=2)

    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == "__main__":
    main()
