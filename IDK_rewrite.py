import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class IDK_rewrite:
    def __init__(self, _X, _psi, _t=100) -> None:
        self.X = _X
        self.psi = _psi
        self.t = _t
        self.center_list = []
        self.radius_list = []
        self.point_fm_list = self.IK_inne_fm()
        self.feature_mean_map: np.ndarray = np.mean(self.point_fm_list, axis=0)

    def IK_inne_fm(self):

        onepoint_matrix = np.zeros(
            (self.X.shape[0], (int)(self.t*self.psi)), dtype=int)
        for time in range(self.t):
            sample_num = self.psi  #
            sample_list = [p for p in range(len(self.X))]
            sample_list = np.random.choice(sample_list, sample_num).tolist()
            # sample_list = random.sample(sample_list, sample_num)
            sample = self.X[sample_list, :]

            tem1 = np.dot(np.square(self.X), np.ones(sample.T.shape))  # n*psi
            tem2 = np.dot(np.ones(self.X.shape), np.square(sample.T))
            point2sample = tem1 + tem2 - 2 * np.dot(self.X, sample.T)  # n*psi

            sample2sample = point2sample[sample_list, :]
            self.center_list += list(sample)
            row, col = np.diag_indices_from(sample2sample)
            sample2sample[row, col] = np.inf
            radius_list = np.min(sample2sample, axis=1)  # 每行的最小值形成一个行向量
            self.radius_list += list(radius_list)

            min_point2sample_index = np.argmin(point2sample, axis=1)
            min_dist_point2sample = min_point2sample_index+time*self.psi
            point2sample_value = point2sample[range(
                len(onepoint_matrix)), min_point2sample_index]
            ind = point2sample_value < radius_list[min_point2sample_index]
            onepoint_matrix[ind, min_dist_point2sample[ind]] = 1
        self.center_list = np.array(self.center_list)
        self.radius_list = np.array(self.radius_list)
        return onepoint_matrix

    def IDK(self):
        return np.dot(self.point_fm_list, self.feature_mean_map)/self.t

    def kappa(self, point):
        feature_map = np.zeros(self.psi * self.t)
        sample2point = np.dot(np.square(self.center_list), np.ones(point.T.shape)) + \
            np.dot(np.ones(self.center_list.shape), np.square(point.T)) - \
            2 * np.dot(self.center_list, point.T)
        sample2point = sample2point.reshape((self.t, self.psi))
        min_point2sample_index = np.argmin(sample2point, axis=1)
        sample2point_value = sample2point[range(
            self.t), min_point2sample_index]
        ind = sample2point_value < self.radius_list[min_point2sample_index]
        for time in range(self.t):
            if ind[time]:
                feature_map[min_point2sample_index[time]+time*self.psi] = 1
        return np.dot(feature_map, self.feature_mean_map)/self.t

    def get_average_threshold(self):
        return np.mean(self.feature_mean_map)

    def get_min_var_threshold(self):
        map_temp = self.feature_mean_map.copy()
        map_temp.sort()
        var_list = np.array([map_temp[:i].var() + map_temp[i:].var()
                            for i in range(1, len(map_temp))])
        return map_temp[np.argmin(var_list)]


if __name__ == "__main__":
    test_data = np.random.randn(1000, 2)
    myx = IDK_rewrite(test_data, 2)
    print(myx.get_min_var_threshold())
    predict = myx.IDK()
    thres = myx.get_average_threshold()
    anomaly = myx.kappa(np.array([3, 0]))
    normal = myx.kappa(np.array([0, 0]))
    print(anomaly/normal)
    # for i in range(len(test_data)):
    #     if myx.kappa(test_data[i]) > 0.000510:
    #         plt.scatter(test_data[i][0], test_data[i][1], c="b")
    #     else:
    #         plt.scatter(test_data[i][0], test_data[i][1], c="r")
    # plt.show()

    # fpr, tpr, thresholds = roc_curve(label, predict, pos_label=2)

    # for i, value in enumerate(thresholds):
    #     print("%f %f %f" % (fpr[i], tpr[i], value))
    # print(thres)
    # roc_auc = auc(fpr, tpr)

    # plt.plot(fpr, tpr, 'k--',
    #          label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()
