import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
import matplotlib.pyplot as plt
import time
import scipy.io as scio


class IDK_rewrite:
    def __init__(self, _X, _psi, _t=100, _weight=None) -> None:
        self.X = _X
        self.psi = _psi
        self.t = _t
        self.weight = _weight
        self.center_list = []
        self.radius_list = []
        self.point_fm_list = self.IK_inne_fm()
        self.feature_mean_map: np.ndarray = self.get_weight_feature_map()

    def IK_inne_fm(self):

        onepoint_matrix = np.zeros(
            (self.X.shape[0], (int)(self.t*self.psi)), dtype=float)
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
        if self.weight is None:
            return onepoint_matrix
        else:
            self.weight = self.norm(self.weight)
            for i in range(len(self.weight)):
                onepoint_matrix[i] *= self.weight[i]
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

    @staticmethod
    def norm(x):
        return x / np.max(x)

    def get_weight_feature_map(self):
        if self.weight is None:
            return np.mean(self.point_fm_list, axis=0)
        else:
            self.weight = self.norm(self.weight)
            temp = np.zeros(self.point_fm_list.shape)
            for i in range(len(self.weight)):
                temp[i] = self.point_fm_list[i] * self.weight[i]
            return np.mean(temp, axis=0)


if __name__ == "__main__":
    # test_data = np.random.randn(1000, 2)
    # anomaly_data = np.random.randn(20, 2)+np.array((10, 0))
    # all_data = np.concatenate((test_data, anomaly_data), axis=0)
    all_data = np.loadtxt("data_set\cover.csv", delimiter=",")
    data = all_data[:, :-1]
    label = all_data[:, -1].tolist()
    myx = IDK_rewrite(all_data, 16, _t=100)
    idk = myx.IDK()
    best_acc = 0
    for thre in np.sort(np.unique(idk)):
        temp = np.zeros(idk.shape)
        temp[idk <= thre] = 1
        temp = temp.tolist()
        best_acc = max(f1_score(label, temp), best_acc)
        print(best_acc)

    # plt.plot(np.sort(idk))
    # plt.scatter(all_data[idk!=0][:, 0], all_data[idk!=0][:, 1], c="b")
    # plt.scatter(all_data[idk==0][:, 0], all_data[idk==0][:, 1], c="r")
    # plt.show()
    # plt.clf()
    for epoch in range(100):
        sort_idk = np.sort(idk)
        thres = sort_idk[int(len(sort_idk)/1020)+1]
        new_idk = np.zeros(idk.shape)
        new_idk[idk > thres] = 1
        new_idk = idk
        myx = IDK_rewrite(all_data, 4, _t=20, _weight=new_idk)
        idk = myx.IDK()
        plt.scatter(all_data[idk != 0][:, 0], all_data[idk != 0][:, 1], c="b")
        plt.scatter(all_data[idk == 0][:, 0], all_data[idk == 0][:, 1], c="r")
        plt.savefig(f"pic_out/{time.time()}.png")
        plt.clf()

        # plt.plot(np.sort(idk))

    # plt.show()
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
