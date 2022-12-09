import numpy as np
from IKBC import ik_bc
from sklearn.metrics import adjusted_mutual_info_score as ami
from IDK_rewrite import IDK_rewrite
import matplotlib.pyplot as plt
import time
import pandas as pd


class ADiDSuCD:
    def __init__(self, _data: np.ndarray, _label: np.ndarray, _psi: int, _t: int = 100, _init_window=500, _buffer_size=300) -> None:
        self.data = _data
        self.label = _label
        self.psi = _psi
        self.t = _t
        self.init_window = _init_window
        self.buffer_size = _buffer_size

        self.current_index = 0
        self.buffer = []
        self.detectors = []
        self.predict_label = np.zeros(self.label.shape)-1

        self.thre_pool = [[0.0004], [0.004], [0.008], [0.008]]

        self.init_detector()
        self.detect()

    def init_detector(self):
        init_index = []
        for _ in range(self.init_window):
            init_index.append(self._get_element())
        init_data = self.data[init_index]
        init_label = self.label[init_index]  # 不可用
        init_n_cluster = 2
        init_predict_label = ik_bc(
            node_features=init_data, n_cluster=init_n_cluster, psi=2, t=100, tau=0.5)  # Searched, seed=514
        classes = []
        for i in range(init_n_cluster):
            classes.append((np.array(init_index)[init_predict_label == i], np.array(
                init_label)[init_predict_label == i]))
        for i, cla in enumerate(classes):
            _idk = IDK_rewrite(self.data[cla[0]], self.psi, self.t)
            _idk_score = _idk.IDK()
            best_thre = self.search_thre(_idk_score, cla[1])
            best_thre = self.thre_pool.pop(0)[0]  # set thre
            self.predict_label[cla[0][np.where(_idk_score < best_thre)]] = 0
            self.predict_label[cla[0]
                               [np.where(_idk_score >= best_thre)]] = i + 1
            self.buffer += cla[0][np.where(_idk_score < best_thre)].tolist()
            self.detectors.append((_idk, best_thre))
            self.plot_idk(_idk, [best_thre], self.data[cla[0]])

            

    def _get_element(self):
        self.current_index += 1
        return self.current_index - 1

    @staticmethod
    def search_thre(idk_score: np.ndarray, true_label: np.ndarray):
        best_thre = idk_score[0]
        max_ami = 0
        for thre in idk_score:
            ano_array = np.zeros(idk_score.shape)
            ano_array[idk_score < thre] = 1
            if ami(true_label.tolist(), ano_array.tolist()) > max_ami:
                best_thre = thre
                max_ami = ami(true_label.tolist(), ano_array.tolist())
        return best_thre

    @property
    def is_over(self):
        return self.current_index == self.label.shape[0]

    def detect(self):
        while not self.is_over:
            # while self.current_index < 1000:
            temp_index = self._get_element()
            temp_data = self.data[temp_index]
            temp_label = self.label[temp_index]
            class_score = [each[0].kappa(temp_data)/each[1]
                           for each in self.detectors]
            class_score_array = np.array(class_score)
            max_score_num = class_score_array.argmax()
            if class_score_array[max_score_num] < 1:  # All detectors say anomaly
                self.buffer.append(temp_index)
                self.predict_label[temp_index] = 0
            else:
                self.predict_label[temp_index] = max_score_num + 1
            if self.is_buffer_full:
                self.emerge_class()
            pass

    def emerge_class(self):
        temp_index = np.array(self.buffer)
        temp_data = self.data[temp_index]
        temp_label = self.label[temp_index]
        temp_idk = IDK_rewrite(temp_data, self.psi, self.t)
        temp_idk_score = temp_idk.IDK()
        best_thre = self.search_thre(temp_idk_score, temp_label)
        best_thre = self.thre_pool.pop(0)  # set thre
        self.plot_idk(temp_idk, best_thre, temp_data)  # TODO
        best_thre = best_thre[0]  # TODO
        self.buffer = temp_index[temp_idk_score < best_thre].tolist()
        self.predict_label[temp_index[temp_idk_score < best_thre]] = 0
        self.predict_label[temp_index[temp_idk_score >= best_thre]] = len(
            self.detectors)
        self.detectors.append((temp_idk, best_thre))
        

        pass

    @property
    def is_buffer_full(self):
        return len(self.buffer) == self.buffer_size

    def plot_idk(self, _idk, _thres, _data):
        x = np.arange(-15, 15, 0.1)
        y = np.arange(-15, 15, 0.1)
        X, Y = np.meshgrid(x, y)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        grid = np.concatenate((X, Y), axis=1)
        for _thre in _thres:
            grid_score = []
            for each in grid:
                grid_score.append(_idk.kappa(each))
            # grid_score = np.array(grid_score).reshape(-1, 1)
            grid_score = np.array(grid_score)
            # data_base = np.concatenate((grid, grid_score), 1)
            # df = pd.DataFrame(data_base)
            # df.plot.scatter(0, 1, c=2, colormap='Greys')
            # plt.scatter(_data[:, 0], _data[:, 1], c=(0, 0, 0, 0.5))
            # plt.show()
            # pass
            # plt.scatter(grid[:, 0], grid[:, 1])
            plt.scatter(grid[grid_score < _thre][:, 0],
                        grid[grid_score < _thre][:, 1], c="r")
            plt.scatter(grid[grid_score >= _thre][:, 0],
                        grid[grid_score >= _thre][:, 1], c="b")
            # plt.scatter(self.data[:, 0], self.data[:, 1], c="y")
            plt.scatter(_data[:, 0], _data[:, 1], c="y")
            plt.suptitle(f"{_thre}")
            # plt.savefig(f"pic_out/{time.time()}.png")
            plt.show()

        pass


if __name__ == "__main__":
    np.random.seed(515)
    S = 0
    # for _ in range(10):
    alldata = np.loadtxt("my_data\stream9direction.csv", delimiter=",")
    data = alldata[:, :2]
    label = alldata[:, 2]

    myx = ADiDSuCD(data, label, 4, _t=1000)
    predict_label = myx.predict_label
    plt.scatter(data[predict_label==0][:, 0], data[predict_label==0][:, 1], c="r")
    plt.scatter(data[predict_label==1][:, 0], data[predict_label==1][:, 1], c="b")
    plt.scatter(data[predict_label==2][:, 0], data[predict_label==2][:, 1], c="g")
    plt.scatter(data[predict_label==3][:, 0], data[predict_label==3][:, 1], c="k")
    # plt.scatter(data[predict_label==4][:, 0], data[predict_label==4][:, 1], c="y")
    # plt.scatter(data[predict_label==5][:, 0], data[predict_label==5][:, 1], c="c")
    plt.show()
    print(ami(label.tolist(), predict_label.tolist()))
    # S += ami(label.tolist(), predict_label.tolist())
    # print(S/10)
