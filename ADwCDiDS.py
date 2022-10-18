from IDK import IDK
from IDK_rewrite import IDK_rewrite
from BaseLineModel import BaseLineModel
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from generate_uniform_circle import generate_uniform_circle
from sklearn.metrics import roc_curve, auc, accuracy_score



class Model_detector:
    def __init__(self,
                 _model: IDK_rewrite,
                 _memory: int,
                 _k: float) -> None:
        self.model = _model
        self.memory = _memory
        self.k = _k
        self.normal_thershold = self.model.get_average_thershold()
        self.anomaly_thershold = self.k * self.normal_thershold
        self.anomarous_index_list = []
        self.is_birth = False
        self.is_death = False

    def add_anomarous(self,
                      data_index: int) -> None:
        self.anomarous_index_list.append(data_index)
        while self.anomarous_index_list[-1] - self.anomarous_index_list[0] >= self.memory:
            self.anomarous_index_list.pop(0)

    @property
    def get_anomarous_rate(self) -> float:
        return len(self.anomarous_index_list) / self.memory

    def check_anomaly(self,
                      index: int,
                      point: np.ndarray) -> bool:
        if self.model.kappa(point) < self.anomaly_thershold:
            self.add_anomarous(index)
            return True
        return False

    def get_score(self,
                  point: np.ndarray) -> float:
        return self.model.kappa(point)


class ADwCDiDS:
    def __init__(self,
                 _data: np.ndarray,
                 _psi: int,
                 _t: int = 100,
                 _window_size: int = 500,
                 _alpha: float = 0.05,
                 _beta: float = 0.4,
                 _k: float = 0.01) -> None:
        """An anomaly detector in data streaming with concept drift based on IDK

        Args:
        ---
            _data (np.ndarray): the streaming data.
            _psi (int): parameter of IDK, the number of hyperspheres.
            _t (int): parameter of IDK, the ensemble size.
            _window_size (int, optional): the size of window.
            _alpha (float, optional): the birth rate.
            _beta (float, optional): the death rate.
            _k (_type_, optional): thershold ratio of anomalous to normal.
        """
        self.data = _data
        self.psi = _psi
        self.t = _t
        self.window_size = _window_size
        self.alpha = _alpha
        self.beta = _beta
        self.k = _k
        self.data_index = 0
        self.size = self.data.shape[0]
        self.model_bucket: List[Model_detector] = []
        self.init_model()


    def detect(self) -> None:
        output = []
        while not self.is_over:
            temp_data_index, temp_data = self._get_element()
            self.check_all_anomaly(temp_data_index, temp_data)
            output.append(self.get_all_score(temp_data))
            self.birth_all_model()
        return output

    def _get_element(self) -> Tuple[int, np.ndarray]:
        output = self.data_index, self.data[self.data_index]
        self.data_index += 1
        return output

    def init_model(self):
        self.model_bucket.append(
            Model_detector(
                IDK_rewrite(self.data[:int(self.window_size * self.alpha), :],
                    self.psi),
                self.window_size,
                self.k))

    # def init_model(self):
    #     self.model_bucket.append(
    #         Model_detector(
    #             BaseLineModel(
    #                 self.data[:int(self.window_size * self.alpha), :]),
    #             self.window_size,
    #             self.k))

    @property
    def is_over(self):
        return self.size == self.data_index

    def check_1_anomaly(self,
                        model: Model_detector,
                        data_index: int,
                        data: np.ndarray) -> bool:
        return model.check_anomaly(data_index, data)

    def check_all_anomaly(self,
                          data_index: int,
                          data: np.ndarray) -> bool:
        flag = [1, 0]
        for each in self.model_bucket:
            if not self.check_1_anomaly(each, data_index, data):
                flag = [0, 1]
        return flag

    def check_rate(self,
                   model: Model_detector) -> str:
        if model.get_anomarous_rate > self.beta:
            return "D"
        elif model.get_anomarous_rate > self.alpha:
            return "B"
        return "N"

    def birth_1_model(self,
                      model: Model_detector) -> None:
        if self.check_rate(model) == "N":
            return
        elif self.check_rate(model) == "B":
            anomalous = self.data[model.anomarous_index_list]
            if not model.is_birth:
                self.model_bucket.append(Model_detector(
                    # BaseLineModel(anomalous),
                    IDK_rewrite(anomalous, self.psi),
                    self.window_size,
                    self.k))
                model.is_birth = True
            return
        elif self.check_rate(model) == "D":
            self.model_bucket.remove(model)
            return

    def birth_all_model(self):
        for _model in self.model_bucket:
            self.birth_1_model(_model)

    def del_all_point(self,
                      data_index_list: List[int]) -> None:
        for _model in self.model_bucket:
            self.del_1_point(_model, data_index_list)

    def del_1_point(self,
                    model: Model_detector,
                    data_index_list: List[int]) -> None:
        for i in range(len(model.anomarous_index_list)):
            if model.anomarous_index_list[i] in data_index_list:
                model.anomarous_index_list[i] = None
        model.anomarous_index_list = list(
            filter(lambda x: x is not None, model.anomarous_index_list))

    def get_1_score(self,
                    model: Model_detector,
                    data: np.ndarray):
        return model.get_score(data)

    def get_all_score(self,
                      data: np.ndarray):
        score = 1
        for each in self.model_bucket:
            score = min(score, each.get_score(data))
        return score


def main():
    for _psi in [2]:
        roc = 0
        max_accuracy = 0
        for _ in range(10):
            test_data = np.loadtxt("data_set\shuttle.csv", delimiter=",")[:, :9]
            ans = np.loadtxt("data_set\shuttle.csv", delimiter=",")[:, 9].reshape(-1)
            # test_data = np.array(generate_uniform_circle((0,0), 1, 1000) + generate_uniform_circle((10, 10), 1, 1000))
            myx = ADwCDiDS(test_data, _psi, _k=0.001)
            predict = myx.detect()
            ans = list(1+ans) # 异常是2，正常是1

            fpr, tpr, thresholds = roc_curve(ans, predict, pos_label=2)
            accuracy_scores = []
            for thresh in thresholds:
                accuracy_scores.append(accuracy_score(ans,
                                                    [1 if m > thresh else 2 for m in predict]))

            accuracies = np.array(accuracy_scores)
            max_accuracy += accuracies.max()
            # max_accuracy_threshold =  thresholds[accuracies.argmax()]
            # print(max_accuracy)
            # print(max_accuracy_threshold)



            # for i, _data in enumerate(test_data):
            #     if predict[i] < thersholds[-5]:
            #         plt.scatter(_data[0], _data[1], c="r")
            #     else:
            #         plt.scatter(_data[0], _data[1], c="b")

            # plt.show()

            # for i, value in enumerate(thersholds):
            #     print("%f %f %f" % (fpr[i], tpr[i], value))

            roc_auc = auc(fpr, tpr)
            roc += roc_auc
        roc /= 10
        max_accuracy /= 10
        print(f"{_psi}\t{roc=:.5f}\t{max_accuracy=:.5f}")

            # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

            # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
            # plt.ylim([-0.05, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
            # plt.title('ROC Curve')
            # plt.legend(loc="lower right")
            # plt.show()

if __name__ == "__main__":
    main()
