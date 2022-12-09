from IDK_rewrite import IDK_rewrite
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc
import scipy.io as scio


class Model_detector:
    def __init__(self,
                 _model: IDK_rewrite,
                # _model: iForest,
                 _memory: int) -> None:
        self.model = _model
        self.memory = _memory
        # self.normal_threshold = self.model.get_average_threshold()
        # self.anomaly_threshold = self.k * self.normal_threshold
        self.anomaly_threshold = self.model.get_min_var_threshold()
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
        if self.get_score(point) <= self.anomaly_threshold:
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
                 _beta: float = 0.5,
                 _init_window=500) -> None:
        """An anomaly detector in data streaming with concept drift based on IDK

        Args:
        ---
            _data (np.ndarray): the streaming data.
            _psi (int): parameter of IDK, the number of hyperspheres.
            _t (int): parameter of IDK, the ensemble size.
            _window_size (int, optional): the size of window.
            _alpha (float, optional): the birth rate.
            _beta (float, optional): the death rate.
            _k (_type_, optional): threshold ratio of anomalous to normal.
        """
        self.data = _data
        self.psi = _psi
        self.t = _t
        self.window_size = _window_size
        self.alpha = _alpha
        self.beta = _beta
        self.init_window_size = _init_window
        self.data_index = 0
        self.size = self.data.shape[0]
        self.model_bucket: List[Model_detector] = []
        self.init_model()

    def detect(self) -> None:
        output = []
        while not self.is_over:
            temp_data_index, temp_data = self._get_element()
            self.check_all_anomaly(temp_data_index, temp_data)
            # output.append(predict[0])
            output.append(self.get_all_score(temp_data))
            self.birth_all_model()
            # ma = 0
            # mi = 1000
            # for model in self.model_bucket:
            #     ma = max(ma, len(model.anomarous_index_list))
            #     mi = min(mi, len(model.anomarous_index_list))
            # with open("new.csv", mode="a+") as f:
            #     f.write(str(len(self.model_bucket))+","+str(ma)+","+str(mi)+"\n")
            # print(len(self.model_bucket))
        return output

    def _get_element(self) -> Tuple[int, np.ndarray]:
        output = self.data_index, self.data[self.data_index]
        self.data_index += 1
        return output

    def init_model(self):
        self.model_bucket.append(
            Model_detector(
                IDK_rewrite(self.data[:int(self.init_window_size), :],
                            self.psi),
                # iForest(self.data[:int(self.init_window_size), :]),
                self.window_size))

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
                    # iForest(anomalous),
                    self.window_size))
                # print("B")
                # print(model.anomarous_index_list)
                model.is_birth = True
            return
        elif self.check_rate(model) == "D":
            # print("D")
            # print(model.anomarous_index_list)
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
        score = 0
        for each in self.model_bucket:
            score = max(score, each.get_score(data))
        return score


def main():
    for _psi in [2,4,8,16,32,64,128,256]:
        roc = 0
        max_f1 = 0
        a = 0.10
        b = 0.60
        for _ in range(10):
            test_data = np.loadtxt(
                "data_set\shuttle.csv", delimiter=",")[:, :9]
            ans = np.loadtxt("data_set\shuttle.csv", delimiter=",")[:, 9].reshape(-1)
            myx = ADwCDiDS(test_data, _psi, _t=500, _window_size=256,
                           _alpha=a, _beta=b, _init_window=256)
            predict = myx.detect()
            # exit()
            predict = 1-np.array(predict)
            predict_1 = predict[0:1000]
            ans_1 = ans[0:1000]
            predict_2 = predict[1000:2000]
            ans_2 = ans[1000:2000]
            predict_3 = predict[2000:3000]
            ans_3 = ans[2000:3000]
            
            for tp, ta in [(predict, ans)]:
                fpr, tpr, thresholds = roc_curve(ta, tp)
                f1_scores = []
                for thresh in thresholds:
                    f1_scores.append(
                        f1_score(ta, [1 if m > thresh else 0 for m in tp]))

                f1s = np.array(f1_scores)
                
                roc_auc = auc(fpr, tpr)
                print(roc_auc, end=",")
                print(f1s.max(), end=",")
            print("")
            # f1_scores = np.array(f1_scores)
            # max_f1 += f1_score(ans_1, predict_1)
            # max_accuracy_threshold =  thresholds[accuracies.argmax()]
            # print(max_accuracy)
            # print(max_accuracy_threshold)

            # for i, _data in enumerate(test_data):
            #     if predict[i] <= thresholds[f1_scores.argmax()]:
            #         plt.scatter(_data[0], _data[1], c="b")
            # for i, _data in enumerate(test_data):
            #     if predict[i] > thresholds[f1_scores.argmax()]:
            #         plt.scatter(_data[0], _data[1], c="r")

            # plt.show()
            # exit()
            # for i, value in enumerate(thresholds):
            #     print("%f %f %f" % (fpr[i], tpr[i], value))
            max_f1 += f1s.max()
            roc += roc_auc
        roc /= 10
        max_f1 /= 10
        print(f"{a=:.2f}\t{b=:.2f}\t{_psi=}\t{roc=:.5f}\t{max_f1=:.5f}")

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
