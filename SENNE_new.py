import scipy.io as scio
import numpy as np
import ADwCDiDS
from IDK_rewrite import IDK_rewrite
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(7)

path = "SENNE-master\Datasets\Synthetic.mat"
data = scio.loadmat(path)["art4"]

c1: np.ndarray = data[0][0]
c2: np.ndarray = data[1][0]
c3: np.ndarray = data[2][0]
c = np.concatenate((c1, c2, c3))

l1 = 79393
l2 = 79393 + 79480
l3 = 79393 + 79480 + 79455

index = [np.random.choice(l3) for _ in range(1500)]

test_data = c[index]
test_label = list()
for _i in index:
    if _i < l1:
        test_label.append(1)
    elif l1 <= _i < l2:
        test_label.append(2)
    elif l2 <= _i < l3:
        test_label.append(3)


_psi = 20
for epoch in range(1):
    pool = [ADwCDiDS.Model_detector(IDK_rewrite(c1, _psi, _t=100), 1000),
            ADwCDiDS.Model_detector(IDK_rewrite(c2, _psi, _t=100), 1000)]
    # for _th1 in [0.0001 * i for i in range(11)]:
    for _th1 in [0.0020]:
        # for _th2 in [0.003 + 0.0001 * i for i in range(41)]:
        for _th2 in [0.0020]:
            output = []
            for point in test_data:
                if pool[0].get_score(point) <= _th1 \
                        and pool[1].get_score(point) <= _th2:
                    output.append(3)
                elif pool[0].get_score(point) > pool[1].get_score(point):
                    output.append(1)
                elif pool[0].get_score(point) <= pool[1].get_score(point):
                    output.append(2)

            acc = accuracy_score(test_label, output)

            output = np.array(output)
            temp = np.where(output == 1)
            plt.scatter(test_data[temp, 0], test_data[temp, 1], c="g")
            temp = np.where(output == 2)
            plt.scatter(test_data[temp, 0], test_data[temp, 1], c="y")
            temp = np.where(output == 3)
            plt.scatter(test_data[temp, 0], test_data[temp, 1], c="b")

            # for i in range(len(output)):
            #     if output[i] == 1:
            #         plt.scatter(test_data[i, 0], test_data[i, 1], c="g")
            #     elif output[i] == 2:
            #         plt.scatter(test_data[i, 0], test_data[i, 1], c="y")
            #     elif output[i] == 3:
            #         plt.scatter(test_data[i, 0], test_data[i, 1], c="b")
            # plt.show()
            # plt.clf()
            plt.savefig(f"pic_out\{epoch}_acc_{_th1:.5f}_{_th2:.5f}_{acc:.4f}.png")
            plt.clf()
            print(f"{_th1:.5f}", f"{_th2:.5f}", f"{acc:.4f}", "=====")
            exit()
