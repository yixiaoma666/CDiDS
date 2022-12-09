from psKC import psKC
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import clustering_metric as cm
from IKBC import ik_bc

# path = r"SENNE-master\Datasets\Synthetic.mat"
# all_data = scio.loadmat(path)

# data = all_data["art4"][:, 0]
# label = all_data["art4"][:, 1]

# path = r"my_data\nine_direction.mat"
# all_data = scio.loadmat(path)["nine_direction"]

# data = all_data[:, 0]
# label = all_data[:, 1]


# test_data_0 = data[0][:1000]
# test_data_1 = data[1][:1000]
# test_data_2 = data[2][:1000]
# test_data_3 = data[3][:1000]


# test_data = np.concatenate(
#     (test_data_0, test_data_1, test_data_2, test_data_3), axis=0)
# test_label = np.concatenate((np.tile([0], 1000), np.tile(
#     [1], 1000), np.tile([2], 1000), np.tile([3], 1000)))

all_data = np.loadtxt(r"my_data\nine_direction.csv", delimiter=",")
test_data = all_data[:, :2]
test_label = all_data[:, 2].reshape(-1).tolist()

for _ in range(20):
    best_tau = 0
    best_ami = 0
    best_predict = []
    for tau in [0.05+i*0.05 for i in range(8, 20)]:
        myx = ik_bc(test_data, n_cluster=9, psi=4, t=100, tau=tau)
        ami = adjusted_mutual_info_score(test_label, myx)
        if ami > best_ami:
            best_ami = ami
            best_tau = tau
            best_predict = myx
    print(best_tau, best_ami)

    # myx = psKC(test_data)
    plt.scatter(test_data[np.where(best_predict == 0), 0],
                test_data[np.where(best_predict == 0), 1], c="b")
    plt.scatter(test_data[np.where(best_predict == 1), 0],
                test_data[np.where(best_predict == 1), 1], c="g")
    plt.scatter(test_data[np.where(best_predict == 2), 0],
                test_data[np.where(best_predict == 2), 1], c="r")
    plt.scatter(test_data[np.where(best_predict == 3), 0],
                test_data[np.where(best_predict == 3), 1], c="y")
    plt.scatter(test_data[np.where(best_predict == 4), 0],
                test_data[np.where(best_predict == 4), 1], c="k")
    plt.scatter(test_data[np.where(best_predict == 5), 0],
                test_data[np.where(best_predict == 5), 1], c="c")

    plt.show()
    # print(adjusted_mutual_info_score(test_label, myx))
    # pass
