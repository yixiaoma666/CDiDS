import scipy.io as scio
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_mutual_info_score


def psKC(data) \
    -> np.ndarray:

    # path = r"data_set\4C.mat"
    # all_data = scio.loadmat(path)

    # data: np.ndarray = all_data["data"]
    # label = all_data["class"]

    psi = 50
    t = 500
    v = 0.1
    tau = 0.000001

    sn = data.shape[0]
    n, d = data.shape
    IDX = np.ndarray((0, sn))

    for i in range(t):
        subIndex = np.random.choice(sn, psi, replace=False)
        tdata = data[subIndex, :]

        dis = cdist(tdata, data)
        centerIdx = np.argmin(dis, axis=0).reshape(1, -1)
        IDX = np.concatenate((IDX, centerIdx+i*psi), axis=0)

    IDR = np.tile(range(n), (t, 1))
    V = IDR - IDR + 1
    ndata = coo_matrix(
        (V.reshape(-1), (IDR.reshape(-1), IDX.reshape(-1))), shape=(n, t*psi))
    ndata = ndata.todense()

    k = -1  # class index
    C = []
    CD = np.array(range(data.shape[0])).reshape(-1, 1)
    D = np.concatenate((CD, ndata), axis=1)

    while D.shape[0] > 1:
        k += 1

        x = np.argmax(np.matmul(D[:, 1:], np.sum(
            D[:, 1:], 0).T) / t / D.shape[0])

        C.append(D[x, :].reshape(1, -1))
        D = np.delete(D, x, 0)

        _temp = np.matmul(D[:, 1:], C[k][:, 1:].T) / t / D.shape[0]
        z = np.max(_temp)
        y = np.argmax(_temp)

        C[k] = np.concatenate((C[k], D[y, :].reshape(1, -1)), axis=0)
        D = np.delete(D, y, 0)

        r = (1 - v) * z
        if r <= tau:
            print("break")
            break

        sizeC = 2

        while r > tau:
            S = (np.matmul(D[:, 1:], np.sum(C[k][:, 1:],
                                            axis=0).T) / t / sizeC).reshape(-1, 1)
            x = np.where(S > r)[0]
            C[k] = np.concatenate((C[k], D[x, :]), axis=0)
            D = np.delete(D, x, 0)
            r = (1 - v) * r
            sizeC += x.shape[0]
        # print(D.shape[0])

    Tclass = np.zeros((data.shape[0], 1))

    Centre = []
    for i in range(len(C)):
        Tclass[C[i][:, 0]] = i
        Centre.append(C[i][0, 0])
    Centre = np.array(Centre)

    return Tclass.reshape(-1)

    # AMI = adjusted_mutual_info_score(label.reshape(-1), Tclass.reshape(-1)+1)
