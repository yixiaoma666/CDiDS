from collections import Counter

import numpy as np

from IDK_rewrite import IK_anne_fm_sparse, IK_inne_fm_sparse, IK_inne_fm


def ANNE_iid_conduct(time_series_train,time_series_test,psi):
    X=time_series_train
    n,m=len(time_series_train),len(time_series_test)
    ts_len = len(time_series_train[0])
    train_num=ts_len*n
    if psi>=train_num:
        return None
    if m>0:
        X=np.concatenate((time_series_train,time_series_test))
    X=X.reshape(-1,1)
    idkfm_train_and_test = np.zeros((n + m, 100 * psi), dtype=float)
    onepointfm_matrix = IK_anne_fm_sparse(X, t=100, psi=psi, train_num=train_num).reshape((n + m, -1))
    counter_list = [Counter(it) for it in onepointfm_matrix]
    key_list = [list(counter.keys()) for counter in counter_list]
    val_list = [list(counter.values()) for counter in counter_list]
    for i in range(n + m):
        idkfm_train_and_test[i][key_list[i]] += val_list[i]
    idkfm_train_and_test /= ts_len
    # tem=np.linalg.norm(idkfm_train_and_test,axis=1)
    # idkfm_train_and_test/=tem[:,None]
    return idkfm_train_and_test

def INNE_iid_conduct(time_series_train,time_series_test,psi):
    X=time_series_train
    n,m=len(time_series_train),len(time_series_test)
    ts_len = len(time_series_train[0])
    train_num=ts_len*n
    if psi>=train_num:
        return None
    if m>0:
        X=np.concatenate((time_series_train,time_series_test))
    X=X.reshape(-1,1)
    idkfm_train_and_test = np.zeros((n + m, 100 * psi), dtype=float)
    onepointfm_matrix = IK_inne_fm_sparse(X, t=100, psi=psi, train_num=train_num).reshape((n + m, -1))
    counter_list = [Counter(it) for it in onepointfm_matrix]
    for counter in counter_list:
        if -1 in counter:
            del counter[-1]
    key_list = [list(counter.keys()) for counter in counter_list]
    val_list = [list(counter.values()) for counter in counter_list]
    for i in range(n + m):
        idkfm_train_and_test[i][key_list[i]] += val_list[i]
    idkfm_train_and_test /= ts_len
    # tem=np.linalg.norm(idkfm_train_and_test,axis=1)
    # idkfm_train_and_test/=tem[:,None]
    return idkfm_train_and_test

def INNE_sliding_iid_conduct(time_series_train,time_series_test,k,psi):
    X = time_series_train
    n, m = len(time_series_train), len(time_series_test)
    ts_len = len(time_series_train[0])
    train_num = ts_len * n
    width=k
    if psi >= train_num:
        return None
    if m > 0:
        X = np.concatenate((time_series_train, time_series_test))
    X = X.reshape(-1, 1)
    idkfm_train_and_test = np.zeros((n + m, 100 * psi), dtype=float)
    onepointfm_matrix = IK_inne_fm_sparse(X, t=100, psi=psi, train_num=train_num).reshape((n + m, ts_len,-1))
    for i,point_fm_list in enumerate(onepointfm_matrix):
        point_fm_list = np.insert(point_fm_list, 0, 0, axis=0)
        cumsum = np.cumsum(point_fm_list, axis=0)
        idkfm_train_and_test[i] = (cumsum[width:] - cumsum[:-width]) / float(width)
    # subsequence_fm_list = np.zeros((window_num, t * psi1))
    # subsequence_fm_list[0] = np.sum(point_fm_list[:width, :], axis=0)
    # for i in range(1, window_num):
    #     subsequence_fm_list[i] = subsequence_fm_list[i - 1] - point_fm_list[i - 1] + point_fm_list[i + width - 1]
    #
    # subsequence_fm_list /= width


    pass



