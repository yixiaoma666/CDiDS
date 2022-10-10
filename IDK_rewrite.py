import random

import numpy as np



def IK_inne_fm(X,psi,t=100):

    onepoint_matrix = np.zeros((X.shape[0], (int)(t*psi)), dtype=int)
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(len(X))]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :]

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

        # tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        # sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)
        sample2sample=point2sample[sample_list,:]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = 99999999
        radius_list = np.min(sample2sample, axis=1)  # 每行的最小值形成一个行向量

        min_point2sample_index=np.argmin(point2sample, axis=1)
        min_dist_point2sample = min_point2sample_index+time*psi
        point2sample_value = point2sample[range(len(onepoint_matrix)), min_point2sample_index]
        ind=point2sample_value < radius_list[min_point2sample_index]
        onepoint_matrix[ind,min_dist_point2sample[ind]]=1
    return onepoint_matrix

def IDK(X,psi,t=100):
    point_fm_list=IK_inne_fm(X=X,psi=psi,t=t)
    feature_mean_map=np.mean(point_fm_list,axis=0)
    return np.dot(point_fm_list,feature_mean_map)/t


def IK_inne_fm_sparse(X,psi,train_num,t=100):
    #return n*t

    onepoint_matrix = np.full((X.shape[0], (int)(t)), -1,dtype=int)
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(train_num)]  # [0, 1, 2, 3]s
        sample_list = random.sample(sample_list, sample_num)  # [1, 2]
        sample = X[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
        # distance between sample
        # tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        # sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)
        #
        # sample2sample[sample2sample < 1e-9] = 99999999;

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

        # tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        # sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)
        sample2sample=point2sample[sample_list,:]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = 99999999
        radius_list = np.min(sample2sample, axis=1)  # 每行的最小值形成一个行向量

        min_point2sample_index=np.argmin(point2sample, axis=1)
        min_dist_point2sample = min_point2sample_index+time*psi
        point2sample_value = point2sample[range(len(onepoint_matrix)), min_point2sample_index]
        ind=point2sample_value < radius_list[min_point2sample_index]
        onepoint_matrix[ind,time]=min_dist_point2sample[ind]


    return onepoint_matrix




def IK_anne_fm_sparse(X,psi,train_num=None,t=100):
    if train_num is None:
        train_num=len(X)
    onepoint_matrix = np.zeros((X.shape[0], (int)(t)), dtype=int)
    for time in range(t):
        sample_num = psi  #
        sample_list = [p for p in range(train_num)]  # [0, 1, 2, 3]s
        sample_list = random.sample(sample_list, sample_num)  # [1, 2]
        sample = X[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
        # distance between sample
        # tem = np.dot(np.square(sample), np.ones(sample.T.shape))
        # sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)
        #
        # sample2sample[sample2sample < 1e-9] = 99999999;

        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi
        min_dist_point2sample = np.argmin(point2sample, axis=1)+time*psi
        onepoint_matrix[:,time]=min_dist_point2sample

    return onepoint_matrix



