from my_IDK import my_IDK
import numpy as np
from tqdm import tqdm

def dict_add(d1:dict, d2:dict):
    output = dict()
    for key in d1.keys():
        output[key] = d1[key] + d2[key]
    return output


dl = np.loadtxt("norm1d_drift.csv", delimiter=",")
data = dl[:10000, :-2].reshape(-1, 1)

now = 0
W = 1000
t = 100
output_dict = dict()
output_dict_list = []

for _ in tqdm(range(t)):
    now = 0
    temp_output_dict = dict()
    while now + W <= data.shape[0]:
        # if now % 1000 == 0:
        #     print(now)
        detector = my_IDK(data[now: now + W], 2, 1)
        scores = detector.get_given_score([0, 333, 666, -1])
        for key in scores.keys():
            if key+now not in temp_output_dict.keys():
                temp_output_dict[key+now] = [scores[key]]
            else:
                temp_output_dict[key+now].append(scores[key])
        # for i in range(now, now + W):
        #     if i not in temp_output_dict.keys():
        #         temp_output_dict[i] = np.array([scores[i - now]])
        #     else:
        #         temp_output_dict[i] = np.append(
        #             temp_output_dict[i], np.array([scores[i - now]]), 0)
        now += 1
    output_dict_list.append(temp_output_dict)
for key in output_dict_list[0].keys():
    output_dict[key] = [
        sum(x)/t for x in zip(*[x[key] for x in output_dict_list])]
pass


