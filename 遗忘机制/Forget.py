import random

forget_list = []
L = 10
p = L / (1 + L)
S = 0

# for i in range(10000):
#     if i < L:
#         forget_list.append(i)
#         continue
#     else:
#         min_num = min(forget_list)
#         for m in forget_list:
#             if (i-m == 1 and random.random() < p**(L-1)) or \
#                 (1 < i-m <= 10 and random.random() < (1-p)*p**(10+m-i)) or \
#                     (i-m > 10 and random.random() < (1-p)):
#                 forget_list.remove(m)
#         forget_list.append(i)
#     S += len(forget_list)

for i in range(100):
    if i < L:
        forget_list.append(i)
    else:
        for m in forget_list:
            if random.random() < 1 / (L-1):
                forget_list.remove(m)
        forget_list.append(i)
        S += len(forget_list)
    print(forget_list)


S /= (100-10)

print(S)