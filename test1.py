from SENNE import SENNE
import random
import matplotlib.pyplot as plt
import math

c1, r1 = (0, 0), 1
c2, r2 = (2, 2), 1
c3, r3 = (5, 5), 2
c4, r4 = (0, 5), 2

DS = list()
class1 = list()
class2 = list()
class3 = list()
class4 = list()
for _ in range(100):
    r = random.random() * r1
    theta = random.random() * 2 * math.pi
    class1.append((c1[0] + r * math.cos(theta), c1[1] + r * math.sin(theta)))
for _ in range(100):
    r = random.random() * r2
    theta = random.random() * 2 * math.pi
    class2.append((c2[0] + r * math.cos(theta), c2[1] + r * math.sin(theta)))
for _ in range(100):
    r = random.random() * r3
    theta = random.random() * 2 * math.pi
    class3.append((c3[0] + r * math.cos(theta), c3[1] + r * math.sin(theta)))
for _ in range(100):
    r = random.random() * r4
    theta = random.random() * 2 * math.pi
    class4.append((c4[0] + r * math.cos(theta), c4[1] + r * math.sin(theta)))


DS.append(class1)
DS.append(class2)
my_class = [class1 + class2 + class3 + class4]
# print(class3)


S = 0
myx = SENNE(my_class, 25, 2, 0.8)
# check_point = (1.35, 1.35)
# print(f"0类N={myx.get_Ni(check_point, 0)}\t 1类N={myx.get_Ni(check_point, 1)}")
# print(f"0类P={myx.get_Pi(check_point, 0)}\t 1类P={myx.get_Pi(check_point, 1)}")
# print(myx.f(check_point))
classified = myx.classify()
print(len(classified))
myx2 = SENNE(classified, 25, 2, 0.8)
# S += len(output)
# if len(output) != 4:
#     continue
# color = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
#         (0, 1, 1), (1, 0, 1), (1, 1, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
# for i in range(len(output)):
#     for each in output[i]:
#         plt.scatter(each[0], each[1], color=color[i])
# plt.show()
# print(f"psi={25}\tt={2}\t{S/100}\n")